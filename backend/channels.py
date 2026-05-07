"""Channel integrations — Telegram, etc.

Manages external messaging channels that forward messages to the agent
and relay answers back. Channel configs are stored in knowledge/channels.json.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHANNELS_PATH = ROOT / "knowledge" / "channels.json"


class _ConflictNoiseFilter(logging.Filter):
    """Collapse python-telegram-bot's `Conflict: terminated by other
    getUpdates request` storms into a single warning line.

    Background: a uvicorn `--reload` race spawns a fresh poller before
    the old child finishes its in-flight long-poll. Telegram
    cancels the old `getUpdates` with `Conflict`; the lib's
    `network_retry_loop` then logs a 30-line stack trace at ERROR
    level on every retry until the situation resolves (usually
    seconds). The trace is alarming but harmless.

    This filter:
      - Drops the stack trace (sets `exc_info` and `exc_text` to None).
      - Throttles repeats: emits one short WARNING per minute even
        if the lib retries every few seconds.
      - Lets all non-Conflict records through unchanged.
    """

    THROTTLE_SECONDS = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._last_log_at: float = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage() if record.args else (record.msg or "")
        if "Conflict" not in msg and (
            not record.exc_info or "Conflict" not in str(record.exc_info[1])
        ):
            return True
        now = time.time()
        if now - self._last_log_at < self.THROTTLE_SECONDS:
            return False
        self._last_log_at = now
        record.levelno = logging.WARNING
        record.levelname = "WARNING"
        record.msg = (
            "Telegram poll preempted by another getUpdates consumer "
            "(Conflict). Usually a dev-reload race; the lib retries. "
            "If it persists, check for a duplicate backend with the "
            "same TELEGRAM token."
        )
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        return True


# Install once, on the python-telegram-bot Updater logger that emits
# the noisy traceback. Module-level so it survives bot restarts within
# the same process.
_TG_UPDATER_LOG = logging.getLogger("telegram.ext.Updater")
if not any(isinstance(f, _ConflictNoiseFilter) for f in _TG_UPDATER_LOG.filters):
    _TG_UPDATER_LOG.addFilter(_ConflictNoiseFilter())


# --------------- storage ---------------

def _load_channels() -> list[dict]:
    if CHANNELS_PATH.exists():
        try:
            data = json.loads(CHANNELS_PATH.read_text(encoding="utf-8"))
            return data.get("channels", [])
        except Exception:
            return []
    return []


def _save_channels(channels: list[dict]) -> None:
    CHANNELS_PATH.write_text(
        json.dumps({"channels": channels}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_channels() -> list[dict]:
    return _load_channels()


def get_channel(channel_id: str) -> Optional[dict]:
    for ch in _load_channels():
        if ch["id"] == channel_id:
            return ch
    return None


def save_channel(channel: dict) -> dict:
    """Create or update a channel config."""
    channels = _load_channels()

    existing = None
    for i, ch in enumerate(channels):
        if ch["id"] == channel["id"]:
            existing = i
            break

    channel.setdefault("created", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    channel["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if existing is not None:
        channels[existing] = channel
    else:
        channels.append(channel)

    _save_channels(channels)
    return channel


def delete_channel(channel_id: str) -> bool:
    channels = _load_channels()
    new = [ch for ch in channels if ch["id"] != channel_id]
    if len(new) == len(channels):
        return False
    _save_channels(new)
    return True


# --------------- Telegram bot ---------------


class _TgProgressStream:
    """Streams agent progress events into a single Telegram message in
    near-real-time, by repeatedly editing one placeholder.

    Why a stream and not one message per event:
      Telegram rate-limits edits to one chat at ~1/sec; sending one new
      message per event would also clutter the chat. Editing one
      placeholder gives a "live" feel without spamming.

    Threading:
      The agent runs sync inside `loop.run_in_executor(...)` so the
      event loop stays responsive. Progress callbacks fire from the
      executor thread; this class bridges with
      `asyncio.run_coroutine_threadsafe(..., loop)` so the actual
      `edit_message_text` calls execute on the bot's loop.

    Throttling:
      An edit happens at most once per `EDIT_INTERVAL_SEC`. If another
      event arrives mid-throttle, a single deferred edit is scheduled
      to flush the latest snapshot — so the user always sees the most
      recent state, but we don't burn rate-limit budget.

    Buffering:
      Only the last `MAX_LINES` events are rendered, so a long trace
      doesn't push past Telegram's 4096-char message cap.
    """

    EDIT_INTERVAL_SEC = 1.2
    MAX_LINES = 30
    MAX_LINE_LEN = 180

    def __init__(self, bot: Any, chat_id: int, message_id: int, loop: Any):
        self.bot = bot
        self.chat_id = chat_id
        self.message_id = message_id
        self.loop = loop
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._last_edit = 0.0
        self._pending = False
        self._closed = False

    def push(self, event: str, message: str) -> None:
        """Sync entry point — called from the agent thread."""
        if self._closed:
            return
        line = f"{event}: {message}"
        if len(line) > self.MAX_LINE_LEN:
            line = line[: self.MAX_LINE_LEN - 1] + "…"
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self.MAX_LINES:
                self._lines = self._lines[-self.MAX_LINES :]
        self._schedule_edit()

    def _schedule_edit(self) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self._maybe_edit(), self.loop)
        except Exception:
            pass

    def _render(self) -> str:
        with self._lock:
            body = "\n".join(self._lines[-self.MAX_LINES :])
        text = "🧠 Thinking…\n" + body
        if len(text) > 3900:
            text = text[:3895] + "…"
        return text

    async def _maybe_edit(self) -> None:
        now = time.time()
        wait = self.EDIT_INTERVAL_SEC - (now - self._last_edit)
        if wait > 0:
            # Coalesce: if a deferred flush is already pending, drop this one.
            if self._pending:
                return
            self._pending = True
            try:
                await asyncio.sleep(wait)
            finally:
                self._pending = False
        self._last_edit = time.time()
        await self._edit(self._render())

    async def _edit(self, text: str) -> None:
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self.message_id,
                text=text,
            )
        except Exception:
            # Swallow — Telegram occasionally returns "message not modified"
            # or 429 on bursts; both are non-fatal for a progress stream.
            pass

    async def finalize(self, summary: str) -> None:
        """Replace placeholder with the final compact summary."""
        self._closed = True
        await self._edit(summary[:4000])


def _format_trace_footer(result: Any) -> str:
    """Compact thinking + tools footer for Telegram replies.

    Telegram has no `<details>` collapsible UI, so we keep this tight:
      - 🧠 Thinking line: stage names from the trace, joined by →,
        plus total step count and elapsed seconds. Tool events are
        excluded so the line stays readable.
      - 🔧 Tools line: counts of each distinct tool used, e.g.
        `read_file(2), web_search(1)`. Omitted when no tools ran.

    Returns "" when there's no trace at all (chat-fast-path replies).
    """
    trace = getattr(result, "thinking_trace", None) or []
    if not trace:
        return ""
    # Stage chain — drop tool events, drop spammy `found:` repeats.
    seen: set[str] = set()
    stages: list[str] = []
    for s in trace:
        ev = s.event or ""
        if ev.startswith("tool"):
            continue
        if ev in seen:
            continue
        seen.add(ev)
        stages.append(ev)
    # Tool tally
    tool_counts: dict[str, int] = {}
    last_ts = 0.0
    for s in trace:
        last_ts = max(last_ts, s.ts or 0.0)
        if s.event in ("tool", "tool_error") and s.tool_call:
            tool_counts[s.tool_call.name] = tool_counts.get(s.tool_call.name, 0) + 1
    lines: list[str] = []
    if stages:
        chain = " → ".join(stages[:8])
        if len(stages) > 8:
            chain += f" → … (+{len(stages) - 8})"
        lines.append(f"🧠 Thinking: {chain}  ({len(trace)} steps · {last_ts:.1f}s)")
    if tool_counts:
        tools = ", ".join(f"{n}({c})" for n, c in sorted(tool_counts.items()))
        lines.append(f"🔧 Tools: {tools}")
    return "\n".join(lines)


class TelegramBot:
    """Runs a Telegram bot that forwards messages to the agent."""

    def __init__(self, token: str, channel_id: str, allowed_users: list[str] | None = None):
        self.token = token
        self.channel_id = channel_id
        self.allowed_users = allowed_users or []
        self._thread: threading.Thread | None = None
        self._running = False
        self._app = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Track the most recent chat_id we've received a message from
        # on this bot. Used by send_text() so the WebUI's
        # "compose-as-telegram" mode can deliver the agent's reply
        # back to the user's TG without us having to enumerate chats.
        self._last_chat_id: int | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"tg-bot-{self.channel_id}")
        self._thread.start()
        log.info("Telegram bot %s starting...", self.channel_id)

    def stop(self) -> None:
        self._running = False
        if self._app and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(self._app.stop(), self._loop)
                asyncio.run_coroutine_threadsafe(self._app.shutdown(), self._loop)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        log.info("Telegram bot %s stopped", self.channel_id)

    def send_text(self, text: str, *, chat_id: int | None = None) -> bool:
        """Round E: deliver `text` to a Telegram chat without going
        through agent.run. Used by the WebUI's "compose-as-telegram"
        mode — the agent processes the message in WebUI but the
        finished answer also lands in the user's Telegram bubble so
        the conversation thread stays continuous in TG.

        `chat_id` defaults to the most-recent chat we've received a
        message from (`_last_chat_id`); pass an explicit id when
        you want to target a specific user. Returns True on
        successful schedule (delivery is async — failures surface
        as warnings in the bot log).

        Splits long bodies at Telegram's 4096-char limit so the call
        doesn't 400 on a long agent answer.
        """
        target = chat_id if chat_id is not None else self._last_chat_id
        if target is None or not self._running or not self._app or not self._loop:
            return False
        body = (text or "").strip()
        if not body:
            return False
        LIMIT = 4000
        chunks: list[str] = []
        i = 0
        while i < len(body):
            chunks.append(body[i : i + LIMIT])
            i += LIMIT

        async def _send_all() -> None:
            for chunk in chunks:
                try:
                    await self._app.bot.send_message(chat_id=target, text=chunk)
                except Exception as e:
                    log.warning("TG send_text failed on bot %s chat %s: %s",
                                self.channel_id, target, e)
                    return
        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
            return True
        except Exception as e:
            log.warning("TG send_text scheduling failed on %s: %s", self.channel_id, e)
            return False

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            app = ApplicationBuilder().token(self.token).build()
            self._app = app

            allowed = self.allowed_users

            async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                user = update.effective_user
                chat_id = update.effective_chat.id
                await update.message.reply_text(
                    f"Connected! Chat ID: {chat_id}\n"
                    f"User: {user.username or user.first_name}\n"
                    "Send me any message and the agent will respond."
                )

            async def _gather_attachments(update: "Update") -> list[str]:
                """Pull photos / voice / documents off the Telegram message,
                stash them via the AttachmentStore, transcribe voice, and
                return the resulting sha256 list ready to feed agent.run().

                Quietly skips anything that fails — network or Telegram
                API hiccups should not block the text part of the message.
                """
                from .attachments import ATTACHMENTS, classify_kind
                from .transcriber import TRANSCRIBER

                msg = update.message
                if msg is None:
                    return []
                shas: list[str] = []

                # Photos (Telegram sends multiple resolutions; take the
                # largest — better for vision models)
                if getattr(msg, "photo", None):
                    try:
                        largest = msg.photo[-1]
                        f = await largest.get_file()
                        data = await f.download_as_bytearray()
                        rec = ATTACHMENTS.save(
                            bytes(data),
                            "image/jpeg",
                            filename=f"telegram_{largest.file_unique_id}.jpg",
                            kind="image",
                        )
                        shas.append(rec.sha256)
                    except Exception as e:
                        log.warning("Telegram photo download failed: %s", e)

                # Voice → store + try to transcribe
                user_sent_voice = bool(getattr(msg, "voice", None))
                if getattr(msg, "voice", None):
                    try:
                        f = await msg.voice.get_file()
                        data = await f.download_as_bytearray()
                        mime = msg.voice.mime_type or "audio/ogg"
                        rec = ATTACHMENTS.save(
                            bytes(data),
                            mime,
                            filename=f"telegram_voice_{msg.voice.file_unique_id}.ogg",
                            kind="audio",
                        )
                        text = TRANSCRIBER.transcribe(
                            bytes(data),
                            mime_type=mime,
                            filename=f"voice.ogg",
                        )
                        if text:
                            ATTACHMENTS.set_transcript(rec.sha256, text)
                        shas.append(rec.sha256)
                    except Exception as e:
                        log.warning("Telegram voice handling failed: %s", e)

                # Audio / documents that look like images or audio
                if getattr(msg, "audio", None):
                    try:
                        f = await msg.audio.get_file()
                        data = await f.download_as_bytearray()
                        mime = msg.audio.mime_type or "audio/mpeg"
                        rec = ATTACHMENTS.save(bytes(data), mime, filename=msg.audio.file_name or "audio", kind="audio")
                        shas.append(rec.sha256)
                    except Exception as e:
                        log.warning("Telegram audio download failed: %s", e)

                if getattr(msg, "document", None):
                    try:
                        f = await msg.document.get_file()
                        data = await f.download_as_bytearray()
                        mime = msg.document.mime_type or "application/octet-stream"
                        rec = ATTACHMENTS.save(
                            bytes(data),
                            mime,
                            filename=msg.document.file_name or "document",
                            kind=classify_kind(mime),
                        )
                        shas.append(rec.sha256)
                    except Exception as e:
                        log.warning("Telegram document download failed: %s", e)

                return shas

            async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                if not update.message:
                    return

                user = update.effective_user
                username = user.username or str(user.id)

                # Check allowed users (empty = allow all)
                if allowed and username not in allowed and str(user.id) not in allowed:
                    await update.message.reply_text("Access denied.")
                    return

                # Round E (TG forward): remember this chat_id so the
                # WebUI's "compose-as-telegram" mode can deliver the
                # agent's reply back here. Last writer wins — the most
                # recent TG conversation is the one a WebUI compose
                # targets. Multi-user TG bots would need a fancier
                # routing layer; for a personal assistant this is
                # exactly the desired behaviour.
                self._last_chat_id = update.message.chat.id

                # Pull text from message OR caption (photos arrive with caption)
                text = (update.message.text or update.message.caption or "").strip()

                # Remember this in handle_message scope too. _gather_attachments()
                # also computes it internally, but that local variable is not
                # visible down in the TTS reply block. Without this, voice
                # replies fail with: NameError: user_sent_voice is not defined.
                user_sent_voice = bool(getattr(update.message, "voice", None))

                # Pick up any media (photos / voice / audio / docs)
                attachment_shas = await _gather_attachments(update)

                # Voice without text → use the transcript as the message body
                if not text and attachment_shas:
                    from .attachments import ATTACHMENTS
                    for sha in attachment_shas:
                        meta = ATTACHMENTS.get_meta(sha)
                        if meta and meta.kind == "audio" and meta.transcript:
                            text = meta.transcript
                            break
                    if not text:
                        # Image-only message — give the agent something to chew on
                        text = "(see attached file)"

                if not text and not attachment_shas:
                    return

                await update.message.chat.send_action("typing")

                try:
                    from .agent import Agent

                    # Live progress placeholder — gets edited in-place as
                    # the agent thinks, runs tools, verifies. Survives as
                    # the trace record after the run; the actual answer
                    # is sent as a separate message.
                    placeholder = await update.message.reply_text("🧠 Thinking…")
                    running_loop = asyncio.get_running_loop()
                    stream = _TgProgressStream(
                        bot=update.message.get_bot(),
                        chat_id=update.message.chat.id,
                        message_id=placeholder.message_id,
                        loop=running_loop,
                    )

                    def _progress_cb(event: str, message: str) -> None:
                        # Called from the executor thread where agent.run runs.
                        stream.push(event, message)

                    agent = Agent(progress=_progress_cb)
                    # Don't block the event loop — run the (sync) agent in
                    # a thread pool so the streamer can keep editing.
                    result = await running_loop.run_in_executor(
                        None,
                        lambda: agent.run(
                            text, project=None,
                            attachments=attachment_shas or None,
                            channel="telegram",
                        ),
                    )
                    answer = result.answer or "(no answer)"

                    # Compact thinking + tools footer (between answer and stats).
                    trace_footer = _format_trace_footer(result)

                    # Token usage statistics — appended to the END of
                    # the main answer (not the placeholder summary)
                    # because that's where the user expects to find
                    # them: at the bottom of the message they're
                    # actually reading. Placeholder gets a minimal
                    # `✅ Done` so it's clear the work finished.
                    stats_block = ""
                    if result.token_usage:
                        tu = result.token_usage
                        stats_lines = [
                            "━━━━━━━━━━━━━━━━━━━━━━",
                            f"🔢 Tokens: {tu.total_tokens:,} (in: {tu.input_tokens:,}, out: {tu.output_tokens:,})",
                        ]
                        if tu.cache_read_tokens > 0:
                            stats_lines.append(f"💾 Cache read: {tu.cache_read_tokens:,}")
                        if tu.cache_creation_tokens > 0:
                            stats_lines.append(f"📝 Cache created: {tu.cache_creation_tokens:,}")
                        stats_lines.append(f"💰 Cost: ${tu.cost_usd:.4f}")
                        stats_lines.append(f"🔄 LLM calls: {tu.llm_calls}")
                        # Per-stage breakdown — top 3 by input tokens. Lets
                        # the user see at a glance which stage owned the
                        # bill ("solve: 220k in" vs "verify: 8k in") so
                        # the next optimisation isn't a guess. Skip when
                        # there's only one stage or all stages are tiny.
                        stages = tu.by_stage or {}
                        if len(stages) > 1 and tu.input_tokens >= 5_000:
                            top = list(stages.items())[:3]
                            parts = []
                            for name, s in top:
                                pct = (
                                    s.get("input_tokens", 0) / tu.input_tokens * 100
                                    if tu.input_tokens else 0
                                )
                                parts.append(
                                    f"{name} {int(s.get('input_tokens', 0)):,}"
                                    f" ({pct:.0f}%)"
                                )
                            stats_lines.append("📊 Stages: " + " · ".join(parts))
                        stats_block = "\n".join(stats_lines)

                    # Build the answer with footer + stats appended.
                    # When the combined message would exceed Telegram's
                    # 4096-char limit, the LAST chunk carries the
                    # footer/stats so the bottom of the conversation
                    # always shows totals — no chunk in the middle.
                    answer_parts: list[str] = [answer]
                    if trace_footer:
                        answer_parts.append(trace_footer)
                    if stats_block:
                        answer_parts.append(stats_block)
                    answer_with_stats = "\n\n".join(answer_parts)

                    # Replace the placeholder with a minimal "done"
                    # marker. Stats and trace are now in the answer,
                    # not here.
                    await stream.finalize("✅ Done")

                    # Smart chunking: keep the trace_footer + stats
                    # block whole in the LAST message. Naive 4000-char
                    # slicing would split the stats block at byte 4000
                    # of the answer body, which looks broken on
                    # Telegram. Strategy:
                    #   - separate the answer body from the tail
                    #     (tail = trace + stats)
                    #   - chunk only the body
                    #   - append the tail to whichever chunk has room,
                    #     otherwise send it as a fresh final message
                    tail_parts: list[str] = []
                    if trace_footer:
                        tail_parts.append(trace_footer)
                    if stats_block:
                        tail_parts.append(stats_block)
                    tail = "\n\n".join(tail_parts)

                    LIMIT = 4000
                    if not tail or len(answer) + len(tail) + 2 <= LIMIT:
                        # Body + tail fit in one message.
                        await update.message.reply_text(
                            answer if not tail else f"{answer}\n\n{tail}"
                        )
                    else:
                        # Chunk the body. Leave room in the LAST body
                        # chunk for the tail when possible.
                        body_chunks: list[str] = []
                        i = 0
                        while i < len(answer):
                            chunk = answer[i:i + LIMIT]
                            body_chunks.append(chunk)
                            i += LIMIT
                        # Try to merge tail into last body chunk.
                        if (
                            body_chunks
                            and len(body_chunks[-1]) + len(tail) + 2 <= LIMIT
                        ):
                            body_chunks[-1] = f"{body_chunks[-1]}\n\n{tail}"
                            tail_msg = None
                        else:
                            tail_msg = tail
                        for c in body_chunks:
                            await update.message.reply_text(c)
                        if tail_msg:
                            await update.message.reply_text(tail_msg)

                    # Round D + voice-fix: voice reply. When the user
                    # sent a voice message AND TTS is configured +
                    # enabled, also send the answer as audio. PTB
                    # needs the bytes wrapped in InputFile with an
                    # explicit filename for Telegram to accept WAV
                    # otherwise the upload silently 400s. We try
                    # reply_voice first (native TG voice bubble; PTB
                    # accepts WAV here in v20+), then fall back to
                    # reply_audio on PTB versions that are stricter.
                    try:
                        from .config import CONFIG as _C
                        from .tts import SYNTHESIZER as _TTS
                        tts_cfg = _C.tts
                        speak = (
                            tts_cfg.get("enabled_always", False)
                            or (
                                user_sent_voice
                                and tts_cfg.get("enabled_on_voice_input", True)
                            )
                        )
                        if speak and answer.strip():
                            cap = int(tts_cfg.get("max_chars", 1000) or 1000)
                            spoken = (answer or "").strip()
                            if len(spoken) > cap:
                                spoken = spoken[:cap]
                            audio_wav = await running_loop.run_in_executor(
                                None,
                                lambda: _TTS.synthesize(spoken),
                            )
                            if audio_wav:
                                # Telegram's native voice bubble needs OGG
                                # container + Opus codec, 48 kHz mono.
                                # WAV plays as a distorted bubble or a
                                # generic audio attachment depending on
                                # the client. Convert through ffmpeg
                                # when available; fall back to raw WAV
                                # so we ship SOMETHING on machines
                                # without ffmpeg installed.
                                from .tts import convert_wav_to_telegram_voice
                                audio, audio_fmt = await running_loop.run_in_executor(
                                    None,
                                    lambda: convert_wav_to_telegram_voice(audio_wav),
                                )
                                import io as _io
                                from telegram import InputFile
                                fname = "reply.ogg" if audio_fmt == "ogg" else "reply.wav"
                                voice_blob = _io.BytesIO(audio)
                                voice_blob.name = fname
                                # Try native voice bubble first.
                                sent = False
                                try:
                                    voice_blob.seek(0)
                                    await update.message.reply_voice(
                                        voice=InputFile(voice_blob, filename=fname),
                                    )
                                    sent = True
                                except Exception as e_voice:
                                    log.info(
                                        "TG reply_voice failed (%s) — "
                                        "falling back to reply_audio",
                                        e_voice,
                                    )
                                if not sent:
                                    voice_blob.seek(0)
                                    await update.message.reply_audio(
                                        audio=InputFile(voice_blob, filename=fname),
                                        title="agent voice reply",
                                    )
                            else:
                                # Synth produced no bytes — surface
                                # the reason so the user sees why
                                # the bot stayed silent on voice
                                # rather than puzzling over a missing
                                # bubble.
                                err = _TTS.status().get("last_error") or "(no detail)"
                                await update.message.reply_text(
                                    f"⚠️ TTS produced no audio: {err}"
                                )
                    except Exception as _tts_err:
                        # Surface the error visibly — debugging silent
                        # TTS failures was painful. Cap the trace so
                        # we don't ship a 5KB stack into TG.
                        log.warning("TTS reply failed: %s", _tts_err, exc_info=True)
                        try:
                            await update.message.reply_text(
                                f"⚠️ Voice reply failed: {str(_tts_err)[:300]}"
                            )
                        except Exception:
                            pass

                    _log_channel_message(self.channel_id, username, text, answer)

                except Exception as e:
                    log.error("Telegram bot error processing message: %s", e)
                    await update.message.reply_text(f"Error: {str(e)[:500]}")

            app.add_handler(CommandHandler("start", handle_start))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            # Media handlers route through the same handle_message so caption
            # text + attachment shas reach the agent in one turn.
            app.add_handler(MessageHandler(filters.PHOTO, handle_message))
            app.add_handler(MessageHandler(filters.VOICE, handle_message))
            app.add_handler(MessageHandler(filters.AUDIO, handle_message))
            app.add_handler(MessageHandler(filters.Document.ALL, handle_message))

            loop.run_until_complete(app.initialize())
            # Defensive: explicitly clear any webhook before polling.
            # If anything ever sets a webhook on this token (manual
            # curl, another deploy), getUpdates would 409 forever.
            # `drop_pending_updates=True` also flushes the queue so we
            # don't blast through old messages on restart.
            try:
                loop.run_until_complete(
                    app.bot.delete_webhook(drop_pending_updates=True)
                )
            except Exception as e:
                log.warning(
                    "Telegram bot %s: delete_webhook on start failed: %s",
                    self.channel_id, e,
                )
            loop.run_until_complete(app.start())
            loop.run_until_complete(app.updater.start_polling(drop_pending_updates=True))
            log.info("Telegram bot %s polling started successfully", self.channel_id)

            # Keep running until stopped
            while self._running:
                loop.run_until_complete(asyncio.sleep(1))

            loop.run_until_complete(app.updater.stop())
            loop.run_until_complete(app.stop())
            loop.run_until_complete(app.shutdown())

        except Exception as e:
            log.error("Telegram bot %s crashed: %s", self.channel_id, e, exc_info=True)
            self._running = False


def _log_channel_message(channel_id: str, user: str, question: str, answer: str) -> None:
    """Append channel interaction to sessions for tracking."""
    try:
        from .sessions import SESSIONS
        turn = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": f"[{channel_id}:{user}] {question}",
            "answer": answer,
            "intent": "channel",
            "is_chat": False,
            "confidence": 0,
            "topics": [],
        }
        SESSIONS.add_turn(turn)
    except Exception:
        pass


# --------------- Channel Manager ---------------

class ChannelManager:
    """Manages all active channel connections."""

    def __init__(self):
        self._bots: dict[str, TelegramBot] = {}

    def start_channel(self, channel_id: str) -> dict:
        """Start a channel by its ID. Returns status."""
        ch = get_channel(channel_id)
        if not ch:
            return {"ok": False, "error": "Channel not found"}

        if not ch.get("enabled", False):
            return {"ok": False, "error": "Channel is disabled"}

        ch_type = ch.get("type", "")

        if ch_type == "telegram":
            token = ch.get("config", {}).get("bot_token", "")
            if not token:
                return {"ok": False, "error": "No bot token configured"}

            if channel_id in self._bots and self._bots[channel_id].is_running:
                return {"ok": True, "status": "already_running"}

            allowed = ch.get("config", {}).get("allowed_users", [])
            bot = TelegramBot(token=token, channel_id=channel_id, allowed_users=allowed)
            bot.start()
            self._bots[channel_id] = bot

            # Update status in storage
            ch["status"] = "running"
            ch["last_started"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_channel(ch)

            return {"ok": True, "status": "started"}
        else:
            return {"ok": False, "error": f"Unknown channel type: {ch_type}"}

    def stop_channel(self, channel_id: str) -> dict:
        """Stop a running channel."""
        if channel_id in self._bots:
            self._bots[channel_id].stop()
            del self._bots[channel_id]

        ch = get_channel(channel_id)
        if ch:
            ch["status"] = "stopped"
            save_channel(ch)

        return {"ok": True, "status": "stopped"}

    def channel_status(self, channel_id: str) -> str:
        if channel_id in self._bots and self._bots[channel_id].is_running:
            return "running"
        return "stopped"

    def status_all(self) -> dict[str, str]:
        return {cid: ("running" if bot.is_running else "stopped") for cid, bot in self._bots.items()}

    def auto_start(self) -> None:
        """Start all enabled channels that have auto_start=True."""
        for ch in _load_channels():
            if ch.get("enabled") and ch.get("auto_start"):
                try:
                    self.start_channel(ch["id"])
                except Exception as e:
                    log.error("Failed to auto-start channel %s: %s", ch["id"], e)

    def stop_all(self) -> None:
        for cid in list(self._bots.keys()):
            self.stop_channel(cid)

    def send_to_first_telegram(self, text: str) -> bool:
        """Round E: forward arbitrary text to the first running
        Telegram bot's most-recent chat. Used by the WebUI's
        compose-as-telegram mode after agent.run completes — the
        answer renders in the WebUI AND lands in the user's TG so
        the conversation thread stays continuous.

        Returns True on successful schedule, False when no bot is
        running or it has never received a message (no chat_id to
        reply to). Multi-bot or multi-user routing would need a
        richer addressing scheme; for a personal assistant the
        first-running-bot heuristic is exactly right.
        """
        for bot in self._bots.values():
            if bot.is_running:
                return bot.send_text(text)
        return False


CHANNELS = ChannelManager()
