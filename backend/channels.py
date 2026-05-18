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

from . import paths
from .config import CONFIG

log = logging.getLogger(__name__)

# Engine-relative ROOT kept for legacy imports; channel state lives
# with the user's other knowledge data, never inside the engine repo.
ROOT = paths.repo_root()

# Eager value for backward compat — existing callers that import
# `CHANNELS_PATH` keep working. Tests that monkeypatch this
# attribute STILL win because `_resolve_channels_path` reads it
# back via the module namespace.
CHANNELS_PATH = paths.knowledge_dir() / "channels.json"


def _resolve_channels_path() -> Path:
    """Audit fix: a previous version of this module bound
    `CHANNELS_PATH = paths.knowledge_dir() / "channels.json"` at
    import time. Tests that monkeypatched HRANT_DATA_DIR AFTER
    import got a phantom mismatch — the module kept writing to the
    dev's real ~/.hrant/data/channels.json instead of the
    tmp_path the test set up.

    Now `_load_channels` / `_save_channels` call THIS helper which
    re-reads the module attribute (so test setattr wins) AND
    re-resolves via paths.knowledge_dir() (so HRANT_DATA_DIR env
    changes win). Cost: a couple of env lookups per call, never
    on a hot path."""
    import sys as _sys
    module = _sys.modules.get(__name__)
    override = getattr(module, "CHANNELS_PATH", None) if module else None
    if override is not None and isinstance(override, Path):
        return override
    return paths.knowledge_dir() / "channels.json"


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
    p = _resolve_channels_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("channels", [])
        except Exception:
            return []
    return []


def _save_channels(channels: list[dict]) -> None:
    p = _resolve_channels_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
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


# Whitelist of agent progress events that may surface to the user.
# The agent emits ~30 different event names (core, chat, strategy,
# tool_starting, found, learning, learned, solve, verify, memory_save,
# cleanup, etc.). Most of those are internal pipeline trace — surfacing
# them in chat reads like a debug log spilled into the conversation.
# We keep only events that name a user-visible action ("searching the
# web", "reading source code") and translate them into a short
# human-readable label. Everything else collapses into the generic
# "🧠 Thinking…" tick so the user still sees the bot is alive.
_USER_VISIBLE_PROGRESS_EVENTS: dict[str, str] = {
    "learning": "📚 Learning a new topic",
    "learned": "📚 Saved a note",
    "subtask": "🧩 Subtask",
    "tool_starting": "🔧 Using a tool",
    "self_critic": "🔁 Re-checking the answer",
    "found": "📎 Found relevant notes",
}


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

    Filtering:
      Only events in `_USER_VISIBLE_PROGRESS_EVENTS` are rendered.
      Internal pipeline trace (core, chat, strategy, solve, verify,
      memory_save, cleanup, raw tool JSON results, …) stays out of
      the chat — those went straight to the user before the filter
      was added, producing a debug-log-in-Telegram UX bug.
    """

    EDIT_INTERVAL_SEC = 1.2
    MAX_LINE_LEN = 120

    def __init__(self, bot: Any, chat_id: int, message_id: int, loop: Any):
        self.bot = bot
        self.chat_id = chat_id
        self.message_id = message_id
        self.loop = loop
        # Single "latest user-visible action" line, not an accumulating
        # list. The placeholder is a status indicator, not a transcript.
        self._latest_label: str = "🧠 Thinking…"
        self._lock = threading.Lock()
        self._last_edit = 0.0
        self._pending = False
        self._closed = False

    def push(self, event: str, message: str) -> None:
        """Sync entry point — called from the agent thread.

        Events outside the user-visible whitelist are dropped. For
        whitelisted events we render `<emoji label>: <short detail>`
        and replace (not append to) the placeholder body, so the user
        only ever sees the agent's CURRENT action, not its full
        breadcrumb trail."""
        if self._closed:
            return
        label = _USER_VISIBLE_PROGRESS_EVENTS.get(event)
        if label is None:
            return
        detail = (message or "").strip()
        if detail:
            line = f"{label}: {detail}"
        else:
            line = label
        if len(line) > self.MAX_LINE_LEN:
            line = line[: self.MAX_LINE_LEN - 1] + "…"
        with self._lock:
            self._latest_label = line
        self._schedule_edit()

    def _schedule_edit(self) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self._maybe_edit(), self.loop)
        except Exception:
            pass

    def _render(self) -> str:
        with self._lock:
            return self._latest_label

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


# Telegram MEDIA: convention — agent writes `MEDIA:/abs/path/to/file`
# on its own line and the bridge converts each one to a real
# attachment. Path is required to be absolute (relative paths are
# ambiguous across the agent process / bot process), and limited
# to whitelisted parent dirs so a hallucinated `MEDIA:/etc/shadow`
# can't leak host files.
import re as _re_media
_MEDIA_LINE_RE = _re_media.compile(r"^[ \t]*MEDIA:[ \t]*([^\r\n]+?)[ \t]*$", _re_media.MULTILINE)
_MEDIA_VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".mkv", ".m4v"})
_MEDIA_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
_MEDIA_AUDIO_EXTS = frozenset({".mp3", ".m4a", ".ogg", ".oga", ".wav", ".flac"})


def _media_path_is_safe(p: Path) -> bool:
    """Allow paths under data_dir and the system /tmp only — guards
    against `MEDIA:/etc/...` or other host-file leaks."""
    try:
        rp = p.resolve()
    except Exception:
        return False
    if not rp.is_absolute() or not rp.exists() or not rp.is_file():
        return False
    try:
        data_root = paths.data_dir(require=False).resolve()
    except Exception:
        data_root = None
    tmp_roots = [Path("/tmp"), Path("/var/tmp")]
    candidates = ([data_root] if data_root else []) + tmp_roots
    for root in candidates:
        try:
            rp.relative_to(root)
            return True
        except ValueError:
            continue
    return False


async def _strip_and_send_media(answer: str, update: "Any") -> tuple[str, int]:
    """Scan `answer` for MEDIA: lines, send each referenced file as
    a Telegram attachment, and return (cleaned_answer, sent_count).

    Best-effort:
      - missing file → keep the path in the textual answer so the
        user sees the error, count NOT incremented
      - send failure → log + leave path inline, count NOT incremented
      - sent successfully → strip the line from the text
    """
    sent = 0
    matches = list(_MEDIA_LINE_RE.finditer(answer))
    if not matches:
        return answer, 0
    # Iterate in REVERSE so the slice indexes stay valid as we cut.
    cleaned = answer
    cuts: list[tuple[int, int]] = []
    for m in matches:
        raw_path = (m.group(1) or "").strip().strip('"').strip("'")
        if not raw_path:
            continue
        p = Path(raw_path)
        if not _media_path_is_safe(p):
            log.info("MEDIA: refused unsafe path %r", raw_path)
            continue
        ext = p.suffix.lower()
        ok = False
        try:
            if ext in _MEDIA_VIDEO_EXTS:
                with p.open("rb") as fh:
                    await update.message.reply_video(video=fh)
                ok = True
            elif ext in _MEDIA_IMAGE_EXTS:
                with p.open("rb") as fh:
                    await update.message.reply_photo(photo=fh)
                ok = True
            elif ext in _MEDIA_AUDIO_EXTS:
                with p.open("rb") as fh:
                    await update.message.reply_audio(audio=fh)
                ok = True
            else:
                with p.open("rb") as fh:
                    await update.message.reply_document(document=fh)
                ok = True
        except Exception as e:
            log.warning("MEDIA: send failed for %s: %s", p, e)
            ok = False
        if ok:
            sent += 1
            cuts.append((m.start(), m.end()))
    # Apply cuts in reverse order.
    for start, end in reversed(cuts):
        cleaned = cleaned[:start] + cleaned[end:]
    # Tidy: collapse 3+ consecutive newlines we may have introduced.
    cleaned = _re_media.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, sent


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


# Audit fix: cap concurrent agent.run calls from a single Telegram
# bot. Pre-fix, a user (or group with many members) spamming the
# bot spawned one executor thread per message → N concurrent
# `agent.run` → N concurrent LLM streams. Cost amplification +
# provider rate-limit thrash. 3 lets a power user keep a few
# parallel queries going while bounding the worst case. Tune via
# the env var if you have many trusted group members.
import os as _os_for_concurrency
_MAX_CONCURRENT_AGENT_RUNS = int(
    _os_for_concurrency.environ.get("HRANT_TELEGRAM_MAX_CONCURRENCY", 3)
)


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
        # Per-bot semaphore. Created lazily because asyncio.Semaphore
        # binds to the loop it's created on, and the loop is set up
        # in `_run_in_thread`.
        self._run_semaphore: asyncio.Semaphore | None = None
        # Track the most recent chat_id we've received a message from
        # on this bot. Used by send_text() so the WebUI's
        # "compose-as-telegram" mode can deliver the agent's reply
        # back to the user's TG without us having to enumerate chats.
        self._last_chat_id: int | None = None
        # Telegram delivers photo albums as ONE update per photo, each
        # carrying the same `media_group_id`. Pre-fix, the bot ran a
        # full agent turn for every photo, producing N disjoint replies
        # for one user "send". Buffer entries by media_group_id and
        # flush after a short debounce so the agent sees the whole
        # batch as a single message. Keyed by media_group_id; each
        # value is { 'shas': [...], 'caption': str, 'first_update':
        # Update, 'flush_task': asyncio.Task | None }.
        self._media_groups: dict[str, dict] = {}

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

    def _notify_owners_of_pairing(self, decision, tg_user, first_message: str) -> None:
        """DM every owner-on-Telegram that an unknown user is asking
        for access. Best-effort — failures are logged, not raised, so
        the stranger still gets their reply.

        Hermes-style: the DM carries 4 inline buttons —
        "Allow Once (1h)" / "Session (24h)" / "Always" / "Deny".
        Pressing any of them lands in `_handle_callback_query` which
        forwards to `access.py::_register_pairing_callback._handler`.

        Telegram bots can only DM users who started a conversation
        with this bot at some point. For the box owner that's almost
        always true. If we can't find their chat_id (they never DMed
        THIS bot), we log and move on — they'll see the request in
        the WebUI Pairing panel.
        """
        from . import roles as _roles
        from . import contacts as _contacts
        from . import tg_interactive as _tg

        try:
            state = _roles._load()
        except Exception as e:
            log.warning("notify_owners_of_pairing: roles load failed: %s", e)
            return

        owner_ids = [
            sid for sid in (state.get("owner_speaker_ids") or [])
            if isinstance(sid, str) and sid.startswith("telegram:")
        ]
        if not owner_ids:
            log.info(
                "notify_owners_of_pairing: no telegram owner — pairing "
                "request from %s (code=%s) will only show in WebUI",
                tg_user.id, decision.pairing_code,
            )
            return

        snippet = (first_message or "").strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        handle_html = _tg.fmt_user_handle(
            tg_user.id,
            username=tg_user.username or "",
            full_name=tg_user.full_name or "",
        )
        snippet_html = _tg.escape_html(snippet)
        code = decision.pairing_code
        text = (
            f"🔐 <b>Pairing request</b>\n\n"
            f"From: {handle_html}\n"
            f"First message: <i>{snippet_html or '(no text)'}</i>\n\n"
            f"Code: <code>{_tg.escape_html(code)}</code>"
        )

        # 4-button approval keyboard (Hermes-style).
        # Layout:
        #   [ 🕒 Allow Once (1h) ] [ ⏳ Session (24h) ]
        #   [ ✅ Always           ] [ ❌ Deny           ]
        buttons = (
            _tg.InlineButtonSet()
            .row(
                _tg.InlineButton("🕒 Allow Once (1h)", callback_data=f"pair:approve:once:{code}"),
                _tg.InlineButton("⏳ Session (24h)",   callback_data=f"pair:approve:session:{code}"),
            )
            .row(
                _tg.InlineButton("✅ Always",          callback_data=f"pair:approve:always:{code}"),
                _tg.InlineButton("❌ Deny",            callback_data=f"pair:deny:{code}"),
            )
        )
        markup = buttons.to_markup()

        for owner_sid in owner_ids:
            chat_id = _contacts.chat_id_for_speaker(owner_sid)
            if chat_id is None:
                log.info(
                    "notify_owners_of_pairing: owner %s has no captured "
                    "chat_id on this bot — skipping DM",
                    owner_sid,
                )
                continue
            try:
                self._send_with_buttons(chat_id, text, markup)
            except Exception as e:
                log.warning(
                    "notify_owners_of_pairing: send_with_buttons(%s) failed: %s",
                    chat_id, e,
                )

    def _on_self_mod_proposal(self, proposal) -> None:
        """Subscribed callback: a new self-mod proposal landed. DM
        every Telegram owner with an inline-button approval prompt.

        Buttons:
          [👀 Show diff] [✅ Approve & Apply] [❌ Reject]

        Owner-only (the callback handler in self_modifier.py refuses
        non-owner clickers); failures are logged, not raised."""
        from . import roles as _roles
        from . import contacts as _contacts
        from . import tg_interactive as _tg

        pid = getattr(proposal, "id", "") or ""
        if not pid:
            return
        try:
            owner_state = _roles._load()
        except Exception:
            return
        owner_ids = [
            sid for sid in (owner_state.get("owner_speaker_ids") or [])
            if isinstance(sid, str) and sid.startswith("telegram:")
        ]
        if not owner_ids:
            return

        title = (proposal.title or proposal.description or proposal.id)[:80]
        risk = getattr(proposal, "risk", "low")
        module = getattr(proposal, "module", "")
        module_clause = f" in <code>{_tg.escape_html(module)}</code>" if module else ""
        text = (
            f"🛠 <b>Self-modification proposal</b>{module_clause}\n\n"
            f"<b>Title:</b> {_tg.escape_html(title)}\n"
            f"<b>Risk:</b> {_tg.escape_html(str(risk))}\n"
            f"<b>Rationale:</b> <i>{_tg.escape_html(proposal.reasoning or '(none)')}</i>\n"
            f"<b>ID:</b> <code>{_tg.escape_html(pid)}</code>"
        )
        buttons = (
            _tg.InlineButtonSet()
            .row(
                _tg.InlineButton("👀 Show diff", callback_data=f"prop:diff:{pid}"),
            )
            .row(
                _tg.InlineButton("✅ Approve & Apply", callback_data=f"prop:apply:{pid}"),
                _tg.InlineButton("❌ Reject", callback_data=f"prop:reject:{pid}"),
            )
        )
        markup = buttons.to_markup()
        for owner_sid in owner_ids:
            chat_id = _contacts.chat_id_for_speaker(owner_sid)
            if chat_id is None:
                continue
            try:
                self._send_with_buttons(chat_id, text, markup)
            except Exception as e:
                log.warning("self-mod notify(%s) failed: %s", chat_id, e)

    def _on_skill_proposed(self, skill) -> None:
        """Self-improvement loop notification — DM every owner-on-
        Telegram about a freshly-proposed skill. Three inline buttons:
        [✅ Activate], [👀 Show], [❌ Delete]. The skill is already on
        disk but disabled; only the owner can switch it live."""
        from . import roles as _roles
        from . import contacts as _contacts
        from . import tg_interactive as _tg

        sname = getattr(skill, "name", "") or ""
        if not sname:
            return
        try:
            state = _roles._load()
        except Exception:
            return
        owner_ids = [
            sid for sid in (state.get("owner_speaker_ids") or [])
            if isinstance(sid, str) and sid.startswith("telegram:")
        ]
        if not owner_ids:
            return

        triggers = ", ".join(skill.triggers or []) or "(none)"
        text = (
            f"🧠 <b>New skill proposed</b>\n\n"
            f"<b>Name:</b> <code>{_tg.escape_html(sname)}</code>\n"
            f"<b>Description:</b> {_tg.escape_html(skill.description or '')}\n"
            f"<b>Triggers:</b> <i>{_tg.escape_html(triggers)}</i>\n\n"
            f"<i>Disabled by default. Activate to make it live for "
            f"future turns.</i>"
        )
        buttons = (
            _tg.InlineButtonSet()
            .row(
                _tg.InlineButton("👀 Show", callback_data=f"skill:show:{sname}"),
            )
            .row(
                _tg.InlineButton("✅ Activate", callback_data=f"skill:enable:{sname}"),
                _tg.InlineButton("❌ Delete", callback_data=f"skill:delete:{sname}"),
            )
        )
        markup = buttons.to_markup()
        for owner_sid in owner_ids:
            chat_id = _contacts.chat_id_for_speaker(owner_sid)
            if chat_id is None:
                continue
            try:
                self._send_with_buttons(chat_id, text, markup)
            except Exception as e:
                log.warning("skill-proposed DM(%s) failed: %s", chat_id, e)

    def _on_message_scheduled(self, row: dict) -> None:
        """Post-create preview DM for a scheduled message — Phase 4.

        The row was just persisted by `scheduled_messages.schedule()`.
        We DM the REQUESTER (the speaker who asked the agent to
        schedule it) with a preview and a [❌ Cancel] button. They
        have until `due_at` to bail out.

        Best-effort: a missing chat_id (the requester never DMed this
        bot) or a Telegram API hiccup is logged, not raised."""
        from . import contacts as _contacts
        from . import tg_interactive as _tg

        requester = row.get("requested_by") or ""
        if not requester.startswith("telegram:"):
            return
        chat_id = _contacts.chat_id_for_speaker(requester)
        if chat_id is None:
            return

        target = row.get("target_speaker") or "?"
        due = row.get("due_at") or ""
        body = (row.get("text") or "").strip()
        preview = (body[:300] + "…") if len(body) > 300 else body
        text = (
            f"📅 <b>Scheduled message queued</b>\n\n"
            f"To: <code>{_tg.escape_html(target)}</code>\n"
            f"Due: <code>{_tg.escape_html(due)}</code>\n"
            f"Body: <i>{_tg.escape_html(preview) or '(empty)'}</i>\n\n"
            f"<i>Tap Cancel to drop it before delivery.</i>"
        )
        buttons = (
            _tg.InlineButtonSet()
            .row(
                _tg.InlineButton("❌ Cancel", callback_data=f"sched:cancel:{row.get('id')}"),
            )
        )
        try:
            self._send_with_buttons(chat_id, text, buttons.to_markup())
        except Exception as e:
            log.warning("schedule preview DM(%s) failed: %s", chat_id, e)

    def _send_with_buttons(self, chat_id: int, text: str, markup) -> None:
        """Schedule an HTML-formatted message with an inline keyboard.

        Wraps `bot.send_message` for the cross-thread case (caller is
        on the agent's thread; the bot owns its own asyncio loop)."""
        if not self._app or not self._loop:
            return

        async def _send() -> None:
            try:
                await self._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=text,
                    parse_mode="HTML",
                    reply_markup=markup,
                )
            except Exception as e:
                log.warning("send_with_buttons inner failed: %s", e)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception as e:
            log.warning("send_with_buttons schedule failed: %s", e)

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import (
                ApplicationBuilder, MessageHandler, CommandHandler,
                CallbackQueryHandler, ContextTypes, filters,
            )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            app = ApplicationBuilder().token(self.token).build()
            self._app = app

            allowed = self.allowed_users

            async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                from . import tg_interactive as _tg
                from telegram import ReplyKeyboardMarkup, KeyboardButton
                user = update.effective_user
                chat_id = update.effective_chat.id
                handle = _tg.fmt_user_handle(
                    user.id,
                    username=user.username or "",
                    full_name=user.full_name or "",
                )
                # Persistent quick-reply keyboard under the input box.
                # Plain `/command` text — the labels DON'T carry
                # emoji because a leading emoji breaks TG's "starts
                # with /" detection for CommandHandler. The slash
                # commands themselves still appear with descriptions
                # in the autocomplete dropdown (set_my_commands above).
                quick = ReplyKeyboardMarkup(
                    [
                        [
                            KeyboardButton("/status"),
                            KeyboardButton("/sessions"),
                            KeyboardButton("/help"),
                        ],
                    ],
                    resize_keyboard=True,
                    is_persistent=True,
                )
                await update.message.reply_text(
                    f"👋 <b>Connected to hrant</b>\n\n"
                    f"Chat ID: <code>{chat_id}</code>\n"
                    f"User: {handle}\n\n"
                    f"Send me any message and I'll respond. "
                    f"The quick-buttons below run the slash commands.",
                    parse_mode="HTML",
                    reply_markup=quick,
                )

            async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                """/status — short bot health snapshot. HTML-formatted."""
                from . import roles as _roles
                from . import tg_interactive as _tg
                user = update.effective_user
                if user is None:
                    return
                speaker_id = f"telegram:{user.id}"
                role = _roles.role_of(speaker_id)
                # Count of trusted users (owner-only info).
                trusted_count = 0
                if role == "owner":
                    state = _roles.list_roles()
                    speakers = state.get("speakers") or {}
                    trusted_count = sum(
                        1 for v in speakers.values()
                        if (v or {}).get("role") == "trusted"
                    )
                lines = [
                    "📊 <b>hrant status</b>",
                    f"You: {_tg.fmt_user_handle(user.id, username=user.username or '', full_name=user.full_name or '')}",
                    f"Role: <code>{role}</code>",
                ]
                if role == "owner":
                    lines.append(f"Trusted users: <code>{trusted_count}</code>")
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")

            async def handle_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                """/sessions — list the user's recent Telegram thread sessions."""
                from . import sessions as _sessions_mod
                from . import tg_interactive as _tg
                user = update.effective_user
                if user is None:
                    return
                speaker_id = f"telegram:{user.id}"
                rows = _sessions_mod.SESSIONS.list_sessions(speaker_id=speaker_id) or []
                if not rows:
                    await update.message.reply_text("No sessions yet.", parse_mode="HTML")
                    return
                lines = ["🧵 <b>Your recent sessions</b>", ""]
                for r in rows[:10]:
                    title = _tg.escape_html((r.get("title") or "(untitled)")[:60])
                    label = _tg.escape_html(r.get("thread_label") or "")
                    turns = r.get("turn_count") or 0
                    started = _tg.escape_html((r.get("started") or "")[:16])
                    lines.append(
                        f"• <i>{title}</i>\n  "
                        f"  <code>{label}</code> · {turns} turns · {started}"
                    )
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")

            async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                """/help — list available commands + capabilities."""
                text = (
                    "🆘 <b>hrant help</b>\n\n"
                    "<b>Commands</b>\n"
                    "/start — connection check\n"
                    "/status — your role + bot health\n"
                    "/sessions — your recent conversation threads\n"
                    "/help — this message\n\n"
                    "<b>What I can do</b>\n"
                    "• Answer questions, search the knowledge base, run code (owner)\n"
                    "• Pair new users — when a stranger writes, you get inline buttons\n"
                    "• Self-modify — apply proposed code changes after your approval\n"
                    "• Accept text / voice / photo / video / documents in any chat\n\n"
                    "Different chats (DM vs group, multiple bots) get separate "
                    "conversation threads."
                )
                await update.message.reply_text(text, parse_mode="HTML")

            async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                """Route an inline-button press through the
                tg_interactive dispatcher and apply the resulting
                CallbackResult (edit message, show toast, drop
                keyboard) to the original message."""
                from . import tg_interactive as _tg
                query = update.callback_query
                if query is None:
                    return
                data = query.data or ""
                clicker = query.from_user
                ctx = {
                    "clicker_speaker_id": f"telegram:{clicker.id}" if clicker else "",
                    "chat_id": update.effective_chat.id if update.effective_chat else None,
                    "message_id": query.message.message_id if query.message else None,
                    "callback_data": data,
                }
                result = _tg.dispatch_callback(data, ctx)

                # Always answer the callback so the spinning UI on
                # the user's button stops; pass `text=` for a toast.
                try:
                    await query.answer(
                        text=(result.toast or "")[:200] or None,
                        show_alert=False,
                    )
                except Exception as e:
                    log.debug("callback_query answer failed: %s", e)

                # Edit the original message OR clear its keyboard.
                if query.message is not None:
                    try:
                        if result.edited_text is not None:
                            await query.message.edit_text(
                                result.edited_text, parse_mode="HTML",
                            )
                        elif result.clear_keyboard:
                            await query.message.edit_reply_markup(reply_markup=None)
                    except Exception as e:
                        log.debug("callback_query edit failed: %s", e)

                # Optional follow-up message (e.g. "Show diff" sends
                # the full diff as a second message so the original
                # approval prompt with buttons stays usable).
                if result.followup_text and query.message is not None:
                    chat_id = update.effective_chat.id if update.effective_chat else None
                    if chat_id is not None:
                        # Split at the 4096-byte Telegram limit; each
                        # chunk gets its own send so a long diff doesn't
                        # 400 us out.
                        text = result.followup_text
                        LIMIT = 3900
                        chunks = (
                            [text] if len(text) <= LIMIT
                            else [text[i:i + LIMIT] for i in range(0, len(text), LIMIT)]
                        )
                        for chunk in chunks:
                            try:
                                await self._app.bot.send_message(
                                    chat_id=int(chat_id),
                                    text=chunk,
                                    parse_mode="HTML",
                                )
                            except Exception as e:
                                log.debug("callback_query followup failed: %s", e)

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

                # Audio files (mp3 / m4a / wav sent as TG audio). Same
                # treatment as msg.voice — auto-transcribe so the LLM
                # can read what the user said without an explicit
                # tool call. The file-type audit caught this: voice
                # auto-transcribed, audio file didn't, so the model
                # had to invent a workaround.
                if getattr(msg, "audio", None):
                    try:
                        f = await msg.audio.get_file()
                        data = await f.download_as_bytearray()
                        mime = msg.audio.mime_type or "audio/mpeg"
                        fname = msg.audio.file_name or f"telegram_audio_{msg.audio.file_unique_id}"
                        rec = ATTACHMENTS.save(bytes(data), mime, filename=fname, kind="audio")
                        try:
                            text = TRANSCRIBER.transcribe(
                                bytes(data), mime_type=mime, filename=fname,
                            )
                            if text:
                                ATTACHMENTS.set_transcript(rec.sha256, text)
                        except Exception as e:
                            log.debug("Telegram audio transcribe failed (non-fatal): %s", e)
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

                # Regular video (msg.video) — Telegram pre-encodes most
                # uploads as mp4. Stored with kind="video"; the LLM
                # build-content step lazily extracts sampled frames +
                # audio transcript via video_processor on first use.
                if getattr(msg, "video", None):
                    try:
                        f = await msg.video.get_file()
                        data = await f.download_as_bytearray()
                        mime = msg.video.mime_type or "video/mp4"
                        rec = ATTACHMENTS.save(
                            bytes(data), mime,
                            filename=msg.video.file_name or f"telegram_video_{msg.video.file_unique_id}.mp4",
                            kind="video",
                        )
                        shas.append(rec.sha256)
                    except Exception as e:
                        log.warning("Telegram video download failed: %s", e)

                # Video note ("кружочек") — square, no caption, always
                # mp4. Same handling as msg.video.
                if getattr(msg, "video_note", None):
                    try:
                        f = await msg.video_note.get_file()
                        data = await f.download_as_bytearray()
                        rec = ATTACHMENTS.save(
                            bytes(data), "video/mp4",
                            filename=f"telegram_videonote_{msg.video_note.file_unique_id}.mp4",
                            kind="video",
                        )
                        shas.append(rec.sha256)
                    except Exception as e:
                        log.warning("Telegram video_note download failed: %s", e)

                return shas

            async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                if not update.message:
                    return

                user = update.effective_user
                username = user.username or str(user.id)

                # Audit #10: group-chat isolation. If the bot was
                # added to a Telegram group (or supergroup/channel),
                # every member can talk to it — they each show up as
                # a distinct `telegram:<id>` speaker and consume the
                # owner's LLM quota. By default we refuse in group
                # chats unless the user is in `allowed_users`. Set
                # `HRANT_TELEGRAM_ALLOW_GROUPS=1` to revert to the
                # pre-fix behaviour (every group member chats freely).
                chat = update.effective_chat
                in_group = (
                    chat is not None
                    and chat.type in ("group", "supergroup", "channel")
                )
                if in_group and not _os_for_concurrency.environ.get(
                    "HRANT_TELEGRAM_ALLOW_GROUPS"
                ):
                    if not allowed or (
                        username not in allowed and str(user.id) not in allowed
                    ):
                        # Quiet refusal — don't reply to every group
                        # message, that'd be its own spam pattern.
                        # Just drop. Owner sees the access-denied
                        # branch fire in the bot's logs.
                        log.info(
                            "TG group-chat %s rejected for %s (not in "
                            "allowed_users; set HRANT_TELEGRAM_ALLOW_GROUPS "
                            "to allow all)",
                            chat.id, username,
                        )
                        return

                # Unified access (Phase B+C): consult access.is_telegram_allowed
                # which checks roles.json FIRST, falls back to legacy
                # allowed_users, then to the pairing system. The bot
                # never has to know about roles.json — it just asks
                # access.py "may this Telegram user speak?"
                #
                # Why: the previous gate only looked at allowed_users,
                # so a user already promoted to `trusted` in roles.json
                # was still silently blocked here. That mismatch is
                # what made the wife-add take 4 turns; access.py is
                # the single source of truth.
                from .access import is_telegram_allowed
                _initial_text = (update.message.text or update.message.caption or "").strip()
                decision = is_telegram_allowed(
                    user.id,
                    username=user.username or "",
                    legacy_allowed=allowed,
                    first_message=_initial_text,
                    label=user.full_name or user.username or "",
                )
                if not decision.allowed:
                    if decision.pairing_pending:
                        # Stranger re-ping: don't re-notify the owner
                        # (idempotent — they already got the code).
                        await update.message.reply_text(
                            "Your access request is still pending. "
                            "The owner has been notified — please wait."
                        )
                    elif decision.pairing_code:
                        await update.message.reply_text(
                            "Your access request has been sent to the "
                            "owner for approval. You'll be able to "
                            "chat once they approve."
                        )
                        try:
                            self._notify_owners_of_pairing(decision, user, _initial_text)
                        except Exception as e:
                            log.warning("owner pairing notification failed: %s", e)
                    else:
                        await update.message.reply_text("Access denied.")
                    return

                # Phase 10: speaker_id partitions per-Telegram-user
                # sessions + conversation memory + user profile. Each
                # Telegram user is a distinct speaker — Gor talking
                # to the bot and his wife talking to the bot get
                # totally separate threads even on the same bot.
                speaker_id = f"telegram:{user.id}"
                user_label = user.full_name or user.username or str(user.id)
                # Phase 11B: remember chat_id so we can deliver
                # scheduled / cross-speaker messages back to this
                # user later, even without their pinging us first.
                try:
                    from .contacts import remember_telegram_user
                    remember_telegram_user(
                        user.id,
                        update.message.chat.id,
                        username=user.username,
                        label=user_label,
                    )
                except Exception as e:
                    log.warning("contacts.remember_telegram_user failed: %s", e)

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

                # Media-group batching. Telegram delivers a photo album
                # as multiple Update objects, each with the same
                # `media_group_id`. Without batching, we ran the agent
                # once per photo and the user got N disjoint replies
                # for one logical message. Strategy:
                #   - first photo of the group: register a flush task
                #     after MEDIA_GROUP_DEBOUNCE_SEC of inactivity,
                #     remember this update as the one we'll reply to
                #   - subsequent photos: append their sha + extend
                #     the caption, then return without running the
                #     agent
                #   - flush task: pop the buffered shas + caption,
                #     overwrite `attachment_shas` / `text` with the
                #     merged values, fall through to the normal turn.
                # `_media_groups` is mutated from the bot's event-loop
                # thread only (this handler runs there), so no extra
                # lock needed.
                MEDIA_GROUP_DEBOUNCE_SEC = 1.2
                mg_id = getattr(update.message, "media_group_id", None)
                if mg_id:
                    entry = self._media_groups.get(mg_id)
                    if entry is None:
                        entry = {
                            "shas": list(attachment_shas),
                            "caption": text,
                            "first_update": update,
                            "speaker_id": f"telegram:{user.id}",
                            "username": username,
                            "user_sent_voice": user_sent_voice,
                            "flush_event": asyncio.Event(),
                        }
                        self._media_groups[mg_id] = entry

                        async def _flush_media_group(group_id: str) -> None:
                            try:
                                await asyncio.sleep(MEDIA_GROUP_DEBOUNCE_SEC)
                            except asyncio.CancelledError:
                                return
                            e = self._media_groups.pop(group_id, None)
                            if e is not None:
                                e["flush_event"].set()

                        asyncio.create_task(_flush_media_group(mg_id))
                        # First update of the group — wait for the
                        # debounce window to close, then proceed with
                        # the merged batch below.
                        await entry["flush_event"].wait()
                        attachment_shas = entry["shas"]
                        text = entry["caption"].strip()
                    else:
                        # Subsequent update for the same group — append
                        # and bail. The first update's handler is still
                        # in its sleep window and will pick up our
                        # appended shas on flush.
                        for sha in attachment_shas:
                            if sha not in entry["shas"]:
                                entry["shas"].append(sha)
                        if text and text not in entry["caption"]:
                            entry["caption"] = (
                                f"{entry['caption']}\n{text}".strip()
                                if entry["caption"] else text
                            )
                        return

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

                # Telegram's typing indicator decays after ~5s. Long
                # agent turns (10-30s+) outlive it, so the user sees
                # "no activity" while we're still working. Spawn a
                # background task that refreshes the action every 4s
                # for the duration of agent.run, then cancel it.
                async def _keep_typing(chat) -> None:
                    try:
                        while True:
                            await asyncio.sleep(4.0)
                            try:
                                await chat.send_action("typing")
                            except Exception:
                                # Quiet on transient network errors —
                                # the next iteration retries; the
                                # missed beat is invisible to the user.
                                pass
                    except asyncio.CancelledError:
                        return

                typing_task = asyncio.create_task(
                    _keep_typing(update.message.chat),
                )

                # `result` is declared here so the outer exception
                # handler can rescue the answer even if post-processing
                # (footer / stats / chunking) crashed AFTER the job
                # was marked completed. Without this, a bug in the
                # post-job stage caused the user to see "Error: ..."
                # and never receive the actual answer the agent had
                # already produced.
                result = None
                try:
                    from .agent import Agent

                    # Live progress placeholder — gets edited in-place as
                    # the agent thinks, runs tools, verifies. Replaced
                    # in-place with the final answer once the run
                    # completes (no separate "Done" + answer pair).
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
                    # Job tracking — every Telegram turn gets a
                    # durable record. reply_to carries the chat_id
                    # so if we ever want to cross-restart-notify
                    # ("sorry, I was interrupted, retry?") we have
                    # the routing info. Don't block the event loop —
                    # run the (sync) agent in a thread pool so the
                    # streamer can keep editing.
                    from .job_runner import run_tracked as _run_tracked
                    reply_to = {
                        "telegram_chat_id": update.message.chat.id,
                        "telegram_user_id": update.effective_user.id if update.effective_user else None,
                    }
                    # session_key isolates conversation threads by
                    # (bot, chat, user) — same user across different
                    # bots OR DM vs group on the same bot get distinct
                    # threads. Identity (speaker_id) stays per-user
                    # so roles / profile / facts remain unified.
                    session_key = (
                        f"telegram:{self.channel_id}:"
                        f"{update.message.chat.id}:{user.id}"
                    )
                    # Audit fix: bound concurrent agent.run calls. A
                    # spam burst (or a group with many active users)
                    # otherwise spawned one executor thread per
                    # message → N concurrent LLM streams. The
                    # semaphore queues excess requests; users still
                    # get responses, just serialised.
                    if self._run_semaphore is None:
                        self._run_semaphore = asyncio.Semaphore(
                            _MAX_CONCURRENT_AGENT_RUNS,
                        )
                    async with self._run_semaphore:
                        result, job_id = await running_loop.run_in_executor(
                            None,
                            lambda: _run_tracked(
                                agent,
                                text, project=None,
                                attachments=attachment_shas or None,
                                channel="telegram",
                                speaker_id=speaker_id,
                                session_key=session_key,
                                reply_to=reply_to,
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

                    # MEDIA: convention — agent can attach files to its
                    # reply by writing a line of the form
                    #   MEDIA:/absolute/path/to/file
                    # anywhere in the answer. Each such path is sent
                    # as a Telegram attachment (video / photo / audio /
                    # document picked by extension) and the line is
                    # stripped from the textual reply so the user
                    # doesn't see the raw path. Mirrors the convention
                    # used by other gateways so a skill written for
                    # one bot ports directly to ours.
                    answer, _media_sent = await _strip_and_send_media(
                        answer or "", update,
                    )
                    if _media_sent:
                        agent.progress(
                            "media",
                            f"sent {_media_sent} attachment(s) from MEDIA: lines",
                        ) if hasattr(agent, "progress") else None

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

                    # Replace the placeholder IN-PLACE with the final
                    # answer instead of leaving a stale `✅ Done` line
                    # next to a separate answer bubble. The user's
                    # mental model is: one message in, one message
                    # out. The placeholder becomes the answer.
                    #
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
                        final_chunks = [
                            answer if not tail else f"{answer}\n\n{tail}"
                        ]
                    else:
                        body_chunks: list[str] = []
                        i = 0
                        while i < len(answer):
                            body_chunks.append(answer[i:i + LIMIT])
                            i += LIMIT
                        if (
                            body_chunks
                            and len(body_chunks[-1]) + len(tail) + 2 <= LIMIT
                        ):
                            body_chunks[-1] = f"{body_chunks[-1]}\n\n{tail}"
                            final_chunks = body_chunks
                        else:
                            final_chunks = body_chunks + [tail]

                    # The placeholder carries the FIRST chunk so the
                    # user sees their answer appear where the "🧠
                    # Thinking…" bubble was. The rest (if any) is
                    # appended as fresh reply_text bubbles. Falls back
                    # to a regular reply if the edit fails (message
                    # too old, etc.) — better to ship two messages
                    # than to drop the answer.
                    head, *rest = final_chunks
                    edit_ok = False
                    try:
                        await stream.finalize(head)
                        edit_ok = True
                    except Exception as e_edit:
                        log.warning(
                            "TG placeholder finalize failed (%s); "
                            "falling back to reply_text",
                            e_edit,
                        )
                    if not edit_ok:
                        await update.message.reply_text(head)
                    for c in rest:
                        await update.message.reply_text(c)

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

                    _log_channel_message(
                        self.channel_id, username, text, answer,
                        speaker_id=speaker_id,
                    )

                except Exception as e:
                    # log.exception (not log.error) so the FULL
                    # traceback lands in journalctl. We had a slice
                    # error reach chat as `Error: slice(None, 200,
                    # None)` and there was no traceback to debug from;
                    # use the exception logger so the next occurrence
                    # is diagnosable.
                    log.exception(
                        "Telegram bot error processing message", exc_info=e,
                    )
                    # Sanitize the user-visible message — `str(e)` can
                    # be empty (NetworkError), a slice repr (a
                    # different bug we're tracking), or a multi-line
                    # traceback. Force a short, single-line, human-
                    # readable error.
                    err_text = (str(e) or type(e).__name__).strip()
                    err_text = err_text.splitlines()[0] if err_text else type(e).__name__
                    if len(err_text) > 300:
                        err_text = err_text[:297] + "…"
                    # If the agent already produced an answer before
                    # the crash (i.e. post-processing failed, not the
                    # agent run itself), ship the answer FIRST so the
                    # user sees the actual reply, then a small error
                    # note. Pre-fix, a post-processing slice bug ate
                    # the entire turn and the user only saw
                    # "Error: ..." with no answer at all.
                    rescued_answer = ""
                    try:
                        if result is not None:
                            rescued_answer = (getattr(result, "answer", "") or "").strip()
                    except Exception:
                        rescued_answer = ""
                    if rescued_answer:
                        try:
                            await update.message.reply_text(rescued_answer[:4000])
                        except Exception:
                            pass
                    await update.message.reply_text(
                        f"⚠️ {type(e).__name__}: {err_text}"
                    )
                finally:
                    # Stop the typing-indicator refresher whether the
                    # turn completed cleanly, crashed, or was cancelled.
                    # Without this, every long message leaks a coroutine
                    # that keeps sending typing every 4s forever.
                    typing_task.cancel()
                    try:
                        await typing_task
                    except (asyncio.CancelledError, Exception):
                        pass

            app.add_handler(CommandHandler("start", handle_start))
            app.add_handler(CommandHandler("status", handle_status))
            app.add_handler(CommandHandler("sessions", handle_sessions))
            app.add_handler(CommandHandler("help", handle_help))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            # Media handlers route through the same handle_message so caption
            # text + attachment shas reach the agent in one turn.
            app.add_handler(MessageHandler(filters.PHOTO, handle_message))
            app.add_handler(MessageHandler(filters.VOICE, handle_message))
            app.add_handler(MessageHandler(filters.AUDIO, handle_message))
            app.add_handler(MessageHandler(filters.Document.ALL, handle_message))
            app.add_handler(MessageHandler(filters.VIDEO, handle_message))
            app.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_message))
            # Inline-button callbacks (pairing approvals, future
            # self-mod approvals, etc.). The tg_interactive dispatcher
            # picks the right handler by callback_data prefix.
            app.add_handler(CallbackQueryHandler(handle_callback_query))

            # Eager-import the modules that register
            # tg_interactive callback handlers at module-load time
            # (`pair:`, `prop:`, `sched:`, `skill:`). Without this,
            # the first time the user pressed a button on a pre-
            # restart message we'd dispatch "no handler for 'pair'"
            # because the imports are otherwise lazy.
            try:
                from . import access as _access_eager  # noqa: F401
                from . import skills as _skills_eager  # noqa: F401
            except Exception as e:
                log.warning("callback module eager-import failed: %s", e)

            # Subscribe to self-mod proposal-created events so the
            # owner sees every new proposal as an inline-button DM.
            # Idempotent: a re-register on bot restart is a no-op.
            try:
                from . import self_modifier as _sm
                _sm.register_on_proposal_created(self._on_self_mod_proposal)
            except Exception as e:
                log.warning("self_modifier subscribe failed: %s", e)

            # Subscribe to scheduled-message events so the requester
            # gets a preview DM with a [❌ Cancel] button. Post-create
            # notification — the message is already scheduled; if the
            # owner spots a typo or a wrong time, one tap rescues it.
            try:
                from . import scheduled_messages as _sched
                _sched.register_on_message_scheduled(self._on_message_scheduled)
            except Exception as e:
                log.warning("scheduled_messages subscribe failed: %s", e)

            # Self-improvement loop: when the agent proposes a new
            # reusable skill, DM the owner with inline buttons —
            # [✅ Activate] [👀 Show] [❌ Delete]. The skill is
            # written disabled by default; only an owner press
            # toggles it live.
            try:
                from . import skills as _skills_mod
                _skills_mod.register_on_skill_proposed(self._on_skill_proposed)
            except Exception as e:
                log.warning("skills subscribe failed: %s", e)

            loop.run_until_complete(app.initialize())
            # Surface the slash-command list to the Telegram UI so
            # the user gets autocomplete when typing /. Best-effort —
            # a transient API error here shouldn't keep the bot down.
            try:
                from telegram import BotCommand as _BC
                _commands = [
                    _BC("start",    "Connection check"),
                    _BC("status",   "Your role + bot health"),
                    _BC("sessions", "Your recent conversation threads"),
                    _BC("help",     "What I can do"),
                ]
                loop.run_until_complete(app.bot.set_my_commands(_commands))
            except Exception as e:
                log.warning("set_my_commands failed: %s", e)
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


def _log_channel_message(
    channel_id: str,
    user: str,
    question: str,
    answer: str,
    *,
    speaker_id: str | None = None,
) -> None:
    """Append a channel interaction to the per-speaker session.
    Each Telegram user has their own session because each gets a
    distinct `speaker_id`."""
    try:
        from .sessions import DEFAULT_SPEAKER, SESSIONS
        sp = speaker_id or f"{channel_id}:{user}"
        turn = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": f"[{channel_id}:{user}] {question}",
            "answer": answer,
            "intent": "channel",
            "is_chat": False,
            "confidence": 0,
            "topics": [],
            "channel": channel_id,
            "speaker_id": sp,
        }
        SESSIONS.add_turn(turn, speaker_id=sp)
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

    def send_to_telegram_chat(self, chat_id: int, text: str) -> bool:
        """Send text to a SPECIFIC Telegram chat via whichever bot is
        running. Used by the boot-time interrupted-job notifier so a
        Telegram user who asked a question right before the server
        died sees a "sorry, I got interrupted" message instead of
        silence. Returns True on successful schedule, False when no
        bot is running."""
        for bot in self._bots.values():
            if bot.is_running:
                return bot.send_text(text, chat_id=chat_id)
        return False

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
