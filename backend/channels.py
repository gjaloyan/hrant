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

                # Pull text from message OR caption (photos arrive with caption)
                text = (update.message.text or update.message.caption or "").strip()

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
                    agent = Agent()
                    result = agent.run(text, project=None, attachments=attachment_shas or None)
                    answer = result.answer or "(no answer)"

                    if len(answer) > 4000:
                        for i in range(0, len(answer), 4000):
                            await update.message.reply_text(answer[i:i + 4000])
                    else:
                        await update.message.reply_text(answer)

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


CHANNELS = ChannelManager()
