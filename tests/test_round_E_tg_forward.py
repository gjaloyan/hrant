"""Round E — TG bot forward when WebUI composes a turn as
channel=telegram.

The WebUI dropdown lets the user pick "📱 telegram" before sending.
That tags conversation memory + the turn artefact under the TG
bucket, but to actually keep the user's TG thread continuous we also
have to deliver the agent's answer THROUGH the Telegram bot. This
module pins:
  - TelegramBot tracks the last chat_id it received a message from
  - TelegramBot.send_text uses that chat_id (or an explicit one) and
    chunks long bodies at the 4096-char Telegram limit
  - ChannelManager.send_to_first_telegram routes through the first
    running bot
  - api/chat.py calls the manager helper after agent.run returns
    when target_channel == "telegram"
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# --- TelegramBot.send_text -----------------------------------------------


def test_send_text_no_loop_returns_false():
    """Bot that hasn't started yet (no _loop, no _app) must refuse
    to send rather than crash. The WebUI surfaces this as a
    'TG forward skipped' progress message."""
    from backend.channels import TelegramBot
    bot = TelegramBot(token="fake", channel_id="x")
    assert bot.send_text("hello", chat_id=42) is False


def test_send_text_no_chat_id_when_never_messaged_returns_false():
    """A running bot that has NEVER received a message has no
    `_last_chat_id` to reply to. send_text must surface that
    cleanly so the WebUI can warn the user 'send a TG message
    first'."""
    from backend.channels import TelegramBot
    bot = TelegramBot(token="fake", channel_id="x")
    bot._running = True
    bot._app = MagicMock()
    bot._loop = MagicMock()
    # _last_chat_id stays None — no incoming messages yet.
    assert bot.send_text("hi") is False


def test_send_text_uses_last_chat_id_when_not_specified(monkeypatch):
    from backend.channels import TelegramBot
    bot = TelegramBot(token="fake", channel_id="x")
    bot._running = True
    bot._app = MagicMock()
    bot._loop = MagicMock()
    bot._last_chat_id = 12345

    captured: list[tuple[int, str]] = []

    async def fake_send(*, chat_id, text):
        captured.append((chat_id, text))

    bot._app.bot = MagicMock()
    bot._app.bot.send_message = fake_send

    # Don't actually schedule on a real loop — capture the coroutine.
    scheduled: list = []

    def fake_run(coro, _loop):
        scheduled.append(coro)

        class _F:
            def result(self, timeout=None):
                pass
        return _F()

    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", fake_run)

    assert bot.send_text("hello world") is True
    # Drain the scheduled coroutine to verify the right chat_id + text.
    import asyncio
    assert len(scheduled) == 1
    asyncio.run(scheduled[0])
    assert captured == [(12345, "hello world")]


def test_send_text_explicit_chat_id_overrides_last(monkeypatch):
    from backend.channels import TelegramBot
    bot = TelegramBot(token="fake", channel_id="x")
    bot._running = True
    bot._app = MagicMock()
    bot._loop = MagicMock()
    bot._last_chat_id = 100

    captured: list[tuple[int, str]] = []

    async def fake_send(*, chat_id, text):
        captured.append((chat_id, text))

    bot._app.bot = MagicMock()
    bot._app.bot.send_message = fake_send

    scheduled: list = []
    monkeypatch.setattr(
        "asyncio.run_coroutine_threadsafe",
        lambda coro, loop: scheduled.append(coro) or MagicMock(),
    )

    assert bot.send_text("hi", chat_id=999) is True
    import asyncio
    asyncio.run(scheduled[0])
    assert captured[0][0] == 999  # explicit chat_id wins


def test_send_text_chunks_long_body(monkeypatch):
    """Telegram caps a single send at 4096 chars; send_text must
    split a 9000-char answer into 3 chunks (~4000 each) and send
    them in sequence. Otherwise long agent answers would 400."""
    from backend.channels import TelegramBot
    bot = TelegramBot(token="fake", channel_id="x")
    bot._running = True
    bot._app = MagicMock()
    bot._loop = MagicMock()
    bot._last_chat_id = 1

    sent: list[str] = []

    async def fake_send(*, chat_id, text):
        sent.append(text)

    bot._app.bot = MagicMock()
    bot._app.bot.send_message = fake_send

    scheduled: list = []
    monkeypatch.setattr(
        "asyncio.run_coroutine_threadsafe",
        lambda coro, loop: scheduled.append(coro) or MagicMock(),
    )

    body = "x" * 9000
    assert bot.send_text(body) is True
    import asyncio
    asyncio.run(scheduled[0])
    assert len(sent) == 3
    assert all(len(s) <= 4000 for s in sent)
    assert "".join(sent) == body


def test_send_text_skips_empty_body():
    from backend.channels import TelegramBot
    bot = TelegramBot(token="fake", channel_id="x")
    bot._running = True
    bot._app = MagicMock()
    bot._loop = MagicMock()
    bot._last_chat_id = 1
    assert bot.send_text("") is False
    assert bot.send_text("   ") is False


# --- ChannelManager.send_to_first_telegram ------------------------------


def test_manager_send_to_first_telegram_no_bots_returns_false():
    from backend.channels import ChannelManager
    mgr = ChannelManager()
    assert mgr.send_to_first_telegram("hi") is False


def test_manager_send_to_first_telegram_routes_to_running_bot(monkeypatch):
    """Multi-bot setup: the manager picks the first one whose
    `is_running` is True. Stopped bots are skipped."""
    from backend.channels import ChannelManager
    mgr = ChannelManager()
    stopped = MagicMock()
    stopped.is_running = False
    stopped.send_text.return_value = True
    running = MagicMock()
    running.is_running = True
    running.send_text.return_value = True
    mgr._bots = {"a": stopped, "b": running}
    assert mgr.send_to_first_telegram("hi") is True
    stopped.send_text.assert_not_called()
    running.send_text.assert_called_once_with("hi")


def test_manager_send_to_first_telegram_propagates_failure():
    from backend.channels import ChannelManager
    mgr = ChannelManager()
    bad = MagicMock()
    bad.is_running = True
    bad.send_text.return_value = False
    mgr._bots = {"a": bad}
    assert mgr.send_to_first_telegram("hi") is False


# --- api/chat.py wiring --------------------------------------------------


def test_chat_module_calls_send_to_first_telegram_on_telegram_channel():
    """Smoke-check at the source level: the runner must call
    CHANNELS.send_to_first_telegram when target_channel ==
    'telegram'. The ai-flow integration test would need a real
    asyncio loop + SSE stream which is overkill for this contract."""
    import inspect
    import backend.api.chat as chat_mod
    src = inspect.getsource(chat_mod)
    assert 'target_channel == "telegram"' in src
    assert "send_to_first_telegram" in src
    # And it surfaces a helpful progress event when no bot is running
    # so the WebUI can show why TG stayed quiet.
    assert "TG forward skipped" in src
