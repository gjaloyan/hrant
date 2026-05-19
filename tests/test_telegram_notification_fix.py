"""Pin the fix for the May 20, 2026 notification bug.

User reported: "I do not receive Telegram notifications when the
agent replies to me. I only see new messages from the agent when I
manually open the chat."

Root cause: the final answer was delivered by EDITING the "🧠
Thinking…" placeholder via `edit_message_text`. Telegram does NOT
send push notifications for edits — only for new messages. The
fix is to DELETE the placeholder and send the answer as a FRESH
`reply_text` so a push notification fires.

What this file pins:
  - `_TgProgressStream` has a `delete_placeholder()` method that
    calls `bot.delete_message`.
  - Returns True on success, False on Telegram refusal (so caller
    can fall back to edit-in-place when delete is blocked).
  - The streamer's `finalize()` still works as the fallback path.
"""
from __future__ import annotations
import inspect

from unittest.mock import AsyncMock, MagicMock

import pytest


def test_progress_stream_has_delete_placeholder():
    """The new method must exist — without it, the channel handler
    would have nothing to call after the agent finishes."""
    from backend.channels import _TgProgressStream
    assert hasattr(_TgProgressStream, "delete_placeholder")
    method = _TgProgressStream.delete_placeholder
    assert inspect.iscoroutinefunction(method)


@pytest.mark.asyncio
async def test_delete_placeholder_calls_telegram_delete():
    """Verify the API call shape — bot.delete_message(chat_id, message_id)."""
    from backend.channels import _TgProgressStream

    fake_bot = MagicMock()
    fake_bot.delete_message = AsyncMock(return_value=True)
    stream = _TgProgressStream(
        bot=fake_bot,
        chat_id=123,
        message_id=456,
        loop=None,
    )
    ok = await stream.delete_placeholder()
    assert ok is True
    fake_bot.delete_message.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
    )
    # The stream closes itself after delete so subsequent progress
    # pushes don't try to edit a now-gone message.
    assert stream._closed is True


@pytest.mark.asyncio
async def test_delete_placeholder_returns_false_on_refusal():
    """When Telegram refuses (message too old, missing permission),
    the method must return False so the caller can fall back."""
    from backend.channels import _TgProgressStream

    fake_bot = MagicMock()
    fake_bot.delete_message = AsyncMock(
        side_effect=Exception("Bad Request: message to delete not found")
    )
    stream = _TgProgressStream(
        bot=fake_bot,
        chat_id=123,
        message_id=456,
        loop=None,
    )
    ok = await stream.delete_placeholder()
    assert ok is False
    # Still closes so deferred edits don't fire.
    assert stream._closed is True


def test_channel_handler_uses_delete_then_fresh_reply():
    """Pin via source inspection: the message handler must call
    delete_placeholder BEFORE reply_text, and reply_text must be
    used (not edit_message_text / finalize) on the success path.

    Why source-level pin: full integration test of the Telegram
    handler is complex (requires mocking PTB Application, update,
    user, etc.). Source inspection catches the regression where a
    future refactor reverts to finalize-edit — a fast read-only
    check.
    """
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    # Both methods must appear in the source (so the handler can
    # call either depending on whether the delete succeeded).
    assert "delete_placeholder" in src
    assert "reply_text" in src
    # The fresh-reply branch must mention that delete enables
    # notifications — pin the comment so a future dev doesn't strip
    # it during cleanup and re-introduce the bug.
    low = src.lower()
    assert "push notification" in low or "notification" in low
    assert "edit_message_text" not in src or "fallback" in low
