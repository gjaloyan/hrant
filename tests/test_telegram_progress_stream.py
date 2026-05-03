"""Telegram realtime progress streamer: enqueues progress events from
the agent thread and edits a single placeholder message in near-real-
time, throttled to avoid Telegram rate limits.
"""
from __future__ import annotations
import asyncio
import threading
import time

import pytest

from backend.channels import _TgProgressStream


class _FakeBot:
    """Stand-in for python-telegram-bot's Bot — only the one method we use."""

    def __init__(self) -> None:
        self.edits: list[dict] = []
        self.lock = threading.Lock()

    async def edit_message_text(self, **kwargs) -> None:
        with self.lock:
            self.edits.append(kwargs)


@pytest.mark.asyncio
async def test_push_renders_lines_and_edits_placeholder():
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    # Trigger from inside the same loop so we can await the edit.
    stream.push("core", "loading")
    # Give the scheduled coroutine a chance to run.
    await asyncio.sleep(0.05)
    assert bot.edits, "first push should produce one edit"
    assert "🧠 Thinking" in bot.edits[0]["text"]
    assert "core: loading" in bot.edits[0]["text"]


@pytest.mark.asyncio
async def test_throttle_coalesces_bursts_into_single_deferred_edit():
    """A burst of pushes inside the throttle window must NOT produce one
    edit per push — they should coalesce into one deferred flush that
    shows the latest snapshot."""
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.EDIT_INTERVAL_SEC = 0.1  # short for tests
    # Burst — 5 events back-to-back.
    for i in range(5):
        stream.push("step", f"event-{i}")
    # First edit goes out immediately; subsequent ones queue behind throttle.
    await asyncio.sleep(0.02)
    first_count = len(bot.edits)
    # Wait past throttle window — the deferred edit must flush.
    await asyncio.sleep(0.2)
    final_count = len(bot.edits)
    # Way fewer edits than pushes (coalesced):
    assert final_count <= 3, f"expected ≤3 edits for 5 pushes, got {final_count}"
    assert final_count >= first_count
    # The latest edit reflects the latest event.
    assert "event-4" in bot.edits[-1]["text"]


@pytest.mark.asyncio
async def test_buffer_caps_lines():
    """Long traces must truncate to MAX_LINES so we don't exceed
    Telegram's 4096-char message cap."""
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.MAX_LINES = 5
    stream.EDIT_INTERVAL_SEC = 0.05
    for i in range(20):
        stream.push("step", f"event-{i}")
    await asyncio.sleep(0.2)
    last = bot.edits[-1]["text"]
    # The first events were dropped — only the tail of 5 remains.
    assert "event-0" not in last
    assert "event-19" in last


@pytest.mark.asyncio
async def test_finalize_replaces_placeholder():
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.push("step", "running")
    await asyncio.sleep(0.05)
    await stream.finalize("✅ Done · 12 steps · 23s")
    # Last edit is the finalize summary, not the running trace.
    assert bot.edits[-1]["text"].startswith("✅ Done")
    # Subsequent pushes are no-ops after close.
    stream.push("post", "ignored")
    n_before = len(bot.edits)
    await asyncio.sleep(0.05)
    assert len(bot.edits) == n_before, "push after finalize must not edit"


@pytest.mark.asyncio
async def test_long_line_is_truncated():
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    long = "x" * 400
    stream.push("tool", long)
    await asyncio.sleep(0.05)
    # MAX_LINE_LEN cap kicks in.
    assert len(bot.edits[-1]["text"]) < len("🧠 Thinking…\n") + 400
    assert "…" in bot.edits[-1]["text"]


@pytest.mark.asyncio
async def test_edit_failure_swallowed_so_agent_doesnt_crash():
    """If Telegram rate-limits or returns 'message not modified', the
    streamer must keep going — the agent's progress stream is best-
    effort, not a hard dependency."""

    class _AngryBot:
        edits = 0
        async def edit_message_text(self, **kwargs):
            _AngryBot.edits += 1
            raise RuntimeError("Bad Request: message is not modified")

    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=_AngryBot(), chat_id=1, message_id=42, loop=loop)
    stream.push("core", "loading")
    stream.push("solve", "composing")
    await asyncio.sleep(0.05)
    # Didn't raise; we just stop seeing UI updates, which is OK.
    assert _AngryBot.edits >= 1
