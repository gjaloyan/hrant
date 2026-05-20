"""Telegram realtime progress streamer.

Pre-fix the streamer accumulated every progress event the agent
emitted (`core`, `chat`, `cleanup`, `solve`, `memory_save`, raw tool
JSON results, …) and rendered them as a multi-line debug log inside
the chat bubble. Users saw a debug trace where they expected an
answer. The streamer now:

  - keeps a SINGLE latest-action line (no accumulation)
  - only renders events in the user-visible whitelist
    (`learning`, `learned`, `subtask`, `tool_starting`, `self_critic`,
    `found`); internal pipeline events are dropped silently
  - replaces (not appends to) the placeholder body on each tick
"""
from __future__ import annotations
import asyncio
import threading

import pytest

from backend.channels import _TgProgressStream, _USER_VISIBLE_PROGRESS_EVENTS


class _FakeBot:
    """Stand-in for python-telegram-bot's Bot — only the one method we use."""

    def __init__(self) -> None:
        self.edits: list[dict] = []
        self.lock = threading.Lock()

    async def edit_message_text(self, **kwargs) -> None:
        with self.lock:
            self.edits.append(kwargs)


@pytest.mark.asyncio
async def test_whitelisted_push_renders_label():
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.push("learning", "Armenian politics")
    await asyncio.sleep(0.05)
    assert bot.edits, "whitelisted event should produce one edit"
    text = bot.edits[0]["text"]
    # Human-readable label from the whitelist.
    expected_label = _USER_VISIBLE_PROGRESS_EVENTS["learning"]
    assert expected_label in text
    assert "Armenian politics" in text


@pytest.mark.asyncio
async def test_non_whitelisted_events_silently_dropped():
    """Internal pipeline events (core / chat / cleanup / memory_save /
    raw tool results) MUST NOT reach the chat. Pre-fix, the user saw
    `core: loading`, `chat: chatting...`, `memory_save: ...` and other
    debug-log noise in their conversation."""
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    for ev, msg in [
        ("core", "loading core memory"),
        ("chat", "chatting..."),
        ("cleanup", "ready"),
        ("memory_save", "remembered 3 facts"),
        ("solve", "composing answer..."),
        ("strategy", "type=task"),
    ]:
        stream.push(ev, msg)
    await asyncio.sleep(0.1)
    # No event in the burst was whitelisted → no edits at all.
    assert bot.edits == [], (
        "internal pipeline events leaked to chat: "
        f"{bot.edits!r}"
    )


@pytest.mark.asyncio
async def test_latest_action_replaces_previous_one():
    """The placeholder is a status indicator, not a transcript.
    A second whitelisted event must REPLACE the first label, not
    append to a growing list."""
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.EDIT_INTERVAL_SEC = 0.05  # short for tests
    stream.push("learning", "topic-A")
    await asyncio.sleep(0.02)
    stream.push("tool_starting", "read_file")
    await asyncio.sleep(0.15)
    last = bot.edits[-1]["text"]
    # The second event replaced the first — no "topic-A" left.
    assert "topic-A" not in last
    assert "read_file" in last


@pytest.mark.asyncio
async def test_throttle_coalesces_bursts():
    """A burst of pushes inside the throttle window must NOT produce
    one edit per push — they should coalesce into a single deferred
    flush that shows the latest snapshot."""
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.EDIT_INTERVAL_SEC = 0.1
    for i in range(5):
        stream.push("subtask", f"step-{i}")
    await asyncio.sleep(0.02)
    first_count = len(bot.edits)
    await asyncio.sleep(0.25)
    final_count = len(bot.edits)
    assert final_count <= 3, f"expected ≤3 edits for 5 pushes, got {final_count}"
    assert final_count >= first_count
    assert "step-4" in bot.edits[-1]["text"]


@pytest.mark.asyncio
async def test_finalize_replaces_placeholder():
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.push("learning", "topic")
    await asyncio.sleep(0.05)
    await stream.finalize("Here is the final answer.")
    assert bot.edits[-1]["text"] == "Here is the final answer."
    # Subsequent pushes are no-ops after close — even whitelisted ones.
    stream.push("learning", "ignored")
    n_before = len(bot.edits)
    await asyncio.sleep(0.05)
    assert len(bot.edits) == n_before, "push after finalize must not edit"


@pytest.mark.asyncio
async def test_long_stage_message_is_truncated():
    """A long non-tool stage message must be truncated so the
    placeholder doesn't blow Telegram's 4096-char cap on a single
    update. The Hermes-style refactor caps stage lines at ~120 chars
    (inline constant in `push()`)."""
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    long = "x" * 400
    stream.push("learning", long)
    await asyncio.sleep(0.05)
    text = bot.edits[-1]["text"]
    assert len(text) <= 130  # ~120-char cap with the label prefix
    assert "…" in text


@pytest.mark.asyncio
async def test_edit_failure_swallowed():
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
    stream.push("learning", "topic")
    stream.push("tool_starting", "read_file")
    await asyncio.sleep(0.05)
    assert _AngryBot.edits >= 1


@pytest.mark.asyncio
async def test_tool_lifecycle_not_double_counted():
    """A single tool execution emits BOTH `tool_starting` (pre-call)
    and `tool` (post-call) progress events. Pre-fix the coalescer
    treated them as two separate calls and the user saw `(×2)` next
    to every tool line — looking like the LLM was running each tool
    twice when it wasn't. After the fix, the lifecycle pair renders
    as `count=1` (the post-call `tool` event updates the entry in
    place instead of opening a new one)."""
    from types import SimpleNamespace
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    detail = SimpleNamespace(name="terminal_exec", args={"cmd": "ls -la"})
    # Pre-call event (live indicator).
    stream.push("tool_starting", "terminal_exec(cmd)", tool_call=detail)
    # Post-call event (callback after the tool returns).
    stream.push("tool", "terminal_exec(cmd) -> output...", tool_call=detail)
    await asyncio.sleep(0.15)
    assert len(stream._entries) == 1, (
        f"lifecycle pair must coalesce; got {stream._entries}"
    )
    assert stream._entries[0]["count"] == 1, (
        f"single execution must render count=1, not ×2: {stream._entries[0]}"
    )


@pytest.mark.asyncio
async def test_tool_lifecycle_error_flips_icon():
    """The post-call `tool_error` event must flip the entry's icon to
    ❌ in place — without bumping the count."""
    from types import SimpleNamespace
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    detail = SimpleNamespace(name="terminal_exec", args={"cmd": "bad"})
    stream.push("tool_starting", "terminal_exec(cmd)", tool_call=detail)
    stream.push("tool_error", "terminal_exec(cmd) -> err", tool_call=detail)
    await asyncio.sleep(0.15)
    assert len(stream._entries) == 1
    assert stream._entries[0]["count"] == 1
    assert stream._entries[0]["icon"] == "❌"


@pytest.mark.asyncio
async def test_two_genuine_consecutive_identical_calls_still_coalesce():
    """A real double-call (LLM emits the same tool_use block twice in
    one iteration) must still render as `count=2`. We need this to
    survive the lifecycle-coalescing fix above."""
    from types import SimpleNamespace
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    detail = SimpleNamespace(name="search_knowledge", args={"q": "armenia"})
    # Two genuine starts back-to-back, BEFORE either completes.
    stream.push("tool_starting", "search_knowledge(q)", tool_call=detail)
    stream.push("tool_starting", "search_knowledge(q)", tool_call=detail)
    await asyncio.sleep(0.15)
    assert len(stream._entries) == 1
    assert stream._entries[0]["count"] == 2, (
        f"genuine duplicate must coalesce to count=2: {stream._entries[0]}"
    )


@pytest.mark.asyncio
async def test_two_genuine_lifecycle_pairs_collapse_to_count():
    """When the LLM REALLY emits the same tool_use block twice in a
    turn, each execution produces a full `tool_starting → tool`
    lifecycle pair. The user wants to see "ran twice" as a single
    line with `(×2)`. That's the explicit signal the LLM duplicated;
    showing two separate lines would just look like more noise."""
    from types import SimpleNamespace
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    detail = SimpleNamespace(name="search_knowledge", args={"q": "x"})
    # Two full lifecycles, same args.
    stream.push("tool_starting", "search_knowledge(q)", tool_call=detail)
    stream.push("tool", "search_knowledge(q) -> ...", tool_call=detail)
    stream.push("tool_starting", "search_knowledge(q)", tool_call=detail)
    stream.push("tool", "search_knowledge(q) -> ...", tool_call=detail)
    await asyncio.sleep(0.15)
    assert len(stream._entries) == 1, (
        f"two lifecycles for same args must coalesce to ONE row; "
        f"got {stream._entries}"
    )
    assert stream._entries[0]["count"] == 2, (
        f"two real executions must show count=2: {stream._entries[0]}"
    )


@pytest.mark.asyncio
async def test_empty_detail_renders_label_only():
    """Some whitelisted events fire with no detail string (e.g. just
    `tool_starting`). We render the label alone instead of a bare
    `tool_starting: ` line with a trailing colon."""
    bot = _FakeBot()
    loop = asyncio.get_running_loop()
    stream = _TgProgressStream(bot=bot, chat_id=1, message_id=42, loop=loop)
    stream.push("tool_starting", "")
    await asyncio.sleep(0.05)
    text = bot.edits[-1]["text"]
    assert _USER_VISIBLE_PROGRESS_EVENTS["tool_starting"] in text
    assert not text.endswith(": ")
