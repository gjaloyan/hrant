"""Tests for the Hermes-style streaming "thinking block" in Telegram.

User requested:
  1. Show running tool calls in real-time inside the thinking
     message (as Hermes does — see screenshots in the chat).
  2. Don't delete the thinking process when the final answer
     arrives — keep it visible above the answer.

This pins:
  - `_TgProgressStream` keeps a list of tool entries and renders
    them as `<icon> <name>: "<preview>" [(×N)]` lines.
  - Consecutive duplicate tool calls (same name + same preview)
    collapse to a single line with a count.
  - `freeze()` exists and stops further edits without deleting
    the placeholder.
  - The 3-arg `push(event, message, tool_call=...)` signature
    threads tool-call payloads through.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _mk_stream():
    """Build a stream with no live loop — _schedule_edit becomes a
    no-op so we can inspect _render() snapshots without async."""
    s = MagicMock()  # the bot
    from backend.channels import _TgProgressStream
    stream = _TgProgressStream(bot=s, chat_id=1, message_id=2, loop=None)
    return stream


def test_initial_render_shows_thinking_placeholder():
    stream = _mk_stream()
    assert stream._render() == "🧠 Thinking…"


def test_tool_event_adds_entry():
    stream = _mk_stream()
    tc = SimpleNamespace(name="read_file", args={"path": "/tmp/x.py"})
    stream.push("tool", "", tool_call=tc)
    rendered = stream._render()
    assert "read_file" in rendered
    assert '/tmp/x.py' in rendered
    # Icon for read_file.
    assert rendered.startswith("📄")


def test_consecutive_duplicate_tools_collapse_to_count():
    """Hermes screenshot showed `read_file: "..." (×2)` — same tool +
    same arg preview collapses with a count."""
    stream = _mk_stream()
    same_args = {"path": "/home/hrant/.hermes/scripts/polymarket.py"}
    tc = SimpleNamespace(name="read_file", args=same_args)
    stream.push("tool", "", tool_call=tc)
    stream.push("tool", "", tool_call=tc)
    rendered = stream._render()
    # Single line, count suffix.
    assert rendered.count("read_file") == 1
    assert "(×2)" in rendered


def test_consecutive_different_args_do_not_collapse():
    """Same tool name but DIFFERENT preview => two separate lines."""
    stream = _mk_stream()
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="read_file", args={"path": "/a.py"},
    ))
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="read_file", args={"path": "/b.py"},
    ))
    rendered = stream._render()
    assert rendered.count("read_file") == 2
    assert "(×" not in rendered  # no collapse


def test_different_tools_do_not_collapse():
    stream = _mk_stream()
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="read_file", args={"path": "/a.py"},
    ))
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="terminal_exec", args={"command": "ls"},
    ))
    rendered = stream._render()
    assert "read_file" in rendered
    assert "terminal_exec" in rendered
    # Different lines.
    lines = [l for l in rendered.split("\n") if l.strip()]
    assert len(lines) >= 2


def test_tool_error_uses_red_marker():
    """Failed tool calls should be visually distinct so the user
    spots the failure at a glance."""
    stream = _mk_stream()
    tc = SimpleNamespace(name="read_file", args={"path": "/missing.py"})
    stream.push("tool_error", "", tool_call=tc)
    rendered = stream._render()
    assert "❌" in rendered


def test_stage_label_used_when_no_tools_yet():
    """If only stage events fire (no tool_call), the stage label
    shows as the placeholder body."""
    stream = _mk_stream()
    # `learning` is in the user-visible whitelist.
    stream.push("learning", "topic X")
    rendered = stream._render()
    assert "Learning" in rendered or "📚" in rendered


def test_tools_take_priority_over_stage():
    """Once any tool fires, the stage label disappears — the
    Hermes block becomes the running tool list and the stage
    becomes redundant."""
    stream = _mk_stream()
    stream.push("learning", "X")
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="read_file", args={"path": "/x.py"},
    ))
    rendered = stream._render()
    # No Learning line anymore — only the tool entry.
    assert "Learning" not in rendered
    assert "read_file" in rendered


def test_max_entries_cap_with_earlier_tail():
    """Past MAX_ENTRIES distinct entries, older ones collapse into
    a "… (+N earlier)" tail line so the message stays under
    Telegram's 4096-char limit."""
    from backend.channels import _TgProgressStream
    stream = _mk_stream()
    cap = _TgProgressStream.MAX_ENTRIES
    for i in range(cap + 5):
        stream.push("tool", "", tool_call=SimpleNamespace(
            name="search_files", args={"pattern": f"unique_pat_{i}"},
        ))
    rendered = stream._render()
    # Tail must surface the +5 dropped.
    assert "+5 earlier" in rendered
    # Cap respected — only `cap` entries after the tail line.
    lines = [l for l in rendered.split("\n") if l.strip()]
    assert len(lines) == cap + 1  # cap entries + tail line


def test_push_accepts_3_arg_signature():
    """The agent's progress callback uses (event, message, tool_call).
    Old test code uses (event, message); both must work without
    raising."""
    stream = _mk_stream()
    # 2-arg form (legacy)
    stream.push("learning", "X")
    # 3-arg form with None
    stream.push("learning", "Y", tool_call=None)
    # 3-arg form with payload
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="read_file", args={"path": "/x.py"},
    ))
    # No exception is the assertion.


@pytest.mark.asyncio
async def test_freeze_does_not_delete():
    """`freeze()` must STOP accepting updates and flush one final
    edit. It must NOT delete the placeholder — the thinking trace
    needs to stay visible above the final answer."""
    stream = _mk_stream()
    stream.bot.edit_message_text = AsyncMock()
    stream.bot.delete_message = AsyncMock()  # should not be called
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="read_file", args={"path": "/a.py"},
    ))
    await stream.freeze()
    # Edit fired (final flush).
    assert stream.bot.edit_message_text.await_count >= 1
    # Delete NOT fired — the trace stays visible.
    stream.bot.delete_message.assert_not_called()
    # Stream is closed → further pushes are no-ops.
    assert stream._closed is True
    stream.push("tool", "", tool_call=SimpleNamespace(
        name="terminal_exec", args={"command": "ls"},
    ))
    # No additional entry added after freeze.
    rendered = stream._render()
    assert "terminal_exec" not in rendered
