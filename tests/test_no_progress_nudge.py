"""Per-turn no-progress nudge.

When the LLM makes N consecutive tool calls without any
state-advancing call (set_setting, start_background_job,
ask_user, MEDIA-emit, …), the next tool result gets a "NUDGE"
banner prepended telling the model to either execute or
ask_user. Catches the 'inspect-without-execute drift' from the
2026-05-26 terminal-bench turns.
"""
from __future__ import annotations

import pytest


def _make_test_registry():
    """Registry with one inspection-class tool (`read_file`) and one
    advancing tool (`set_setting`). Both use stubbed handlers."""
    from backend.tool_registry import ToolRegistry
    reg = ToolRegistry()

    def _read(**k):
        return "<file content>"

    def _set(**k):
        return {"ok": True}

    reg.register_func(
        name="read_file", description="",
        input_schema={"type": "object", "properties": {
            "path": {"type": "string"}}},
        handler=_read,
    )
    reg.register_func(
        name="set_setting", description="",
        input_schema={"type": "object", "properties": {
            "key": {"type": "string"}, "value": {"type": "string"}}},
        handler=_set,
    )
    return reg


def test_first_four_inspections_do_not_nudge():
    reg = _make_test_registry()
    for i in range(4):
        text, _ = reg.execute("read_file", {"path": f"f{i}.txt"})
        assert "NUDGE" not in text, f"nudge fired prematurely at call {i+1}"


def test_fifth_inspection_call_triggers_nudge():
    reg = _make_test_registry()
    for i in range(4):
        reg.execute("read_file", {"path": f"f{i}.txt"})
    text, _ = reg.execute("read_file", {"path": "f5.txt"})
    assert "NUDGE" in text or "no execute" in text.lower(), (
        "nudge should fire at the 5th non-advancing call"
    )


def test_nudge_does_not_repeat_until_counter_resets():
    """Once nudged, calling the same inspection tool again should
    NOT re-nudge — the agent has been told once; re-nudging is
    noise. The next nudge fires after the counter resets via an
    advancing call."""
    reg = _make_test_registry()
    for i in range(5):
        reg.execute("read_file", {"path": f"f{i}.txt"})
    text, _ = reg.execute("read_file", {"path": "f6.txt"})
    assert "NUDGE" not in text, (
        "nudge should not repeat consecutively; one warning is enough"
    )


def test_advancing_call_resets_counter():
    """An advancing tool (set_setting, start_background_job, etc.)
    resets the no-progress counter. After reset, the agent gets
    another 4 inspection calls before the next nudge."""
    reg = _make_test_registry()
    for i in range(5):
        reg.execute("read_file", {"path": f"f{i}.txt"})  # nudge fires here
    reg.execute("set_setting", {"key": "x", "value": "y"})  # reset
    for i in range(4):
        text, _ = reg.execute("read_file", {"path": f"g{i}.txt"})
        assert "NUDGE" not in text, f"nudge fired too early after reset at {i+1}"
    text, _ = reg.execute("read_file", {"path": "g5.txt"})
    assert "NUDGE" in text or "no execute" in text.lower()


def test_advancing_call_result_has_no_nudge():
    reg = _make_test_registry()
    for i in range(5):
        reg.execute("read_file", {"path": f"f{i}.txt"})
    text, _ = reg.execute("set_setting", {"key": "x", "value": "y"})
    assert "NUDGE" not in text


def test_nudge_state_resets_at_turn_boundary():
    """Each turn starts with a fresh counter."""
    from backend import tool_registry as tr
    reg = _make_test_registry()
    for i in range(5):
        reg.execute("read_file", {"path": f"f{i}.txt"})
    tr.reset_per_turn_call_cache()  # also resets nudge state
    for i in range(4):
        text, _ = reg.execute("read_file", {"path": f"g{i}.txt"})
        assert "NUDGE" not in text


def test_nudge_message_names_concrete_recovery_options():
    """The nudge text must tell the agent its options — otherwise
    it's just complaining without direction."""
    reg = _make_test_registry()
    for i in range(4):
        reg.execute("read_file", {"path": f"f{i}.txt"})
    text, _ = reg.execute("read_file", {"path": "f5.txt"})
    low = text.lower()
    # The nudge must mention either ask_user OR an execute-class
    # action so the agent has a concrete escape hatch.
    assert "ask_user" in text or "execute" in low
