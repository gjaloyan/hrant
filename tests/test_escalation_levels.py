"""Unit tests for backend.escalation — the level decision is pure logic."""
from __future__ import annotations

from backend.escalation import (
    Level, decide_level, should_run_verifier, tool_names_from_trace,
)


class _Step:
    """Minimal stand-in for a ThinkingStep with a ToolCallDetail."""
    def __init__(self, event, name):
        self.event = event
        self.tool_call = type("TC", (), {"name": name})() if name else None


def test_decide_level_fast_chat_is_l0():
    assert decide_level(was_fast_chat=True, tool_names=[]) is Level.L0_CHAT


def test_decide_level_pure_action_is_l1():
    assert decide_level(
        was_fast_chat=False, tool_names=["save_user_fact"]
    ) is Level.L1_ACTION
    assert decide_level(
        was_fast_chat=False, tool_names=["set_setting", "schedule_message"]
    ) is Level.L1_ACTION


def test_decide_level_info_tool_is_l2():
    # any non-pure-action tool drags the turn to L2
    assert decide_level(
        was_fast_chat=False, tool_names=["save_user_fact", "web_search"]
    ) is Level.L2_TASK
    assert decide_level(
        was_fast_chat=False, tool_names=["terminal_exec"]
    ) is Level.L2_TASK
    # sandbox_exec is an execute tool but PRODUCES verifiable output -> L2
    assert decide_level(
        was_fast_chat=False, tool_names=["sandbox_exec"]
    ) is Level.L2_TASK


def test_decide_level_no_tools_full_path_is_l2():
    # escalated off the fast path but used no tools -> verify the reasoning
    assert decide_level(
        was_fast_chat=False, tool_names=[]
    ) is Level.L2_TASK


def test_should_run_verifier_by_level():
    assert should_run_verifier(Level.L0_CHAT) is False
    assert should_run_verifier(Level.L1_ACTION) is False
    assert should_run_verifier(Level.L2_TASK) is True


def test_tool_names_from_trace_counts_completed_steps_only():
    trace = [
        _Step("tool_starting", "save_user_fact"),  # not yet run -> ignored
        _Step("tool", "save_user_fact"),
        _Step("tool", "web_search"),
        _Step("assistant", None),                  # not a tool step
    ]
    assert tool_names_from_trace(trace) == ["save_user_fact", "web_search"]


def test_tool_names_from_trace_dict_fallback():
    class _DictStep:
        event = "tool"
        tool_call = {"name": "fetch_url"}
    assert tool_names_from_trace([_DictStep()]) == ["fetch_url"]


def test_tool_names_from_trace_empty():
    assert tool_names_from_trace([]) == []
    assert tool_names_from_trace(None) == []
