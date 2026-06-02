"""Tests for the mid-turn ACTION DRIFT marker injected into tool
results.

The end-of-turn `_decide_self_correction` only inspects the final
answer — so a turn that drifts through 40 read-only probes still
pays for 40 LLM round-trips before the corrective fires. The drift
marker nudges the LLM mid-flow: at 6 consecutive read-only tool
calls without any execute-class action, append a warning to the
tool result; at 12, escalate it.

Tests construct a tiny in-process loop emulating what the
unified-agent code does per tool call (increment counter on
read-only, reset on execute-class) and assert the marker text
appears at the right thresholds.
"""
from __future__ import annotations


def _drift_loop(tool_names: list[str]) -> list[str]:
    """Emulate the drift detector for a sequence of tool calls.

    Returns the list of markers (empty string when no marker fired
    for that step) — same shape as `marker_drift` in
    `_execute_with_progress` for each iteration.
    """
    from backend.endpoint_check import _EXECUTE_TOOLS as _EXEC_T
    state = {"consecutive_readonly": 0, "marker_fired": 0}
    markers: list[str] = []
    for name in tool_names:
        if name in _EXEC_T:
            state["consecutive_readonly"] = 0
        else:
            state["consecutive_readonly"] += 1
        cur = state["consecutive_readonly"]
        marker = ""
        if cur == 6 and state["marker_fired"] == 0:
            state["marker_fired"] = 1
            marker = "ACTION DRIFT"
        elif cur == 12 and state["marker_fired"] < 2:
            state["marker_fired"] = 2
            marker = "STILL DRIFTING"
        markers.append(marker)
    return markers


def test_no_marker_under_threshold():
    """5 read-only calls in a row stay below the 6-threshold — no
    marker. Quick lookups should not nag the LLM."""
    out = _drift_loop(["read_file"] * 5)
    assert out == [""] * 5


def test_first_marker_fires_at_six():
    """6th consecutive read-only tool gets the ACTION DRIFT warning."""
    out = _drift_loop(["read_file"] * 8)
    # markers at positions 0..4: empty. Position 5 (6th call) = ACTION DRIFT.
    # Positions 6 and 7: empty (single firing per threshold).
    assert out[:5] == [""] * 5
    assert out[5] == "ACTION DRIFT"
    assert out[6] == ""
    assert out[7] == ""


def test_second_marker_fires_at_twelve():
    """If the agent ignores ACTION DRIFT and pushes past 12 read-only
    calls in a row, escalate to STILL DRIFTING."""
    out = _drift_loop(["terminal_exec"] * 14)
    assert out[5] == "ACTION DRIFT"
    assert out[6:11] == [""] * 5
    assert out[11] == "STILL DRIFTING"
    assert out[12:] == [""] * 2


def test_execute_class_action_resets_counter():
    """An execute-class action (start_background_job) resets the
    consecutive count — investigation that LEADS to action is fine,
    we only flag investigation that goes nowhere."""
    seq = (
        ["read_file"] * 5             # 5 reads, no marker yet
        + ["start_background_job"]    # action → counter reset
        + ["terminal_exec"] * 5       # 5 more reads, still no marker
    )
    out = _drift_loop(seq)
    assert all(m == "" for m in out), out


def test_marker_does_not_double_fire():
    """Once the 6-marker has fired, it doesn't re-fire — even if the
    agent keeps drifting past 6 with no progress."""
    out = _drift_loop(["read_file"] * 10)
    assert out.count("ACTION DRIFT") == 1
    assert out.count("STILL DRIFTING") == 0


def test_kick_supervisor_counts_as_execute_class():
    """Sanity: kick_supervisor (added in v0.16.413) is in the
    execute-class set, so calling it resets the drift counter."""
    seq = (
        ["read_file"] * 5
        + ["kick_supervisor"]
        + ["terminal_exec"] * 5
    )
    out = _drift_loop(seq)
    assert all(m == "" for m in out)
