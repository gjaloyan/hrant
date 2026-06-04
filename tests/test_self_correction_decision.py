"""Tests for `_decide_self_correction` — the gating logic for the
unified-agent self-correction re-prompt.

The branch covers two failure modes:

  (A) Zero-tool turn that claims an action without backing it (the
      original self-correction; existing tests in
      `test_self_correction.py` cover the `unbacked_action_claim`
      judge underneath).

  (B) Toolful turn that ran a long read-only investigation and gave
      up without taking the state-changing action the user's
      request required. This is the failure mode that caused the
      2026-06-02 bench giveup (16 read_file / terminal_exec /
      grep / locate_symbol calls + "no verifiable result" answer
      with no `start_background_job`).

The gating function is pure logic + LLM judgments, so we stub the
judges (`unbacked_action_claim` and `endpoint_met`) to drive each
branch.
"""
from __future__ import annotations

import pytest


def _patch_judges(monkeypatch, *, claim="", endpoint_met=True):
    """Stub both LLM judges with deterministic returns."""
    monkeypatch.setattr(
        "backend.endpoint_check.unbacked_action_claim",
        lambda task, answer, tool_names: claim,
    )
    monkeypatch.setattr(
        "backend.endpoint_check.endpoint_met",
        lambda *, task, answer, tool_names: endpoint_met,
    )


def test_empty_answer_no_correction():
    """An empty answer never triggers a correction, regardless of
    tool history. Decision short-circuits before any judge call."""
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="anything", answer="", turn_tools=["read_file"],
    )
    assert tag == ""
    assert corrective == ""


def test_zero_tools_no_claim_no_correction(monkeypatch):
    """Zero tools + clean informational answer (judge says no claim)
    -> no correction."""
    _patch_judges(monkeypatch, claim="")
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="What is the capital of France?",
        answer="Paris.",
        turn_tools=[],
    )
    assert tag == ""
    assert corrective == ""


def test_zero_tools_with_unbacked_claim_triggers_correction(monkeypatch):
    """Zero tools + answer claims an action -> branch (A) fires."""
    _patch_judges(monkeypatch, claim="saving your name")
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="Remember that my name is Gor.",
        answer="Done, I've saved that.",
        turn_tools=[],
    )
    assert "unbacked claim" in tag
    assert "saving your name" in tag
    assert "no tool this turn" in corrective
    # Must NOT include the toolful failure-mode language.
    assert "read-only" not in corrective


def test_toolful_with_execute_tool_no_correction(monkeypatch):
    """Even if endpoint_met says False, a turn that called an
    execute-class tool (e.g. start_background_job) is considered to
    have taken action — no second-guessing."""
    _patch_judges(monkeypatch, endpoint_met=False)
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="Run terminal-bench on 2 tasks.",
        answer="Job bg-abc123 dispatched.",
        turn_tools=["read_file", "start_background_job"],
    )
    assert tag == ""
    assert corrective == ""


def test_toolful_readonly_endpoint_not_met_triggers_correction(monkeypatch):
    """The 2026-06-02 bench giveup: 16 read-only tools, agent gives
    up with "no verifiable result", endpoint_met says False -> the
    new branch (B) fires."""
    _patch_judges(monkeypatch, endpoint_met=False)
    from backend.unified_agent import _decide_self_correction
    tools = [
        "load_skill", "terminal_exec", "terminal_exec",
        "get_background_job", "terminal_exec", "terminal_exec",
        "terminal_exec", "terminal_exec", "terminal_exec",
    ]
    tag, corrective = _decide_self_correction(
        task="Run terminal-bench on 2 tasks.",
        answer="I cannot confirm a real Terminal-Bench result.",
        turn_tools=tools,
    )
    assert "toolful no-deliver" in tag
    assert str(len(tools)) in tag
    # The corrective must reference both paths the agent can take.
    assert "start_background_job" in corrective
    assert "ask_user" in corrective
    assert "read-only" in corrective


def test_toolful_readonly_endpoint_met_no_correction(monkeypatch):
    """Read-only tools + honest blocker statement (endpoint_met says
    the request was satisfied by an honest 'cannot do X because Y +
    here is a plan') -> no correction."""
    _patch_judges(monkeypatch, endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="Run terminal-bench on 2 tasks.",
        answer=(
            "I can't run it: OPENAI_API_KEY is missing in the container "
            ".env. Two options: (a) set it via `set_setting`, "
            "(b) switch to --agent oracle for a sanity check."
        ),
        turn_tools=["read_file", "terminal_exec"],
    )
    assert tag == ""
    assert corrective == ""


def test_toolful_correction_lists_tool_names_capped(monkeypatch):
    """When >6 tools fired, the corrective truncates the list with
    a +N more counter so the prompt stays bounded."""
    _patch_judges(monkeypatch, endpoint_met=False)
    from backend.unified_agent import _decide_self_correction
    tools = [f"tool_{i}" for i in range(10)]
    _, corrective = _decide_self_correction(
        task="Do the thing.",
        answer="Investigation summary, no action taken.",
        turn_tools=tools,
    )
    # First 6 tool names appear; the rest is summarized.
    for t in tools[:6]:
        assert t in corrective
    assert "(+4 more)" in corrective
    # 10th tool name was beyond cap — should NOT appear verbatim.
    assert "tool_9" not in corrective


def test_build_rules_for_turn_always_includes_verify_tests_rule():
    """The 'verify with the real test suite before declaring done'
    rule must appear in the per-turn system prompt regardless of
    attachments / sticky / refusal context. Always-on by design —
    universally useful, not bench-specific."""
    from backend.unified_agent import _build_rules_for_turn
    out = _build_rules_for_turn(
        ctx=None,
        has_attachments=False,
        sticky_fired=False,
        repeat_refusal=False,
    )
    assert "Before declaring a task done" in out
    assert "/tests/" in out
    assert "pytest" in out.lower() or "test suite" in out.lower()


def test_build_rules_for_turn_verify_tests_rule_present_with_attachments_too():
    """The verify-tests rule does NOT depend on structural signals
    — it must still appear when has_attachments / sticky / refusal
    add their own blocks."""
    from backend.unified_agent import _build_rules_for_turn
    out = _build_rules_for_turn(
        ctx=None,
        has_attachments=True,
        sticky_fired=True,
        repeat_refusal=True,
    )
    assert "Before declaring a task done" in out
    assert "/tests/" in out
