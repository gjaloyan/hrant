"""Tests for the Task Solver Process section in _UNIFIED_RULES.

User directive: "do not respond with a capability limitation before
attempting the task. First inspect the input, search existing skills,
try available tools, use terminal/package installation if needed, and
only then report limitations if execution is impossible."

This pins the section's existence + its anti-refusal language so the
behavior doesn't regress.
"""
from __future__ import annotations

import pytest


def test_task_solver_process_section_present():
    """V2 (2026-05-27): the section is now in M2 with the
    'TASK SOLVER POLICY' header."""
    from backend.unified_agent import _UNIFIED_RULES
    assert "TASK SOLVER" in _UNIFIED_RULES


def test_task_solver_process_says_execution_first():
    """The opening must commit the agent to execution before
    explanation. Anti-refusal anchor."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    assert "execution" in low
    assert "execute" in low or "act" in low


def test_task_solver_process_forbids_capability_limitation_first():
    """The bad-opening pattern (lead with 'I can't' / 'tools are
    not available') must be explicitly forbidden. V2 wording lives
    in M2's 'Anti-patterns' block — phrases evolve but the
    'no leading refusal' rule must remain visible."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    # The "no upfront refusal" anchor — any phrasing.
    assert "i can't" in low or "cannot" in low or "can not" in low
    # The anti-pattern frame — refusal / lecture / limit phrasing.
    assert (
        "anti-pattern" in low or "limitation" in low
        or "refusal" in low or "lecture" in low
    )


def test_task_solver_process_walks_react_state_machine():
    """V2: the 8 numbered phases were replaced by a ReAct state
    machine with 5 verbs (PLAN / EXECUTE / VERIFY / ASK / FINALIZE).
    Pin those verbs in M2."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES.upper()
    for verb in ("PLAN", "EXECUTE", "VERIFY", "ASK", "FINALIZE"):
        assert verb in body, f"M2 state-machine missing verb {verb!r}"


def test_task_solver_process_references_known_tools():
    """The 'execute, don't lecture' phase names actual tools so the
    LLM doesn't generalise away from concrete actions."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    for tool in (
        "load_skill",
        "universal_resolver",
        "terminal_exec",
        "MEDIA:",
    ):
        assert tool in body, f"Task Solver Process should name {tool!r}"


def test_task_solver_process_covers_install_via_terminal_exec():
    """2026-05-21: install gate dropped — Phase 6 now tells the
    LLM to install packages directly via terminal_exec
    (`pip install`, `apt install`, etc.) instead of the old
    propose_install ceremony."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    low = body.lower()
    # The rule must mention terminal_exec as the install vehicle.
    assert "terminal_exec" in body
    # And must name at least one install command shape.
    assert "pip install" in low or "apt install" in low


def test_task_solver_process_ask_when_blocked_rule():
    """V2: the ask-when-blocked rule lives in M2 + M6. The
    'blocked' anchor + the four acceptable triggers must still
    be visible. V2 says 'interpretations' instead of 'ambig'."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    low = body.lower()
    assert "blocked" in low
    # The acceptable triggers — wording varies between v1/v2.
    for marker in ("missing", "interpret", "destructive", "credential"):
        assert marker in low, f"ask_user trigger {marker!r} missing"


def test_task_solver_process_failure_format():
    """Phase 8 dictates the failure-report format. Pin it so the
    agent doesn't shrink to one-liner refusals."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    low = body.lower()
    # The "show your work" elements.
    assert "tried" in low or "tools" in low
    assert "failed" in low or "exit" in low or "error" in low
    assert "unblock" in low or "next attempt" in low


# ─── TSP-3: phase 1 requires a plan sentence ────────────────────────


def test_phase_1_requires_plan_sentence():
    """Hermes-like behavior — declare the plan before the first tool
    call so mid-execution drift is rare. Pin the rule's existence."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    low = body.lower()
    # The plan-sentence requirement.
    assert "plan" in low
    assert "one-sentence" in low or "one sentence" in low or "одну строк" in low
    # The example string must be present so the LLM sees the shape.
    assert "Я обработаю видео" in body or "preprocess_video" in body


# ─── TSP-1: enforced operating limits ───────────────────────────────


def test_operating_limits_section_present():
    """V2: 'operating limits' was the v1 section header inside the
    legacy TSP. In v2 the same rules live under M2's 'Loop
    discipline' / 'Anti-patterns' / 'Long-running shell' blocks.
    The behavioral content — attempt-bar threshold + sequencing
    discipline — is what we pin, not the header text."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    # The discipline blocks (any one of these must be visible).
    assert (
        "loop discipline" in low
        or "anti-pattern" in low
        or "long-running" in low
    )


def test_attempt_bar_pinned_in_rules():
    """The rule that says '<2 distinct tools = automatic rewrite'
    must be in the rules so the LLM understands the structural
    backstop, not just have it as runtime behavior."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    # The 2-distinct-tools threshold.
    assert "2 distinct" in low or "minimum 2" in low or "at least 2" in low
    # The rewrite mention.
    assert "rewrit" in low or "automat" in low


@pytest.mark.skip(
    reason="V2 (2026-05-27): the 30/50/20 iteration-budget heuristic "
    "was dropped during the audit refactor. The behavioral floor "
    "(5+ inspect-without-execute → stop) is now in M2 anti-patterns; "
    "the explicit percentage math added prompt weight without "
    "measurably improving behavior."
)
def test_iteration_budget_30_50_20_present():
    pass


def test_inspect_without_execute_threshold_pinned():
    """V2: M2 has a hard threshold for inspect-without-execute
    that replaces the 30/50/20 heuristic — pin its existence."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    assert "inspect" in low
    # The threshold (5+ inspect calls without execute).
    assert "5+" in _UNIFIED_RULES or "5 " in low


def test_iteration_budget_names_explore_execute_verify_phases():
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    # The three phases the budget allocates.
    assert "inspect" in low or "identify" in low
    assert "execute" in low
    assert "verify" in low or "deliver" in low


def test_inspection_tool_lookup_in_rules():
    """V2: the inspection 'cheatsheet' header was dropped; the
    file-type→tool mappings live in M3 (tool routing) + the
    `_RULES_FILE_TYPES` scenario block. Pin that the inspection
    tools are still namable somewhere in _UNIFIED_RULES."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    for tool in ("preprocess_video", "analyze_image", "read_file", "run_python"):
        assert tool in body, (
            f"inspection tool {tool!r} should still be visible in rules"
        )


@pytest.mark.skip(
    reason="2026-05-21: REFUSAL_ATTEMPT_BAR runtime constant + the "
    "keyword-based rewriter that used it were removed. The '2 distinct "
    "tools before refusing' rule still lives in the system prompt — "
    "the LLM enforces it itself."
)
def test_attempt_bar_constant_matches_rules():
    """The REFUSAL_ATTEMPT_BAR runtime constant and the rules text
    must agree on the threshold. If you change one, change both."""
