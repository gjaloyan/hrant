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
    from backend.unified_agent import _UNIFIED_RULES
    assert "Task Solver Process" in _UNIFIED_RULES


def test_task_solver_process_says_execution_first():
    """The opening must commit the agent to execution before
    explanation. Anti-refusal anchor."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    assert "execution" in low
    assert "execute" in low or "act" in low


def test_task_solver_process_forbids_capability_limitation_first():
    """The bad-opening example must explicitly call out the failure
    mode the user keeps seeing in production logs."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    # Anti-refusal language anchors.
    assert "limitation" in low
    assert "can't do" in low or "cannot" in low or "can not" in low


def test_task_solver_process_walks_eight_phases():
    """The body must enumerate the phases. Tests don't pin exact
    wording (it can evolve), only that the markers are there."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    # 8 numbered phases must each be marked at start-of-line.
    for n in (1, 2, 3, 4, 5, 6, 7, 8):
        assert f"\n{n}." in body or body.startswith(f"{n}."), (
            f"missing phase {n} marker"
        )


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
    """Phase 7 must restrict question-asking to specific scenarios so
    the agent stops asking 'do you want me to try?' on routine tasks."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    low = body.lower()
    assert "blocked" in low or "truly blocked" in low
    # The 4 acceptable-question categories.
    for marker in ("missing", "ambig", "destructive"):
        assert marker in low, f"phase 7 should mention {marker!r}"


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
    from backend.unified_agent import _UNIFIED_RULES
    assert "TSP operating limits" in _UNIFIED_RULES or \
           "operating limits" in _UNIFIED_RULES.lower()


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


def test_iteration_budget_30_50_20_present():
    """The 30/50/20 phased budget is the planning heuristic — pin
    its presence so a future edit doesn't quietly drop it."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    # Either as percentages or as phases.
    assert "30/50/20" in body or "30%" in body and "50%" in body


def test_iteration_budget_names_explore_execute_verify_phases():
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    # The three phases the budget allocates.
    assert "inspect" in low or "identify" in low
    assert "execute" in low
    assert "verify" in low or "deliver" in low


def test_inspection_cheatsheet_in_tsp_section():
    """Phase 2 inspection cheatsheet — quick lookup so the agent
    doesn't reinvent inspection per turn."""
    from backend.unified_agent import _UNIFIED_RULES
    body = _UNIFIED_RULES
    assert "Inspection cheatsheet" in body or "cheatsheet" in body.lower()
    # The file-type → tool mappings must be visible.
    for tool in ("preprocess_video", "analyze_image", "read_file", "run_python"):
        assert tool in body, f"cheatsheet should name {tool!r}"


@pytest.mark.skip(
    reason="2026-05-21: REFUSAL_ATTEMPT_BAR runtime constant + the "
    "keyword-based rewriter that used it were removed. The '2 distinct "
    "tools before refusing' rule still lives in the system prompt — "
    "the LLM enforces it itself."
)
def test_attempt_bar_constant_matches_rules():
    """The REFUSAL_ATTEMPT_BAR runtime constant and the rules text
    must agree on the threshold. If you change one, change both."""
