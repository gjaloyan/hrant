"""Tests for the Task Solver Process section in _UNIFIED_RULES.

User directive: "do not respond with a capability limitation before
attempting the task. First inspect the input, search existing skills,
try available tools, use terminal/package installation if needed, and
only then report limitations if execution is impossible."

This pins the section's existence + its anti-refusal language so the
behavior doesn't regress.
"""
from __future__ import annotations


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
        "propose_install",
        "MEDIA:",
    ):
        assert tool in body, f"Task Solver Process should name {tool!r}"


def test_task_solver_process_covers_auto_install_via_gate():
    """Phase 6 must tell the LLM about the auto-install path so it
    doesn't try `pip install` via terminal_exec (which is deny-listed
    anyway)."""
    from backend.unified_agent import _UNIFIED_RULES
    low = _UNIFIED_RULES.lower()
    assert "propose_install" in _UNIFIED_RULES
    assert "auto-propose" in low or "auto-install" in low or \
           "auto-propos" in low


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
