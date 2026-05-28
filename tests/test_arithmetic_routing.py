"""Arithmetic routing regression — kept alive tests only.

Dead tests (testing _looks_like_arithmetic, _arithmetic_marker,
_classify_intent) were removed when those symbols were deleted from
backend/agent.py (legacy pre-unified pipeline cluster).
"""
from __future__ import annotations


def test_solver_system_prompt_has_hard_arithmetic_rule():
    """A hard rule in the solver system prompt belt-and-suspenders the
    per-turn arithmetic handling. The MUST is the strong wording; 'consider'
    or 'you can' would be too soft."""
    from backend.agent import SOLVER_SYSTEM_BASE
    s = SOLVER_SYSTEM_BASE.lower()
    assert "arithmetic" in s
    assert "must call" in s or "you must" in s
