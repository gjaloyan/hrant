"""Arithmetic must not be answered from training data — it must reach
the solver and trigger calc / run_python. The agent's tracked goal
'Producing nonsensical output for basic arithmetic (2+2)' is the
ground-truth motivation for these regressions."""
from __future__ import annotations

from backend.agent import Agent, _looks_like_arithmetic


def test_looks_like_arithmetic_basic_ops():
    for s in [
        "2+2", "2 + 2", "15 * 3", "(5+3)/2", "10 - 7",
        "100 % of 200", "100% of 200",  # percent forms
        "2^10",
        "  3.14 * 2  ",  # decimals + whitespace
    ]:
        assert _looks_like_arithmetic(s), f"missed: {s!r}"


def test_looks_like_arithmetic_negatives():
    """Don't false-positive on prose with isolated numbers."""
    for s in [
        "hello", "what time is it?", "tell me about 1984",
        "i live at 5 pine st", "version 2 is better than 1",
    ]:
        assert not _looks_like_arithmetic(s), f"false positive: {s!r}"


def test_classify_intent_routes_arithmetic_to_task():
    """The arithmetic short-circuit at the top of _classify_intent
    bypasses chitchat regex AND the LLM classifier. '2+2' looks short
    enough that an LLM might call it 'chat'; we route to 'task' so the
    solver path runs and calc is reachable."""
    agent = Agent()
    # No router patching needed — arithmetic short-circuits before LLM.
    assert agent._classify_intent("2+2") == "task"
    assert agent._classify_intent("сколько будет 15 * 3") == "task"


def test_arithmetic_marker_present_for_arithmetic_only():
    agent = Agent()
    assert agent._arithmetic_marker("2+2").startswith("[ARITHMETIC DETECTED]")
    assert agent._arithmetic_marker("hello there") == ""
    assert agent._arithmetic_marker("") == ""


def test_arithmetic_marker_mentions_calc_and_run_python():
    """The marker has to NAME the right tools — model needs to know
    which to call, not just that it should call something."""
    m = Agent()._arithmetic_marker("5 * 5")
    assert "calc" in m
    assert "run_python" in m


def test_solver_system_prompt_has_hard_arithmetic_rule():
    """A separate hard rule in the solver system prompt belt-and-
    suspenders the per-turn marker. If the marker ever gets truncated
    out (e.g. context window pressure), the hard rule still survives."""
    from backend.agent import SOLVER_SYSTEM_BASE
    s = SOLVER_SYSTEM_BASE.lower()
    assert "arithmetic" in s
    # MUST is the strong wording; "consider" or "you can" would be too soft.
    assert "must call" in s or "you must" in s
