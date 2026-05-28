"""Round 10 — token-usage findings from a self-review where `_solve`
hit 278k input tokens on a single self-analysis turn even with the
Round 9 caps in place.

Three changes covered here:
  1. `read_file` / `view_file` per-call cap tightened 16k -> 12k
     (the cumulative-across-iterations bill dominates, so the cap
     has to bite harder than the single-turn budget seems to need).
  2. New per-loop input-token budget: once a tool-use loop has
     burned `CONFIG.router.tool_loop_input_budget` cumulative input
     tokens, break out and let the forced synthesis call wrap up.
  3. Solver prompt on self-analysis turns now demands
     `start_line`/`end_line` for files >=1000 lines so the model
     stops dumping `agent.py` whole into the loop.

Plus a contradiction-detector regression: agent.progress's docstring
no longer claims it carries the FULL tool result (it carries a
truncated preview).
"""
from __future__ import annotations

import inspect

import pytest


# --- #2: read_file/view_file cap (Round 10 set 12k; Round 11 reverted to 16k
# because the tighter cap was costing more in answer quality than it saved
# — the real fix landed in Round 11 with curated forced-synthesis payloads.
# These tests now guard against re-tightening without revisiting that
# trade-off; the 16k cap belongs in test_round11 conceptually but sits here
# for chronological clarity.) -----------------------------------------------


def test_read_file_cap_is_16k():
    from backend.llm import _TOOL_LLM_RESULT_CAPS
    assert _TOOL_LLM_RESULT_CAPS["read_file"] == 16_000
    assert _TOOL_LLM_RESULT_CAPS["view_file"] == 16_000


def test_read_file_compact_respects_cap():
    from backend.llm import _compact_tool_result_for_llm
    big = "z" * 60_000
    out = _compact_tool_result_for_llm("read_file", big)
    # 16k cap + truncation marker; safely under 17k.
    assert len(out) < 17_000
    # And the marker still points at the line-range alternative
    # (carried over from Round 9 — guard against regression).
    assert "start_line" in out and "end_line" in out


# --- #4: per-loop input-token budget --------------------------------------


def test_tool_loop_input_budget_default_is_disabled():
    """2026-05-21: default flipped to 0 ("no limits, agent need to
    have a free work opportunity"). The cap mechanism stays so an
    operator can opt back in, but no turn is broken mid-loop by
    default."""
    from backend.config import CONFIG
    val = CONFIG.router.get("tool_loop_input_budget")
    assert isinstance(val, int)
    assert val == 0, (
        f"tool_loop_input_budget default is 0 (disabled); got {val}"
    )


def test_budget_helper_false_when_under_cap():
    from backend.llm import TOKENS, _tool_loop_input_budget_exceeded
    TOKENS.reset_request()
    try:
        # Cold start: zero usage, guard does NOT fire.
        assert _tool_loop_input_budget_exceeded() is False
    finally:
        TOKENS.reset_request()


def test_budget_helper_does_not_fire_when_cap_zero():
    """With the cap disabled (default), the helper must always
    return False even if usage is astronomically high. This pins
    the "no limits" behaviour."""
    from backend.config import CONFIG
    from backend.llm import TOKENS, _tool_loop_input_budget_exceeded
    TOKENS.reset_request()
    try:
        TOKENS._request_input = 10_000_000  # 10M tokens, way past any old cap
        # CONFIG default is 0 → helper must say False
        assert CONFIG.router.get("tool_loop_input_budget") == 0
        assert _tool_loop_input_budget_exceeded() is False
    finally:
        TOKENS.reset_request()


def test_budget_helper_fires_when_operator_opts_in():
    """If an operator sets a non-zero cap via runtime_config, the
    enforcement mechanism must still work — pins the opt-in path."""
    from backend.config import CONFIG
    from backend.llm import TOKENS, _tool_loop_input_budget_exceeded

    prev = CONFIG.router.get("tool_loop_input_budget")
    CONFIG.router["tool_loop_input_budget"] = 50_000
    TOKENS.reset_request()
    try:
        TOKENS._request_input = 60_000
        assert _tool_loop_input_budget_exceeded() is True
    finally:
        TOKENS.reset_request()
        CONFIG.router["tool_loop_input_budget"] = prev


def test_budget_helper_safe_when_tracker_blows_up(monkeypatch):
    """Whatever happens inside TokenTracker, the helper must NEVER
    raise — a tool loop that crashes on its own budget guard would
    be worse than no guard at all."""
    from backend.llm import TOKENS, _tool_loop_input_budget_exceeded

    def boom():  # pragma: no cover - just raises
        raise RuntimeError("tracker offline")

    monkeypatch.setattr(TOKENS, "request_usage", boom)
    # Should swallow and return False — let the loop continue
    # rather than exit prematurely.
    assert _tool_loop_input_budget_exceeded() is False


def test_budget_guard_applied_at_every_tool_loop():
    """All five `complete_with_tools` implementations (Anthropic,
    OpenAI-compat, Codex Responses, Bedrock, Cohere) must call the
    guard — otherwise the loop in one provider runs unbounded
    while the others honour the cap."""
    import backend.llm as llm_mod

    src = inspect.getsource(llm_mod)
    # Rough lower bound: helper call appears once per provider's
    # loop. Five providers + the helper definition itself = 6.
    occurrences = src.count("_tool_loop_input_budget_exceeded()")
    assert occurrences >= 6, (
        f"expected the budget guard at every tool-loop site, "
        f"saw only {occurrences} call sites"
    )


# --- docstring contradiction (caught by detect_false_absence) -------------


def test_progress_docstring_no_longer_claims_full_result():
    """The contradiction detector flagged this in the last review
    cycle: docstring said `tool_call carries (name, args, full
    result)` but the implementation truncates to a 4k preview.
    Guard against the stale phrasing creeping back."""
    from backend.agent import Agent

    doc = Agent.progress.__doc__ or ""
    lower = doc.lower()
    # The phrase "full result" should not appear ungated; a stale
    # claim there mis-describes the trace payload.
    assert "full result" not in lower, (
        "progress() docstring must not claim it carries the full "
        "tool result — it carries a truncated preview"
    )
    # Positive marker so a future rewrite can't accidentally drop
    # the truncation contract from the doc.
    assert "truncat" in lower
