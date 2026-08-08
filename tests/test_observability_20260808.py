"""Section 6 of the audit: numbers that were not measuring anything.

  * Router call/cost accounting was UNREACHABLE. `call_with_tools` dispatched
    to the provider chain and returned, so everything below — the active-model
    branch, _track_active_model_call, _save_state, and the A/B path with its
    daily-budget check — never ran in production. The give-away was the
    function's own docstring sitting after that return as a dead string
    literal. Measured on prod: router_state.json unwritten for 54 days,
    frozen at 27 calls on 2026-06-15, while the token tracker counted 197
    calls that same day. The owner's $5/day budget gate reads api_cost_today
    from that dead file.
  * Every live model missed the pricing table, so all reported cost was a
    flat $3/$15 fiction — including for a model name that does not exist.
  * "Which model answered" was decided by Counter.most_common over the
    turn's calls, i.e. the most FREQUENT model rather than the one that
    produced the answer.
"""
from __future__ import annotations

import backend.llm as llm
from backend.providers import get_model_pricing, unpriced_models


class _P:
    def __init__(self, name, model, die=None):
        self.name = self.provider_id = name
        self.model = model
        self._die = die

    def call_with_tools(self, tt, s, u, *a, **kw):
        if self._die:
            raise llm.LLMError(self._die)
        return f"answer from {self.name}"


_ABORT = "provider aborted mid-stream: an error occurred while processing your request"


def test_the_chain_records_which_adapter_answered():
    out = llm._run_with_safety_fallback(
        [_P("codex", "gpt-5.5", die=_ABORT), _P("openrouter", "xiaomi/mimo-v2.5-pro")],
        "call_with_tools", "complex_solving", "s", "u",
        execute_tool=lambda *a, **k: ("", False))
    assert out == "answer from openrouter"
    assert llm.turn_chain_winner()["model"] == "xiaomi/mimo-v2.5-pro"


def test_the_recorded_winner_beats_the_popularity_vote():
    """The failed primary made SIX calls; the fallback that actually answered
    made three. Frequency would report the wrong one."""
    from backend.model_report import primary_model_used
    llm._run_with_safety_fallback(
        [_P("codex", "gpt-5.5", die=_ABORT), _P("openrouter", "xiaomi/mimo-v2.5-pro")],
        "call_with_tools", "complex_solving", "s", "u",
        execute_tool=lambda *a, **k: ("", False))
    calls = [{"model": "gpt-5.5"}] * 6 + [{"model": "xiaomi/mimo-v2.5-pro"}] * 3
    assert primary_model_used(calls) == "xiaomi/mimo-v2.5-pro"


def test_the_counter_still_answers_when_no_chain_ran():
    from backend.model_report import primary_model_used
    llm._TURN_CHAIN_WINNER.set(None)
    assert primary_model_used([{"model": "gpt-5.5"}] * 3) == "gpt-5.5"
    assert primary_model_used([]) == ""


# ── pricing ───────────────────────────────────────────────────────────

def test_a_vendor_prefixed_slug_resolves_to_real_pricing():
    """OpenRouter names it "anthropic/claude-sonnet-4-5"; the table is keyed
    on the bare name, and the lookup was exact-match."""
    assert get_model_pricing("anthropic/claude-sonnet-4-5") == \
        get_model_pricing("claude-sonnet-4-5")


def test_suffixed_variants_resolve():
    assert get_model_pricing("gpt-4o:free") == get_model_pricing("gpt-4o")


def test_an_unknown_model_is_recorded_as_a_guess():
    get_model_pricing("definitely-not-a-real-model-xyz")
    assert "definitely-not-a-real-model-xyz" in unpriced_models()


def test_a_known_model_is_not_recorded_as_a_guess():
    get_model_pricing("gpt-4o")
    assert "gpt-4o" not in unpriced_models()
