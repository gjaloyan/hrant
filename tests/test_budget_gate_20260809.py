"""The owner's daily spend cap, on the path production actually takes.

2026-08-09 dead-code audit: the cap was checked inside `Router._pick()`,
which is called only from the tails of call() and call_with_tools() — both of
which return earlier at `if chain: ...`. Prod proof: total_a_calls=0 and
total_b_calls=0 against total_active_model_calls=2619, i.e. that tail has
never executed once. So daily_api_budget_usd was configurable in the UI,
validated by runtime_config, persisted, documented in the module header — and
completely inert.

These tests deliberately do NOT rely on the autouse `_isolate_active_provider_chain`
fixture's empty chain, because that fixture is exactly what hid this: it pins
the chain empty for the whole suite, so every router test exercised the dead
branch and never the live one.
"""
from __future__ import annotations

import backend.llm as llm


class _Prov:
    def __init__(self, name, model):
        self.name = self.provider_id = name
        self.model = model


def test_under_the_cap_the_chain_order_is_untouched():
    cheap = _Prov("openrouter", "gpt-4o-mini")
    dear = _Prov("anthropic", "claude-sonnet-4-5")
    r = llm.DualModelRouter.__new__(llm.DualModelRouter)
    r.cfg_router = {"daily_api_budget_usd": 5.0}
    r.state = {"api_cost_today": 1.0}
    assert r._apply_budget_policy([dear, cheap])[0] is dear


def test_over_the_cap_the_cheapest_provider_goes_first():
    cheap = _Prov("openrouter", "gpt-4o-mini")
    dear = _Prov("anthropic", "claude-sonnet-4-5")
    r = llm.DualModelRouter.__new__(llm.DualModelRouter)
    r.cfg_router = {"daily_api_budget_usd": 5.0}
    r.state = {"api_cost_today": 7.5}
    out = r._apply_budget_policy([dear, cheap])
    assert out[0] is cheap
    assert "over budget" in r.state["last_reason"]


def test_an_unpriced_model_is_not_assumed_cheap():
    """Guessing "cheap" for an unknown model is how a cap becomes decorative."""
    known = _Prov("a", "gpt-4o-mini")
    unknown = _Prov("b", "some-model-nobody-priced")
    out = llm._cheapest_first([unknown, known])
    assert out[0] is known


def test_the_cap_degrades_rather_than_bricking_the_agent():
    """The documented rule is "exceeded -> fallback to B", i.e. degrade. A
    hard stop would leave the agent unusable for the rest of the day."""
    only = _Prov("solo", "claude-sonnet-4-5")
    r = llm.DualModelRouter.__new__(llm.DualModelRouter)
    r.cfg_router = {"daily_api_budget_usd": 1.0}
    r.state = {"api_cost_today": 99.0}
    out = r._apply_budget_policy([only])
    assert out == [only], "over budget must still return a usable chain"
    assert "no cheaper provider" in r.state["last_reason"]


def test_no_cap_configured_is_a_no_op():
    p = _Prov("a", "gpt-4o-mini")
    r = llm.DualModelRouter.__new__(llm.DualModelRouter)
    r.cfg_router = {"daily_api_budget_usd": 0.0}
    r.state = {"api_cost_today": 999.0}
    assert r._apply_budget_policy([p]) == [p]
    over, _s, _c = llm._budget_exceeded(r.cfg_router, r.state)
    assert over is False


def test_garbage_state_never_raises():
    r = llm.DualModelRouter.__new__(llm.DualModelRouter)
    r.cfg_router = {"daily_api_budget_usd": "not-a-number"}
    r.state = {}
    assert llm._budget_exceeded(r.cfg_router, r.state) == (False, 0.0, 0.0)
