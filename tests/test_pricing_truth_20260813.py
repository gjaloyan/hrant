"""Reported cost must not be invented.

The owner, reading his own OpenRouter dashboard: "in openrouter i se 21m tokes
use and pay for it 0,77 usd" — against figures this system had been reporting
more than twenty times higher.

Every model actually in use was missing from the static pricing table, so all
of them were billed at the $3/$15 default: tencent/hy3 (really $0.132/$0.528),
xiaomi/mimo-v2.5-pro, z-ai/glm-5.2, and gpt-5.5 — which is covered by a
ChatGPT subscription and has no per-token charge at all.

The 2026-08-08 audit had already added a warning for exactly this ("so
reported cost for this model is a guess"). It fired on every model, every run.
The numbers were quoted as facts anyway.
"""
import pytest

from backend.providers import (
    DEFAULT_PRICING, _SUBSCRIPTION_MODELS, get_model_pricing,
)


def test_subscription_models_cost_no_tokens():
    """A ChatGPT-subscription model bills a flat fee. Pricing it per token
    fabricates spend that never happened — and it dominated the daily total."""
    for m in ("gpt-5.5", "gpt-5.6-sol", "gpt-5.4-mini"):
        p = get_model_pricing(m)
        assert p["input"] == 0.0 and p["output"] == 0.0, m


def test_the_subscription_set_covers_the_codex_family():
    for m in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"):
        assert m in _SUBSCRIPTION_MODELS


def test_a_metered_model_is_not_billed_at_the_default(monkeypatch):
    """The bug: everything resolved to $3/$15 regardless of what ran, so a
    cheap model could not be told from an expensive one."""
    import backend.providers as pr
    monkeypatch.setattr(pr, "_openrouter_pricing",
                        lambda: {"tencent/hy3": {"input": 0.132,
                                                 "output": 0.528}})
    p = get_model_pricing("tencent/hy3")
    assert p["input"] == pytest.approx(0.132)
    assert p["output"] == pytest.approx(0.528)
    assert p != DEFAULT_PRICING


def test_live_prices_win_over_the_static_table(monkeypatch):
    """A gateway adds models weekly; a hand-maintained table cannot keep up,
    so the source of truth wins when it answers."""
    import backend.providers as pr
    monkeypatch.setattr(pr, "_openrouter_pricing",
                        lambda: {"z-ai/glm-5.2": {"input": 0.5,
                                                  "output": 3.15}})
    assert get_model_pricing("z-ai/glm-5.2")["output"] == pytest.approx(3.15)


def test_a_dead_network_falls_back_instead_of_crashing(monkeypatch):
    """Wrong-but-old beats breaking a cost calculation."""
    import backend.providers as pr
    monkeypatch.setattr(pr, "_openrouter_pricing", lambda: {})
    p = get_model_pricing("something-nobody-has-priced")
    assert p == DEFAULT_PRICING


def test_the_fetch_never_raises(monkeypatch):
    import backend.providers as pr

    def _boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(pr, "_OR_PRICING_CACHE", {"at": 0.0, "data": {}})
    import httpx
    monkeypatch.setattr(httpx, "get", _boom)
    assert pr._openrouter_pricing() == {}
