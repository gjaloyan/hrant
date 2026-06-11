"""Model cascade — small tier first, strong-verifier gate, escalate.

AGI roadmap #1 (2026-06-12). Ships disabled; the owner opts in via
/api/cascade. The small tier today is OpenRouter qwen (pseudo-local
until the box can serve a model); the cascade is provider-agnostic.
"""
from __future__ import annotations

import pytest

from backend.models import VerificationResult


@pytest.fixture
def cas(tmp_path, monkeypatch):
    from backend import cascade as c
    monkeypatch.setattr(c, "_config_path", lambda: tmp_path / "cascade.json")
    c.load_config(force=True)
    yield c
    c.load_config(force=True)


# ─── Config / route ───────────────────────────────────────────────


def test_default_disabled(cas):
    assert cas.load_config(force=True)["enabled"] is False
    assert cas.route() is None


def test_route_when_configured(cas):
    cas.save_config({
        "enabled": True,
        "provider_id": "openrouter-x",
        "model": "qwen/qwen3.6-35b-a3b",
        "confidence_gate": 75,
    })
    assert cas.route() == ("openrouter-x", "qwen/qwen3.6-35b-a3b", 75)


def test_enabled_but_unconfigured_routes_none(cas):
    cas.save_config({"enabled": True, "provider_id": "", "model": "m"})
    assert cas.route() is None


def test_gate_clamped(cas):
    cas.save_config({
        "enabled": True, "provider_id": "p", "model": "m",
        "confidence_gate": 250,
    })
    assert cas.route()[2] == 100


# ─── Gate ─────────────────────────────────────────────────────────


def test_gate_accepts_confident_clean(cas):
    ok, why = cas.gate_passes(
        VerificationResult(confidence=85), confidence_gate=70,
    )
    assert ok is True and "confidence-85" in why


def test_gate_rejects_low_confidence(cas):
    ok, why = cas.gate_passes(
        VerificationResult(confidence=60), confidence_gate=70,
    )
    assert ok is False and "below-70" in why


def test_gate_rejects_content_contradictions(cas):
    ok, why = cas.gate_passes(
        VerificationResult(
            confidence=90,
            contradictions=["answer says X absent, file shows X"],
        ),
        confidence_gate=70,
    )
    assert ok is False and "contradiction" in why


def test_gate_ignores_delivery_markers(cas):
    """endpoint_not_met markers are delivery-class — the cascade gate
    judges CONTENT (delivery is the self-correction loop's job and
    fires identically on either tier)."""
    ok, _ = cas.gate_passes(
        VerificationResult(
            confidence=90,
            contradictions=[
                "endpoint_not_met: action-verb request without "
                "execute-class tool call or MEDIA: delivery",
            ],
        ),
        confidence_gate=70,
    )
    assert ok is True


def test_gate_rejects_none_vr(cas):
    ok, why = cas.gate_passes(None, confidence_gate=70)
    assert ok is False and why == "no-verifier-result"


# ─── call_with_tools model_override ───────────────────────────────


def test_call_with_tools_override_uses_resolved_llm(monkeypatch):
    from backend.llm import DualModelRouter as Router
    import backend.llm as llm_mod

    class _OvLLM:
        def __init__(self):
            self.calls = 0

        def complete_with_tools(self, system, user, tools, execute_tool, **kw):
            self.calls += 1
            return "override-reply"

    ov = _OvLLM()
    r = Router.__new__(Router)
    r._routed_llms = {}
    r.state = {"last_reason": ""}
    monkeypatch.setattr(r, "_save_state", lambda: None, raising=False)
    monkeypatch.setattr(
        r, "_track_active_model_call", lambda **kw: None, raising=False,
    )
    import backend.failover as _fo
    monkeypatch.setattr(
        _fo, "resolve_entry_cfg", lambda pid, m: {"provider_id": pid, "model": m},
    )
    monkeypatch.setattr(llm_mod, "create_llm", lambda cfg: ov)
    monkeypatch.setattr(llm_mod, "_supports_tools", lambda *a, **kw: True)
    # Chain rigged to explode — override must bypass it.
    monkeypatch.setattr(
        llm_mod, "_active_provider_chain",
        lambda tt: [object()],
    )
    monkeypatch.setattr(
        llm_mod, "_run_with_safety_fallback",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("chain must not run for an override"),
        ),
    )

    from backend.llm import TaskType
    out = r.call_with_tools(
        TaskType.COMPLEX_SOLVING, "sys", "user",
        tools=[{"name": "t"}], execute_tool=lambda n, a: ("", False),
        model_override=("openrouter-x", "qwen/test"),
    )
    assert out == "override-reply"
    assert ov.calls == 1
    assert "override: openrouter-x:qwen/test" in r.state["last_reason"]


def test_call_with_tools_override_unresolvable_raises(monkeypatch):
    from backend.llm import DualModelRouter as Router, LLMError, TaskType
    import backend.failover as _fo

    r = Router.__new__(Router)
    r._routed_llms = {}
    r.state = {"last_reason": ""}
    monkeypatch.setattr(_fo, "resolve_entry_cfg", lambda pid, m: None)

    with pytest.raises(LLMError, match="not resolvable"):
        r.call_with_tools(
            TaskType.COMPLEX_SOLVING, "sys", "user",
            tools=[{"name": "t"}], execute_tool=lambda n, a: ("", False),
            model_override=("ghost", "model"),
        )
