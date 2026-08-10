"""`_ProviderChainAdapter._build` — the live path's error contract.

The chain runner advances to the next provider only for errors it recognises
as LLMError. So an adapter that lets a raw exception escape (a bad key, an
httpx failure at client construction, a provider record that vanished) does
not fail over — it aborts the turn.

That contract used to be tested only through `Router.call`'s legacy A/B tail,
which prod never executed and which was deleted on 2026-08-10. Tested here on
the code that actually runs.
"""
import pytest

from backend.llm import LLMError, _ProviderChainAdapter


def _adapter(prov=None):
    return _ProviderChainAdapter(prov or {"id": "openrouter-x",
                                          "models": ["qwen/test"]})


def test_a_missing_or_disabled_provider_raises_llm_error(monkeypatch):
    """`resolve_entry_cfg` returns None when the provider is gone or turned
    off. The chain must be able to walk past it, which means LLMError."""
    import backend.failover as _fo
    monkeypatch.setattr(_fo, "resolve_entry_cfg", lambda pid, m: None)
    with pytest.raises(LLMError, match="missing/disabled"):
        _adapter()._build()


def test_a_crash_while_building_is_wrapped_not_leaked(monkeypatch):
    """A bare ValueError/KeyError from create_llm classifies as 'unknown' and
    stops the chain dead. Wrapping is what keeps failover working."""
    import backend.failover as _fo
    import backend.llm as llm_mod
    monkeypatch.setattr(_fo, "resolve_entry_cfg",
                        lambda pid, m: {"provider": "openai", "model": m})

    def _boom(cfg):
        raise KeyError("api_key")

    monkeypatch.setattr(llm_mod, "create_llm", _boom)
    with pytest.raises(LLMError) as ei:
        _adapter()._build()
    assert "create_llm(openai/qwen/test)" in str(ei.value)


def test_an_llm_error_passes_through_unchanged(monkeypatch):
    """Double-wrapping would bury the message the classifier reads."""
    import backend.failover as _fo
    import backend.llm as llm_mod
    monkeypatch.setattr(_fo, "resolve_entry_cfg",
                        lambda pid, m: {"provider": "openai", "model": m})

    def _boom(cfg):
        raise LLMError("Anthropic API 429: rate_limit_error")

    monkeypatch.setattr(llm_mod, "create_llm", _boom)
    with pytest.raises(LLMError, match="^Anthropic API 429"):
        _adapter()._build()


def test_the_built_llm_is_reused(monkeypatch):
    """One httpx client / OAuth handshake per adapter, not per call."""
    import backend.failover as _fo
    import backend.llm as llm_mod
    monkeypatch.setattr(_fo, "resolve_entry_cfg",
                        lambda pid, m: {"provider": "openai", "model": m})
    built = {"n": 0}

    def _make(cfg):
        built["n"] += 1
        return object()

    monkeypatch.setattr(llm_mod, "create_llm", _make)
    a = _adapter()
    assert a._build() is a._build()
    assert built["n"] == 1
