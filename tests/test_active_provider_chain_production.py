"""_active_provider_chain must return a non-empty iterable in
production so the safety/quota fallback actually engages on real
LLMError. Previously it returned [] by default, making the
fallback path dead code in production.
"""
from __future__ import annotations

import pytest

# Opt out of the autouse `_isolate_active_provider_chain` conftest
# fixture — this module tests the chain factory itself.
pytestmark = pytest.mark.uses_real_provider_chain


def test_active_provider_chain_non_empty_with_configured_providers(monkeypatch):
    """When at least one enabled provider is configured (the normal
    production state), _active_provider_chain MUST return a non-empty
    list. Otherwise the safety/quota fallback in
    _run_with_safety_fallback is dead code."""
    from backend import llm as _llm

    fake_providers = [
        {
            "id": "anthropic-default",
            "name": "Anthropic (default)",
            "type": "anthropic",
            "enabled": True,
            "is_default": True,
        },
    ]
    monkeypatch.setattr(
        "backend.providers.get_providers", lambda: fake_providers
    )
    chain = _llm._active_provider_chain()
    chain_list = list(chain)
    assert chain_list, (
        f"_active_provider_chain returned {chain_list!r} — the "
        f"safety/quota fallback path will be dead code in production. "
        f"It must return the ordered list the router walks for a real "
        f"call."
    )
    first = chain_list[0]
    has_name = (
        getattr(first, "name", None)
        or (isinstance(first, dict) and (first.get("name") or first.get("id")))
    )
    assert has_name, f"chain entry has no identifiable name: {first!r}"


def test_active_provider_chain_empty_when_no_providers(monkeypatch):
    """If no providers are configured at all, the chain is empty —
    callers must handle this (router falls through to its legacy
    single-provider path)."""
    from backend import llm as _llm
    monkeypatch.setattr("backend.providers.get_providers", lambda: [])
    chain = _llm._active_provider_chain()
    assert list(chain) == []


def test_active_provider_chain_skips_disabled_providers(monkeypatch):
    """Disabled providers are excluded — only enabled ones are in
    the fallback chain."""
    from backend import llm as _llm
    fake_providers = [
        {"id": "p1", "type": "openai_codex", "enabled": True},
        {"id": "p2", "type": "anthropic", "enabled": False},
        {"id": "p3", "type": "openrouter", "enabled": True},
    ]
    monkeypatch.setattr(
        "backend.providers.get_providers", lambda: fake_providers
    )
    chain = _llm._active_provider_chain()
    ids = []
    for entry in chain:
        ids.append(
            getattr(entry, "id", None)
            or (entry.get("id") if isinstance(entry, dict) else None)
        )
    assert "p2" not in ids
    assert "p1" in ids or "p3" in ids
