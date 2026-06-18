"""The active-model selection must actually control execution.

Bug: `_active_provider_chain` walked enabled providers in registration order
using each provider's `default_model`, ignoring `ACTIVE_MODEL` (the UI
selector). So picking a model in the UI never changed which model ran — the
provider's `default_model` always won. Fix: when a model is pinned, it becomes
the chain PRIMARY (that provider, forced to the pinned model); the rest follow
as fallbacks.
"""
from __future__ import annotations

import pytest

import backend.llm as llm
from backend.providers import ACTIVE_MODEL


_PROVIDERS = [
    {"id": "codex", "type": "openai_codex", "enabled": True,
     "default_model": "gpt-5.5"},
    {"id": "openrouter", "type": "openrouter", "enabled": True,
     "default_model": "nex-agi/nex-n2-pro:free",
     "models": ["anthropic/claude-sonnet-4-5", "nex-agi/nex-n2-pro:free"]},
]


@pytest.mark.uses_real_provider_chain
def test_chain_no_pin_uses_registration_order(monkeypatch):
    monkeypatch.setattr("backend.providers.get_providers", lambda: _PROVIDERS)
    monkeypatch.setattr(ACTIVE_MODEL, "get", lambda: {})
    chain = llm._active_provider_chain()
    assert [a.id for a in chain] == ["codex", "openrouter"]
    assert chain[0].model == "gpt-5.5"                       # codex default
    assert chain[1].model == "nex-agi/nex-n2-pro:free"       # openrouter default


@pytest.mark.uses_real_provider_chain
def test_pin_becomes_chain_primary_with_its_model(monkeypatch):
    monkeypatch.setattr("backend.providers.get_providers", lambda: _PROVIDERS)
    monkeypatch.setattr(
        ACTIVE_MODEL, "get",
        lambda: {"provider_id": "openrouter",
                 "model": "anthropic/claude-sonnet-4-5"},
    )
    chain = llm._active_provider_chain()
    # pinned provider is the HEAD, forced to the PINNED model (not its default)
    assert chain[0].id == "openrouter"
    assert chain[0].model == "anthropic/claude-sonnet-4-5"
    # the other providers follow as fallbacks; the pinned one isn't duplicated
    assert [a.id for a in chain] == ["openrouter", "codex"]


@pytest.mark.uses_real_provider_chain
def test_pin_to_disabled_provider_is_ignored(monkeypatch):
    provs = [dict(_PROVIDERS[0], enabled=False), _PROVIDERS[1]]
    monkeypatch.setattr("backend.providers.get_providers", lambda: provs)
    monkeypatch.setattr(
        ACTIVE_MODEL, "get",
        lambda: {"provider_id": "codex", "model": "gpt-5.5"},
    )
    chain = llm._active_provider_chain()
    # codex is disabled -> pin can't apply; only the enabled provider remains
    assert [a.id for a in chain] == ["openrouter"]
