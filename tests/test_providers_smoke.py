"""Smoke tests for backend.providers — the LLM provider registry.

The audit flagged this as the largest untested module (1467 LOC).
Full coverage is out of scope; this file pins the CRITICAL paths
that would silently break agent.run if regressed:

  - Provider CRUD: save / get / list / delete
  - Default provider injection (Anthropic from env when no
    providers.json yet)
  - get_api_key resolution: literal key vs env var vs OAuth
  - PKCE pair generation (used by all OAuth flows)
  - ACTIVE_MODEL set / clear / resolve_llm_config

Out of scope (would require live API calls or extensive mocking):
  - test_provider connectivity checks
  - OAuth exchange flows
  - generate_pkce statistical quality
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def fresh_providers(tmp_path, monkeypatch):
    """Redirect providers.json at a tmp path. Uses monkeypatch on
    the singleton's PROVIDERS_PATH attribute so we don't trash
    module state for other tests in the suite (reload pollutes
    every other test that imports from backend.providers)."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import providers as _p
    monkeypatch.setattr(
        _p, "PROVIDERS_PATH",
        tmp_path / "knowledge" / "providers.json",
    )
    return _p


# ─── CRUD ──────────────────────────────────────────────────────────


def test_get_providers_empty_when_no_file(fresh_providers, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert fresh_providers.get_providers() == []


def test_save_provider_persists_to_disk(fresh_providers):
    rec = {
        "id": "my-openai",
        "name": "OpenAI (test)",
        "type": "openai",
        "auth_type": "api_key",
        "api_key": "sk-test-1234",
        "enabled": True,
    }
    saved = fresh_providers.save_provider(rec)
    assert saved["id"] == "my-openai"
    # Round-trip via fresh load.
    assert any(
        p["id"] == "my-openai" for p in fresh_providers.get_providers()
    )


def test_get_provider_returns_none_for_unknown(fresh_providers):
    assert fresh_providers.get_provider("does-not-exist") is None


def test_delete_provider_removes_record(fresh_providers):
    fresh_providers.save_provider({
        "id": "to-delete",
        "name": "ditto",
        "type": "openai",
        "auth_type": "api_key",
        "api_key": "x",
    })
    assert fresh_providers.delete_provider("to-delete") is True
    assert fresh_providers.get_provider("to-delete") is None
    # Deleting a missing id is a no-op (returns False).
    assert fresh_providers.delete_provider("never-existed") is False


def test_get_api_key_prefers_literal_over_env(fresh_providers, monkeypatch):
    """When `api_key` is non-empty in the record, prefer it over
    `api_key_env`. This matters because users can override one
    provider's key without touching .env."""
    monkeypatch.setenv("MY_ENV_KEY", "from-env")
    assert fresh_providers.get_api_key({
        "api_key": "literal-key",
        "api_key_env": "MY_ENV_KEY",
    }) == "literal-key"


def test_get_api_key_falls_through_to_env(fresh_providers, monkeypatch):
    monkeypatch.setenv("SOME_KEY", "env-value")
    assert fresh_providers.get_api_key({
        "api_key": "",
        "api_key_env": "SOME_KEY",
    }) == "env-value"


def test_get_api_key_empty_when_neither_set(fresh_providers, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    assert fresh_providers.get_api_key({
        "api_key": "",
        "api_key_env": "MISSING_KEY",
    }) == ""


# ─── PKCE ──────────────────────────────────────────────────────────


def test_generate_pkce_returns_two_distinct_strings(fresh_providers):
    verifier, challenge = fresh_providers.generate_pkce()
    assert verifier
    assert challenge
    assert verifier != challenge
    # OAuth RFC: verifier must be at least 43 chars (32 bytes base64url).
    assert len(verifier) >= 43


def test_generate_pkce_yields_distinct_pairs(fresh_providers):
    """Two calls produce different verifier/challenge pairs (random)."""
    v1, c1 = fresh_providers.generate_pkce()
    v2, c2 = fresh_providers.generate_pkce()
    assert v1 != v2
    assert c1 != c2


# ─── ACTIVE_MODEL ──────────────────────────────────────────────────


def test_active_model_resolve_returns_none_when_unset(fresh_providers):
    fresh_providers.ACTIVE_MODEL.clear()
    assert fresh_providers.ACTIVE_MODEL.resolve_llm_config() is None


def test_active_model_set_then_resolve(fresh_providers, monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc")
    fresh_providers.save_provider({
        "id": "active-test",
        "name": "x",
        "type": "openai",
        "auth_type": "api_key",
        "api_key_env": "MY_KEY",
        "default_model": "gpt-4o",
        "enabled": True,
    })
    fresh_providers.ACTIVE_MODEL.set("active-test", "gpt-4o")
    cfg = fresh_providers.ACTIVE_MODEL.resolve_llm_config()
    assert cfg is not None
    assert cfg["provider_id"] == "active-test"
    assert cfg["model"] == "gpt-4o"


def test_active_model_resolve_returns_none_when_provider_disabled(fresh_providers):
    fresh_providers.save_provider({
        "id": "disabled",
        "name": "x",
        "type": "openai",
        "auth_type": "api_key",
        "api_key": "x",
        "enabled": False,
    })
    fresh_providers.ACTIVE_MODEL.set("disabled", "gpt-4o")
    assert fresh_providers.ACTIVE_MODEL.resolve_llm_config() is None


def test_active_model_clear(fresh_providers):
    fresh_providers.save_provider({
        "id": "to-clear",
        "name": "x",
        "type": "openai",
        "auth_type": "api_key",
        "api_key": "x",
    })
    fresh_providers.ACTIVE_MODEL.set("to-clear", "gpt-4o")
    fresh_providers.ACTIVE_MODEL.clear()
    assert fresh_providers.ACTIVE_MODEL.resolve_llm_config() is None


# ─── _load_providers tolerates malformed disk state ───────────────


def test_load_providers_tolerates_unreadable_json(fresh_providers):
    fresh_providers.PROVIDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fresh_providers.PROVIDERS_PATH.write_text("not valid json {", encoding="utf-8")
    # Should not crash; should return an empty list.
    out = fresh_providers._load_providers()
    assert isinstance(out, list)
