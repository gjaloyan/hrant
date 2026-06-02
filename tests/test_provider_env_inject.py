"""Tests for auto-injection of configured provider API keys into
bg-job subprocess environments.

Bug caught during the 2026-06-03 real-bench attempt: bench scripts
(harbor run --agent hermes -m anthropic/...) authenticate against
`OPENROUTER_API_KEY` via the shell `os.environ`. The Hrant agent
already has a working OpenRouter key stored under
`providers.json`, but it lived only in the agent's process — the
bg-job subprocess inherited the systemd-service env (.env file
was empty), so `harbor` couldn't authenticate and the supervisor
spent retries on "no usable API key".

`_collect_provider_env()` resolves each enabled provider's key via
the same `get_api_key` path the agent uses, keyed by the type's
`key_env_default` (OPENROUTER_API_KEY for openrouter, etc.).
`start_job` merges these into the subprocess env so bench tools see
them without manual `export` ceremony in the command.

Operator-set env vars are never overwritten — only filled.
"""
from __future__ import annotations

import pytest


def _stub_providers(monkeypatch, items):
    """Stub backend.providers with a fixed provider list + canned keys."""
    import backend.providers as _p

    monkeypatch.setattr(_p, "get_providers", lambda: items)

    def fake_get_key(prov):
        return prov.get("__test_key__", "")

    monkeypatch.setattr(_p, "get_api_key", fake_get_key)


def test_collect_provider_env_returns_known_keys(monkeypatch):
    """An enabled OpenRouter provider with a key contributes
    OPENROUTER_API_KEY=<key>."""
    _stub_providers(monkeypatch, [
        {
            "id": "openrouter-x", "type": "openrouter",
            "enabled": True, "__test_key__": "sk-or-XXXX",
        },
    ])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from backend.tools.background_jobs import _collect_provider_env
    out = _collect_provider_env()
    assert out.get("OPENROUTER_API_KEY") == "sk-or-XXXX"


def test_collect_provider_env_respects_operator_set(monkeypatch):
    """If the operator already exported OPENROUTER_API_KEY in the
    service env, the auto-collect must NOT overwrite it."""
    _stub_providers(monkeypatch, [
        {
            "id": "or", "type": "openrouter",
            "enabled": True, "__test_key__": "sk-AUTO-INJECTED",
        },
    ])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-OPERATOR-WINS")
    from backend.tools.background_jobs import _collect_provider_env
    out = _collect_provider_env()
    # Auto-collect skipped this var because operator pre-set it.
    assert "OPENROUTER_API_KEY" not in out


def test_collect_provider_env_skips_disabled_provider(monkeypatch):
    """Disabled providers contribute nothing — they're explicitly off."""
    _stub_providers(monkeypatch, [
        {
            "id": "off", "type": "openrouter",
            "enabled": False, "__test_key__": "sk-NOPE",
        },
    ])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from backend.tools.background_jobs import _collect_provider_env
    assert "OPENROUTER_API_KEY" not in _collect_provider_env()


def test_collect_provider_env_skips_empty_keys(monkeypatch):
    """An enabled provider with no resolvable key contributes nothing
    — better silence than `OPENROUTER_API_KEY=` which would lie to
    downstream tools (a present-but-empty env var defeats the
    `os.environ.get(...) or default` fallback pattern)."""
    _stub_providers(monkeypatch, [
        {
            "id": "or", "type": "openrouter",
            "enabled": True, "__test_key__": "",
        },
    ])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from backend.tools.background_jobs import _collect_provider_env
    assert "OPENROUTER_API_KEY" not in _collect_provider_env()


def test_collect_provider_env_uses_explicit_api_key_env_override(monkeypatch):
    """A provider record may pin a non-default env var name via
    `api_key_env`. Auto-collect must honour that, not the type default."""
    _stub_providers(monkeypatch, [
        {
            "id": "or-custom", "type": "openrouter",
            "api_key_env": "MY_OPENROUTER_KEY",
            "enabled": True, "__test_key__": "sk-CUSTOM",
        },
    ])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MY_OPENROUTER_KEY", raising=False)
    from backend.tools.background_jobs import _collect_provider_env
    out = _collect_provider_env()
    assert out.get("MY_OPENROUTER_KEY") == "sk-CUSTOM"
    # Type default NOT used because provider-level override wins.
    assert "OPENROUTER_API_KEY" not in out


def test_collect_provider_env_skips_provider_without_known_env_name(monkeypatch):
    """A type with no `key_env_default` (e.g. codex_subscription —
    uses OAuth, not an env var) is skipped."""
    _stub_providers(monkeypatch, [
        {
            "id": "codex", "type": "openai_codex",
            "enabled": True, "__test_key__": "ignored",
        },
    ])
    from backend.tools.background_jobs import _collect_provider_env
    out = _collect_provider_env()
    # openai_codex has key_env_default="" in PROVIDER_TYPES.
    # Nothing should land for it.
    assert all("CODEX" not in k for k in out.keys())


def test_collect_provider_env_safe_when_get_providers_raises(monkeypatch):
    """If `providers.get_providers()` raises (corrupt providers.json,
    permission error), `_collect_provider_env()` must return {} so
    bg-job launch is unaffected — never block a launch on a
    config-side issue."""
    import backend.providers as _p

    def _boom():
        raise RuntimeError("simulated providers.json corruption")

    monkeypatch.setattr(_p, "get_providers", _boom)
    from backend.tools.background_jobs import _collect_provider_env
    assert _collect_provider_env() == {}


def test_collect_provider_env_per_provider_failure_is_skipped(monkeypatch):
    """One bad provider record (e.g. get_api_key raises) must not
    poison the whole collect — other providers still contribute."""
    _stub_providers(monkeypatch, [
        {"id": "bad", "type": "openrouter", "enabled": True,
         "__test_key__": "sk-OK1"},
        {"id": "good", "type": "anthropic", "enabled": True,
         "__test_key__": "sk-OK2"},
    ])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import backend.providers as _p
    real_get_key = _p.get_api_key

    def selective_boom(prov):
        if prov.get("id") == "bad":
            raise RuntimeError("simulated")
        return real_get_key(prov)

    monkeypatch.setattr(_p, "get_api_key", selective_boom)
    from backend.tools.background_jobs import _collect_provider_env
    out = _collect_provider_env()
    assert "OPENROUTER_API_KEY" not in out
    assert out.get("ANTHROPIC_API_KEY") == "sk-OK2"
