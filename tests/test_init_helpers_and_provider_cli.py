"""Tests for Phase 13 — init wizard provider flows + `hrant provider` CLI.

Network-free: the connection-test helpers patch `httpx.get`; the
auto-register helpers run against an isolated providers.json under
HRANT_DATA_DIR.
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    """Redirect data + force the providers module to use the tmp
    providers.json. Without this the tests would write into the
    user's real ~/.hrant/data/."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    # Force providers.PROVIDERS_PATH recomputation by patching it
    # directly. (It's resolved at import time via paths.knowledge_dir().)
    from backend import providers as _p
    monkeypatch.setattr(_p, "PROVIDERS_PATH", tmp_path / "providers.json")
    return tmp_path


# --- Connection tests --------------------------------------------------


def test_anthropic_test_ok():
    from backend.init_helpers import test_anthropic_key
    fake = MagicMock(status_code=200)
    fake.json.return_value = {"data": [{"id": "a"}, {"id": "b"}]}
    with patch("httpx.get", return_value=fake):
        ok, msg = test_anthropic_key("sk-ant-abc123")
    assert ok is True
    assert "2 models" in msg


def test_anthropic_test_401_rejects():
    from backend.init_helpers import test_anthropic_key
    fake = MagicMock(status_code=401)
    with patch("httpx.get", return_value=fake):
        ok, msg = test_anthropic_key("bad-key")
    assert ok is False
    assert "key rejected" in msg


def test_anthropic_test_empty_key():
    from backend.init_helpers import test_anthropic_key
    ok, msg = test_anthropic_key("")
    assert ok is False
    assert msg == "(no key)"


def test_anthropic_test_network_error():
    """A timeout / connection error must not crash the wizard — just
    return (False, network message)."""
    from backend.init_helpers import test_anthropic_key
    with patch("httpx.get", side_effect=ConnectionError("refused")):
        ok, msg = test_anthropic_key("sk-x")
    assert ok is False
    assert "network" in msg.lower()


def test_openai_test_rate_limit_still_ok():
    """429 means the key works but we're throttled — count as ok so
    the wizard doesn't warn the user about a valid key."""
    from backend.init_helpers import test_openai_key
    fake = MagicMock(status_code=429)
    with patch("httpx.get", return_value=fake):
        ok, msg = test_openai_key("sk-x")
    assert ok is True
    assert "rate-limited" in msg


# --- Auto-register OpenAI ----------------------------------------------


def test_auto_register_openai_creates_entry(isolated_kb):
    from backend.init_helpers import auto_register_openai
    entry = auto_register_openai("sk-x")
    assert entry is not None
    assert entry["type"] == "openai"
    assert entry["api_key_env"] == "OPENAI_API_KEY"
    # Persisted to providers.json.
    from backend.providers import _load_providers
    saved = _load_providers()
    assert any(p["id"] == entry["id"] for p in saved)


def test_auto_register_openai_idempotent(isolated_kb):
    """Re-running with the same key shouldn't append duplicate
    entries — we already have one, return it."""
    from backend.init_helpers import auto_register_openai
    first = auto_register_openai("sk-x")
    second = auto_register_openai("sk-x")
    assert first is not None and second is not None
    assert first["id"] == second["id"]
    from backend.providers import _load_providers
    saved = _load_providers()
    openai_count = sum(1 for p in saved if p["type"] == "openai")
    assert openai_count == 1


def test_auto_register_skips_empty_key(isolated_kb):
    from backend.init_helpers import auto_register_openai
    assert auto_register_openai("") is None
    assert auto_register_openai("   ") is None


# --- Discover + apply --------------------------------------------------


def test_discover_and_apply_empty_host():
    from backend.init_helpers import discover_and_apply
    r = discover_and_apply("")
    assert r["found"] == {}
    assert r["applied"] == {}
    assert "error" not in r


def test_discover_and_apply_runs(isolated_kb):
    from backend.init_helpers import discover_and_apply
    with patch(
        "backend.discovery.discover_services",
        return_value={
            "whisper": {"ok": True, "url": "http://1.2.3.4:8016"},
            "piper": {"ok": False, "reason": "down"},
        },
    ), patch(
        "backend.discovery.apply_discovery",
        return_value={"whisper": "applied", "piper": "skipped"},
    ):
        r = discover_and_apply("1.2.3.4")
    assert r["found"]["whisper"]["ok"] is True
    assert r["applied"]["whisper"] == "applied"


# --- `hrant provider` CLI ----------------------------------------------


def test_provider_list_runs(isolated_kb, capsys):
    """Smoke test — even with no user providers in providers.json,
    the auto-injected Anthropic default (from env) shows up. Set
    the env var to force the auto-inject path."""
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-test"
    try:
        from backend import cli as cli_mod
        rc = cli_mod.cmd_provider_list(argparse.Namespace())
        assert rc == 0
        out = capsys.readouterr().out
        assert "anthropic-default" in out
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_provider_login_no_type_prints_list(capsys):
    from backend import cli as cli_mod
    rc = cli_mod.cmd_provider_login(argparse.Namespace(provider_type=""))
    assert rc == 2
    out = capsys.readouterr().out
    assert "anthropic" in out
    assert "ollama" in out
    assert "openai_codex" in out


def test_provider_test_unknown_id_errors(isolated_kb, capsys):
    from backend import cli as cli_mod
    rc = cli_mod.cmd_provider_test(
        argparse.Namespace(provider_id="no-such-thing"),
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "no provider" in err


def test_provider_use_sets_active(isolated_kb, capsys):
    """End-to-end: register a fake provider then `hrant provider use`
    flips ACTIVE_MODEL."""
    from backend import cli as cli_mod
    from backend import providers as _p
    _p._save_providers([{
        "id": "fake-1",
        "name": "Fake",
        "type": "ollama",
        "auth_type": "none",
        "enabled": True,
        "is_default": False,
        "default_model": "qwen2.5:7b",
        "models": ["qwen2.5:7b"],
        "base_url": "http://localhost:11434",
        "api_key": "",
        "api_key_env": "",
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": "2026-05-14 00:00:00",
    }])
    rc = cli_mod.cmd_provider_use(
        argparse.Namespace(provider_id="fake-1", model=None),
    )
    assert rc == 0
    active = _p.ACTIVE_MODEL.get() or {}
    assert active.get("provider_id") == "fake-1"
    assert active.get("model") == "qwen2.5:7b"


def test_provider_logout_codex_prints_hint(isolated_kb, capsys):
    """Codex/Copilot logout doesn't actually touch providers.json
    (the auth lives in upstream CLI files); it just instructs the
    user where to log out."""
    from backend import cli as cli_mod
    from backend import providers as _p
    _p._save_providers([{
        "id": "openai-codex",
        "name": "Codex",
        "type": "openai_codex",
        "auth_type": "codex_subscription",
        "enabled": True,
        "is_default": False,
        "api_key": "",
        "api_key_env": "",
        "base_url": "",
        "models": [],
        "default_model": "",
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": "2026-05-14 00:00:00",
    }])
    rc = cli_mod.cmd_provider_logout(
        argparse.Namespace(provider_id="openai-codex"),
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "codex logout" in out


def test_provider_logout_api_key_clears_credentials(isolated_kb):
    from backend import cli as cli_mod
    from backend import providers as _p
    _p._save_providers([{
        "id": "openai-x",
        "name": "OpenAI test",
        "type": "openai",
        "auth_type": "api_key",
        "enabled": True,
        "is_default": False,
        "api_key": "sk-secret",
        "api_key_env": "",
        "base_url": "",
        "models": [],
        "default_model": "",
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": "2026-05-14 00:00:00",
    }])
    rc = cli_mod.cmd_provider_logout(
        argparse.Namespace(provider_id="openai-x"),
    )
    assert rc == 0
    saved = _p._load_providers()
    entry = next(p for p in saved if p["id"] == "openai-x")
    assert entry["api_key"] == ""  # cleared
