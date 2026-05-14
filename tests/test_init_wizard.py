"""Tests for backend.init_wizard — the interactive Hrant setup
wizard. Builds the wizard against a non-TTY stdin so every prompt
takes the default path, then asserts the deterministic shape of
the resulting choices.

Network-bound tests (live API key validation, Ollama probe) are
patched out — the wizard's job here is the routing, not the
upstream connectivity.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect data + force providers.PROVIDERS_PATH at the tmp
    knowledge dir so the wizard's auto-register doesn't touch the
    user's real ~/.hrant/data/providers.json."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import providers as _p
    monkeypatch.setattr(_p, "PROVIDERS_PATH", tmp_path / "providers.json")
    return tmp_path


def test_display_helpers_handle_non_tty(monkeypatch):
    """_bold / _dim should NOT inject ANSI escapes when stdout
    isn't a TTY — keeps log capture / script output clean."""
    from backend import init_wizard as iw
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert iw._bold("x") == "x"
    assert iw._dim("x") == "x"


def test_ask_yes_no_returns_default_on_non_tty(monkeypatch):
    from backend import init_wizard as iw
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert iw._ask_yes_no("?", default=True) is True
    assert iw._ask_yes_no("?", default=False) is False


def test_ask_str_returns_default_on_non_tty(monkeypatch):
    from backend import init_wizard as iw
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert iw._ask_str("?", default="hello") == "hello"
    assert iw._ask_str("?", default="") == ""


def test_ask_choice_returns_default_on_non_tty(monkeypatch):
    from backend import init_wizard as iw
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    opts = [("a", "first"), ("b", "second"), ("c", "third")]
    assert iw._ask_choice("?", opts, default_idx=0) == 0
    assert iw._ask_choice("?", opts, default_idx=2) == 2


def test_provider_menu_shape():
    """The 6 starter providers must always include the 'skip'
    escape so a user can finish init without committing to a
    provider."""
    from backend.init_wizard import _ascii_provider_menu
    menu = _ascii_provider_menu()
    keys = [k for k, _, _ in menu]
    assert "anthropic" in keys
    assert "openai" in keys
    assert "openai_codex" in keys
    assert "github_copilot" in keys
    assert "ollama" in keys
    assert "skip" in keys
    # First option is Claude (recommended) — the default index 0.
    assert keys[0] == "anthropic"


def test_run_wizard_skip_path(isolated, monkeypatch, capsys):
    """Non-TTY = every prompt takes the default. Default first
    choice is 'anthropic' which triggers _wizard_provider_api_key
    which then prompts for the key — non-TTY returns empty key,
    which the sub-flow handles by returning None.

    We assert the wizard runs end-to-end without raising and the
    summary structure is correct."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    # Patch the model picker to avoid an interactive prompt deeper
    # in (defensive — _ask_choice already returns default 0 on
    # non-TTY, but if we ever change that, the test should still
    # not block).
    from backend import init_wizard
    result = init_wizard.run_wizard({})
    assert isinstance(result, dict)
    # Anthropic path tried to read a key from stdin; got empty;
    # provider was not registered.
    assert result["provider_registered"] is None
    assert result["env_updates"] == {}


def test_model_picker_with_known_models(monkeypatch):
    """When the provider entry has populated `models`, picker
    returns the first one on non-TTY (default index 0)."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    from backend.init_wizard import _wizard_model_picker
    provider = {
        "type": "anthropic",
        "models": ["claude-sonnet-4-5", "claude-opus-4-7"],
    }
    assert _wizard_model_picker(provider) == "claude-sonnet-4-5"


def test_model_picker_falls_back_to_known_table(monkeypatch):
    """Empty `models` falls back to _KNOWN_MODELS curated list."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    from backend.init_wizard import _wizard_model_picker
    provider = {"type": "openai", "models": []}
    # Default index 0 → 'gpt-4o-mini' from the curated table.
    assert _wizard_model_picker(provider) == "gpt-4o-mini"


def test_cmd_init_routes_to_wizard_on_tty(isolated, monkeypatch, capsys):
    """When stdin is a TTY and --skip-wizard isn't set, cmd_init
    should call into init_wizard.run_wizard exactly once."""
    import argparse
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    fake_result = {
        "env_updates": {"ANTHROPIC_API_KEY": "sk-test"},
        "provider_registered": "anthropic-default",
        "active_model": "claude-sonnet-4-5",
        "voice_enabled": False,
        "telegram_enabled": False,
        "tailscale_host": "",
    }
    with patch("backend.init_wizard.run_wizard", return_value=fake_result) as m_wiz, \
         patch("backend.init_wizard.print_final_summary"):
        from backend import cli as cli_mod
        rc = cli_mod.cmd_init(argparse.Namespace(reset=False, skip_wizard=False))
    assert rc == 0
    m_wiz.assert_called_once()


def test_cmd_init_legacy_path_when_skip_wizard(isolated, monkeypatch):
    """--skip-wizard → falls through to the flat-prompt path. Even
    on a TTY, the wizard module should NOT be called. Patch
    _read_input so the flat-path prompts don't try to block on
    real stdin in test mode."""
    import argparse
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("backend.cli._read_input", lambda prompt, default="": default or "")
    with patch("backend.init_wizard.run_wizard") as m_wiz:
        from backend import cli as cli_mod
        rc = cli_mod.cmd_init(argparse.Namespace(reset=False, skip_wizard=True))
    assert rc == 0
    m_wiz.assert_not_called()


def test_validate_telegram_token_ok():
    from backend.init_wizard import _validate_telegram_token
    fake = MagicMock(status_code=200)
    fake.json.return_value = {
        "ok": True,
        "result": {"username": "hrant_bot", "first_name": "Hrant"},
    }
    with patch("httpx.get", return_value=fake):
        ok, msg = _validate_telegram_token("123:abc")
    assert ok is True
    assert "@hrant_bot" in msg


def test_validate_telegram_token_rejected():
    from backend.init_wizard import _validate_telegram_token
    fake = MagicMock(status_code=200)
    fake.json.return_value = {"ok": False, "description": "Unauthorized"}
    with patch("httpx.get", return_value=fake):
        ok, msg = _validate_telegram_token("bad-token")
    assert ok is False
    assert "Unauthorized" in msg


def test_validate_telegram_token_empty():
    from backend.init_wizard import _validate_telegram_token
    ok, msg = _validate_telegram_token("")
    assert ok is False
    assert "no token" in msg


def test_validate_telegram_token_network_error():
    from backend.init_wizard import _validate_telegram_token
    with patch("httpx.get", side_effect=ConnectionError("dns fail")):
        ok, msg = _validate_telegram_token("123:abc")
    assert ok is False
    assert "network" in msg


def test_cmd_init_legacy_path_on_non_tty(isolated, monkeypatch):
    """Non-TTY (cron, CI) → also uses the legacy path so prompts
    silently take defaults without trying to render a wizard."""
    import argparse
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with patch("backend.init_wizard.run_wizard") as m_wiz:
        from backend import cli as cli_mod
        rc = cli_mod.cmd_init(argparse.Namespace(reset=False, skip_wizard=False))
    assert rc == 0
    m_wiz.assert_not_called()
