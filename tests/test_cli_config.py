"""Tests for `hrant config` — the friendly knob-twiddling surface.

The module is supposed to be a thin face over .env + JSON files, so
the tests pin three properties:

  1. Each registered key resolves to a real backing file and round-
     trips through write_value → read_value cleanly.
  2. Secrets are redacted on display (never the raw key in `list` or
     `get` output).
  3. The argparse wiring routes `hrant config <action>` to the right
     handler with the right key/value args.

We isolate by pointing HRANT_DATA_DIR at a tmp_path — every read +
write goes there, never the dev machine's real ~/.hrant/data/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from backend import cli as cli_mod
from backend import cli_config as cc
from backend import paths


# ─── Fixture: clean data_dir per test ────────────────────────────────


@pytest.fixture
def hrant_home(tmp_path, monkeypatch):
    """Redirect HRANT_DATA_DIR so cli_config reads + writes inside
    tmp_path. paths.knowledge_dir() / env_path() re-read the env var
    on every call, so just setting it is enough — no cache to bust."""
    data_dir = tmp_path / "hrant_data"
    data_dir.mkdir()
    (data_dir / "knowledge").mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(data_dir))
    return data_dir


# ─── Registry sanity ────────────────────────────────────────────────


def test_registry_has_expected_groups():
    """Spec: API keys / Voice / Telegram / Discovery / Autonomic.
    A missing group means a refactor lost user-facing settings."""
    groups = {k.group for k in cc.REGISTRY}
    for expected in ("API keys", "Voice", "Telegram", "Discovery", "Autonomic"):
        assert expected in groups, f"group missing from registry: {expected}"


def test_every_key_has_a_label_and_source():
    for k in cc.REGISTRY:
        assert k.key, f"empty key name in registry"
        assert k.label, f"key {k.key} missing label"
        assert k.source.startswith(("env:", "json:")), (
            f"{k.key}: unsupported source DSL `{k.source}`"
        )


def test_find_key_returns_matching_entry():
    k = cc.find_key("anthropic.api_key")
    assert k is not None
    assert k.secret is True
    assert k.source == "env:ANTHROPIC_API_KEY"


def test_find_key_returns_none_for_unknown():
    assert cc.find_key("does.not.exist") is None


# ─── Read/write round-trip: env-backed keys ─────────────────────────


def test_env_key_round_trip(hrant_home):
    k = cc.find_key("anthropic.api_key")
    assert cc.read_value(k) is None  # fresh env
    cc.write_value(k, "sk-ant-test-xxxx-yyyy-zzzz")
    assert cc.read_value(k) == "sk-ant-test-xxxx-yyyy-zzzz"
    # The .env file actually exists and contains the line.
    env_text = paths.env_path().read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=sk-ant-test-xxxx-yyyy-zzzz" in env_text


def test_env_key_delete_removes_line(hrant_home):
    k = cc.find_key("tailscale.host")
    cc.write_value(k, "100.64.0.5")
    assert cc.read_value(k) == "100.64.0.5"
    cc.delete_value(k)
    assert cc.read_value(k) is None


# ─── Read/write round-trip: JSON-backed keys ────────────────────────


def test_json_key_round_trip_creates_file(hrant_home):
    k = cc.find_key("tts.backend")
    assert cc.read_value(k) is None
    cc.write_value(k, "edge_tts")
    assert cc.read_value(k) == "edge_tts"
    # File created at the right place with the right shape.
    p = Path(hrant_home / "knowledge" / "tts_config.json")
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == {"backend": "edge_tts"}


def test_json_key_supports_nested_dotted_paths(hrant_home):
    k = cc.find_key("tts.edge_voice")
    cc.write_value(k, "en-US-AriaNeural")
    p = hrant_home / "knowledge" / "tts_config.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == {"edge_tts": {"voice": "en-US-AriaNeural"}}


def test_json_key_does_not_clobber_sibling_paths(hrant_home):
    """Writing tts.edge_voice must keep tts.backend (`edge_tts`)
    intact. The dotted-set helper merges, not replaces."""
    cc.write_value(cc.find_key("tts.backend"), "edge_tts")
    cc.write_value(cc.find_key("tts.edge_voice"), "en-US-AriaNeural")
    cc.write_value(cc.find_key("tts.piper_url"), "http://localhost:8017")
    p = hrant_home / "knowledge" / "tts_config.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["backend"] == "edge_tts"
    assert data["edge_tts"]["voice"] == "en-US-AriaNeural"
    assert data["local_piper"]["url"] == "http://localhost:8017"


# ─── Type coercion + validation ─────────────────────────────────────


def test_float_key_rejects_out_of_range(hrant_home):
    k = cc.find_key("autonomic.heartbeat_seconds")
    with pytest.raises(ValueError):
        cc.write_value(k, "0.5")          # below min=1
    with pytest.raises(ValueError):
        cc.write_value(k, "99999")        # above max=3600


def test_float_key_stores_numeric_value(hrant_home):
    k = cc.find_key("autonomic.heartbeat_seconds")
    cc.write_value(k, "60")
    assert cc.read_value(k) == 60.0
    p = hrant_home / "knowledge" / "autonomic_settings.json"
    assert json.loads(p.read_text(encoding="utf-8")) == {"tick_interval_seconds": 60.0}


def test_choice_key_rejects_unknown_value(hrant_home):
    k = cc.find_key("tts.backend")
    with pytest.raises(ValueError):
        cc.write_value(k, "espeak")  # not in choices


# ─── Redaction ──────────────────────────────────────────────────────


def test_redact_short_keys_fully_masked():
    assert cc._redact("hi") == "**"


def test_redact_long_keys_show_last_4():
    out = cc._redact("sk-ant-secret-long-key-12345")
    assert out.endswith("2345")
    assert "secret" not in out


def test_list_redacts_secrets(hrant_home, capsys):
    cc.write_value(cc.find_key("anthropic.api_key"), "sk-ant-supersecretvalue-9999")
    cc.print_list()
    out = capsys.readouterr().out
    # Last 4 chars are shown, the secret body must NOT leak.
    assert "9999" in out
    assert "supersecretvalue" not in out


def test_get_redacts_secrets(hrant_home, capsys):
    cc.write_value(cc.find_key("openai.api_key"), "sk-openai-supersecret-1234")
    cc.print_get("openai.api_key")
    out = capsys.readouterr().out
    assert "1234" in out
    assert "supersecret" not in out


# ─── Argparse wiring ────────────────────────────────────────────────


def test_config_top_level_in_help():
    parser = cli_mod.build_parser()
    help_text = parser.format_help()
    assert "config" in help_text


def test_config_list_dispatcher_present():
    parser = cli_mod.build_parser()
    args = parser.parse_args(["config", "list"])
    assert args.func is cli_mod.cmd_config
    assert args.config_cmd == "list"


def test_config_get_passes_key():
    parser = cli_mod.build_parser()
    args = parser.parse_args(["config", "get", "anthropic.api_key"])
    assert args.func is cli_mod.cmd_config
    assert args.config_cmd == "get"
    assert args.key == "anthropic.api_key"


def test_config_set_passes_key_and_value():
    parser = cli_mod.build_parser()
    args = parser.parse_args(["config", "set", "tts.backend", "edge_tts"])
    assert args.func is cli_mod.cmd_config
    assert args.config_cmd == "set"
    assert args.key == "tts.backend"
    assert args.value == "edge_tts"


def test_cmd_config_set_unknown_key_returns_error(capsys):
    rc = cli_mod.cmd_config(argparse.Namespace(
        config_cmd="set", key="bogus.key", value="x",
    ))
    assert rc == 1
    out = capsys.readouterr().out
    assert "unknown key" in out


def test_cmd_config_get_unknown_key_returns_error(capsys):
    rc = cli_mod.cmd_config(argparse.Namespace(
        config_cmd="get", key="bogus.key",
    ))
    assert rc == 1


def test_cmd_config_set_persists_via_argparse_path(hrant_home):
    """End-to-end through the dispatcher — argparse args land on
    the right key + value and the value reaches the .env file."""
    rc = cli_mod.cmd_config(argparse.Namespace(
        config_cmd="set", key="tailscale.host", value="100.64.0.5",
    ))
    assert rc == 0
    assert cc.read_value(cc.find_key("tailscale.host")) == "100.64.0.5"
