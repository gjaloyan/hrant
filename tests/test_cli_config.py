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
    on every call. backend.channels caches CHANNELS_PATH at import
    time, so monkeypatch that too — otherwise the telegram-token tests
    would write to the dev machine's real channels.json."""
    data_dir = tmp_path / "hrant_data"
    data_dir.mkdir()
    (data_dir / "knowledge").mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(data_dir))
    from backend import channels as _ch
    monkeypatch.setattr(_ch, "CHANNELS_PATH", data_dir / "knowledge" / "channels.json")
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
        # Either the source DSL is the standard env:/json: shape, OR
        # it's `custom` AND custom reader/writer callables are wired.
        if k.source == "custom":
            assert k.reader is not None, f"{k.key}: custom source needs reader"
            assert k.writer is not None, f"{k.key}: custom source needs writer"
            continue
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


# ─── Telegram bot token: channels.json storage ──────────────────────


def test_telegram_token_reads_from_existing_channel(hrant_home):
    """When a telegram channel exists in channels.json, `hrant config
    get telegram.bot_token` returns that channel's bot_token — NOT
    whatever's in .env. This was the bug the user reported: the bot
    was configured but `config list` showed (not set)."""
    from backend import channels as _ch
    _ch.save_channel({
        "id": "telegram-default",
        "type": "telegram",
        "enabled": True,
        "auto_start": True,
        "config": {"bot_token": "1234567890:AAA-bbb-ccc-secret", "allowed_users": []},
    })
    assert cc.read_value(cc.find_key("telegram.bot_token")) == "1234567890:AAA-bbb-ccc-secret"


def test_telegram_token_write_updates_existing_channel(hrant_home):
    from backend import channels as _ch
    _ch.save_channel({
        "id": "telegram-default",
        "type": "telegram",
        "enabled": True,
        "config": {"bot_token": "old-token", "allowed_users": ["preserved"]},
    })
    cc.write_value(cc.find_key("telegram.bot_token"), "new-token-9999")
    ch = next(c for c in _ch.get_channels() if c["type"] == "telegram")
    assert ch["config"]["bot_token"] == "new-token-9999"
    # Sibling fields stay intact — write must not clobber allowed_users.
    assert ch["config"]["allowed_users"] == ["preserved"]


def test_telegram_token_write_creates_channel_when_missing(hrant_home):
    """No telegram channel yet → writer creates one with sane defaults
    so the user can `hrant config set telegram.bot_token X` as their
    first step without running the init wizard."""
    from backend import channels as _ch
    assert [c for c in _ch.get_channels() if c["type"] == "telegram"] == []
    cc.write_value(cc.find_key("telegram.bot_token"), "first-token")
    chans = [c for c in _ch.get_channels() if c["type"] == "telegram"]
    assert len(chans) == 1
    assert chans[0]["config"]["bot_token"] == "first-token"
    assert chans[0]["enabled"] is True
    assert chans[0]["auto_start"] is True


def test_telegram_token_delete_disables_channel(hrant_home):
    """unset must NOT remove the whole channel record (loses
    allowed_users etc.); it disables it + clears the token."""
    from backend import channels as _ch
    _ch.save_channel({
        "id": "telegram-default",
        "type": "telegram",
        "enabled": True,
        "config": {"bot_token": "secret", "allowed_users": ["user1"]},
    })
    cc.delete_value(cc.find_key("telegram.bot_token"))
    ch = next(c for c in _ch.get_channels() if c["type"] == "telegram")
    assert ch["enabled"] is False
    assert ch["config"]["bot_token"] == ""
    assert ch["config"]["allowed_users"] == ["user1"]


def test_telegram_token_redacted_in_list(hrant_home, capsys):
    from backend import channels as _ch
    _ch.save_channel({
        "id": "telegram-default",
        "type": "telegram",
        "enabled": True,
        "config": {"bot_token": "1851234567:AAFverysecretsupersecretWP-U", "allowed_users": []},
    })
    cc.print_list()
    out = capsys.readouterr().out
    # Last 4 chars are visible; the secret body is NOT.
    assert "WP-U" in out
    assert "supersecret" not in out


# ─── Arrow-key menu: non-TTY fallback ──────────────────────────────


def test_cli_menu_falls_back_to_numbered_on_non_tty(monkeypatch, capsys):
    """When stdin/stdout aren't TTYs (cron, pipe, captured stdin in
    tests), the arrow menu must degrade to a numbered prompt that
    reads one line. Otherwise scripted runs would hang waiting for
    arrow keys that never come."""
    from backend import cli_menu
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    idx = cli_menu.select("Pick one", [("A", ""), ("B", ""), ("C", "")], 0)
    assert idx == 1  # "2" → 0-based index 1


def test_cli_menu_returns_default_on_empty_input(monkeypatch):
    from backend import cli_menu
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: "")
    idx = cli_menu.select("?", [("A", ""), ("B", "")], default_idx=1)
    assert idx == 1


def test_cli_menu_returns_default_on_eof(monkeypatch):
    """EOF on stdin (pipe closed) → default index. Don't crash."""
    from backend import cli_menu

    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", raise_eof)
    idx = cli_menu.select("?", [("A", ""), ("B", "")], default_idx=0)
    assert idx == 0


# ─── Arrow-key parsing: regression tests for the read-from-pty path
#
# Bug fixed in this commit: pressing arrow keys cancelled the menu
# instead of moving the selection. The terminal sends `\x1b[A` (up)
# as one burst, but `sys.stdin.read(1)` consumed only `\x1b` and
# parked `[A` in Python's stdin buffer. `select.select` then
# checked the kernel FD (empty — bytes were in the buffer) and
# reported "no follow-up", so the menu treated the arrow as lone-
# Esc cancel. Fix: read with `os.read(fd, ...)` directly.
#
# These tests pipe a real pty pair so the buffer-vs-FD distinction
# matters; a `monkeypatch.setattr(os, "read", ...)` style mock
# would let the bug recur without anyone noticing.


def _type_after_raw_mode(master_fd, payload, delay=0.05):
    """Deliver `payload` to the pty AFTER the code under test has entered raw
    mode, and return the thread so the caller can join it.

    `_read_key_unix` calls `tty.setraw(fd)`, which is
    `termios.tcsetattr(fd, TCSAFLUSH, ...)` — and TCSAFLUSH DISCARDS pending
    input. Writing the payload before the call therefore races: the bytes are
    flushed away and the following `os.read` blocks forever on input that no
    longer exists. On Windows both tests are skipped, so the suite looked green
    while every full run on the Linux box hung here (measured 2026-08-07: the
    prod suite sat at 18% for 66 minutes on this file).

    Writing from a timer thread also models reality better — a human presses
    the key after the menu has entered raw mode, not before.
    """
    import os as _os
    import threading

    t = threading.Timer(delay, lambda: _os.write(master_fd, payload))
    t.daemon = True
    t.start()
    return t


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="pty + termios are Unix-only; the Windows arrow path "
           "uses msvcrt.getch which we test separately"
)
def test_read_key_unix_decodes_arrow_keys_from_real_pty():
    """Open a pty, write the exact byte sequence that an arrow key
    sends, call `_read_key_unix`, assert it returns the right
    direction. Regression for the buffer-vs-FD bug."""
    import os as _os
    import pty
    import sys as _sys
    from backend import cli_menu

    cases = [
        (b"\x1b[A", "up"),
        (b"\x1b[B", "down"),
        (b"\x1b[C", "right"),
        (b"\x1b[D", "left"),
        (b"\r",      "enter"),
        (b"\n",      "enter"),
        (b"q",       "q"),
        (b"\x03",    "ctrl_c"),
    ]
    for payload, expected in cases:
        master, slave = pty.openpty()
        try:
            # Delivered from a timer thread: tty.setraw() flushes pending
            # input, so a write issued before the call is discarded and the
            # read blocks forever. See _type_after_raw_mode.
            writer = _type_after_raw_mode(master, payload)
            # Replace sys.stdin with the slave-side file object for
            # the duration of the call. `_read_key_unix` reads via
            # `sys.stdin.fileno()`, then does the actual read with
            # `os.read(fd, ...)` — so this is the production code
            # path, not a mock.
            slave_file = _os.fdopen(slave, "rb", buffering=0)
            real_stdin = _sys.stdin
            try:
                _sys.stdin = slave_file
                got = cli_menu._read_key_unix()
            finally:
                _sys.stdin = real_stdin
                writer.join(timeout=1)
            assert got == expected, (
                f"payload={payload!r}: expected {expected!r}, got {got!r}"
            )
        finally:
            try:
                _os.close(master)
            except OSError:
                pass


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="pty test is Unix-only"
)
def test_read_key_unix_lone_esc_returns_esc():
    """A solo Esc keypress (no follow-up bytes) must still return
    'esc' so q/Esc cancel works. The 0.1s select timeout in
    `_read_key_unix` handles this."""
    import os as _os
    import pty
    import sys as _sys
    from backend import cli_menu

    master, slave = _os.openpty() if hasattr(_os, "openpty") else pty.openpty()
    try:
        _os.write(master, b"\x1b")  # only ESC, no [X follow-up
        slave_file = _os.fdopen(slave, "rb", buffering=0)
        real_stdin = _sys.stdin
        try:
            _sys.stdin = slave_file
            got = cli_menu._read_key_unix()
        finally:
            _sys.stdin = real_stdin
            writer.join(timeout=1)
        assert got == "esc"
    finally:
        try:
            _os.close(master)
        except OSError:
            pass
