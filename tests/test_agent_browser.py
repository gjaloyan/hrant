"""Tests for the `agent_browser` tool wrapper around the Vercel
Labs `agent-browser` Rust CLI.

Integration shape: deep-research browser the LLM picks WHEN
`fetch_url` falls short (JS-rendered, SPA, login-walled). For
plain HTML pages the cheap `fetch_url` stays default.

Pinned:
  - empty command refused with structured error, exit_code=-1
  - binary missing on PATH → `binary_missing=True` + install hint
  - successful run shells out `agent-browser <cmd> --json`,
    captures stdout/stderr, sets ok per exit code
  - `--json` is auto-appended if not already in the command
  - timeout clamped to [1, MAX_TIMEOUT_SEC]
  - output truncation marker on >MAX_OUTPUT_CHARS stdout
  - handler refuses non-owner callers (owner-only gate)
  - tool registered in the global registry as `agent_browser`
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from backend.tools import agent_browser as ab


# ─── _validate_command + binary-missing path ──────────────────────


def test_empty_command_refused():
    res = ab.run_agent_browser("")
    assert res.ok is False
    assert res.exit_code == -1
    assert "empty" in res.error.lower()


def test_whitespace_only_refused():
    res = ab.run_agent_browser("   \t\n  ")
    assert res.ok is False
    assert "empty" in res.error.lower()


def test_binary_missing_returns_structured_hint():
    with patch.object(ab, "_resolve_binary", return_value=None):
        res = ab.run_agent_browser("navigate https://example.com")
    assert res.ok is False
    assert res.binary_missing is True
    assert "agent-browser" in res.error
    assert "install" in res.error.lower()
    assert "npm install -g" in res.error


# ─── command construction + --json auto-append ────────────────────


def test_auto_appends_json_flag():
    """The wrapper should add `--json` if the LLM forgot, so
    stdout is always structured."""
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = b'{"ok":true}'
        stderr = b""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    with patch.object(ab, "_resolve_binary", return_value="/usr/bin/agent-browser"), \
         patch.object(ab.subprocess, "run", side_effect=_fake_run):
        ab.run_agent_browser("navigate https://example.com")

    assert "--json" in captured["cmd"]


def test_preserves_existing_json_flag():
    """If the LLM already added `--json`, don't double-add it."""
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = b"{}"
        stderr = b""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    with patch.object(ab, "_resolve_binary", return_value="/usr/bin/agent-browser"), \
         patch.object(ab.subprocess, "run", side_effect=_fake_run):
        ab.run_agent_browser("navigate https://x.com --json")

    # Should appear exactly once.
    assert captured["cmd"].count("--json") == 1


def test_prefixes_binary_path():
    """argv[0] is the resolved binary path, not a bare `agent-browser` —
    important when the binary lives outside PATH (e.g. the npm global bin
    dir, which is exactly where it sits on prod).

    Updated 2026-08-10: the wrapper stopped composing a shell string and now
    passes an argv list, because /bin/sh was eating `&` in query URLs and
    parentheses in `eval` JS. The invariant is unchanged."""
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = b""
        stderr = b""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    with patch.object(ab, "_resolve_binary", return_value="/opt/agent-browser/bin/agent-browser"), \
         patch.object(ab.subprocess, "run", side_effect=_fake_run):
        ab.run_agent_browser("open https://x.com")

    assert captured["cmd"][0] == "/opt/agent-browser/bin/agent-browser"
    assert captured["cmd"][1] == "open"


# ─── successful + failed exec paths ───────────────────────────────


def test_successful_exec_returns_ok():
    class _FakeProc:
        returncode = 0
        stdout = b'{"action":"navigate","status":"loaded"}'
        stderr = b""

    with patch.object(ab, "_resolve_binary", return_value="/usr/bin/agent-browser"), \
         patch.object(ab.subprocess, "run", return_value=_FakeProc()):
        res = ab.run_agent_browser("navigate https://example.com")

    assert res.ok is True
    assert res.exit_code == 0
    assert "loaded" in res.stdout
    assert res.binary_missing is False


def test_failed_exec_reports_exit_code():
    class _FakeProc:
        returncode = 1
        stdout = b""
        stderr = b"error: page failed to load"

    with patch.object(ab, "_resolve_binary", return_value="/usr/bin/agent-browser"), \
         patch.object(ab.subprocess, "run", return_value=_FakeProc()):
        res = ab.run_agent_browser("navigate https://broken.example")

    assert res.ok is False
    assert res.exit_code == 1
    assert "exit code 1" in res.error
    assert "failed to load" in res.stderr


def test_timeout_yields_structured_error():
    import subprocess as _sp
    with patch.object(ab, "_resolve_binary", return_value="/usr/bin/agent-browser"), \
         patch.object(ab.subprocess, "run",
                      side_effect=_sp.TimeoutExpired(cmd="x", timeout=5)):
        res = ab.run_agent_browser("navigate https://slow.example",
                                   timeout_seconds=5)

    assert res.ok is False
    assert res.exit_code == -1
    assert "timed out" in res.error.lower()
    assert res.binary_missing is False


def test_timeout_clamped_to_max():
    """Requested timeout > MAX_TIMEOUT_SEC is clamped, not respected
    verbatim. Prevents the LLM from setting a 1-hour cap that ties
    up the agent."""
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = b""
        stderr = b""

    def _fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return _FakeProc()

    with patch.object(ab, "_resolve_binary", return_value="/usr/bin/agent-browser"), \
         patch.object(ab.subprocess, "run", side_effect=_fake_run):
        ab.run_agent_browser("navigate https://x.com",
                             timeout_seconds=10_000)

    assert captured["timeout"] == ab.MAX_TIMEOUT_SEC


# ─── output truncation ─────────────────────────────────────────────


def test_output_truncation_marker():
    big = b"x" * (40 * 1024)

    class _FakeProc:
        returncode = 0
        stdout = big
        stderr = b""

    with patch.object(ab, "_resolve_binary", return_value="/usr/bin/agent-browser"), \
         patch.object(ab.subprocess, "run", return_value=_FakeProc()):
        res = ab.run_agent_browser("screenshot https://x.com")

    assert res.truncated is True
    assert "truncated" in res.stdout.lower() or "…" in res.stdout


# ─── owner-only gate via the builtin handler ──────────────────────


def test_handler_owner_only(monkeypatch):
    """The handler refuses non-owner callers before touching
    subprocess."""
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)
    out = builtin_tools._agent_browser_handler("navigate https://x.com")
    data = json.loads(out)
    assert data["ok"] is False
    assert "owner-only" in data["error"]


def test_handler_happy_path_owner(monkeypatch):
    from backend import builtin_tools, roles

    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)

    class _FakeProc:
        returncode = 0
        stdout = b'{"ok":true,"url":"https://example.com"}'
        stderr = b""

    monkeypatch.setattr(ab, "_resolve_binary",
                         lambda: "/usr/bin/agent-browser")
    monkeypatch.setattr(ab.subprocess, "run",
                         lambda *a, **kw: _FakeProc())

    out = builtin_tools._agent_browser_handler("navigate https://example.com")
    data = json.loads(out)
    assert data["ok"] is True
    assert data["exit_code"] == 0


# ─── registry exposure ────────────────────────────────────────────


def test_agent_browser_tool_is_registered():
    from backend.tool_registry import get_registry
    names = {t["name"] for t in get_registry().to_anthropic_list()}
    assert "agent_browser" in names
