"""Tests for M4 — `hrant status` channel-state fetched via HTTP API.

Pinned behaviour:
  - `_fetch_runtime_channel_state` hits the daemon at
    `http://127.0.0.1:{HRANT_API_PORT or 3333}/api/channels` and parses
    `channels[].runtime_status` into `{channel_id: state}`.
  - On unreachable daemon (refused / timeout / non-200), returns a
    dict with `__error__` set; never raises.
  - Honors the `HRANT_API_PORT` env var override.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


def test_fetch_runtime_channel_state_parses_api_response(monkeypatch):
    from backend.cli import _fetch_runtime_channel_state
    import io

    fake_payload = {
        "channels": [
            {"id": "hrant", "type": "telegram",
             "runtime_status": "running", "enabled": True},
            {"id": "secondary", "type": "telegram",
             "runtime_status": "stopped", "enabled": False},
        ]
    }
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: False

    with patch("urllib.request.urlopen", return_value=fake_resp):
        out = _fetch_runtime_channel_state()
    assert out == {"hrant": "running", "secondary": "stopped"}


def test_fetch_runtime_channel_state_handles_connection_refused():
    from backend.cli import _fetch_runtime_channel_state
    import urllib.error
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("Connection refused")):
        out = _fetch_runtime_channel_state()
    assert "__error__" in out
    assert "not reachable" in out["__error__"].lower()


def test_fetch_runtime_channel_state_handles_timeout():
    from backend.cli import _fetch_runtime_channel_state
    with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
        out = _fetch_runtime_channel_state()
    assert "__error__" in out


def test_fetch_runtime_channel_state_handles_non_200():
    from backend.cli import _fetch_runtime_channel_state
    fake_resp = MagicMock()
    fake_resp.status = 503
    fake_resp.read.return_value = b"oops"
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: False
    with patch("urllib.request.urlopen", return_value=fake_resp):
        out = _fetch_runtime_channel_state()
    assert "__error__" in out
    assert "503" in out["__error__"]


def test_fetch_runtime_channel_state_honors_port_env_var(monkeypatch):
    from backend.cli import _fetch_runtime_channel_state

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url if hasattr(req, "full_url") else str(req)
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b'{"channels": []}'
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *a: False
        return resp

    monkeypatch.setenv("HRANT_API_PORT", "9999")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _fetch_runtime_channel_state()
    assert ":9999" in captured.get("url", ""), (
        f"expected port 9999 in URL, got {captured.get('url')!r}"
    )


def test_fetch_runtime_channel_state_skips_channel_without_id():
    """Malformed daemon response shouldn't crash the CLI."""
    from backend.cli import _fetch_runtime_channel_state
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = json.dumps({
        "channels": [
            {"runtime_status": "running"},  # no id
            {"id": "ok", "runtime_status": "running"},
        ]
    }).encode("utf-8")
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: False
    with patch("urllib.request.urlopen", return_value=fake_resp):
        out = _fetch_runtime_channel_state()
    assert out == {"ok": "running"}
