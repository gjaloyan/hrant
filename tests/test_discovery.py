"""Tests for backend.discovery — service probing + apply.

Covered:
  - probe_service success / non-200 / wrong-body / timeout / connection error
  - discover_services dispatches every known service against the host
  - discover_services falls back to TAILSCALE_HOST when host=None
  - discover_services returns _error sentinel when no host available
  - discover_services filters by `services=` subset
  - apply_discovery writes Whisper URL into transcriber config
  - apply_discovery writes Piper URL into tts config
  - apply_discovery skips entries without ok=True
  - apply_discovery always marks ollama as "skipped" (UI-managed)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend import discovery


# --- probe_service ------------------------------------------------------


def _fake_response(status: int, text: str) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


def test_probe_service_ok_when_marker_present():
    spec = discovery.KNOWN_SERVICES["whisper"]
    with patch.object(httpx, "get", return_value=_fake_response(200, '{"model":"base"}')):
        r = discovery.probe_service(spec, host="1.2.3.4")
    assert r["ok"] is True
    assert r["url"] == "http://1.2.3.4:8016"


def test_probe_service_fails_on_missing_marker():
    """The marker filters out random servers (e.g. an nginx default
    page) that happen to listen on the probed port."""
    spec = discovery.KNOWN_SERVICES["whisper"]
    with patch.object(httpx, "get", return_value=_fake_response(200, "<html>nope</html>")):
        r = discovery.probe_service(spec, host="1.2.3.4")
    assert r["ok"] is False
    assert "marker" in r["reason"]


def test_probe_service_fails_on_non_200():
    spec = discovery.KNOWN_SERVICES["whisper"]
    with patch.object(httpx, "get", return_value=_fake_response(404, "")):
        r = discovery.probe_service(spec, host="1.2.3.4")
    assert r["ok"] is False
    assert "HTTP 404" in r["reason"]


def test_probe_service_handles_timeout():
    spec = discovery.KNOWN_SERVICES["whisper"]
    with patch.object(httpx, "get", side_effect=httpx.TimeoutException("nope")):
        r = discovery.probe_service(spec, host="1.2.3.4", timeout=0.5)
    assert r["ok"] is False
    assert "timeout" in r["reason"]


def test_probe_service_handles_connection_error():
    spec = discovery.KNOWN_SERVICES["whisper"]
    with patch.object(httpx, "get", side_effect=ConnectionError("refused")):
        r = discovery.probe_service(spec, host="1.2.3.4")
    assert r["ok"] is False
    assert "connection failed" in r["reason"]


def test_probe_service_honors_explicit_port():
    spec = discovery.KNOWN_SERVICES["whisper"]
    with patch.object(httpx, "get", return_value=_fake_response(200, '{"model":"x"}')) as m:
        r = discovery.probe_service(spec, host="1.2.3.4", port=9999)
    assert r["url"] == "http://1.2.3.4:9999"
    # The HTTP probe URL also has to point at the override.
    called_url = m.call_args[0][0]
    assert called_url.startswith("http://1.2.3.4:9999")


# --- discover_services --------------------------------------------------


def test_discover_services_probes_all_known_when_no_filter():
    with patch.object(
        discovery, "probe_service",
        side_effect=lambda spec, host, **kw: {"ok": True, "url": f"http://{host}:{spec.default_port}"},
    ):
        out = discovery.discover_services(host="1.2.3.4")
    assert set(out.keys()) == {"whisper", "piper", "ollama"}
    assert out["whisper"]["url"] == "http://1.2.3.4:8016"


def test_discover_services_filters_by_subset():
    with patch.object(
        discovery, "probe_service",
        side_effect=lambda spec, host, **kw: {"ok": True, "url": f"http://{host}:{spec.default_port}"},
    ):
        out = discovery.discover_services(host="1.2.3.4", services=["whisper"])
    assert set(out.keys()) == {"whisper"}


def test_discover_services_returns_error_when_no_host(monkeypatch):
    monkeypatch.delenv("TAILSCALE_HOST", raising=False)
    out = discovery.discover_services()
    assert "_error" in out
    assert "TAILSCALE_HOST" in out["_error"]


def test_discover_services_uses_tailscale_env_when_host_none(monkeypatch):
    monkeypatch.setenv("TAILSCALE_HOST", "100.64.0.1")
    seen_hosts: list[str] = []

    def fake_probe(spec, host, **kw):
        seen_hosts.append(host)
        return {"ok": False, "url": f"http://{host}:{spec.default_port}", "reason": "x"}

    with patch.object(discovery, "probe_service", side_effect=fake_probe):
        discovery.discover_services()
    # Every known service must have been probed against the env-provided host.
    assert seen_hosts and all(h == "100.64.0.1" for h in seen_hosts)


def test_discover_services_unknown_name_yields_explicit_error():
    out = discovery.discover_services(host="1.2.3.4", services=["bogus"])
    assert out["bogus"]["ok"] is False
    assert "unknown" in out["bogus"]["reason"]


# --- apply_discovery ----------------------------------------------------


def test_apply_discovery_writes_whisper_url(tmp_path, monkeypatch):
    """Patch transcriber.load_config/save_config so the test doesn't
    touch the real knowledge/transcriber_config.json."""
    state: dict = {}

    def fake_load():
        return dict(state)

    def fake_save(cfg):
        state.clear()
        state.update(cfg)

    monkeypatch.setattr("backend.transcriber.load_config", fake_load)
    monkeypatch.setattr("backend.transcriber.save_config", fake_save)
    monkeypatch.setattr("backend.tts.load_config", lambda: {})
    monkeypatch.setattr("backend.tts.save_config", lambda c: None)

    found = {
        "whisper": {"ok": True, "url": "http://1.2.3.4:8016"},
        "piper": {"ok": False, "reason": "x"},
    }
    result = discovery.apply_discovery(found)
    assert result["whisper"] == "applied"
    assert state["local_whisper"]["url"] == "http://1.2.3.4:8016"


def test_apply_discovery_writes_piper_url(monkeypatch):
    piper_state: dict = {}

    monkeypatch.setattr("backend.transcriber.load_config", lambda: {})
    monkeypatch.setattr("backend.transcriber.save_config", lambda c: None)

    def fake_p_load():
        return dict(piper_state)

    def fake_p_save(cfg):
        piper_state.clear()
        piper_state.update(cfg)

    monkeypatch.setattr("backend.tts.load_config", fake_p_load)
    monkeypatch.setattr("backend.tts.save_config", fake_p_save)

    found = {
        "whisper": {"ok": False, "reason": "x"},
        "piper": {"ok": True, "url": "http://1.2.3.4:8017"},
    }
    result = discovery.apply_discovery(found)
    assert result["piper"] == "applied"
    assert piper_state["local_piper"]["url"] == "http://1.2.3.4:8017"


def test_apply_discovery_skips_when_not_ok(monkeypatch):
    monkeypatch.setattr("backend.transcriber.load_config", lambda: {})
    monkeypatch.setattr("backend.transcriber.save_config", lambda c: None)
    monkeypatch.setattr("backend.tts.load_config", lambda: {})
    monkeypatch.setattr("backend.tts.save_config", lambda c: None)
    found = {
        "whisper": {"ok": False, "reason": "down"},
        "piper": {"ok": False, "reason": "down"},
    }
    result = discovery.apply_discovery(found)
    assert "skipped" in result["whisper"]
    assert "skipped" in result["piper"]


def test_apply_discovery_never_auto_applies_ollama():
    """Even when ollama is up, we don't auto-write — the agent's
    model_b shape is richer than just a URL."""
    found = {"ollama": {"ok": True, "url": "http://x:11434"}}
    result = discovery.apply_discovery(found)
    assert "skipped" in result["ollama"]
