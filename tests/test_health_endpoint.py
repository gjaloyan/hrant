"""Tests for backend.api.health — aggregated /api/health endpoint.

Each component check is patched in isolation so we don't depend on
real Whisper / Piper / Ollama instances. The aggregate rollup is
tested directly against synthetic component dicts.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api import health as health_mod


def test_aggregate_all_ok_is_ok():
    components = {
        "a": {"status": "ok", "detail": ""},
        "b": {"status": "ok", "detail": ""},
    }
    assert health_mod._aggregate(components) == "ok"


def test_aggregate_any_degraded_pulls_to_degraded():
    components = {
        "a": {"status": "ok"},
        "b": {"status": "degraded", "detail": "x"},
    }
    assert health_mod._aggregate(components) == "degraded"


def test_aggregate_any_down_pulls_to_down():
    components = {
        "a": {"status": "ok"},
        "b": {"status": "degraded"},
        "c": {"status": "down", "detail": "x"},
    }
    assert health_mod._aggregate(components) == "down"


def test_aggregate_not_configured_does_not_drag_down():
    """A deployment without Piper isn't unhealthy — it's text-only.
    not_configured components must NOT count against the aggregate."""
    components = {
        "a": {"status": "ok"},
        "b": {"status": "not_configured", "detail": "no URL"},
    }
    assert health_mod._aggregate(components) == "ok"


def test_check_agent_core_ok_when_importable():
    r = health_mod._check_agent_core()
    assert r["status"] == "ok"


def test_check_stt_not_configured_when_no_url(monkeypatch):
    monkeypatch.setattr("backend.transcriber.load_config", lambda: {})
    r = health_mod._check_stt()
    assert r["status"] == "not_configured"


def test_check_stt_ok_when_probe_succeeds(monkeypatch):
    monkeypatch.setattr(
        "backend.transcriber.load_config",
        lambda: {"local_whisper": {"url": "http://1.2.3.4:8016"}},
    )
    monkeypatch.setattr(
        "backend.discovery.probe_service",
        lambda *a, **kw: {"ok": True, "url": "http://1.2.3.4:8016"},
    )
    r = health_mod._check_stt()
    assert r["status"] == "ok"


def test_check_stt_down_when_probe_fails(monkeypatch):
    monkeypatch.setattr(
        "backend.transcriber.load_config",
        lambda: {"local_whisper": {"url": "http://1.2.3.4:8016"}},
    )
    monkeypatch.setattr(
        "backend.discovery.probe_service",
        lambda *a, **kw: {"ok": False, "reason": "connection refused"},
    )
    r = health_mod._check_stt()
    assert r["status"] == "down"
    assert "connection refused" in r["detail"]


def test_check_tts_not_configured_when_no_url(monkeypatch):
    monkeypatch.setattr("backend.tts.load_config", lambda: {})
    r = health_mod._check_tts()
    assert r["status"] == "not_configured"


def test_check_ffmpeg_ok_when_present(monkeypatch):
    monkeypatch.setattr("backend.tts._ffmpeg_available", lambda: True)
    r = health_mod._check_ffmpeg()
    assert r["status"] == "ok"


def test_check_ffmpeg_degraded_when_missing(monkeypatch):
    """ffmpeg missing means WAV fallback works → degraded, not down."""
    monkeypatch.setattr("backend.tts._ffmpeg_available", lambda: False)
    r = health_mod._check_ffmpeg()
    assert r["status"] == "degraded"
    assert "WAV" in r["detail"]


def test_check_telegram_not_configured_when_no_telegram_channels(monkeypatch):
    monkeypatch.setattr("backend.channels.get_channels", lambda: [])
    r = health_mod._check_telegram()
    assert r["status"] == "not_configured"


def test_check_telegram_ok_when_at_least_one_running(monkeypatch):
    monkeypatch.setattr(
        "backend.channels.get_channels",
        lambda: [{"id": "t1", "type": "telegram"}],
    )
    from backend.channels import CHANNELS
    monkeypatch.setattr(CHANNELS, "status_all", lambda: {"t1": "running"})
    r = health_mod._check_telegram()
    assert r["status"] == "ok"
    assert "1/1" in r["detail"]


def test_check_telegram_degraded_when_configured_but_stopped(monkeypatch):
    monkeypatch.setattr(
        "backend.channels.get_channels",
        lambda: [{"id": "t1", "type": "telegram"}],
    )
    from backend.channels import CHANNELS
    monkeypatch.setattr(CHANNELS, "status_all", lambda: {"t1": "stopped"})
    r = health_mod._check_telegram()
    assert r["status"] == "degraded"


def test_check_workspace_ok_when_writable(tmp_path, monkeypatch):
    """Build a synthetic workspace via tmp_path so we don't touch the
    real ./workspace folder (and so the test cleans up after itself)."""
    class _WS:
        root = tmp_path
    monkeypatch.setattr("backend.workspace.get_workspace", lambda: _WS())
    r = health_mod._check_workspace()
    assert r["status"] == "ok"


def test_check_workspace_down_when_root_missing(tmp_path, monkeypatch):
    class _WS:
        root = tmp_path / "does-not-exist"
    monkeypatch.setattr("backend.workspace.get_workspace", lambda: _WS())
    r = health_mod._check_workspace()
    assert r["status"] == "down"


# --- end-to-end via TestClient -----------------------------------------


def _build_app():
    """Tiny FastAPI app with only the health router mounted, so the
    test doesn't pay for the full lifespan startup of main.app
    (autonomic scheduler, channel auto-start, etc.)."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(health_mod.router)
    return app


def test_health_endpoint_shape(monkeypatch):
    # Patch every component check to a stable ok-ish state so the
    # response shape is deterministic.
    monkeypatch.setattr(health_mod, "_check_agent_core", lambda: {"status": "ok", "detail": ""})
    monkeypatch.setattr(health_mod, "_check_model", lambda: {"status": "ok", "detail": "x"})
    monkeypatch.setattr(health_mod, "_check_stt", lambda: {"status": "not_configured", "detail": ""})
    monkeypatch.setattr(health_mod, "_check_tts", lambda: {"status": "not_configured", "detail": ""})
    monkeypatch.setattr(health_mod, "_check_ffmpeg", lambda: {"status": "ok", "detail": ""})
    monkeypatch.setattr(health_mod, "_check_telegram", lambda: {"status": "not_configured", "detail": ""})
    monkeypatch.setattr(health_mod, "_check_workspace", lambda: {"status": "ok", "detail": ""})
    monkeypatch.setattr(health_mod, "_check_autonomic", lambda: {"status": "ok", "detail": "last tick 5s ago"})

    client = TestClient(_build_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"  # nothing degraded/down
    assert "version" in body
    assert "uptime_seconds" in body
    assert "components" in body
    assert set(body["components"].keys()) == {
        "agent_core", "model", "stt", "tts", "ffmpeg",
        "telegram", "workspace", "autonomic",
    }


def test_health_endpoint_aggregate_down_when_model_down(monkeypatch):
    monkeypatch.setattr(health_mod, "_check_agent_core", lambda: {"status": "ok"})
    monkeypatch.setattr(health_mod, "_check_model", lambda: {"status": "down", "detail": "no provider"})
    monkeypatch.setattr(health_mod, "_check_stt", lambda: {"status": "not_configured"})
    monkeypatch.setattr(health_mod, "_check_tts", lambda: {"status": "not_configured"})
    monkeypatch.setattr(health_mod, "_check_ffmpeg", lambda: {"status": "ok"})
    monkeypatch.setattr(health_mod, "_check_telegram", lambda: {"status": "not_configured"})
    monkeypatch.setattr(health_mod, "_check_workspace", lambda: {"status": "ok"})
    monkeypatch.setattr(health_mod, "_check_autonomic", lambda: {"status": "ok"})

    client = TestClient(_build_app())
    r = client.get("/api/health")
    assert r.json()["status"] == "down"


def test_check_autonomic_ok_when_recent_tick(tmp_path, monkeypatch):
    """Recent tick within 2x interval → ok."""
    from datetime import datetime, timezone, timedelta
    tl = tmp_path / "tick_log.jsonl"
    recent = (datetime.now(timezone.utc) - timedelta(seconds=10)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ",
    )
    tl.write_text(f'{{"ts":"{recent}","fired":1}}\n', encoding="utf-8")
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tl))
    monkeypatch.setattr("backend.autonomic.settings.resolve_tick_interval", lambda: 30.0)
    r = health_mod._check_autonomic()
    assert r["status"] == "ok"
    assert "last tick" in r["detail"]


def test_check_autonomic_down_when_stale(tmp_path, monkeypatch):
    """Tick older than 10x interval → down."""
    from datetime import datetime, timezone, timedelta
    tl = tmp_path / "tick_log.jsonl"
    stale = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ",
    )
    tl.write_text(f'{{"ts":"{stale}","fired":1}}\n', encoding="utf-8")
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tl))
    monkeypatch.setattr("backend.autonomic.settings.resolve_tick_interval", lambda: 30.0)
    r = health_mod._check_autonomic()
    assert r["status"] == "down"


def test_check_autonomic_down_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "missing.jsonl"))
    r = health_mod._check_autonomic()
    assert r["status"] == "down"
