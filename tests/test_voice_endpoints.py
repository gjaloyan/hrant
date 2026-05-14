"""Tests for backend.api.voice (TTS config + /api/discover) and the
extended transcribe config endpoints in backend.api.attachments.

These endpoints are the WebUI Voice tab's contract; if any of the
fields change shape without the front-end being updated, the tab
silently breaks. The tests pin the shape so a backend refactor
surfaces immediately."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app():
    """Tiny FastAPI app with only the voice + attachments routers
    mounted. Skips the full main.app lifespan (autonomic loop,
    channel auto-start, etc.) so the test stays fast and isolated."""
    from backend.api import attachments as attachments_api
    from backend.api import voice as voice_api

    app = FastAPI()
    app.include_router(voice_api.router)
    app.include_router(attachments_api.router)
    return app


# --- TTS config ---------------------------------------------------------


def test_tts_config_get_returns_current_config(monkeypatch):
    """The endpoint binds `load_tts_config` at module-import time via
    `from ..tts import load_config as load_tts_config`, so a patch on
    `backend.tts.load_config` only takes effect if the voice module
    hasn't been imported yet. To stay deterministic regardless of
    test order, patch the voice module's bound name directly."""
    fake = {"backend": "local_piper", "local_piper": {"url": "http://x:8017"}}
    monkeypatch.setattr("backend.tts.load_config", lambda: dict(fake))
    monkeypatch.setattr("backend.api.voice.load_tts_config", lambda: dict(fake))
    client = TestClient(_build_app())
    r = client.get("/api/tts/config")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "local_piper"
    assert body["local_piper"]["url"] == "http://x:8017"


def test_tts_config_put_merges_partial_update(monkeypatch):
    """PUT with only local_piper.voice must keep the existing url."""
    state: dict = {"local_piper": {"url": "http://x:8017", "voice": "old"}}

    monkeypatch.setattr("backend.tts.load_config", lambda: dict(state))

    def fake_save(cfg):
        state.clear()
        state.update(cfg)
        return cfg

    monkeypatch.setattr("backend.tts.save_config", fake_save)
    # The voice module re-imports save_config at module load; patch
    # the rebound names as well.
    monkeypatch.setattr("backend.api.voice.load_tts_config", lambda: dict(state))
    monkeypatch.setattr("backend.api.voice.save_tts_config", fake_save)
    from backend.tts import SYNTHESIZER
    monkeypatch.setattr(SYNTHESIZER, "reset", lambda: None)
    monkeypatch.setattr(SYNTHESIZER, "status", lambda: {"backend": "local_piper"})

    client = TestClient(_build_app())
    r = client.put("/api/tts/config", json={"local_piper": {"voice": "new_voice"}})
    assert r.status_code == 200
    # Shallow merge: voice replaced, url preserved.
    assert state["local_piper"]["voice"] == "new_voice"
    assert state["local_piper"]["url"] == "http://x:8017"


def test_tts_config_put_replaces_backend_choice(monkeypatch):
    state: dict = {"backend": "auto"}
    monkeypatch.setattr("backend.tts.load_config", lambda: dict(state))
    monkeypatch.setattr("backend.api.voice.load_tts_config", lambda: dict(state))

    def fake_save(cfg):
        state.clear()
        state.update(cfg)
        return cfg

    monkeypatch.setattr("backend.tts.save_config", fake_save)
    monkeypatch.setattr("backend.api.voice.save_tts_config", fake_save)
    from backend.tts import SYNTHESIZER
    monkeypatch.setattr(SYNTHESIZER, "reset", lambda: None)
    monkeypatch.setattr(SYNTHESIZER, "status", lambda: {"backend": "disabled"})

    client = TestClient(_build_app())
    r = client.put("/api/tts/config", json={"backend": "disabled"})
    assert r.status_code == 200
    assert state["backend"] == "disabled"


def test_tts_edge_tts_probe_passes_with_package(monkeypatch):
    """When `edge_tts` package is importable, the probe should mark
    the backend active even with no network call (probe is just an
    import check + voice defaults)."""
    import sys
    # Pretend edge_tts is installed.
    fake_mod = type(sys)("edge_tts")
    monkeypatch.setitem(sys.modules, "edge_tts", fake_mod)
    from backend.tts import Synthesizer
    s = Synthesizer()
    monkeypatch.setattr("backend.tts.load_config", lambda: {"backend": "edge_tts"})
    s._pick_backend()
    assert s._backend == "edge_tts"
    assert s._voice  # default present
    assert s._voice_ru


def test_tts_edge_tts_probe_fails_without_package(monkeypatch):
    """Without the edge_tts package the probe should not pick this
    backend (sets last_error, falls through). The next backend in
    the auto chain gets a turn."""
    import sys
    monkeypatch.delitem(sys.modules, "edge_tts", raising=False)
    # Force `import edge_tts` to fail.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kw):
        if name == "edge_tts":
            raise ImportError("not installed")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from backend.tts import Synthesizer
    s = Synthesizer()
    monkeypatch.setattr(
        "backend.tts.load_config", lambda: {"backend": "edge_tts"},
    )
    s._pick_backend()
    assert s._backend == "disabled"
    assert "not installed" in (s._last_error or "")


def test_tts_status_endpoint(monkeypatch):
    from backend.tts import SYNTHESIZER
    monkeypatch.setattr(
        SYNTHESIZER, "status",
        lambda: {"backend": "local_piper", "voice": "en_US-lessac-medium"},
    )
    client = TestClient(_build_app())
    r = client.get("/api/tts/status")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "local_piper"


def test_tts_reset_endpoint(monkeypatch):
    """POST /api/tts/reset must call SYNTHESIZER.reset() and return
    the post-reset status."""
    from backend.tts import SYNTHESIZER
    called = {"reset": 0}

    def fake_reset():
        called["reset"] += 1

    monkeypatch.setattr(SYNTHESIZER, "reset", fake_reset)
    monkeypatch.setattr(SYNTHESIZER, "status", lambda: {"backend": "auto"})

    client = TestClient(_build_app())
    r = client.post("/api/tts/reset")
    assert r.status_code == 200
    assert called["reset"] == 1


# --- Transcribe (Whisper) config — local_whisper field regression -----


def test_transcribe_config_accepts_local_whisper_field(monkeypatch):
    """Regression: TranscriberConfigUpdate used to omit local_whisper,
    so the UI couldn't write the URL of the FastAPI Whisper wrapper.
    Setting it now must be a 200 OK and persist."""
    state: dict = {}

    monkeypatch.setattr("backend.transcriber.load_config", lambda: dict(state))
    monkeypatch.setattr("backend.api.attachments.load_transcriber_config", lambda: dict(state))

    def fake_save(cfg):
        state.clear()
        state.update(cfg)
        return cfg

    monkeypatch.setattr("backend.transcriber.save_config", fake_save)
    monkeypatch.setattr("backend.api.attachments.save_transcriber_config", fake_save)
    from backend.transcriber import TRANSCRIBER
    monkeypatch.setattr(TRANSCRIBER, "reset", lambda: None)
    monkeypatch.setattr(TRANSCRIBER, "status", lambda: {"backend": "local_whisper"})

    client = TestClient(_build_app())
    r = client.put(
        "/api/transcribe/config",
        json={"local_whisper": {"url": "http://1.2.3.4:8016", "model": "medium"}},
    )
    assert r.status_code == 200, r.text
    assert state["local_whisper"]["url"] == "http://1.2.3.4:8016"
    assert state["local_whisper"]["model"] == "medium"


def test_transcribe_config_get_endpoint(monkeypatch):
    """The UI needs GET to populate the form. Was missing before."""
    monkeypatch.setattr(
        "backend.api.attachments.load_transcriber_config",
        lambda: {"backend": "auto", "local_whisper": {"url": "http://x:8016"}},
    )
    client = TestClient(_build_app())
    r = client.get("/api/transcribe/config")
    assert r.status_code == 200
    assert r.json()["local_whisper"]["url"] == "http://x:8016"


# --- /api/discover ------------------------------------------------------


def test_discover_returns_error_when_no_host(monkeypatch):
    monkeypatch.delenv("TAILSCALE_HOST", raising=False)
    monkeypatch.setattr(
        "backend.discovery.discover_services",
        lambda host=None, services=None, **kw: {"_error": "no host provided"},
    )
    client = TestClient(_build_app())
    r = client.get("/api/discover")
    assert r.status_code == 200
    body = r.json()
    assert "error" in body


def test_discover_returns_found_dict(monkeypatch):
    fake = {
        "whisper": {"ok": True, "url": "http://1.2.3.4:8016"},
        "piper": {"ok": False, "reason": "down"},
    }
    monkeypatch.setattr(
        "backend.discovery.discover_services",
        lambda host=None, services=None, **kw: fake,
    )
    client = TestClient(_build_app())
    r = client.get("/api/discover?host=1.2.3.4")
    assert r.status_code == 200
    body = r.json()
    assert body["host"] == "1.2.3.4"
    assert body["found"]["whisper"]["ok"] is True
    assert "applied" not in body  # no apply=true


def test_discover_apply_writes_and_resets(monkeypatch):
    """With apply=true the endpoint must call apply_discovery AND
    reset both Transcriber and Synthesizer so a subsequent /status
    reflects the new config."""
    fake = {"whisper": {"ok": True, "url": "http://x:8016"}}
    monkeypatch.setattr(
        "backend.discovery.discover_services",
        lambda host=None, services=None, **kw: fake,
    )
    monkeypatch.setattr(
        "backend.discovery.apply_discovery",
        lambda found, **kw: {"whisper": "applied", "piper": "skipped", "ollama": "skipped"},
    )
    from backend.transcriber import TRANSCRIBER
    from backend.tts import SYNTHESIZER
    seen = {"t_reset": 0, "s_reset": 0}
    monkeypatch.setattr(TRANSCRIBER, "reset", lambda: seen.update(t_reset=seen["t_reset"] + 1))
    monkeypatch.setattr(SYNTHESIZER, "reset", lambda: seen.update(s_reset=seen["s_reset"] + 1))

    client = TestClient(_build_app())
    r = client.get("/api/discover?host=x&apply=true")
    assert r.status_code == 200
    body = r.json()
    assert body["applied"]["whisper"] == "applied"
    assert seen["t_reset"] == 1
    assert seen["s_reset"] == 1


def test_discover_filters_by_services_query(monkeypatch):
    """services=whisper,piper passes a parsed list through."""
    seen: dict = {}

    def fake_discover(host=None, services=None, **kw):
        seen["services"] = services
        return {"whisper": {"ok": False, "reason": "x"}}

    monkeypatch.setattr("backend.discovery.discover_services", fake_discover)
    client = TestClient(_build_app())
    r = client.get("/api/discover?host=x&services=whisper,piper")
    assert r.status_code == 200
    assert seen["services"] == ["whisper", "piper"]
