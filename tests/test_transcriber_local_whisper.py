"""Local-whisper backend — no-auth FastAPI Whisper wrapper.

The user runs Whisper Medium on a private network behind a FastAPI
shim that exposes:
  GET  /health                    → liveness JSON ({"status":"ok",...})
  POST /v1/audio/transcriptions   → multipart upload, returns {"text",...}

The existing `whisper_cpp` backend uses /inference (different path) and
the existing `openai_whisper` backend requires an api key via providers.py
— neither fits a no-auth local server. This module pins the new backend
selection + transport against mocked HTTP so a future refactor can't
silently regress the live integration.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from backend import transcriber as tx_mod
from backend.transcriber import (
    DEFAULT_LOCAL_WHISPER_MODEL,
    Transcriber,
)


def _ok_health():
    """Minimal /health response that mirrors the user's server."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"status": "ok", "model": "medium"}
    return m


def _ok_transcribe(text: str = "hello world"):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.status_code = 200
    m.json.return_value = {"text": text, "language": "en"}
    return m


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    """Don't let an inherited LOCAL_WHISPER_URL contaminate tests."""
    monkeypatch.delenv("LOCAL_WHISPER_URL", raising=False)
    monkeypatch.delenv("WHISPER_CPP_URL", raising=False)
    monkeypatch.delenv("AGI_TRANSCRIBER_BACKEND", raising=False)


# --- _try_local_whisper ---------------------------------------------------


def test_local_whisper_picked_when_url_configured_and_health_ok(monkeypatch):
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    assert t._backend == "local_whisper"
    assert t._local_whisper_base == "http://10.0.0.1:8016"
    assert t._model == DEFAULT_LOCAL_WHISPER_MODEL


def test_local_whisper_url_trailing_slash_stripped(monkeypatch):
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016/"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    assert t._local_whisper_base == "http://10.0.0.1:8016"


def test_local_whisper_uses_env_var_when_config_empty(monkeypatch):
    monkeypatch.setattr(tx_mod, "load_config", lambda: {})
    monkeypatch.setenv("LOCAL_WHISPER_URL", "http://10.0.0.2:8016")
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    assert t._backend == "local_whisper"
    assert t._local_whisper_base == "http://10.0.0.2:8016"


def test_local_whisper_skipped_when_no_url(monkeypatch):
    monkeypatch.setattr(tx_mod, "load_config", lambda: {})
    t = Transcriber()
    # Stub openai_whisper too so chain doesn't try the real check.
    monkeypatch.setattr(t, "_try_openai_whisper", lambda cfg: False)
    with patch("backend.transcriber.httpx.get") as get:
        t._pick_backend()
    # No URL configured → local_whisper not even probed (no /health call).
    get.assert_not_called()
    assert t._backend == "disabled"


def test_local_whisper_health_non_200_falls_through(monkeypatch):
    """If /health returns 503/500, this backend is unavailable —
    chain must fall through to whisper_cpp / openai_whisper rather
    than wedge on a flapping server."""
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    monkeypatch.setattr(t, "_try_whisper_cpp", lambda cfg: False)
    monkeypatch.setattr(t, "_try_openai_whisper", lambda cfg: False)
    bad = MagicMock()
    bad.status_code = 503
    with patch("backend.transcriber.httpx.get", return_value=bad):
        t._pick_backend()
    assert t._backend == "disabled"
    assert t._last_error and "503" in t._last_error


def test_local_whisper_network_error_records_last_error(monkeypatch):
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    monkeypatch.setattr(t, "_try_whisper_cpp", lambda cfg: False)
    monkeypatch.setattr(t, "_try_openai_whisper", lambda cfg: False)
    with patch("backend.transcriber.httpx.get", side_effect=Exception("boom")):
        t._pick_backend()
    assert t._backend == "disabled"
    assert "probe failed" in (t._last_error or "")


def test_local_whisper_custom_model_from_config(monkeypatch):
    cfg = {
        "local_whisper": {
            "url": "http://10.0.0.1:8016",
            "model": "whisper-large-v3",
        }
    }
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    assert t._model == "whisper-large-v3"


# --- backend chain priority ----------------------------------------------


def test_local_whisper_wins_over_whisper_cpp_in_auto_mode(monkeypatch):
    """Both URLs configured → local_whisper takes priority. The user's
    local server is the explicit, declared first choice."""
    cfg = {
        "backend": "auto",
        "local_whisper": {"url": "http://10.0.0.1:8016"},
        "whisper_cpp": {"url": "http://10.0.0.99:8080"},
    }
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    assert t._backend == "local_whisper"
    # Did NOT switch over to whisper_cpp.
    assert t._whisper_cpp_base is None


def test_forced_backend_skips_other_candidates(monkeypatch):
    """`backend: "openai_whisper"` in config must short-circuit the
    auto chain — even if local_whisper is configured + reachable."""
    cfg = {
        "backend": "openai_whisper",
        "local_whisper": {"url": "http://10.0.0.1:8016"},
    }
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    # Stub openai_whisper to fail so backend ends up "disabled" — the
    # point of this test is that local_whisper was NEVER tried.
    monkeypatch.setattr(t, "_try_openai_whisper", lambda cfg: False)
    with patch("backend.transcriber.httpx.get") as get:
        t._pick_backend()
    get.assert_not_called()
    assert t._backend == "disabled"


# --- _tx_local_whisper transport -----------------------------------------


def test_tx_local_whisper_posts_to_v1_audio_transcriptions(monkeypatch):
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["files"] = kw.get("files")
        captured["data"] = kw.get("data")
        captured["headers"] = kw.get("headers")
        return _ok_transcribe("hello world")

    with patch("backend.transcriber.httpx.post", side_effect=fake_post):
        out = t.transcribe(b"BYTES", mime_type="audio/ogg", filename="x.ogg")

    assert out == "hello world"
    assert captured["url"] == "http://10.0.0.1:8016/v1/audio/transcriptions"
    # `model` form field present, default is whisper-medium.
    assert captured["data"]["model"] == DEFAULT_LOCAL_WHISPER_MODEL
    # No bearer token — local server runs on a private network.
    assert not captured.get("headers")
    # File MIME passed through.
    fname, body, mime = captured["files"]["file"]
    assert fname == "x.ogg" and body == b"BYTES" and mime == "audio/ogg"


def test_tx_local_whisper_passes_language_when_provided(monkeypatch):
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    captured = {}

    def fake_post(url, **kw):
        captured.update(kw)
        return _ok_transcribe("привет")

    with patch("backend.transcriber.httpx.post", side_effect=fake_post):
        out = t.transcribe(b"x", mime_type="audio/wav", filename="a.wav", language="ru")
    assert out == "привет"
    assert captured["data"]["language"] == "ru"


def test_tx_local_whisper_returns_none_for_empty_text(monkeypatch):
    """Whisper returns text="" for silence — keep behaviour
    consistent with other backends: empty string → None so the caller
    knows there's nothing usable."""
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    with patch("backend.transcriber.httpx.post", return_value=_ok_transcribe("")):
        out = t.transcribe(b"x", filename="silence.wav")
    assert out is None


def test_tx_local_whisper_strips_whitespace(monkeypatch):
    """Server pads with whitespace sometimes — caller shouldn't have
    to strip again."""
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    with patch("backend.transcriber.httpx.post", return_value=_ok_transcribe("   hi.   ")):
        out = t.transcribe(b"x", filename="a.wav")
    assert out == "hi."


def test_tx_local_whisper_records_error_on_failure(monkeypatch):
    """A 5xx response → transcribe() returns None and last_error is
    populated so the dev panel / Settings UI can show it."""
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    bad = MagicMock()
    bad.status_code = 500
    bad.raise_for_status.side_effect = Exception("Internal Server Error")
    with patch("backend.transcriber.httpx.post", return_value=bad):
        out = t.transcribe(b"x", filename="a.wav")
    assert out is None
    assert t._last_error and "local_whisper" in t._last_error


# --- reset() drops the per-backend state ---------------------------------


def test_reset_clears_local_whisper_base(monkeypatch):
    cfg = {"local_whisper": {"url": "http://10.0.0.1:8016"}}
    monkeypatch.setattr(tx_mod, "load_config", lambda: cfg)
    t = Transcriber()
    with patch("backend.transcriber.httpx.get", return_value=_ok_health()):
        t._pick_backend()
    assert t._local_whisper_base == "http://10.0.0.1:8016"
    t.reset()
    assert t._local_whisper_base is None
    assert t._backend is None


# --- saved on-disk config points at user's server ------------------------


def test_persisted_config_targets_user_server():
    """The transcriber_config.json under knowledge/ should already
    name the user's server — written when this skill was wired up.
    Catches accidental regression of the on-disk pointer."""
    from backend.transcriber import load_config
    cfg = load_config()
    if not cfg:
        pytest.skip("no transcriber_config.json on disk in this checkout")
    lw = cfg.get("local_whisper") or {}
    assert lw.get("url"), "local_whisper.url must be set in config"
    # Whisper-medium is what's running on the server today.
    assert lw.get("model", "whisper-medium") == "whisper-medium"
