"""Smoke tests for backend.tts — TTS configuration + voice picker.

Audit flagged 493 LOC untested. Real TTS calls hit external
services (Edge TTS, Piper, OpenAI) — out of scope here. What we
DO test:

  - `_pick_voice` chooses Russian vs default based on text content
  - Config load/save round-trips
  - Synthesizer.reset re-probes without crashing
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_tts(tmp_path, monkeypatch):
    """Isolate tts_config.json under tmp_path. The TTS module
    re-resolves paths through `paths.knowledge_dir()` on every
    call, so setting HRANT_DATA_DIR is enough — no reload needed.
    (Reload pollutes other tests' singletons.)"""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import tts as _t
    return _t


# ─── Voice picker ──────────────────────────────────────────────────


def test_pick_voice_default_when_no_cyrillic(fresh_tts):
    assert fresh_tts._pick_voice(
        "Hello there how are you",
        default="en-US-AriaNeural",
        ru="ru-RU-SvetlanaNeural",
    ) == "en-US-AriaNeural"


def test_pick_voice_picks_russian_for_cyrillic_text(fresh_tts):
    assert fresh_tts._pick_voice(
        "Привет, как дела сегодня?",
        default="en-US-AriaNeural",
        ru="ru-RU-SvetlanaNeural",
    ) == "ru-RU-SvetlanaNeural"


def test_pick_voice_falls_back_to_default_when_no_ru_set(fresh_tts):
    """If `ru=None`, Cyrillic text still gets the default voice
    rather than crashing."""
    out = fresh_tts._pick_voice(
        "Привет",
        default="en-US-AriaNeural",
        ru=None,
    )
    assert out == "en-US-AriaNeural"


def test_pick_voice_handles_empty_text(fresh_tts):
    out = fresh_tts._pick_voice("", default="en-US-AriaNeural", ru="ru-X")
    assert out == "en-US-AriaNeural"


# ─── Config persistence ────────────────────────────────────────────


def test_load_config_returns_empty_dict_on_missing_file(fresh_tts):
    assert fresh_tts.load_config() == {}


def test_save_then_load_round_trips(fresh_tts):
    fresh_tts.save_config({
        "backend": "edge_tts",
        "edge_tts": {"voice": "en-US-AriaNeural"},
    })
    out = fresh_tts.load_config()
    assert out["backend"] == "edge_tts"
    assert out["edge_tts"]["voice"] == "en-US-AriaNeural"


def test_load_config_tolerates_invalid_json(fresh_tts):
    p = fresh_tts._config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("garbage {", encoding="utf-8")
    # Should fall back to empty dict rather than crash.
    out = fresh_tts.load_config()
    assert isinstance(out, dict)


# ─── Synthesizer ───────────────────────────────────────────────────


def test_synthesizer_status_returns_dict(fresh_tts):
    """status() always returns a dict with at minimum a `backend`
    key — the WebUI status banner depends on this shape."""
    s = fresh_tts.SYNTHESIZER.status()
    assert isinstance(s, dict)
    assert "backend" in s


def test_synthesizer_reset_does_not_crash(fresh_tts):
    """Reset re-probes with the current config. With no config on
    disk this falls through to the default chain — must not raise."""
    fresh_tts.SYNTHESIZER.reset()
    s = fresh_tts.SYNTHESIZER.status()
    assert isinstance(s, dict)
