"""Smoke tests for backend.transcriber — STT backend selection."""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_transcriber(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import transcriber as _t
    return _t


def test_load_config_empty_when_missing(fresh_transcriber):
    assert fresh_transcriber.load_config() == {}


def test_save_then_load_round_trips(fresh_transcriber):
    fresh_transcriber.save_config({
        "backend": "local_whisper",
        "local_whisper": {"url": "http://localhost:8016", "model": "whisper-medium"},
    })
    out = fresh_transcriber.load_config()
    assert out["backend"] == "local_whisper"
    assert out["local_whisper"]["url"] == "http://localhost:8016"


def test_load_config_tolerates_invalid_json(fresh_transcriber):
    p = fresh_transcriber._config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json", encoding="utf-8")
    assert isinstance(fresh_transcriber.load_config(), dict)


def test_transcriber_status_returns_dict(fresh_transcriber):
    s = fresh_transcriber.TRANSCRIBER.status()
    assert isinstance(s, dict)
    assert "backend" in s


def test_transcriber_reset_does_not_crash(fresh_transcriber):
    fresh_transcriber.TRANSCRIBER.reset()
    s = fresh_transcriber.TRANSCRIBER.status()
    assert isinstance(s, dict)


def test_transcribe_empty_bytes_returns_none(fresh_transcriber):
    """Empty input shouldn't reach the backend — return None
    cleanly rather than crashing on a 0-byte upload."""
    out = fresh_transcriber.TRANSCRIBER.transcribe(
        b"", mime_type="audio/ogg", filename="empty.ogg",
    )
    assert out is None
