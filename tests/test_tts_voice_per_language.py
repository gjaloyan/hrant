"""TTS voice selection by text language.

User reported: TTS replied to Russian answers using the
`en_US-lessac-medium` voice — Piper tried to pronounce Cyrillic
text with English phonemes and produced unintelligible audio.

Fix: detect Cyrillic in the text and route to a Russian voice
(`ru_RU-irina-medium` by default; configurable via
`local_piper.voice_ru` in tts_config.json).

These tests pin the picker behaviour at the unit level so a future
refactor of language detection doesn't silently regress voice
quality on the user's primary language.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# --- _pick_voice helper -------------------------------------------------


def test_pick_voice_english_text_returns_default():
    from backend.tts import _pick_voice
    out = _pick_voice("hello world", default="en_voice", ru="ru_voice")
    assert out == "en_voice"


def test_pick_voice_cyrillic_text_returns_ru():
    from backend.tts import _pick_voice
    out = _pick_voice("Привет мир", default="en_voice", ru="ru_voice")
    assert out == "ru_voice"


def test_pick_voice_single_cyrillic_letter_triggers_ru():
    """A single Cyrillic char anywhere in the text routes to ru.
    Mixed-language reply ("ok, я понял") should NOT come out in
    half-broken English voice."""
    from backend.tts import _pick_voice
    out = _pick_voice("ok я ready", default="en_voice", ru="ru_voice")
    assert out == "ru_voice"


def test_pick_voice_empty_text_returns_default():
    from backend.tts import _pick_voice
    assert _pick_voice("", default="en_voice", ru="ru_voice") == "en_voice"


def test_pick_voice_no_ru_voice_configured_falls_back_to_default():
    """User who hasn't set up a Russian voice still gets SOMETHING
    even on Cyrillic text — distorted English is better than the
    voice reply silently failing."""
    from backend.tts import _pick_voice
    out = _pick_voice("Привет", default="en_voice", ru=None)
    assert out == "en_voice"


def test_pick_voice_handles_yo_letter():
    """Russian Ё/ё falls outside the basic А-я range in some regex
    patterns. Make sure it still routes to ru."""
    from backend.tts import _pick_voice
    out = _pick_voice("ёж", default="en_voice", ru="ru_voice")
    assert out == "ru_voice"


def test_pick_voice_emoji_only_text_uses_default():
    """Emoji-only text (no script chars) → default voice. Edge
    case: don't accidentally classify emoji as Cyrillic."""
    from backend.tts import _pick_voice
    out = _pick_voice("👋🎉", default="en_voice", ru="ru_voice")
    assert out == "en_voice"


# --- Synthesizer wiring -------------------------------------------------


def _ok_health():
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"status": "ok"}
    return m


def _ok_speech():
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.status_code = 200
    m.content = b"RIFF\x00\x00\x00\x00WAVE"
    return m


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    monkeypatch.delenv("LOCAL_PIPER_URL", raising=False)
    monkeypatch.delenv("AGI_TTS_BACKEND", raising=False)


def test_synthesizer_loads_voice_ru_from_config(monkeypatch):
    """tts_config.json local_piper.voice_ru is honoured at probe
    time; default is `ru_RU-irina-medium` when not set."""
    from backend import tts as tts_mod

    cfg_with = {
        "local_piper": {
            "url": "http://10.0.0.1:8017",
            "voice": "en_US-lessac-medium",
            "voice_ru": "ru_RU-denis-medium",
        }
    }
    monkeypatch.setattr(tts_mod, "load_config", lambda: cfg_with)
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    assert s._voice_ru == "ru_RU-denis-medium"


def test_synthesizer_voice_ru_defaults_when_unset(monkeypatch):
    from backend import tts as tts_mod

    cfg = {"local_piper": {"url": "http://10.0.0.1:8017"}}
    monkeypatch.setattr(tts_mod, "load_config", lambda: cfg)
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    assert s._voice_ru == tts_mod.DEFAULT_LOCAL_PIPER_VOICE_RU


def test_synthesize_routes_russian_text_to_ru_voice(monkeypatch):
    """End-to-end on the Synthesizer: synthesizing Cyrillic text
    must POST with voice=ru_voice. Same instance handling English
    text must POST with voice=en_voice."""
    from backend import tts as tts_mod

    cfg = {
        "local_piper": {
            "url": "http://10.0.0.1:8017",
            "voice": "en_US-lessac-medium",
            "voice_ru": "ru_RU-irina-medium",
        }
    }
    monkeypatch.setattr(tts_mod, "load_config", lambda: cfg)
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()

    captured: list[dict] = []

    def fake_post(url, **kw):
        captured.append(kw.get("json", {}))
        return _ok_speech()

    with patch("backend.tts.httpx.post", side_effect=fake_post):
        s.synthesize("hello world")
        s.synthesize("Привет, как дела?")

    assert captured[0]["voice"] == "en_US-lessac-medium"
    assert captured[1]["voice"] == "ru_RU-irina-medium"


def test_synthesize_explicit_voice_overrides_autopick(monkeypatch):
    """Caller-supplied voice always wins — used when the agent
    deliberately picks a voice for some reason."""
    from backend import tts as tts_mod

    cfg = {
        "local_piper": {
            "url": "http://10.0.0.1:8017",
            "voice_ru": "ru_RU-irina-medium",
        }
    }
    monkeypatch.setattr(tts_mod, "load_config", lambda: cfg)
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()

    captured: list[dict] = []
    with patch(
        "backend.tts.httpx.post",
        side_effect=lambda url, **kw: (
            captured.append(kw.get("json", {})) or _ok_speech()
        ),
    ):
        # Cyrillic text BUT explicit English voice → English wins.
        s.synthesize("Привет", voice="en_US-lessac-medium")

    assert captured[0]["voice"] == "en_US-lessac-medium"
