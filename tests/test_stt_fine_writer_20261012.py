"""`medium` decides the language; a stronger model writes Russian and English.

Measured 2026-10-12 on the owner's notes. "Привет, друг, как ты?" reached the
agent as "Привет, ты только как-то." and "распознавание речи" as
"распознавание дичи" — generic `medium` writing Russian. large-v3-turbo read
both exactly, but given Armenian audio it decided the language was Russian
and wrote Cyrillic nonsense, where `medium` decides Armenian and the
specialist reads it. So each does the part it is good at.
"""
from __future__ import annotations

import sys
import types

import pytest

from backend import transcriber as tr


class _Seg:
    def __init__(self, text):
        self.text = text


class _Model:
    def __init__(self, text, lang="ru"):
        self.text, self.lang, self.calls = text, lang, 0

    def detect_language(self, audio):
        return self.lang, 0.9, [(self.lang, 0.9)]

    def transcribe(self, source, language=None, vad_filter=True):
        self.calls += 1
        return [_Seg(self.text)], types.SimpleNamespace(language=language)


@pytest.fixture
def engine(monkeypatch):
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = lambda *a, **k: pytest.fail("no model loads in this test")
    fw.decode_audio = lambda path, sampling_rate=16000: "AUDIO"
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setattr(tr, "expected_languages", lambda cfg=None: ["hy", "ru", "en"])
    t = tr.Transcriber()
    t._fw_alt = _Model("Պրիվետ դրուկ կակտ է", "hy")
    return t


def test_russian_is_written_by_the_fine_model(engine):
    engine._fw_model = _Model("Привет, ты только как-то.", "ru")
    engine._fw_fine = _Model("Привет, друг, как ты?", "ru")
    assert engine._tx_faster_whisper(b"x", filename="v.ogg") == "Привет, друг, как ты?"
    assert engine._fw_model.calls == 0, "medium only decided the language"


def test_armenian_still_goes_to_the_specialist(engine):
    engine._fw_model = _Model("Հայդան ատկարկավոտվեց", "hy")
    engine._fw_fine = _Model("Исту айден хаскан, мис.", "ru")
    engine._fw_alt = _Model("Իսկ դու հայերեն հասկանում ես", "hy")
    assert engine._tx_faster_whisper(b"x", filename="v.ogg") == "Իսկ դու հայերեն հասկանում ես"
    assert engine._fw_fine.calls == 0, "the fine model never sees Armenian"


def test_without_the_fine_model_medium_writes_as_before(engine, monkeypatch):
    engine._fw_model = _Model("Привет, ты только как-то.", "ru")
    engine._fw_fine_failed = True
    assert engine._tx_faster_whisper(b"x", filename="v.ogg") == "Привет, ты только как-то."


def test_the_fine_model_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(tr, "load_config", lambda: {"faster_whisper": {"fine_model": ""}})
    assert tr.Transcriber()._fine_model() is None


def test_the_fine_model_is_never_downloaded_by_a_voice_note(monkeypatch):
    seen = {}

    def _load(name, **kw):
        seen.update(kw, name=name)
        raise OSError("not in the local cache")

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = _load
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setattr(tr, "load_config", lambda: {})
    t = tr.Transcriber()
    assert t._fine_model() is None
    assert seen["local_files_only"] is True and seen["name"] == tr.FINE_MODEL
    assert t._fine_model() is None and t._fw_fine_failed, "given up on, not retried per note"
