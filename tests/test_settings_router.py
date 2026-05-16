"""Tests for the SettingsRouter — Phase 2.A.

Pinned behaviour:
  - registry has the expected canonical keys
  - list_settings returns key/current/choices for every entry
  - get() reads the current value via the spec's get_fn
  - set() validates value (choices / type), writes through, and
    returns {ok, key, old, new, note}
  - set() with an unknown key raises KeyError
  - set() with an invalid value raises ValueError
  - the TTS-specific shorthand `tts.voice_gender` resolves to
    concrete voice IDs and writes both EN + RU
  - subsystem reset fires on TTS writes (the cached singleton
    voice is cleared so the next synth picks up the new config)
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from backend import settings as s_mod


# --- registry shape ---------------------------------------------------


def test_router_has_canonical_settings():
    keys = {s["key"] for s in s_mod.SETTINGS.list_settings()}
    assert "tts.backend" in keys
    assert "tts.voice" in keys
    assert "tts.voice_ru" in keys
    assert "tts.voice_gender" in keys
    assert "response.language" in keys


def test_list_settings_returns_complete_records():
    for s in s_mod.SETTINGS.list_settings():
        assert "key" in s and isinstance(s["key"], str)
        assert "description" in s and s["description"]
        assert "current" in s         # may be None
        assert "choices" in s         # always a list
        assert "value_type" in s


# --- unknown key / invalid value error paths --------------------------


def test_get_unknown_key_raises():
    with pytest.raises(KeyError):
        s_mod.SETTINGS.get("doesnt.exist")


def test_set_unknown_key_raises():
    with pytest.raises(KeyError):
        s_mod.SETTINGS.set("doesnt.exist", "x")


def test_set_invalid_choice_raises():
    with pytest.raises(ValueError):
        s_mod.SETTINGS.set("tts.backend", "no-such-backend")


def test_set_invalid_voice_gender_raises():
    with pytest.raises(ValueError):
        s_mod.SETTINGS.set("tts.voice_gender", "weird")


# --- happy paths via patched TTS layer --------------------------------


@pytest.fixture
def fake_tts(monkeypatch):
    """Replace tts.load_config / save_config + SYNTHESIZER.reset
    with an in-memory dict so settings.set actually persists
    something we can observe without writing to ~/.hrant."""
    state = {"cfg": {}}
    reset_count = {"n": 0}

    def _load():
        return dict(state["cfg"])

    def _save(cfg):
        state["cfg"] = dict(cfg)

    class _FakeSynth:
        def reset(self):
            reset_count["n"] += 1

    # Patch on backend.tts so the settings module's late imports see
    # the fakes.
    import backend.tts as _tts
    monkeypatch.setattr(_tts, "load_config", _load)
    monkeypatch.setattr(_tts, "save_config", _save)
    monkeypatch.setattr(_tts, "SYNTHESIZER", _FakeSynth())
    return state, reset_count


def test_set_tts_voice_writes_to_backend_block(fake_tts):
    state, _ = fake_tts
    res = s_mod.SETTINGS.set("tts.voice", "en-US-GuyNeural")
    assert res["ok"]
    assert res["key"] == "tts.voice"
    assert res["new"] == "en-US-GuyNeural"
    # On-disk shape: cfg['edge_tts']['voice'] = 'en-US-GuyNeural'.
    assert state["cfg"]["edge_tts"]["voice"] == "en-US-GuyNeural"


def test_set_resets_singleton(fake_tts):
    _, reset_count = fake_tts
    s_mod.SETTINGS.set("tts.voice", "en-US-GuyNeural")
    assert reset_count["n"] >= 1, "tts.reset() should have fired"


def test_voice_gender_shorthand_writes_both_languages(fake_tts):
    state, _ = fake_tts
    res = s_mod.SETTINGS.set("tts.voice_gender", "male")
    assert res["ok"]
    backend_cfg = state["cfg"]["edge_tts"]
    assert backend_cfg["voice"] == "en-US-GuyNeural"
    assert backend_cfg["voice_ru"] == "ru-RU-DmitryNeural"


def test_voice_gender_female_shorthand(fake_tts):
    state, _ = fake_tts
    s_mod.SETTINGS.set("tts.voice_gender", "female")
    assert state["cfg"]["edge_tts"]["voice"] == "en-US-AriaNeural"
    assert state["cfg"]["edge_tts"]["voice_ru"] == "ru-RU-SvetlanaNeural"


def test_voice_gender_auto_clears_pins(fake_tts):
    state, _ = fake_tts
    # Set first
    s_mod.SETTINGS.set("tts.voice_gender", "male")
    assert "voice" in state["cfg"]["edge_tts"]
    # Now reset
    s_mod.SETTINGS.set("tts.voice_gender", "auto")
    assert "voice" not in state["cfg"]["edge_tts"]
    assert "voice_ru" not in state["cfg"]["edge_tts"]


def test_set_backend_validates_choices(fake_tts):
    state, _ = fake_tts
    res = s_mod.SETTINGS.set("tts.backend", "local_piper")
    assert res["ok"]
    assert state["cfg"]["backend"] == "local_piper"


def test_set_returns_note_when_value_unchanged(fake_tts):
    state, _ = fake_tts
    s_mod.SETTINGS.set("tts.voice", "en-US-GuyNeural")
    res = s_mod.SETTINGS.set("tts.voice", "en-US-GuyNeural")
    assert "already" in (res["note"] or "")


# --- tts.rate (added after the second Telegram audit) -----------------


def test_tts_rate_registered():
    """The audit caught the agent hand-patching tts.py for rate
    because tts.rate wasn't in the SETTINGS registry. It is now."""
    keys = {s["key"] for s in s_mod.SETTINGS.list_settings()}
    assert "tts.rate" in keys


def test_tts_rate_default_is_zero_percent(fake_tts):
    state, _ = fake_tts
    assert s_mod.SETTINGS.get("tts.rate") == "+0%"


def test_tts_rate_writes_to_backend_block(fake_tts):
    state, _ = fake_tts
    res = s_mod.SETTINGS.set("tts.rate", "+25%")
    assert res["ok"]
    assert state["cfg"]["edge_tts"]["rate"] == "+25%"


def test_tts_rate_accepts_bare_number():
    """User input is often '25' or '+25' — normalise to '+25%'."""
    # Use a stub via monkeypatch — simpler than the fake_tts fixture
    # because we just want to observe the normalised value.
    saved = {}
    from backend import tts as _tts

    def _save(cfg):
        saved.update(cfg)

    def _load():
        return dict(saved)

    class _Synth:
        def reset(self):
            pass

    with patch.object(_tts, "load_config", _load), \
         patch.object(_tts, "save_config", _save), \
         patch.object(_tts, "SYNTHESIZER", _Synth()):
        s_mod.SETTINGS.set("tts.rate", "25")
        assert saved["edge_tts"]["rate"] == "+25%"


def test_tts_rate_rejects_out_of_range(fake_tts):
    """+200% / -200% are outside the regex range and refused."""
    with pytest.raises(ValueError):
        s_mod.SETTINGS.set("tts.rate", "+200%")


def test_tts_rate_resets_singleton(fake_tts):
    _, reset_count = fake_tts
    s_mod.SETTINGS.set("tts.rate", "+50%")
    assert reset_count["n"] >= 1, "rate change must reset the synth"


# --- end-to-end via the set_setting tool handler ----------------------


def test_set_setting_tool_owner_only(monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)
    out = builtin_tools._set_setting_handler("tts.voice_gender", "male")
    data = json.loads(out)
    assert data["ok"] is False
    assert "owner" in data["error"].lower()


def test_set_setting_tool_owner_applies(monkeypatch, fake_tts):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._set_setting_handler("tts.voice_gender", "male")
    data = json.loads(out)
    assert data["ok"]
    assert data["new"] == "male"


def test_set_setting_tool_returns_available_keys_on_unknown(monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._set_setting_handler("not.a.real.key", "x")
    data = json.loads(out)
    assert data["ok"] is False
    assert "available_keys" in data
    assert "tts.voice_gender" in data["available_keys"]
