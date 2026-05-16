"""Tests for the AGENT STATE SNAPSHOT that hangs in every system prompt.

Regression: pre-fix, the agent had no line-of-sight to its own
current settings — it answered "change voice to male" with "Понял"
and stored a preference fact, without changing anything, because
it didn't know the TTS config existed or that it could mutate it.
Production audit: the user repeated the same request 4 times.

This module's `current_self_state()` runs on every turn and gets
rendered into a SNAPSHOT block in the system prompt. Tests pin
its shape, its never-raises contract, and the rendered output's
key cues (active voice + tools + config path).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend import self_state as ss


# --- shape ------------------------------------------------------------


def test_snapshot_has_required_top_level_keys():
    s = ss.current_self_state()
    for k in (
        "data_dir", "repo_root", "active_model",
        "active_tts", "response_language", "tools_available",
    ):
        assert k in s, f"snapshot missing key {k!r}"


def test_active_tts_has_known_subkeys():
    s = ss.current_self_state()
    tts = s["active_tts"]
    for k in ("backend", "voice"):
        assert k in tts


def test_active_model_has_known_subkeys():
    s = ss.current_self_state()
    m = s["active_model"]
    for k in ("provider", "model"):
        assert k in m


def test_tools_available_is_a_list():
    s = ss.current_self_state()
    assert isinstance(s["tools_available"], list)


# --- never-raises contract -------------------------------------------


def test_state_resolver_swallows_subsystem_failure():
    """The snapshot is on the hot turn-startup path. A broken TTS
    subsystem or a missing config file must NOT crash the turn —
    the snapshot just reports the failure as 'unknown' / empty."""
    with patch.object(ss, "_tts_status", side_effect=RuntimeError("boom")):
        s = ss.current_self_state()
        assert "active_tts" in s
        assert s["active_tts"] == {}  # _safe fallback


def test_render_handles_missing_fields_gracefully():
    """A partial snapshot (testing scenarios, plugin shutdown) still
    renders to a usable string."""
    out = ss.render_snapshot({})
    assert "AGENT STATE SNAPSHOT" in out
    assert "active_model" in out
    assert "active_tts" in out


# --- rendered snapshot content ---------------------------------------


def test_render_includes_voice_gender_hint_when_known():
    """The snapshot's voice line should carry a gender hint when the
    voice name is recognised — that's how the LLM realises a `male
    voice` request is currently NOT in effect."""
    out = ss.render_snapshot({
        "active_tts": {
            "backend": "edge_tts",
            "voice": "en-US-AriaNeural",
            "voice_gender": "female",
            "config_path": "/x/tts_config.json",
            "config_exists": True,
        },
        "active_model": {"provider": "anthropic", "model": "claude"},
        "tools_available": ["terminal_exec", "run_python"],
    })
    assert "en-US-AriaNeural" in out
    assert "female" in out
    assert "/x/tts_config.json" in out
    assert "terminal_exec" in out


def test_render_warns_about_missing_config():
    """When a config file is absent we explicitly tag it (missing) so
    the LLM doesn't pretend it's already correct."""
    out = ss.render_snapshot({
        "active_tts": {
            "backend": "edge_tts",
            "voice": "en-US-AriaNeural",
            "config_path": "/x/tts_config.json",
            "config_exists": False,
        },
        "active_model": {"provider": "anthropic", "model": "claude"},
        "tools_available": [],
    })
    assert "(missing)" in out


def test_render_carries_rules_block():
    """The RULES section is what tells the LLM 'use tools, don't just
    acknowledge'. Without it the snapshot is just inventory."""
    out = ss.render_snapshot({})
    assert "# RULES" in out
    assert "USE TOOLS" in out
    assert "don't just acknowledge" in out
    assert "don't pretend" in out


# --- voice gender hint -----------------------------------------------


def test_voice_gender_hint_recognises_common_edge_voices():
    assert ss._voice_gender_hint("en-US-AriaNeural") == "female"
    assert ss._voice_gender_hint("en-US-GuyNeural") == "male"
    assert ss._voice_gender_hint("ru-RU-SvetlanaNeural") == "female"
    assert ss._voice_gender_hint("ru-RU-DmitryNeural") == "male"


def test_voice_gender_hint_returns_empty_for_unknown():
    assert ss._voice_gender_hint("custom-voice-x") == ""
    assert ss._voice_gender_hint("") == ""
