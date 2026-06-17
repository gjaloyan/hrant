"""Honest model reporting: the model that ACTUALLY served the turn + a
notice when it silently differs from the selected one (a quota fallback)."""
from __future__ import annotations

from backend.model_report import primary_model_used, fallback_note


def test_primary_model_picks_the_served_model():
    calls = [
        {"label": "_unified:tool_iter_0", "model": "openai/gpt-5"},
        {"label": "_unified:tool_iter_1", "model": "openai/gpt-5"},
        {"label": "_unified", "model": "openai/gpt-5"},
    ]
    assert primary_model_used(calls) == "openai/gpt-5"


def test_primary_model_empty():
    assert primary_model_used([]) == ""
    assert primary_model_used(None) == ""
    assert primary_model_used([{"label": "x"}]) == ""  # no model field


def test_no_note_when_selected_model_served():
    # selected gpt-5.5 (codex) and it actually served -> no notice
    assert fallback_note("gpt-5.5", "gpt-5.5", "") == ""
    # provider-qualified leaf matches the bare selection
    assert fallback_note("gpt-5.5", "openai-codex/gpt-5.5", "") == ""


def test_note_when_fell_back_to_other_model():
    note = fallback_note("gpt-5.5", "openai/gpt-5", "usage_limit_reached")
    assert "gpt-5.5" in note and "openai/gpt-5" in note
    assert "usage_limit_reached" in note


def test_note_without_reason():
    note = fallback_note("gpt-5.5", "nex-agi/nex-n2-pro:free", "")
    assert "nex-agi/nex-n2-pro:free" in note
    assert note  # non-empty


def test_no_note_on_missing_data():
    assert fallback_note("", "openai/gpt-5", "x") == ""
    assert fallback_note("gpt-5.5", "", "x") == ""
