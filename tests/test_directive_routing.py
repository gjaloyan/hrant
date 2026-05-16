"""Directive vs preference routing.

Pre-fix, "измени голос на мужской" classified as `preference` →
saved a fact to user_profile.md → no actual change. The user
repeated the same imperative 4 times before the agent did
anything tool-shaped. New regex `_looks_like_system_directive`
catches the imperative shape + system-attribute mention pattern
and forces those messages to the `task` path.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.agent import _looks_like_system_directive


# --- directive positives (route to task) -----------------------------


def test_russian_voice_change_directive():
    cases = [
        "Измени голос на мужской",
        "измени голос",
        "Смени голос",
        "Поменяй голос на мужской",
        "Сделай мужской голос",
        "Поставь голос мужской",
        "Переключи голос на мужской",
    ]
    for msg in cases:
        assert _looks_like_system_directive(msg), msg


def test_english_voice_change_directive():
    cases = [
        "Change the voice to male",
        "Switch voice to male",
        "Set TTS voice to male",
        "Change your voice",
        "Make the voice male",
        "Please change the voice to a male voice",
    ]
    for msg in cases:
        assert _looks_like_system_directive(msg), msg


def test_model_change_directive():
    assert _looks_like_system_directive("Switch the model to gpt-4o")
    assert _looks_like_system_directive("Измени модель на claude")
    assert _looks_like_system_directive("Поменяй провайдер на openai")


def test_language_change_directive():
    assert _looks_like_system_directive("Set the response language to Russian")
    assert _looks_like_system_directive("Поставь язык русский")
    assert _looks_like_system_directive("Включи русский язык")


def test_enable_disable_directives():
    assert _looks_like_system_directive("Turn off voice replies")
    assert _looks_like_system_directive("Disable TTS")
    assert _looks_like_system_directive("Включи tts")
    assert _looks_like_system_directive("Выключи озвучку")


# --- preference negatives (stay on preference path) ------------------


def test_pure_preferences_are_not_directives():
    """Statements with no imperative verb (e.g. 'I prefer X', 'always
    use Y') ARE preferences and should NOT route to task."""
    cases = [
        "I prefer a male voice",
        "I like male voices",
        "Always respond with a male voice",
        "My favorite voice is the male one",
        "У меня предпочтение к мужским голосам",
        "Мне больше нравится мужской голос",
    ]
    for msg in cases:
        assert not _looks_like_system_directive(msg), msg


def test_general_questions_are_not_directives():
    cases = [
        "What voice are you using?",
        "Что за модель ты используешь?",
        "How do I change my voice?",
        "Tell me about the TTS config",
    ]
    for msg in cases:
        assert not _looks_like_system_directive(msg), msg


def test_directive_without_system_attribute_is_not_caught():
    """`change my mind`, `set the table` etc. — directive verb but
    no system-attribute target → not our case."""
    cases = [
        "Change my mind about this",
        "Set the table",
        "Make breakfast",
        "Поменяй тему",  # subject would be ambiguous; but not a system attr
    ]
    for msg in cases:
        assert not _looks_like_system_directive(msg), msg


def test_long_messages_are_not_treated_as_directives():
    """A 300+ char message is almost certainly a task with surrounding
    context, not a snap directive. Length guard mirrors the same
    rule in profile-recall."""
    msg = "Change the voice to male. " * 20
    assert not _looks_like_system_directive(msg)


def test_empty_message_returns_false():
    assert not _looks_like_system_directive("")
    assert not _looks_like_system_directive("   ")


# --- end-to-end through _classify_intent -----------------------------


def test_classify_intent_routes_directive_to_task(monkeypatch):
    """The full classifier path: directive regex fires BEFORE the
    LLM, so this test doesn't need a router mock."""
    from backend.pipeline.intent import IntentClassifierMixin

    class _Stub(IntentClassifierMixin):
        def _attachment_marker(self): return ""
        def _record_llm_call(self, **kwargs): return None
    stub = _Stub()

    assert stub._classify_intent("Измени голос на мужской") == "task"
    assert stub._classify_intent("Switch model to gpt-4o") == "task"


def test_preference_about_system_setting_is_caught_for_escalation():
    """Preference-shaped messages that touch a system attribute
    are caught by `_looks_like_system_setting_preference` so the
    preference branch can escalate them to task. The user's
    actual production messages were of this shape — no imperative
    verb, but a clear voice-config request."""
    from backend.agent import _looks_like_system_setting_preference
    cases = [
        "Respond using a male voice",
        "Respond using a male voice for audio/TTS messages",
        "Always answer with a male voice",
        "I want you to use a male voice",
        "Отвечай мужским голосом",
        "Используй мужской голос",
    ]
    for msg in cases:
        assert _looks_like_system_setting_preference(msg), msg


def test_preference_without_system_setting_is_not_caught():
    """A normal preference (no system attribute) stays on the
    preference path — gets saved to user_profile.md as before."""
    from backend.agent import _looks_like_system_setting_preference
    cases = [
        "My favorite color is teal",
        "I prefer terse answers",
        "Always remind me of important deadlines",
        "Мой любимый фрукт — абрикос",
    ]
    for msg in cases:
        assert not _looks_like_system_setting_preference(msg), msg


def test_classify_intent_keeps_preferences_on_preference_path(monkeypatch):
    """Statements that ARE preferences (no imperative, just a
    standing rule) must still reach the LLM classifier — the regex
    short-circuits don't apply, the LLM decides."""
    from backend.pipeline.intent import IntentClassifierMixin

    class _Stub(IntentClassifierMixin):
        def _attachment_marker(self): return ""
        def _record_llm_call(self, **kwargs): return None
    stub = _Stub()

    # The short-circuit chain must NOT fire on a true preference,
    # so the call would reach the LLM. Router is locally imported
    # inside _classify_intent from `backend.agent`, so we patch
    # there.
    fake_data = {"intent": "preference", "reason": "stable trait"}
    with patch("backend.agent.router") as r:
        r.return_value.call_json.return_value = fake_data
        out = stub._classify_intent("I prefer male voice")
        # Should reach the LLM (not short-circuited)
        r.return_value.call_json.assert_called_once()
        assert out == "preference"
