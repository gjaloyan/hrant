"""Regression: profile-recall questions ('what is my X', 'помнишь мой Y')
must route to fast_chat, not task_mode.

Pre-audit, the LLM intent classifier consistently labelled them
'task' even though fast_chat already has user profile + recent
conversation in context. The result was 4 LLM calls / ~$0.04 per
recall instead of 2 / $0.007.

The fix is a regex short-circuit in `_classify_intent` BEFORE the
LLM call — patterns covering English + Russian, restricted to short
messages so we don't accidentally short-circuit long tasks that
happen to contain a "remember" substring.
"""
from __future__ import annotations

from backend.agent import _looks_like_profile_recall


# --- pattern positives -------------------------------------------------


def test_english_what_is_my():
    assert _looks_like_profile_recall("What is my favorite color?")
    assert _looks_like_profile_recall("What's my home server called?")
    assert _looks_like_profile_recall("Where is my office?")
    assert _looks_like_profile_recall("Who is my brother?")
    assert _looks_like_profile_recall("How is my apricot variety called?")


def test_english_do_you_remember():
    assert _looks_like_profile_recall("Do you remember my brother's name?")
    assert _looks_like_profile_recall("do you know my favorite color")
    assert _looks_like_profile_recall("Do you recall what I told you?")
    assert _looks_like_profile_recall("What did I tell you about Lusine?")


def test_english_tell_me_about():
    assert _looks_like_profile_recall("Tell me about myself.")
    assert _looks_like_profile_recall("Tell me my favorite color.")


def test_russian_variants():
    assert _looks_like_profile_recall("что мой любимый цвет?")
    assert _looks_like_profile_recall("что моё любимое блюдо")
    assert _looks_like_profile_recall("какой мой адрес?")
    assert _looks_like_profile_recall("помнишь мой день рождения?")
    assert _looks_like_profile_recall("помните что я говорил?")
    assert _looks_like_profile_recall("что ты обо мне знаешь?")
    assert _looks_like_profile_recall("что я тебе говорил вчера?")


# --- pattern negatives -------------------------------------------------


def test_does_not_match_general_knowledge():
    assert not _looks_like_profile_recall("What is the capital of Armenia?")
    assert not _looks_like_profile_recall("Who invented the radio?")
    assert not _looks_like_profile_recall("How does TCP work?")


def test_does_not_match_long_messages():
    """Long messages are almost certainly tasks. We don't want a
    300-char paragraph that happens to contain 'remember' to skip
    the LLM classifier."""
    long = "what is my plan for the next month " * 20
    assert not _looks_like_profile_recall(long)


def test_does_not_match_empty():
    assert not _looks_like_profile_recall("")
    assert not _looks_like_profile_recall("   ")


# --- end-to-end through _classify_intent ------------------------------


def test_classify_intent_routes_recall_to_chat(monkeypatch):
    """The full classifier path: recall regex must fire BEFORE the
    LLM is contacted, so the test doesn't need a router mock."""
    from backend.pipeline.intent import IntentClassifierMixin

    class _Stub(IntentClassifierMixin):
        def _attachment_marker(self): return ""
        def _record_llm_call(self, **kwargs): return None
    stub = _Stub()

    assert stub._classify_intent("What is my favorite color?") == "chat"
    assert stub._classify_intent("что мой любимый цвет?") == "chat"


def test_classify_intent_keeps_arithmetic_as_task(monkeypatch):
    """Arithmetic must still go to task path so the solver can call
    calc / run_python. The profile-recall regex must not snap up
    things like '2 + 2 = ?'."""
    from backend.pipeline.intent import IntentClassifierMixin

    class _Stub(IntentClassifierMixin):
        def _attachment_marker(self): return ""
        def _record_llm_call(self, **kwargs): return None
    stub = _Stub()

    assert stub._classify_intent("2 + 2 = ?") == "task"
    assert stub._classify_intent("87654 * 12345") == "task"
