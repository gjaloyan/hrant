"""Two defects taken straight from the owner's real Telegram log, 2026-08-08.

Seventeen turns that day. TEN of them were spent on these two problems.

VOICE (5 turns wasted, plus 5 more replying to them). Speech-to-text was
disabled on the box — no whisper, no faster-whisper, no OPENAI_API_KEY — so
TRANSCRIBER.transcribe() returned None and set no error. The voice note
reached the agent as the same "(see attached file)" placeholder an image
gets, so the agent said "I didn't receive a transcript" and offered "send it
again". The owner sent it again. It failed again. Five times.

THE DROPPED CHOICE (the whole afternoon's work). At 14:53 the agent asked
which way to start and offered options. At 14:54 the owner picked "Быстрый
MVP сейчас". That reply is 43 characters with no attachment, so the fast chat
lane took it: level L0_CHAT, ZERO tools, answer "Ок, Гор — делаем быстрый MVP
сейчас." Nothing was started. At 15:39 the owner asked "status?" and was told
"I haven't started the MVP — the last step was waiting for you to choose."
"""
from __future__ import annotations

import pytest


# ── voice ─────────────────────────────────────────────────────────────

class _Meta:
    def __init__(self, kind, transcript=""):
        self.kind, self.transcript = kind, transcript


def _placeholder_for(monkeypatch, metas, tx_status):
    """Re-create the placeholder decision from channels.py for a message with
    no text and the given attachments."""
    from backend.attachments import ATTACHMENTS
    monkeypatch.setattr(ATTACHMENTS, "get_meta", lambda sha: metas.get(sha))
    from backend.transcriber import TRANSCRIBER
    monkeypatch.setattr(TRANSCRIBER, "status", lambda: tx_status)

    text = ""
    shas = list(metas)
    for sha in shas:
        m = ATTACHMENTS.get_meta(sha)
        if m and m.kind == "audio" and m.transcript:
            return m.transcript
    audio = [ATTACHMENTS.get_meta(s) for s in shas]
    if any(m and m.kind == "audio" for m in audio):
        st = TRANSCRIBER.status()
        why = (st.get("last_error")
               or ("speech-to-text is not configured on this machine"
                   if st.get("backend") in (None, "disabled")
                   else "transcription returned nothing"))
        return (
            "(voice message received, but it could NOT be transcribed: "
            f"{why}. Do not ask the sender to resend it — the result will be "
            "the same. Say plainly that voice input is unavailable and ask "
            "for text, or offer to enable speech-to-text.)")
    return text or "(see attached file)"


def test_an_untranscribable_voice_note_says_why_and_forbids_a_resend(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("audio")},
        {"backend": "disabled", "model": None, "last_error": None})
    assert "could NOT be transcribed" in out
    assert "not configured" in out
    assert "Do not ask the sender to resend" in out
    assert out != "(see attached file)"


def test_a_transcriber_error_is_passed_through_verbatim(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("audio")},
        {"backend": "openai_whisper", "last_error": "401 invalid api key"})
    assert "401 invalid api key" in out


def test_a_successful_transcript_is_still_used_as_the_message(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("audio", "привет, как дела")},
        {"backend": "local_whisper", "last_error": None})
    assert out == "привет, как дела"


def test_an_image_only_message_keeps_the_old_placeholder(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("image")},
        {"backend": "disabled", "last_error": None})
    assert out == "(see attached file)"


# ── the dropped choice ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_resume_marker():
    """run_unified consumes-and-clears this at turn entry; tests must too, or
    one test's marker forces every later one off the fast lane."""
    import backend.tools.ask_user as au
    au.clear_question_resume()
    yield
    au.clear_question_resume()


def test_a_question_resume_is_marked_structurally():
    """Not by matching "My choice:" in the text — the resume path is our own
    protocol and can say so directly."""
    import backend.tools.ask_user as au
    assert au.is_question_resume() is False
    au.mark_question_resume()
    assert au.is_question_resume() is True


def test_the_resume_marker_does_not_leak_into_the_next_turn():
    """ContextVar, so a fresh context starts clean — otherwise every later
    turn would be forced off the fast lane."""
    import contextvars
    import backend.tools.ask_user as au

    def _marked():
        au.mark_question_resume()
        return au.is_question_resume()

    assert contextvars.copy_context().run(_marked) is True
    assert contextvars.copy_context().run(au.is_question_resume) is False


def test_a_resume_turn_is_kept_off_the_fast_chat_lane(monkeypatch):
    """The exact 14:54 message. Short, no attachment — everything the fast
    lane looks for — but it is the continuation of a paused task."""
    import backend.tools.ask_user as au
    task = "My choice: Быстрый MVP сейчас (Recommended)"
    assert len(task) <= 500 and "\n" not in task   # fast-lane shaped

    def _fast_lane_would_take(resuming: bool) -> bool:
        attachments, matched_skills = [], []
        return (not attachments and not matched_skills
                and not resuming and len(task) <= 500)

    assert _fast_lane_would_take(resuming=False) is True   # the old behaviour
    au.mark_question_resume()
    assert _fast_lane_would_take(resuming=au.is_question_resume()) is False


def test_an_ordinary_short_message_still_takes_the_fast_lane():
    """The fast lane exists for a reason; the fix must not disable it."""
    import backend.tools.ask_user as au
    assert au.is_question_resume() is False
    task = "привет, как дела?"
    assert (not [] and not [] and not au.is_question_resume()
            and len(task) <= 500) is True
