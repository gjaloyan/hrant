"""Sticky-request detector — Phase 2.B.

Pinned behaviour:
  - Detector inspects per-speaker recent conversation memory.
  - Triggers when M-of-K turns share a system attribute AND the
    agent's previous replies looked like short acks without tools.
  - When triggered, produces a render block telling the LLM to
    ESCALATE to tool use this turn.
  - When not triggered, returns sticky=False (the prompt stays
    clean — no false alarms on first-time requests).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend import sticky_requests as sr


# --- attribute detection ---------------------------------------------


def test_attributes_mentioned_picks_voice():
    assert "voice" in sr._attributes_mentioned("change the voice to male")
    assert "voice" in sr._attributes_mentioned("Измени голос")


def test_attributes_mentioned_picks_multiple():
    out = sr._attributes_mentioned("Switch model and language")
    assert "model" in out
    assert "language" in out


def test_attributes_mentioned_empty_when_unrelated():
    assert sr._attributes_mentioned("Hi there!") == set()
    assert sr._attributes_mentioned("What's 2+2?") == set()


# --- ack-only detection ----------------------------------------------


def test_ack_only_detects_russian_acks():
    assert sr._looks_like_ack_only("Понял, буду мужским голосом")
    assert sr._looks_like_ack_only("Запомнил.")
    assert sr._looks_like_ack_only("Хорошо.")


def test_ack_only_detects_english_acks():
    assert sr._looks_like_ack_only("Got it, I'll do that")
    assert sr._looks_like_ack_only("Okay, understood")
    assert sr._looks_like_ack_only("Noted")


def test_ack_only_rejects_long_substantive_answers():
    long = (
        "Switched the voice to en-US-GuyNeural via set_setting. "
        "Config file at /home/hrant/.hrant/data/knowledge/tts_config.json "
        "now reflects the new voice. The synthesiser singleton was reset "
        "so the next reply uses the new voice without a restart."
    )
    assert not sr._looks_like_ack_only(long)


def test_ack_only_treats_empty_as_ack():
    # An empty agent answer is even worse than an ack — definitely
    # not an applied change.
    assert sr._looks_like_ack_only("")


# --- end-to-end detection --------------------------------------------


def _fake_turns(*pairs):
    """Build the shape `CONVERSATION.recent` returns: list of dicts
    with `user` + `answer` keys (we ignore the rest of the fields)."""
    return [
        {"user": u, "answer": a, "speaker_id": "test:speaker"}
        for (u, a) in pairs
    ]


def test_sticky_detected_on_repeated_voice_request():
    """Matches the actual production case from the Telegram audit —
    user said «измени голос» 4 times with «голос» word in each turn."""
    with patch.object(
        sr.CONVERSATION, "recent",
        return_value=_fake_turns(
            ("Измени голос на мужской", "Понял, буду мужским голосом"),
            ("Измени свой голос на мужской", "Окей, буду говорить мужским"),
        ),
    ):
        info = sr.detect_sticky_request(
            current_user_message="Почему ты не меняешь голос?",
            speaker_id="test:speaker",
        )
    assert info["sticky"] is True
    assert info["attribute"] == "voice"
    assert info["repeats"] >= 2


def test_sticky_not_triggered_on_first_voice_request():
    """First time the user mentions voice → no priors → no sticky."""
    with patch.object(sr.CONVERSATION, "recent", return_value=[]):
        info = sr.detect_sticky_request(
            current_user_message="Change voice to male",
            speaker_id="test:speaker",
        )
    assert info["sticky"] is False


def test_sticky_not_triggered_when_prior_was_substantive():
    """If the previous agent answer was a real change (not an ack),
    the detector stays quiet — the user's repeat is a NEW request,
    not a frustrated re-issue."""
    real_answer = (
        "Switched voice to GuyNeural via set_setting; config updated, "
        "singleton reset. Next audio reply will use the new voice."
    )
    with patch.object(
        sr.CONVERSATION, "recent",
        return_value=_fake_turns(
            ("Change voice to male", real_answer),
            ("Change voice to female", real_answer),
        ),
    ):
        info = sr.detect_sticky_request(
            current_user_message="Make the voice male again",
            speaker_id="test:speaker",
        )
    # The prior answers were SUBSTANTIVE so the sticky detector
    # decides "OK, prior changes happened; this is a fresh request,
    # not a stuck loop."
    assert info["sticky"] is False


def test_sticky_not_triggered_for_unrelated_repeats():
    """Two prior turns mentioned `voice` but with substantive answers,
    plus today's question about `language`. Different attribute → no
    sticky."""
    with patch.object(
        sr.CONVERSATION, "recent",
        return_value=_fake_turns(
            ("change voice", "ok will do"),  # ack but unrelated attribute
            ("change voice", "ok"),
        ),
    ):
        info = sr.detect_sticky_request(
            current_user_message="Switch the language to Russian",
            speaker_id="test:speaker",
        )
    assert info["sticky"] is False


def test_sticky_swallows_conversation_errors():
    """`CONVERSATION.recent` raising must not crash the turn."""
    with patch.object(sr.CONVERSATION, "recent", side_effect=RuntimeError("boom")):
        info = sr.detect_sticky_request(
            current_user_message="change voice",
            speaker_id="test:speaker",
        )
    assert info["sticky"] is False


# --- rendered block ---------------------------------------------------


def test_render_sticky_block_empty_when_not_sticky():
    assert sr.render_sticky_block({"sticky": False}) == ""


def test_render_sticky_block_contains_escalation_instruction():
    info = {
        "sticky": True,
        "attribute": "voice",
        "repeats": 3,
        "reason": "user asked about voice 3 times — ESCALATE to set_setting",
    }
    out = sr.render_sticky_block(info)
    assert "STICKY REQUEST DETECTED" in out
    assert "ESCALATE" in out
    assert "set_setting" in out
