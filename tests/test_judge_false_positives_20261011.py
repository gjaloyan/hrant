"""Two per-turn judges flagged what the agent already had in hand.

Measured on prod 2026-10-10, on test turns after the thread deploy:
  * "Remember the codeword for this conversation, just confirm" took 8 LLM
    calls: the unbacked-action judge wanted save_user_fact for a fact the
    conversation thread already carries to every later turn.
  * "What time is it in Yerevan?" took 10 calls: the claims judge saw only
    the question and the answer, called the time an unchecked fact, and the
    agent searched the web for what its own clock said. The final verifier
    then scored the answer 33 for the same reason.
Both judges now see the agent's clock, and the action judge knows that a
"remember" scoped to this conversation is backed by the conversation.
"""
from __future__ import annotations

import backend.llm as llm
import backend.unified_agent as ua
from backend import endpoint_check, settings, verifier

_CLOCK = "Saturday 2026-10-10 20:40 (Asia/Yerevan, UTC+04:00); UTC 2026-10-10 16:40"


class _Spy:
    def __init__(self, reply):
        self.reply, self.seen = reply, {}

    def call_json(self, task_type, system, user, **kw):
        self.seen.update(system=system, user=user)
        return self.reply


def test_the_clock_line_is_in_the_users_zone(monkeypatch):
    monkeypatch.setattr(settings, "user_timezone", lambda: "Asia/Yerevan")
    line = settings.clock_line()
    assert "Asia/Yerevan" in line and "UTC+04:00" in line and "UTC 20" in line


def test_a_bad_zone_still_gives_a_clock(monkeypatch):
    monkeypatch.setattr(settings, "user_timezone", lambda: "Not/AZone")
    assert "UTC" in settings.clock_line()


def test_the_claims_judge_sees_the_clock(monkeypatch):
    spy = _Spy({"claims": []})
    monkeypatch.setattr(llm, "router", lambda: spy)
    monkeypatch.setattr(settings, "clock_line", lambda: _CLOCK)
    ua._ungrounded_factual_claims("what time is it in Yerevan?", "It is 20:40.")
    assert _CLOCK in spy.seen["user"]
    assert "ASSISTANT'S CLOCK" in spy.seen["system"]


def test_the_verifier_sees_the_clock(monkeypatch):
    spy = _Spy({"verified_claims": [], "unverified_claims": [], "contradictions": [],
                "projections": [], "notes_used": []})
    monkeypatch.setattr(verifier, "router", lambda: spy)
    monkeypatch.setattr(settings, "clock_line", lambda: _CLOCK)
    verifier.verify("what time is it?", "It is 20:40.", "", [],
                    tool_context="web_search: no results")
    assert _CLOCK in spy.seen["user"]


def test_the_action_judge_knows_the_conversation_is_recorded(monkeypatch):
    spy = _Spy({"unbacked_claim": ""})
    monkeypatch.setattr(llm, "router", lambda: spy)
    endpoint_check.unbacked_action_claim(
        "remember the codeword for this conversation: APRICOT", "Confirmed: APRICOT.", [])
    assert "THIS conversation" in spy.seen["system"]
