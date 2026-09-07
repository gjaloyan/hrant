"""The fast lane's two remaining ways to call unchecked text verified.

From the 2026-09-07 re-audit.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import backend.unified_agent as ua


class _Agent:
    def progress(self, *a, **k):
        pass


@pytest.mark.parametrize("reply", [None, {}, [], {"other": 1}, {"claims": "x"}])
def test_a_judge_that_did_not_answer_is_not_a_clean_bill(reply):
    """Only an exception was caught. A reply of the wrong shape — no
    `claims` key, a bare list, None — read as "checked, nothing to
    settle", which is the same false confidence a crashed judge used to
    produce."""
    with patch.object(ua, "_claims_judge_call", return_value=reply):
        got = ua._ungrounded_factual_claims("q", "an answer")
    assert got.status == "failed", reply
    assert got.claims == []


def test_a_real_empty_verdict_is_still_a_pass():
    with patch.object(ua, "_claims_judge_call", return_value={"claims": []}):
        got = ua._ungrounded_factual_claims("q", "an answer")
    assert (got.status, got.claims) == ("checked", [])


def test_the_redraft_is_checked_not_the_draft_it_replaced():
    """The audit's case: the search said 500, the rewrite kept 123 and
    added a new unbacked sentence, and the turn reported 85/verified —
    because the only check had run one draft earlier."""
    agent = _Agent()
    seen = []

    def judge(task, answer):
        seen.append(answer)
        if answer == "draft":
            return {"claims": ["the price is 123"]}
        return {"claims": ["the price is still 123", "and a new one"]}

    with patch.object(ua, "_claims_judge_call", side_effect=judge), \
         patch.object(ua, "_web_search_for_lane",
                      return_value='[{"title": "t", "url": "http://a.example", "snippet": "the price is 500"}, {"title": "u", "url": "http://b.example", "snippet": "also 500"}]'), \
         patch.object(ua, "_try_chat_path", return_value="redrafted"):
        out = ua._ground_fast_answer(task="q", answer="draft", agent=agent,
                                     speaker_id="s", snapshot="", convo="")

    assert out == "redrafted"
    assert "redrafted" in seen, "the final text was never judged"
    conf, status = ua._lane_check_state(agent)
    assert status == "partial", "claims left in the final text -> partial"
    assert conf < ua.CHECKED_CONFIDENCE
    assert agent._claim_leftovers == ["the price is still 123", "and a new one"]


def test_a_clean_redraft_is_verified():
    agent = _Agent()

    def judge(task, answer):
        return {"claims": ["the price is 123"]} if answer == "draft" else {"claims": []}

    with patch.object(ua, "_claims_judge_call", side_effect=judge), \
         patch.object(ua, "_web_search_for_lane",
                      return_value='[{"title": "t", "url": "http://a.example", "snippet": "the price is 123"}, {"title": "u", "url": "http://b.example", "snippet": "also 123"}]'), \
         patch.object(ua, "_try_chat_path", return_value="redrafted"):
        ua._ground_fast_answer(task="q", answer="draft", agent=agent,
                               speaker_id="s", snapshot="", convo="")

    assert ua._lane_check_state(agent) == (ua.CHECKED_CONFIDENCE, "verified")
