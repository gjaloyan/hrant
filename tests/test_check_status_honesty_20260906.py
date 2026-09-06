"""A verification that never ran must not report itself as 85.

From the GPT-6 Astra audit, 2026-09-05, finding 3. `confidence` starts
at 85 and stays there whether the verifier agreed, found nothing to
check, was skipped for want of tool output, or crashed. The auditor's
point is not that the answers were wrong — it is that the number is a
technical default being read as measured quality, by the WebUI badge,
by the skill statistics, and by anything gating on it.

The policy stays soft, deliberately: an answer is still served when the
judge is unavailable. Only the label changes.
"""
from __future__ import annotations

from unittest.mock import patch

import backend.unified_agent as ua


def test_a_crashed_judge_is_not_an_empty_verdict():
    """The whole defect in one assertion: 'nothing to check' and 'could
    not check' used to be the same empty list."""
    with patch.object(ua, "_claims_judge_call",
                      side_effect=RuntimeError("provider down")):
        got = ua._ungrounded_factual_claims("q", "an answer")
    assert got.claims == []
    assert got.status == "failed"

    with patch.object(ua, "_claims_judge_call", return_value={"claims": []}):
        got = ua._ungrounded_factual_claims("q", "an answer")
    assert got.claims == []
    assert got.status == "checked"


def test_an_empty_answer_is_not_applicable():
    got = ua._ungrounded_factual_claims("q", "   ")
    assert got.status == "not_applicable"


class _Agent:
    def progress(self, *a, **k):
        pass


def test_the_lane_serves_the_answer_but_labels_it_failed():
    """Soft policy, honest label — the audit asked for exactly this
    split, and it is also what the owner asked for generally: the agent
    keeps working, the report stops pretending."""
    agent = _Agent()
    with patch.object(ua, "_claims_judge_call",
                      side_effect=RuntimeError("provider down")):
        out = ua._ground_fast_answer(task="q", answer="draft", agent=agent,
                                     speaker_id="s", snapshot="", convo="")
    assert out == "draft", "the answer is still served"
    assert agent._claim_check == "failed"

    conf, status = ua._lane_check_state(agent)
    assert status == "failed"
    assert conf == ua.UNMEASURED_CONFIDENCE
    assert conf < 85


def test_a_real_check_still_reports_85():
    agent = _Agent()
    with patch.object(ua, "_claims_judge_call", return_value={"claims": []}):
        out = ua._ground_fast_answer(task="q", answer="draft", agent=agent,
                                     speaker_id="s", snapshot="", convo="")
    assert out == "draft"
    conf, status = ua._lane_check_state(agent)
    assert (conf, status) == (ua.CHECKED_CONFIDENCE, "verified")


def test_a_lane_that_never_reached_the_judge_says_not_checked():
    conf, status = ua._lane_check_state(_Agent())
    assert status == "not_checked"
    assert conf == ua.UNMEASURED_CONFIDENCE


def test_the_unmeasured_marker_sits_below_the_finetune_gate():
    """Output nothing verified must not become training data for the
    next model. The collector's threshold is the reason this constant
    has the value it has."""
    from backend.finetune import FinetuneStore
    assert ua.UNMEASURED_CONFIDENCE < FinetuneStore().confidence_threshold


def test_the_full_cycle_keeps_its_scalar_and_tells_the_truth_beside_it():
    """`MetaLearner.analyze_failure` fires under 60 and costs an LLM
    call per turn; the memory extractor stops trusting the answer at the
    same line. "Nobody checked this" is not "this went wrong", so the
    full cycle reports the unchanged number with an honest status rather
    than dropping the scalar and manufacturing failures.

    The fast lane can drop it because it returns before either consumer.
    """
    from backend.models import VerificationResult
    vr = VerificationResult(confidence=85, check_status="not_checked")
    assert vr.confidence >= 60, "must not read as a failure"
    assert vr.check_status == "not_checked", "must not read as verified"


def test_the_field_defaults_to_unlabelled_for_old_turns():
    from backend.models import VerificationResult
    assert VerificationResult(confidence=85).check_status is None


def test_the_lane_usage_dict_actually_fits_the_answer_field():
    """The artifact carried these numbers all along; the ANSWER did not,
    so about a third of turns reported `token_usage: null` and read as
    free. The conversion is wrapped in a try/except so a shape mismatch
    would fail silently back to null — pin the shapes instead.
    """
    from backend.llm import TokenTracker
    from backend.models import AgentAnswer, TokenUsage, VerificationResult

    tracker = TokenTracker()
    tracker.reset_request()
    tracker.record(task_type="chat", model="m", provider="p",
                   usage={"input_tokens": 10, "output_tokens": 4})
    usage = tracker.request_usage()

    assert set(usage) <= set(TokenUsage.model_fields), (
        "request_usage() grew a key TokenUsage cannot take; the lane's "
        "conversion would swallow it and report null"
    )
    a = AgentAnswer(
        answer="x", is_chat=True,
        verification=VerificationResult(confidence=85),
        token_usage=TokenUsage(**usage),
    )
    assert a.token_usage.input_tokens == 10
    assert a.token_usage.llm_calls == 1


def test_a_claim_the_lane_could_not_afford_to_check_is_not_dropped():
    """The judge reports up to three claims; the lane searches two. The
    third used to vanish — a claim known to be unbacked, left standing
    in the answer with nothing said about it (2026-09-05 audit, finding
    4). The budget stays; the remainder is now on the record."""
    agent = _Agent()
    three = ["claim one", "claim two", "claim three"]
    with patch.object(ua, "_claims_judge_call",
                      return_value={"claims": three}), \
         patch.object(ua, "_web_search_for_lane", return_value=[]), \
         patch.object(ua, "_try_chat_path", return_value="redrafted"):
        ua._ground_fast_answer(task="q", answer="draft", agent=agent,
                               speaker_id="s", snapshot="", convo="")

    assert agent._claim_leftovers == ["claim three"]
    conf, status = ua._lane_check_state(agent)
    assert status == "partial", "not the same label as a fully settled answer"
    assert conf < ua.CHECKED_CONFIDENCE


def test_everything_checked_is_still_reported_as_verified():
    agent = _Agent()
    with patch.object(ua, "_claims_judge_call",
                      return_value={"claims": ["only one"]}), \
         patch.object(ua, "_web_search_for_lane", return_value=[]), \
         patch.object(ua, "_try_chat_path", return_value="redrafted"):
        ua._ground_fast_answer(task="q", answer="draft", agent=agent,
                               speaker_id="s", snapshot="", convo="")
    assert agent._claim_leftovers == []
    assert ua._lane_check_state(agent)[1] == "verified"
