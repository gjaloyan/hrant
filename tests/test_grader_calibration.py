"""Grader calibration — split delivery misses from content failures.

Audit 2026-06-11: evaluator avg confidence was 47.7 because the
endpoint cap clips every delivery miss to 30, and the meta-learner
then routed those turns through the LLM failure analyst — which,
seeing a fine answer with a low score, guessed "hallucination"
(35 of 96 in the 2026-06-10 self-reflection were this mislabel).
The learning loop was training on grader strictness, not on real
answer badness.

The split: `confidence` stays the conservative display scalar;
`content_confidence` preserves the pre-clip claim score;
`endpoint_met` carries the delivery judgment. The meta-learner
records known delivery misses directly (root_cause=endpoint_miss)
WITHOUT an LLM call.
"""
from __future__ import annotations

import json

import pytest

from backend.models import VerificationResult


def _ml(tmp_path):
    from backend.meta_learner import MetaLearner
    return MetaLearner(
        path=tmp_path / "error_log.jsonl",
        patterns_path=tmp_path / "error_patterns.json",
    )


def test_verification_result_split_fields_default_none():
    """Old serialized artifacts (no new fields) must load unchanged."""
    vr = VerificationResult(confidence=85)
    assert vr.content_confidence is None
    assert vr.endpoint_met is None
    # Round-trip via dict (as saved in turn artifacts).
    vr2 = VerificationResult(**{"confidence": 40, "contradictions": ["x"]})
    assert vr2.content_confidence is None


def test_eval_entry_carries_split_fields():
    from backend.evaluator import EvalEntry

    e = EvalEntry(
        question="run the bench", intent="unified", confidence=30,
        topics_used=[], contradictions=1, unverified=0, verified=2,
        content_confidence=85, endpoint_met=False,
    )
    d = e.to_dict()
    assert d["confidence"] == 30
    assert d["content_confidence"] == 85
    assert d["endpoint_met"] is False
    # Old rows without the keys read fine via .get().
    old_row = {"confidence": 75}
    assert old_row.get("content_confidence") is None


def test_endpoint_miss_with_good_content_skips_llm(tmp_path, monkeypatch):
    """confidence=30 (clipped), content_confidence=85, endpoint_met=False
    -> root_cause recorded directly as endpoint_miss; the LLM analyst
    must NOT be called."""
    import backend.meta_learner as ml_mod

    def _no_llm():
        raise AssertionError("LLM analyst must not run for a known miss")

    monkeypatch.setattr(ml_mod, "router", _no_llm)

    ml = _ml(tmp_path)
    vr = VerificationResult(
        confidence=30, content_confidence=85, endpoint_met=False,
        contradictions=["endpoint_not_met: action-verb request ..."],
    )
    analysis = ml.analyze_failure("run the bench", "I looked around.", vr)

    assert analysis is not None
    assert analysis["root_cause"] == "endpoint_miss"
    # Logged to error_log.jsonl with the precomputed analysis.
    rows = [
        json.loads(l)
        for l in (tmp_path / "error_log.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["analysis"]["root_cause"] == "endpoint_miss"
    assert rows[0]["content_confidence"] == 85


def test_bad_content_still_routes_to_llm_analyst(tmp_path, monkeypatch):
    """Genuinely low content score keeps today's LLM analysis path —
    even when the endpoint also missed (content problem dominates)."""
    import backend.meta_learner as ml_mod

    calls = {"n": 0}

    def _fake_router():
        class R:
            @staticmethod
            def call_json(*a, **kw):
                calls["n"] += 1
                return {
                    "root_cause": "hallucination",
                    "fix_action": "none",
                    "severity": 6,
                }
        return R()

    monkeypatch.setattr(ml_mod, "router", _fake_router)

    ml = _ml(tmp_path)
    vr = VerificationResult(
        confidence=30, content_confidence=40, endpoint_met=False,
        unverified_claims=["made-up API"],
    )
    analysis = ml.analyze_failure("explain the API", "It has a frobnicate().", vr)

    assert calls["n"] == 1
    assert analysis["root_cause"] == "hallucination"


def test_unclipped_low_confidence_routes_to_llm(tmp_path, monkeypatch):
    """No clip happened (content_confidence None) and confidence < 60
    -> the content itself is the problem; LLM path as before."""
    import backend.meta_learner as ml_mod

    calls = {"n": 0}

    def _fake_router():
        class R:
            @staticmethod
            def call_json(*a, **kw):
                calls["n"] += 1
                return {"root_cause": "wrong_reasoning", "fix_action": "none",
                        "severity": 5}
        return R()

    monkeypatch.setattr(ml_mod, "router", _fake_router)

    ml = _ml(tmp_path)
    vr = VerificationResult(confidence=45)
    analysis = ml.analyze_failure("question", "answer", vr)
    assert calls["n"] == 1
    assert analysis["root_cause"] == "wrong_reasoning"


def test_high_confidence_still_returns_none(tmp_path):
    ml = _ml(tmp_path)
    vr = VerificationResult(confidence=85, endpoint_met=True)
    assert ml.analyze_failure("q", "a", vr) is None
