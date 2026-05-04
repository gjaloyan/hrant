"""Meta-learner: extract_patterns must run automatically every Nth
analyzed failure, not wait for a manual trigger. Without this the
feedback loop never closes — patterns accumulate but never become
goals.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.meta_learner import MetaLearner
from backend.models import VerificationResult


class _FailingResult(VerificationResult):
    pass


def _bad(confidence: int = 30) -> VerificationResult:
    return VerificationResult(
        confidence=confidence,
        verified_claims=[],
        unverified_claims=["something"],
        contradictions=["something else"],
    )


def test_extract_patterns_triggered_every_nth_failure(tmp_path, monkeypatch):
    ml = MetaLearner(
        path=tmp_path / "log.jsonl",
        patterns_path=tmp_path / "patterns.json",
    )
    ml.AUTO_EXTRACT_EVERY_N_FAILURES = 3
    extract_calls = {"n": 0}

    def fake_extract():
        extract_calls["n"] += 1
        return []

    # Stub the LLM call inside analyze_failure so it returns a fake analysis.
    fake_analysis = {
        "root_cause": "missing knowledge",
        "fix_action": "none",
        "severity": 5,
    }

    class _Router:
        def call_json(self, *a, **kw):
            return fake_analysis

    monkeypatch.setattr(ml, "extract_patterns", fake_extract)

    with patch("backend.meta_learner.router", return_value=_Router()):
        ml.analyze_failure("q1", "a1", _bad())  # count=1
        ml.analyze_failure("q2", "a2", _bad())  # count=2
        assert extract_calls["n"] == 0, "should NOT trigger before Nth"
        ml.analyze_failure("q3", "a3", _bad())  # count=3 → trigger
        assert extract_calls["n"] == 1
        ml.analyze_failure("q4", "a4", _bad())  # count=4
        ml.analyze_failure("q5", "a5", _bad())  # count=5
        ml.analyze_failure("q6", "a6", _bad())  # count=6 → trigger again
        assert extract_calls["n"] == 2


def test_no_auto_extract_for_high_confidence_results(tmp_path, monkeypatch):
    """analyze_failure short-circuits on confidence >= 60 — those
    aren't failures, so they MUST NOT bump the counter."""
    ml = MetaLearner(
        path=tmp_path / "log.jsonl",
        patterns_path=tmp_path / "patterns.json",
    )
    ml.AUTO_EXTRACT_EVERY_N_FAILURES = 2
    calls = {"n": 0}
    monkeypatch.setattr(ml, "extract_patterns", lambda: calls.update(n=calls["n"] + 1) or [])
    for i in range(10):
        ml.analyze_failure(f"q{i}", "ok", _bad(confidence=85))  # not a failure
    assert calls["n"] == 0
    assert ml._failure_count == 0


def test_auto_extract_swallows_errors(tmp_path, monkeypatch):
    """If extract_patterns raises (e.g., LLM offline), analyze_failure
    must still return its analysis — the counter is best-effort."""
    ml = MetaLearner(
        path=tmp_path / "log.jsonl",
        patterns_path=tmp_path / "patterns.json",
    )
    ml.AUTO_EXTRACT_EVERY_N_FAILURES = 1

    def boom():
        raise RuntimeError("LLM down")

    monkeypatch.setattr(ml, "extract_patterns", boom)

    class _Router:
        def call_json(self, *a, **kw):
            return {"root_cause": "x", "fix_action": "none", "severity": 5}

    with patch("backend.meta_learner.router", return_value=_Router()):
        result = ml.analyze_failure("q", "a", _bad())
    # analyze_failure returned successfully despite extract_patterns blowing up.
    assert result is not None
    assert result.get("root_cause") == "x"
