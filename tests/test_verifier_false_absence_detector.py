"""Belt-and-suspenders: deterministic detector for the
'agent claims X is missing, but X is in the code' hallucination.
The LLM verifier was observed missing these even with explicit
guidance, so we add a Python-side regex sweep that promotes
detected matches to contradictions automatically.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.verifier import detect_false_absence_contradictions, verify


def test_detects_add_x_when_x_is_in_identifiers():
    answer = "Recommendation: add _save_preference handler to the router."
    out = detect_false_absence_contradictions(
        answer, ["_save_preference", "RouterTable"],
    )
    assert out, "should flag 'add _save_preference' when it is in identifiers"
    assert "_save_preference" in out[0]


def test_detects_missing_x_in_russian():
    answer = "Проблема: GOALS.should_check_proactive не вызывается в agent.py"
    out = detect_false_absence_contradictions(
        answer, ["should_check_proactive", "GOALS"],
    )
    assert out
    assert "should_check_proactive" in out[0]


def test_detects_x_is_missing_with_backticks():
    answer = "`TokenTracker` doesn't exist in the code yet."
    out = detect_false_absence_contradictions(answer, ["TokenTracker"])
    assert out
    assert "TokenTracker" in out[0]


def test_no_match_when_identifier_not_extracted():
    """If the alleged-missing thing isn't in the extracted identifiers,
    we have no evidence it's actually present — leave it alone."""
    answer = "Recommendation: add SomeNewThing."
    out = detect_false_absence_contradictions(answer, ["other_class", "Foo"])
    assert out == []


def test_no_match_on_legitimate_add_suggestions():
    """When the agent suggests adding something we DON'T have an
    extracted identifier for, that's a legitimate suggestion — not
    a false-absence hallucination."""
    answer = "I'd recommend adding rate limiting to the API."
    out = detect_false_absence_contradictions(answer, ["VerifierResult", "compute_confidence"])
    assert out == []


def test_normalizes_case_and_leading_underscore():
    """Same concept can appear under different conventions:
      FILE_CACHE (module const) vs _file_cache (private dict)
      TokenTracker (class) vs _token_tracker (instance var)
    The detector must match across these forms."""
    answer = (
        "Recommendation: add `_file_cache` to avoid double reads. "
        "Also create `_token_tracker` for usage logging."
    )
    out = detect_false_absence_contradictions(
        answer, ["FILE_CACHE", "TokenTracker"],
    )
    assert len(out) == 2
    # Both source-form and answer-form should appear in the message
    # so the operator can see what got matched.
    assert any("_file_cache" in c and "FILE_CACHE" in c for c in out)
    assert any("_token_tracker" in c and "TokenTracker" in c for c in out)


def test_exact_match_no_double_naming_in_message():
    """When candidate == identifier exactly, the contradiction message
    should not say 'matches' redundantly."""
    answer = "Add TokenTracker class."
    out = detect_false_absence_contradictions(answer, ["TokenTracker"])
    assert len(out) == 1
    assert "matches" not in out[0]


def test_dedups_same_identifier():
    """Two phrasings of the same fault don't double-count."""
    answer = (
        "Add TokenTracker here. Also note: TokenTracker is missing from llm.py."
    )
    out = detect_false_absence_contradictions(answer, ["TokenTracker"])
    assert len(out) == 1


def test_empty_inputs_safe():
    assert detect_false_absence_contradictions("", ["foo"]) == []
    assert detect_false_absence_contradictions("anything", []) == []
    assert detect_false_absence_contradictions(None, ["foo"]) == []  # type: ignore[arg-type]


def test_ignores_short_identifiers():
    """We don't want 'add a' or 'add to' triggering on stop-words —
    the regex requires at least 3 chars after the verb."""
    answer = "add x by"
    # 'x' is too short to be an identifier match
    out = detect_false_absence_contradictions(answer, ["x"])
    assert out == []


def test_verify_promotes_auto_contradiction_to_result():
    """End-to-end: when the LLM verifier misses a false-absence claim
    but the deterministic detector catches it, the contradiction lands
    in the result and confidence drops accordingly."""
    # The LLM (mocked) is sloppy and returns no contradictions despite
    # the answer claiming 'reject is missing' against tool output that
    # has 'class Reject' / a 'reject' constant.
    fake_json = {
        "verified_claims": [],
        "unverified_claims": [],
        "contradictions": [],
        "notes_used": [],
    }

    class FakeRouter:
        def call_json(self, task_type, system, user, **kw):
            return fake_json

    answer = "fix: add TokenTracker to llm.py."
    tool_output = (
        'class TokenTracker:\n'
        '    def record(self, ...): ...\n'
    )

    with patch("backend.verifier.router", return_value=FakeRouter()):
        res = verify("review", answer, "", [], tool_context=tool_output)

    assert res.contradictions, "deterministic detector must add a contradiction"
    assert any("TokenTracker" in c for c in res.contradictions)
    # 0 verified, 0 unverified, 1 contradiction → confidence 0.
    assert res.confidence == 0


def test_verify_dedups_against_llm_contradictions():
    """If the LLM already flagged the same identifier as a
    contradiction, the auto detector must not double-count it —
    confidence formula would over-penalize otherwise."""
    fake_json = {
        "verified_claims": [],
        "unverified_claims": [],
        "contradictions": [
            "Answer says 'TokenTracker is missing' but llm.py defines it.",
        ],
        "notes_used": [],
    }

    class FakeRouter:
        def call_json(self, task_type, system, user, **kw):
            return fake_json

    answer = "Recommendation: add TokenTracker to handler."
    tool_output = "class TokenTracker: ...\n"

    with patch("backend.verifier.router", return_value=FakeRouter()):
        res = verify("q", answer, "", [], tool_context=tool_output)

    # Only ONE contradiction, not two — the auto detector saw
    # 'TokenTracker' but the LLM's contradiction already mentions it,
    # so dedup.
    assert len(res.contradictions) == 1
