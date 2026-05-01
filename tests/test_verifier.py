from unittest.mock import patch

from backend.llm import TaskType
from backend.verifier import verify


def test_verifier_no_notes():
    res = verify("вопрос", "ответ", "", [])
    assert res.confidence == 0
    assert res.unverified_claims


def test_verifier_parses_json():
    # Confidence is computed in Python from claim list lengths, not from the LLM.
    # Formula: 100 * verified / (verified + unverified + 2 * contradictions)
    # Here: 100 * 1 / (1 + 1 + 0) = 50
    fake_json = {
        "verified_claims": ["A"],
        "unverified_claims": ["B"],
        "contradictions": [],
        "notes_used": ["RS-485"],
    }

    class FakeRouter:
        def call_json(self, task_type, system, user, **kw):
            assert task_type == TaskType.VERIFICATION
            return fake_json

    with patch("backend.verifier.router", return_value=FakeRouter()):
        res = verify("q", "a", "some notes", ["RS-485"])

    assert res.confidence == 50
    assert res.verified_claims == ["A"]
    assert res.unverified_claims == ["B"]


def test_verifier_confidence_formula():
    """Deterministic confidence: contradictions weighted 2x, no LLM-supplied number."""
    cases = [
        # (verified, unverified, contradictions) -> expected confidence
        ([], [], [], 0),  # nothing said -> 0
        (["a"], [], [], 100),  # all verified -> 100
        (["a", "b", "c"], ["x"], [], 75),  # 100*3/4 = 75
        (["a"], [], ["bad"], 33),  # 100*1/(1+0+2) = 33.33 -> 33
        ([], ["x", "y"], [], 0),  # nothing verified -> 0
    ]
    for verified, unverified, contradictions, expected in cases:
        fake_json = {
            "verified_claims": verified,
            "unverified_claims": unverified,
            "contradictions": contradictions,
            "notes_used": [],
        }

        class FakeRouter:
            def __init__(self, payload):
                self.payload = payload

            def call_json(self, task_type, system, user, **kw):
                return self.payload

        with patch("backend.verifier.router", return_value=FakeRouter(fake_json)):
            res = verify("q", "a", "notes", [])
        assert res.confidence == expected, (
            f"({verified},{unverified},{contradictions}) -> {res.confidence}, expected {expected}"
        )
