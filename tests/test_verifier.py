from unittest.mock import patch

from backend.llm import TaskType
from backend.verifier import verify


def test_verifier_no_notes():
    res = verify("вопрос", "ответ", "", [])
    assert res.confidence == 0
    assert res.unverified_claims


def test_verifier_parses_json():
    fake_json = {
        "confidence": 88,
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

    assert res.confidence == 88
    assert res.verified_claims == ["A"]
    assert res.unverified_claims == ["B"]
