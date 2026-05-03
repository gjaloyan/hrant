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


def test_verifier_system_prompt_handles_negative_existence():
    """Review-mode regression: the verifier prompt must explicitly tell
    the LLM to look for contradictions when the assistant claims a thing
    is missing or proposes a 'fix' that adds it. Without this guidance
    the verifier was rubber-stamping hallucinations like 'add reject
    category' (already there) and 'add fuzzy_threshold check' (already
    there) — both real cases observed in production self-reviews."""
    from backend.verifier import VERIFIER_SYSTEM
    s = VERIFIER_SYSTEM.lower()
    # Contradictions are the right bucket for false-absence claims.
    assert "contradiction" in s
    # Explicit guidance on negative existence / "missing X" patterns.
    assert "missing" in s
    # Explicit guidance on "fix" suggestions for non-existent problems.
    assert "fix" in s
    # The "absence of evidence is not evidence of absence" rule.
    assert "absence" in s


def test_verifier_marks_false_missing_claim_as_contradiction():
    """End-to-end: when the assistant claims a feature is missing but
    the tool output shows it IS present, the verifier should produce a
    contradiction (low confidence), NOT a verified claim."""
    # Simulate: assistant said "code has no reject category, add it".
    # Tool output (file contents the assistant supposedly read) shows
    # the reject category is right there in the source.
    answer = (
        "Problem #1: agent.py is missing the 'reject' category. "
        "Fix: add a reject branch to _save_preference."
    )
    tool_output = (
        "agent.py:248: \"category\": \"language\" | \"style\" | \"about_user\" | \"rule\" | \"reject\",\n"
        "agent.py:750: return \"reject\", task.strip(), ...\n"
    )

    # An attentive verifier sees the contradiction.
    fake_json = {
        "verified_claims": [],
        "unverified_claims": [],
        "contradictions": [
            "Answer claims 'reject' category is missing, but tool output "
            "shows it defined at agent.py:248 and used at :750.",
        ],
        "notes_used": [],
    }

    class FakeRouter:
        def call_json(self, task_type, system, user, **kw):
            # The verifier MUST be passing the tool_output to the LLM
            # — otherwise it can't catch the contradiction.
            assert "agent.py:248" in user
            return fake_json

    from unittest.mock import patch as _p
    with _p("backend.verifier.router", return_value=FakeRouter()):
        res = verify("review my code", answer, "irrelevant note", [], tool_context=tool_output)

    assert res.contradictions, "verifier should flag a contradiction here"
    # confidence formula: 0 verified, 0 unverified, 1 contradiction
    # => 100 * 0 / (0 + 0 + 2) = 0
    assert res.confidence == 0


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
