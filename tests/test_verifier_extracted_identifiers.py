"""The verifier should not have to re-discover identifiers in 12k-char
file dumps. We pre-extract them with a regex and stuff them into the
user prompt as 'EXTRACTED IDENTIFIERS — ALREADY PRESENT IN THE CODE'
so a 'fix: add reject category' claim becomes a trivial keyword check."""
from __future__ import annotations
from unittest.mock import patch

from backend.verifier import _extract_code_identifiers, verify


def test_extract_finds_classes_and_defs():
    src = """
    class Foo:
        def bar(self):
            self.baz = 1

    def helper(): ...
    CONSTANT = 7
    """
    idents = _extract_code_identifiers(src)
    assert "Foo" in idents
    assert "bar" in idents
    assert "helper" in idents
    assert "baz" in idents
    assert "CONSTANT" in idents


def test_extract_returns_empty_for_empty_input():
    assert _extract_code_identifiers("") == []
    assert _extract_code_identifiers(None) == []


def test_extract_caps_at_max_idents():
    """A 12k-char dump shouldn't blow up the verifier prompt."""
    src = "\n".join(f"def fn_{i}(): ..." for i in range(500))
    idents = _extract_code_identifiers(src, max_idents=50)
    assert len(idents) == 50


def test_verifier_user_prompt_includes_extracted_identifiers():
    """The full verify() flow MUST surface extracted identifiers to the
    LLM via the user prompt — that's the whole point of this fix."""
    captured: dict[str, str] = {}

    class FakeRouter:
        def call_json(self, task_type, system, user, **kw):
            captured["user"] = user
            return {
                "verified_claims": [],
                "unverified_claims": [],
                "contradictions": ["claims X is missing, but X is in tool output"],
                "notes_used": [],
            }

    tool_output = '''
    class TokenTracker:
        def record(self, ...):
            self.total = 0
    '''
    with patch("backend.verifier.router", return_value=FakeRouter()):
        verify("review", "Add TokenTracker class", "", [], tool_context=tool_output)

    user = captured["user"]
    assert "EXTRACTED IDENTIFIERS" in user
    assert "TokenTracker" in user
    # The label tells the LLM how to use them.
    assert "ALREADY PRESENT" in user


def test_verifier_user_prompt_skips_identifiers_when_no_tool_output():
    """No tool output → no identifiers section. Don't pollute the
    prompt with an empty header."""
    captured: dict[str, str] = {}

    class FakeRouter:
        def call_json(self, task_type, system, user, **kw):
            captured["user"] = user
            return {
                "verified_claims": ["A"],
                "unverified_claims": [],
                "contradictions": [],
                "notes_used": [],
            }

    with patch("backend.verifier.router", return_value=FakeRouter()):
        verify("q", "a", "some notes", [])

    assert "EXTRACTED IDENTIFIERS" not in captured["user"]
