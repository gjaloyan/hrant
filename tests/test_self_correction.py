"""Unit tests for unbacked_action_claim in backend.endpoint_check.

Tests are deterministic: the router is monkeypatched so no network
calls are made. Covers three scenarios:
  1. Judge returns a non-empty unbacked_claim -> function returns it.
  2. Empty answer -> "" without calling the LLM at all.
  3. LLM raises an exception -> "" (fail open).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRouter:
    """Minimal router stub that implements call_json."""

    def __init__(self, return_value=None, raise_exc=None):
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.calls: list[tuple] = []

    def call_json(self, task_type, system, user, **kw):
        self.calls.append((task_type, system, user))
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unbacked_claim_returned_when_judge_fires(monkeypatch):
    """When the judge returns a non-empty unbacked_claim the function
    returns that string unchanged."""
    fake = _FakeRouter(return_value={"unbacked_claim": "saving user name"})
    monkeypatch.setattr("backend.llm.router", lambda: fake)

    from backend.endpoint_check import unbacked_action_claim

    result = unbacked_action_claim(
        task="Remember that my name is Gor",
        answer="Got it, I've saved that your name is Gor.",
        tool_names=[],
    )
    assert result == "saving user name"
    assert len(fake.calls) == 1


def test_empty_answer_returns_empty_without_llm_call(monkeypatch):
    """An empty (or whitespace-only) answer must short-circuit to ""
    without touching the router at all."""
    fake = _FakeRouter(return_value={"unbacked_claim": "should not be reached"})
    monkeypatch.setattr("backend.llm.router", lambda: fake)

    from backend.endpoint_check import unbacked_action_claim

    assert unbacked_action_claim(task="do something", answer="", tool_names=[]) == ""
    assert unbacked_action_claim(task="do something", answer="   ", tool_names=[]) == ""
    # Router must NOT have been called.
    assert fake.calls == []


def test_llm_exception_fails_open(monkeypatch):
    """When the judge raises any exception the function returns ""
    (fail open) so infra failures never fabricate corrections."""
    fake = _FakeRouter(raise_exc=RuntimeError("network timeout"))
    monkeypatch.setattr("backend.llm.router", lambda: fake)

    from backend.endpoint_check import unbacked_action_claim

    result = unbacked_action_claim(
        task="Set my language to Spanish",
        answer="Done, I've updated your language setting to Spanish.",
        tool_names=[],
    )
    assert result == ""


def test_backed_claim_returns_empty(monkeypatch):
    """When the judge returns an empty unbacked_claim (action was backed
    by a tool call) the function returns ""."""
    fake = _FakeRouter(return_value={"unbacked_claim": ""})
    monkeypatch.setattr("backend.llm.router", lambda: fake)

    from backend.endpoint_check import unbacked_action_claim

    result = unbacked_action_claim(
        task="Remember that my name is Gor",
        answer="Got it, I've saved that your name is Gor.",
        tool_names=["save_user_fact"],
    )
    assert result == ""


def test_informational_answer_returns_empty(monkeypatch):
    """A purely informational answer (no action claimed) returns ""."""
    fake = _FakeRouter(return_value={"unbacked_claim": ""})
    monkeypatch.setattr("backend.llm.router", lambda: fake)

    from backend.endpoint_check import unbacked_action_claim

    result = unbacked_action_claim(
        task="What is the capital of France?",
        answer="The capital of France is Paris.",
        tool_names=[],
    )
    assert result == ""


def test_judge_missing_key_returns_empty(monkeypatch):
    """If the judge returns JSON without the unbacked_claim key the
    function safely returns ""."""
    fake = _FakeRouter(return_value={"some_other_key": "value"})
    monkeypatch.setattr("backend.llm.router", lambda: fake)

    from backend.endpoint_check import unbacked_action_claim

    result = unbacked_action_claim(
        task="Run a script",
        answer="Done, I ran your script.",
        tool_names=[],
    )
    assert result == ""
