"""Memory extractor must not absorb agent-derived claims when the
verifier flagged the answer as unreliable. The user-stated message
side is always safe to mine; the agent_answer side is dropped on
low confidence or any contradictions.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.memory_extractor import MEMORY


def _capture_extraction_prompt(monkeypatch):
    """Helper: stub the LLM so we can inspect what got sent for extraction."""
    captured = {"prompt": None}

    class _Router:
        def call_json(self, task_type, system, user, **kw):
            captured["prompt"] = user
            return {"has_facts": False, "facts": []}

    monkeypatch.setattr("backend.memory_extractor.router", lambda: _Router())
    return captured


def test_high_confidence_answer_is_extracted(monkeypatch):
    captured = _capture_extraction_prompt(monkeypatch)
    MEMORY.extract_and_store(
        "user said something",
        "agent answered with a fact",
        intent="task",
        confidence=90,
        contradictions=0,
    )
    assert "agent answered with a fact" in captured["prompt"]


def test_low_confidence_answer_is_dropped(monkeypatch):
    """confidence < 60 → agent_answer NOT in the extraction prompt."""
    captured = _capture_extraction_prompt(monkeypatch)
    MEMORY.extract_and_store(
        "user said something",
        "agent guessed wrong fact here",
        intent="task",
        confidence=40,
        contradictions=0,
    )
    assert "agent guessed wrong fact here" not in captured["prompt"]
    # User message side still mined
    assert "user said something" in captured["prompt"]


def test_contradictions_drop_agent_answer(monkeypatch):
    """Any contradictions count → agent_answer dropped even if confidence
    is otherwise OK. Contradictions are evidence-against, not just
    absence-of-evidence; we can't trust the answer."""
    captured = _capture_extraction_prompt(monkeypatch)
    MEMORY.extract_and_store(
        "user fact",
        "agent claim that contradicts source",
        intent="task",
        confidence=85,  # high, but...
        contradictions=2,  # ...verifier said no
    )
    assert "agent claim that contradicts source" not in captured["prompt"]


def test_default_args_keep_old_behavior(monkeypatch):
    """Backward compat: callers that don't pass confidence/contradictions
    get the previous behavior (always extract from both sides)."""
    captured = _capture_extraction_prompt(monkeypatch)
    MEMORY.extract_and_store("user msg", "agent answer", intent="task")
    assert "agent answer" in captured["prompt"]
