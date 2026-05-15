"""Smoke tests for backend.memory_extractor — fact extraction from
turns + recall queries.

Real `extract_and_store` calls the LLM; we mock the router so the
test is deterministic and offline. The pinned behaviour:

  - extract_and_store with no facts returned doesn't write anything
  - extract_and_store with facts writes to the memory log
  - recall returns dicts shaped for the agent's context loader
  - stats returns a dict (used by the WebUI Memory tab)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fresh_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import memory_extractor as _m
    # Re-point the singleton's log path; don't reload the module
    # (that pollutes every other test that imports backend.memory_extractor).
    fresh = _m.MemoryExtractor(log_path=tmp_path / "knowledge" / "memory_log.jsonl")
    monkeypatch.setattr(_m, "MEMORY", fresh)
    return _m


def test_memory_extractor_singleton_exists(fresh_memory):
    assert fresh_memory.MEMORY is not None


def test_recall_returns_list_when_no_facts(fresh_memory):
    """With no log file yet, recall must return an empty list (not
    raise). Used by the agent's context-loader before the first
    turn."""
    out = fresh_memory.MEMORY.recall("anything")
    assert isinstance(out, list)


def test_recall_block_returns_string(fresh_memory):
    """recall_block is what gets injected into the system prompt —
    must be a string (empty is fine)."""
    out = fresh_memory.MEMORY.recall_block("query")
    assert isinstance(out, str)


def test_recent_facts_returns_list(fresh_memory):
    out = fresh_memory.MEMORY.recent_facts()
    assert isinstance(out, list)


def test_stats_returns_dict(fresh_memory):
    s = fresh_memory.MEMORY.stats()
    assert isinstance(s, dict)


def test_extract_and_store_no_facts_writes_nothing(fresh_memory, monkeypatch):
    """LLM returns has_facts=false → no log lines added."""
    fake = MagicMock()
    fake.call_json.return_value = {"has_facts": False, "facts": []}
    # NB: memory_extractor does `from .llm import router` at module
    # load, so patching `backend.llm.router` doesn't reach it.
    # Patch the local binding instead.
    monkeypatch.setattr("backend.memory_extractor.router", lambda: fake)
    pre = len(fresh_memory.MEMORY.recent_facts())
    fresh_memory.MEMORY.extract_and_store(
        user_message="hi there",
        agent_answer="hello",
    )
    assert len(fresh_memory.MEMORY.recent_facts()) == pre


def test_extract_and_store_writes_facts(fresh_memory, monkeypatch):
    """LLM returns one fact → it lands in recent_facts."""
    fake = MagicMock()
    fake.call_json.return_value = {
        "has_facts": True,
        "facts": [{
            "summary": "User prefers Edge TTS Russian voice.",
            "triples": [["user", "prefers", "ru-RU-SvetlanaNeural"]],
            "tags": ["voice", "preferences"],
            "category": "preference",
            "confidence": 0.95,
        }],
    }
    # NB: memory_extractor does `from .llm import router` at module
    # load, so patching `backend.llm.router` doesn't reach it.
    # Patch the local binding instead.
    monkeypatch.setattr("backend.memory_extractor.router", lambda: fake)
    fresh_memory.MEMORY.extract_and_store(
        user_message="i want Russian voice",
        agent_answer="set it to ru-RU-SvetlanaNeural",
    )
    recent = fresh_memory.MEMORY.recent_facts()
    assert any("Edge TTS" in (f.get("summary") or "") for f in recent)


def test_extract_and_store_swallows_llm_error(fresh_memory, monkeypatch):
    """LLMError during extraction must not crash the turn —
    extraction is a side-effect, not a critical path. The current
    contract is to swallow LLMError specifically (other exceptions
    propagate as programming errors)."""
    from backend.llm import LLMError
    fake = MagicMock()
    fake.call_json.side_effect = LLMError("LLM down")
    monkeypatch.setattr("backend.memory_extractor.router", lambda: fake)
    # Should not raise.
    fresh_memory.MEMORY.extract_and_store(
        user_message="x", agent_answer="y",
    )
