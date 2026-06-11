"""_auto_recall_block surfaces consolidated facts (audit 2026-06-11).

Pre-fix: memory_facts.jsonl (1400+ rows distilled by nightly
consolidation) was reachable only via an explicit `search_facts`
tool call. Now the pre-flight auto-recall block merges top fact
hits next to note hits, so consolidated memory influences every
turn without the LLM having to think of the tool.
"""
from __future__ import annotations

import pytest


class _NoteEntry:
    def __init__(self, topic):
        self.topic = topic
        self.category = "tech"
        self.path = f"/notes/{topic}.md"


class _NoteHit:
    def __init__(self, topic):
        self.entry = _NoteEntry(topic)
        self.score = 0.9
        self.source = "vector"


def test_facts_appended_to_recall_block(monkeypatch):
    from backend import unified_agent as ua

    import backend.hybrid_searcher as _hs
    class _H:
        def search(self, q, limit=3):
            return [_NoteHit("tailscale-setup")]
    monkeypatch.setattr(_hs, "HYBRID", _H())

    import backend.fact_search as _fs
    monkeypatch.setattr(
        _fs, "search_facts",
        lambda q, limit=5: [
            {"summary": "User runs Hrant on Ubuntu box 100.124.210.21."},
            {"summary": "User prefers male TTS voice at +20% rate."},
        ],
    )

    block = ua._auto_recall_block("how do I reach the ubuntu server again?")
    assert "tailscale-setup" in block
    assert "Long-term facts (consolidated memory):" in block
    assert "100.124.210.21" in block
    assert "male TTS voice" in block


def test_facts_alone_produce_block_without_notes(monkeypatch):
    """Fact hits must surface even when note search finds nothing."""
    from backend import unified_agent as ua

    import backend.hybrid_searcher as _hs
    class _H:
        def search(self, q, limit=3):
            return []
    monkeypatch.setattr(_hs, "HYBRID", _H())

    import backend.fact_search as _fs
    monkeypatch.setattr(
        _fs, "search_facts",
        lambda q, limit=5: [{"summary": "User favorite color is teal."}],
    )

    block = ua._auto_recall_block("what color should the dashboard be?")
    assert "teal" in block


def test_no_hits_no_block(monkeypatch):
    from backend import unified_agent as ua

    import backend.hybrid_searcher as _hs
    class _H:
        def search(self, q, limit=3):
            return []
    monkeypatch.setattr(_hs, "HYBRID", _H())

    import backend.fact_search as _fs
    monkeypatch.setattr(_fs, "search_facts", lambda q, limit=5: [])

    assert ua._auto_recall_block("a question long enough to search") == ""


def test_fact_search_exception_degrades_to_notes_only(monkeypatch):
    from backend import unified_agent as ua

    import backend.hybrid_searcher as _hs
    class _H:
        def search(self, q, limit=3):
            return [_NoteHit("docker-notes")]
    monkeypatch.setattr(_hs, "HYBRID", _H())

    import backend.fact_search as _fs
    def _boom(q, limit=5):
        raise RuntimeError("embedder exploded")
    monkeypatch.setattr(_fs, "search_facts", _boom)

    block = ua._auto_recall_block("how to restart the docker container?")
    assert "docker-notes" in block
    assert "Long-term facts" not in block
