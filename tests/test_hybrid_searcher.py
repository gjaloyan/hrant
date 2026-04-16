"""Tests for backend.hybrid_searcher module."""
from dataclasses import dataclass

import pytest

from backend.hybrid_searcher import HybridSearcher
from backend.knowledge_graph import KnowledgeGraph
from backend.models import IndexEntry
from backend.searcher import SearchHit, Searcher


# ---- Fakes ----

class FakeSearcher:
    """Searcher stub that returns pre-configured hits."""

    def __init__(self, hits: list[SearchHit] | None = None):
        self._hits = hits or []
        self.threshold = 50

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        return self._hits[:limit]

    def find_best(self, topic: str) -> IndexEntry | None:
        return self._hits[0].entry if self._hits else None


def _entry(topic: str, **kw) -> IndexEntry:
    return IndexEntry(
        topic=topic,
        category=kw.get("category", "profession"),
        path=kw.get("path", f"knowledge/{topic.lower()}.md"),
        keywords=kw.get("keywords", [topic.lower()]),
        access_count=0,
        updated="2026-01-01",
        project=None,
    )


# ---- Tests ----

@pytest.fixture
def graph(tmp_path):
    return KnowledgeGraph(path=tmp_path / "graph.json")


def test_fuzzy_only(graph):
    entry = _entry("Python")
    searcher = FakeSearcher([SearchHit(entry=entry, score=0.9)])
    hybrid = HybridSearcher(searcher=searcher, graph=graph)

    results = hybrid.search("python")
    assert len(results) == 1
    assert results[0].entry.topic == "Python"
    assert results[0].source == "fuzzy"
    assert results[0].score > 0


def test_graph_only(tmp_path, tmp_kb):
    """Graph finds a note that fuzzy search missed."""
    graph = KnowledgeGraph(path=tmp_path / "graph.json")
    # Save a note so KM.get_note works
    tmp_kb.save_note(
        topic="GIL",
        body="# GIL\nGlobal Interpreter Lock",
        category="profession",
        keywords=["gil", "python"],
        source="test",
    )
    # Add graph edges pointing to this note
    graph.add_relations(
        [("python", "has", "gil")],
        source_note="gil",
    )
    searcher = FakeSearcher([])  # fuzzy finds nothing
    hybrid = HybridSearcher(searcher=searcher, graph=graph)

    results = hybrid.search("python")
    assert len(results) >= 1
    graph_hits = [r for r in results if r.source == "graph"]
    assert len(graph_hits) >= 1


def test_both_sources_merged(tmp_path):
    graph = KnowledgeGraph(path=tmp_path / "graph.json")
    entry = _entry("Python")
    # Fuzzy finds "Python" with score 0.8
    searcher = FakeSearcher([SearchHit(entry=entry, score=0.8)])
    # Graph also has a note slug "python" pointing to something
    graph.add_relations([("python", "has", "gil")], source_note="python")
    hybrid = HybridSearcher(searcher=searcher, graph=graph)

    results = hybrid.search("python")
    both = [r for r in results if r.source == "both"]
    assert len(both) >= 1
    # Combined score should be higher than fuzzy-only
    assert both[0].score > 0.8 * 0.6  # more than just fuzzy contribution


def test_empty_query(graph):
    searcher = FakeSearcher([])
    hybrid = HybridSearcher(searcher=searcher, graph=graph)
    results = hybrid.search("")
    assert results == []


def test_find_best_returns_top(graph):
    entry = _entry("RS-485")
    searcher = FakeSearcher([SearchHit(entry=entry, score=0.95)])
    hybrid = HybridSearcher(searcher=searcher, graph=graph)
    best = hybrid.find_best("RS-485")
    assert best is not None
    assert best.topic == "RS-485"


def test_find_best_none(graph):
    searcher = FakeSearcher([])
    hybrid = HybridSearcher(searcher=searcher, graph=graph)
    assert hybrid.find_best("nonexistent") is None


def test_limit_respected(graph):
    entries = [_entry(f"Topic{i}") for i in range(10)]
    hits = [SearchHit(entry=e, score=0.9 - i * 0.05) for i, e in enumerate(entries)]
    searcher = FakeSearcher(hits)
    hybrid = HybridSearcher(searcher=searcher, graph=graph)
    results = hybrid.search("topic", limit=3)
    assert len(results) <= 3


def test_custom_weights(graph):
    entry = _entry("Test")
    searcher = FakeSearcher([SearchHit(entry=entry, score=1.0)])
    # All weight on fuzzy
    hybrid = HybridSearcher(searcher=searcher, graph=graph, fuzzy_weight=1.0, graph_weight=0.0)
    results = hybrid.search("test")
    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0, abs=0.01)
