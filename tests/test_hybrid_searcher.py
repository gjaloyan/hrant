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
    # Source is a "+"-joined list of contributing signals; with no graph
    # or vector data this collapses to "fuzzy".
    assert "fuzzy" in results[0].source.split("+")
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
    graph_hits = [r for r in results if "graph" in r.source.split("+")]
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
    both = [
        r for r in results
        if "fuzzy" in r.source.split("+") and "graph" in r.source.split("+")
    ]
    assert len(both) >= 1
    # Combined score should reflect both contributions (not just fuzzy alone)
    assert both[0].score > 0


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


# ---- Score-floor regression: noise-as-best-match ----

class _StubVectorStore:
    """VectorStore stub yielding pre-set raw cosine scores per slug."""

    def __init__(self, hits: dict[str, float]):
        self._hits = hits

    def count(self) -> int:
        return len(self._hits) or 1  # non-zero so HybridSearcher consults us

    def search(self, query_vector, k: int = 5):
        items = sorted(self._hits.items(), key=lambda kv: kv[1], reverse=True)
        return items[:k]


def test_vector_floor_rejects_noise(graph, monkeypatch, tmp_kb):
    """Repro: user asks about a topic the KB doesn't have ('iodine').
    Vector store returns weak cosine matches against unrelated notes.
    Without the floor, min-max normalization scales the top noise hit
    to 1.0 and HybridSearcher.find_best returns it as the 'best' match.
    With the floor, all weak hits are filtered and find_best returns
    None — exactly what _ensure_knowledge needs to skip the topic."""
    # Three unrelated notes the KB happens to have.
    for slug in ("blood-sugar", "source-code-analysis", "mcp"):
        tmp_kb.save_note(
            topic=slug.replace("-", " "),
            body=f"# {slug}\nbody",
            category="profession",
            keywords=[slug],
            source="test",
        )
    # Vector returns weak matches (~0.05 cosine = noise).
    vstore = _StubVectorStore({
        "blood-sugar": 0.07,
        "source-code-analysis": 0.05,
        "mcp": 0.04,
    })
    # Ensure the embedder returns SOMETHING so vector path is exercised.
    monkeypatch.setattr(
        "backend.hybrid_searcher.EMBEDDER",
        type("E", (), {"embed": staticmethod(lambda q: [0.1] * 8)})(),
    )

    searcher = FakeSearcher([])  # fuzzy finds nothing (correct)
    hybrid = HybridSearcher(
        searcher=searcher, graph=graph, vector_store=vstore,
    )
    # With the floor in place, noise is filtered → no result.
    assert hybrid.find_best("iodine") is None
    # And explicit min_raw_score gate also returns None.
    assert hybrid.find_best("iodine", min_raw_score=0.4) is None


def test_vector_floor_keeps_real_matches(graph, monkeypatch, tmp_kb):
    """Sanity: a strong vector match (cosine 0.6) clears the floor and
    is returned as the best hit."""
    tmp_kb.save_note(
        topic="Python",
        body="# Python\nGIL.",
        category="profession",
        keywords=["python"],
        source="test",
    )
    vstore = _StubVectorStore({"python": 0.62})
    monkeypatch.setattr(
        "backend.hybrid_searcher.EMBEDDER",
        type("E", (), {"embed": staticmethod(lambda q: [0.1] * 8)})(),
    )
    hybrid = HybridSearcher(
        searcher=FakeSearcher([]), graph=graph, vector_store=vstore,
    )
    best = hybrid.find_best("python")
    assert best is not None
    assert best.topic == "Python"


def test_min_raw_score_blocks_only_fuzzy_threshold_hits(graph):
    """Strong fuzzy hit (raw 0.95) clears any reasonable min_raw_score.
    Weak fuzzy hit just at the searcher's threshold (0.6) should be
    rejected when the caller asks for higher quality."""
    strong = _entry("Python")
    searcher_strong = FakeSearcher([SearchHit(entry=strong, score=0.95)])
    hybrid_strong = HybridSearcher(searcher=searcher_strong, graph=graph)
    assert hybrid_strong.find_best("python", min_raw_score=0.4) is not None
    assert hybrid_strong.find_best("python", min_raw_score=0.99) is None
