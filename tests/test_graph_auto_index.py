"""Tests for graph auto-indexing on save + smarter query extraction.

Covers the regressions that turned `graph only` retrieval from 0% to 76%
on the bench:
  - KM.save_note adds keyword + body edges automatically
  - delete_note cleans them up
  - re-save replaces edges (no accumulation)
  - stop words are filtered out of query entity extraction
  - direct topic-slug match gets a bonus over BFS-only

The conftest tmp_kb fixture monkeypatches kg_mod.GRAPH onto a fresh
graph, so we read it via the module reference (not a top-level import
that would freeze before patching).
"""
from __future__ import annotations

import pytest

from backend import knowledge_graph as kg_mod
from backend.knowledge_graph import KnowledgeGraph


def test_save_note_creates_keyword_edges(tmp_kb):
    tmp_kb.save_note(
        topic="Python GIL",
        body="The Global Interpreter Lock is a mutex.",
        keywords=["python", "gil", "threading"],
        source="test",
    )
    # Subject is normalized to "python gil" (lowercase, spaces preserved).
    edges = kg_mod.GRAPH._edges.get("python gil") or []
    keyword_targets = {e["target"] for e in edges if e["relation"] == "keyword"}
    assert "python" in keyword_targets
    assert "gil" in keyword_targets
    assert "threading" in keyword_targets


def test_resave_replaces_edges(tmp_kb):
    tmp_kb.save_note(topic="Topic", body="x", keywords=["one", "two"], source="t")
    first = sorted(
        e["target"]
        for e in (kg_mod.GRAPH._edges.get("topic") or [])
        if e["relation"] == "keyword"
    )
    assert first == ["one", "two"]
    # Re-save with different keywords; old ones must be gone.
    tmp_kb.save_note(topic="Topic", body="x", keywords=["three"], source="t")
    second = sorted(
        e["target"]
        for e in (kg_mod.GRAPH._edges.get("topic") or [])
        if e["relation"] == "keyword"
    )
    assert second == ["three"]


def test_delete_note_removes_edges(tmp_kb):
    tmp_kb.save_note(topic="Foo", body="b", keywords=["bar"], source="t")
    assert any(e["target"] == "bar" for e in kg_mod.GRAPH._edges.get("foo") or [])
    tmp_kb.delete_note("Foo")
    # No edges left tagged with note=foo.
    for edges in kg_mod.GRAPH._edges.values():
        for e in edges:
            assert e.get("note") != "foo"


def test_query_filters_stop_words(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("python_gil", "keyword", "python")], source_note="python_gil")
    g.add_relations([("python_gil", "keyword", "gil")], source_note="python_gil")
    # "What is the python GIL" should still find python_gil after filtering
    # out 'what', 'is', 'the'.
    hits = g.find_related_notes("What is the python gil", max_results=3)
    slugs = [h[0] for h in hits]
    assert "python_gil" in slugs


def test_topic_slug_match_bonus(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    # Two notes: one whose slug exactly matches the query, one whose
    # keywords match. The exact-slug note should rank ≥ the keyword note.
    g.add_relations([("python_gil", "keyword", "python")], source_note="python_gil")
    g.add_relations([("threading_basics", "keyword", "python")], source_note="threading_basics")
    hits = g.find_related_notes("python gil", max_results=5)
    slugs = [h[0] for h in hits]
    assert slugs.index("python_gil") <= slugs.index("threading_basics")


def test_bfs_decay_is_gentler(tmp_path):
    """A 1-hop neighbor should still score meaningfully (not ~0.25)."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    # Build: node_a → keyword → kw_x ; query "kw_x" should reach node_a
    # via the inverse-keyword edge (1 hop).
    g.add_relations([("node_a", "keyword", "kw_x")], source_note="node_a")
    hits = g.find_related_notes("kw_x", max_results=3)
    assert any(slug == "node_a" and score >= 0.5 for slug, score in hits)
