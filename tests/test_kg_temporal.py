"""Tests for temporal validity in KnowledgeGraph (valid_from / valid_to / invalidate / timeline)."""
from __future__ import annotations

import pytest

from backend.knowledge_graph import KnowledgeGraph


def test_open_edge_visible_in_current_view(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("max", "child_of", "alice")], "n1", valid_from="2015-04-01")
    out = g.query_entity("max")
    assert any(e["target"] == "alice" and e["relation"] == "child_of" for e in out)


def test_as_of_filters_future_facts(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("max", "loves", "chess")], "n1", valid_from="2025-10-01")
    # Before chess started — should NOT see it
    out = g.query_entity("max", as_of="2025-06-01")
    assert not any(e["target"] == "chess" for e in out)
    # After it started — should see it
    out2 = g.query_entity("max", as_of="2025-12-01")
    assert any(e["target"] == "chess" for e in out2)


def test_invalidate_closes_edge(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("max", "does", "swimming")], "n1", valid_from="2025-01-01")
    closed = g.invalidate("max", "does", "swimming", ended_at="2025-09-30")
    assert closed >= 1
    # Current view (no as_of) should NOT include closed edges
    out = g.query_entity("max")
    assert not any(e["target"] == "swimming" for e in out)
    # Mid-window query should still see it
    out2 = g.query_entity("max", as_of="2025-06-15")
    assert any(e["target"] == "swimming" for e in out2)


def test_invalidate_closes_inverse_edge_too(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("max", "does", "swimming")], "n1", valid_from="2025-01-01")
    g.invalidate("max", "does", "swimming", ended_at="2025-09-30")
    # Reverse-direction lookup from "swimming" should not list current "max"
    inverse = g.query_entity("swimming")
    assert not any(e["target"] == "max" for e in inverse)


def test_timeline_is_sorted_by_valid_from(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("max", "loves", "chess")], "n3", valid_from="2025-10-01")
    g.add_relations([("max", "child_of", "alice")], "n1", valid_from="2015-04-01")
    g.add_relations([("max", "does", "swimming")], "n2", valid_from="2025-01-01")
    timeline = g.timeline("max")
    # Filter to outgoing edges only for ordering check
    outs = [e for e in timeline if e["subject"] == "max"]
    valid_froms = [e.get("valid_from") for e in outs]
    assert valid_froms == sorted(valid_froms)
    # All three facts present
    targets = {e["target"] for e in outs}
    assert {"alice", "swimming", "chess"} <= targets


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "g.json"
    a = KnowledgeGraph(path=p)
    a.add_relations(
        [("max", "loves", "chess")],
        "n1",
        valid_from="2025-10-01",
        confidence=0.85,
    )
    a.invalidate("max", "loves", "chess", ended_at="2026-04-01")

    b = KnowledgeGraph(path=p)
    out = b.query_entity("max", as_of="2025-12-01")
    assert any(
        e["target"] == "chess" and e.get("valid_to") == "2026-04-01"
        for e in out
    )
