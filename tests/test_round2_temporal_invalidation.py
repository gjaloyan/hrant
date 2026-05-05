"""Knowledge graph: when the same (subject, relation) gets a new
target on a single-valued relation (lives_in, costs, status, …),
the OLD edge auto-closes with valid_to=today and the NEW edge
opens with valid_from=today. Without this, both old and new
facts coexist as 'current' and the agent would equally cite both.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from backend.knowledge_graph import KnowledgeGraph


def test_single_valued_relation_invalidates_old_target(tmp_path: Path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("user", "lives_in", "moscow")], source_note="t1")
    # Initially open
    edges = [e for e in g._edges["user"] if e["relation"] == "lives_in"]
    assert len(edges) == 1
    assert edges[0].get("valid_to") is None

    # Move
    g.add_relations([("user", "lives_in", "yerevan")], source_note="t2")
    edges = [e for e in g._edges["user"] if e["relation"] == "lives_in"]
    today = datetime.utcnow().strftime("%Y-%m-%d")
    by_target = {e["target"]: e for e in edges}
    # Old edge closed
    assert by_target["moscow"]["valid_to"] == today
    # New edge open with valid_from
    assert by_target["yerevan"].get("valid_to") is None
    assert by_target["yerevan"].get("valid_from") == today


def test_multi_valued_relation_keeps_both(tmp_path: Path):
    """`brother_of` is NOT in SINGLE_VALUED_RELATIONS — multiple
    targets can be true at once, so adding a second one does NOT
    invalidate the first."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("user", "brother_of", "tigran")], source_note="t1")
    g.add_relations([("user", "brother_of", "armen")], source_note="t2")
    edges = [
        e for e in g._edges["user"]
        if e["relation"] == "brother_of" and e.get("valid_to") is None
    ]
    targets = {e["target"] for e in edges}
    assert targets == {"tigran", "armen"}


def test_inverse_edge_also_closed(tmp_path: Path):
    """Closing user->moscow should mirror onto moscow->user (inverse:lives_in)
    so traversal from the target side stays consistent."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("user", "lives_in", "moscow")], source_note="t1")
    g.add_relations([("user", "lives_in", "yerevan")], source_note="t2")
    inv_old = [
        e for e in g._edges.get("moscow", [])
        if e["target"] == "user" and e["relation"] == "inverse:lives_in"
    ]
    today = datetime.utcnow().strftime("%Y-%m-%d")
    assert inv_old, "inverse edge for old target must exist"
    assert inv_old[0].get("valid_to") == today


def test_auto_invalidate_can_be_disabled(tmp_path: Path):
    """auto_invalidate=False keeps both edges open even on a single-
    valued relation. Used for back-filling history."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("user", "lives_in", "moscow")], source_note="t1")
    g.add_relations(
        [("user", "lives_in", "yerevan")], source_note="t2",
        auto_invalidate=False,
    )
    open_edges = [
        e for e in g._edges["user"]
        if e["relation"] == "lives_in" and e.get("valid_to") is None
    ]
    targets = {e["target"] for e in open_edges}
    assert targets == {"moscow", "yerevan"}


def test_same_target_no_op(tmp_path: Path):
    """Re-adding the SAME (s, r, o) is a no-op — no spurious
    invalidation when the new target equals an existing one."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("user", "lives_in", "moscow")], source_note="t1")
    g.add_relations([("user", "lives_in", "moscow")], source_note="t1")
    edges = [
        e for e in g._edges["user"]
        if e["relation"] == "lives_in" and e.get("valid_to") is None
    ]
    assert len(edges) == 1, "must dedup, not invalidate"


def test_target_index_built_on_load(tmp_path: Path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("a", "uses", "redis")], source_note="t")
    # Reload from disk
    g2 = KnowledgeGraph(path=g.path)
    pairs = g2.find_facts_by_target("redis")
    assert any(s == "a" for s, _ in pairs)


def test_target_index_used_by_find_facts_by_target(tmp_path: Path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([
        ("python", "tagged", "language"),
        ("rust", "tagged", "language"),
    ], source_note="t")
    pairs = g.find_facts_by_target("language")
    subjects = {s for s, _ in pairs}
    assert subjects == {"python", "rust"}
