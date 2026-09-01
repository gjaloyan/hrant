"""A graph with no edges is not a graph.

Prod 2026-09-01: `knowledge/graph.json` held 6968 nodes and an empty edge
map. The note graph had never been built — the Graph screen read "0
entities, 0 edges, 0 notes" — and the autonomic rule meant to notice that
had been reading "healthy" for months, because it counted only nodes and a
different subsystem (memory consolidation) had filled those.
"""
import json

import pytest

from backend.autonomic.layer0 import default_rules
from backend.autonomic.state import StateSnapshotBuilder


def _rule():
    r = [x for x in default_rules() if x.name == "graph_collapsed"]
    assert r, "the graph-collapse rule is gone"
    return r[0]


class Snap:
    """Only the fields the predicate reads."""

    def __init__(self, notes, nodes, edges):
        self.kb_notes_count = notes
        self.kb_graph_nodes = nodes
        self.kb_graph_edges = edges


def test_nodes_without_edges_counts_as_collapsed():
    # The exact prod state: plenty of nodes, no links, notes to index.
    assert _rule().predicate(Snap(notes=28, nodes=6968, edges=0)) is True


def test_a_healthy_graph_is_left_alone():
    assert _rule().predicate(Snap(notes=28, nodes=6968, edges=238)) is False


def test_few_nodes_still_counts_as_collapsed():
    # The original signal must keep working.
    assert _rule().predicate(Snap(notes=28, nodes=12, edges=5)) is True


def test_no_notes_means_nothing_to_rebuild():
    # An empty knowledge base is not a broken graph; rebuilding forever
    # would be the failure mode this guard prevents.
    assert _rule().predicate(Snap(notes=0, nodes=0, edges=0)) is False


def test_the_edge_counter_reads_the_edge_map(tmp_path):
    (tmp_path / "graph.json").write_text(
        json.dumps({"version": 1, "nodes": {"a": 1, "b": 2},
                    "edges": {"x": ["y"], "y": ["x"], "z": []}}),
        encoding="utf-8")
    sp = StateSnapshotBuilder.__new__(StateSnapshotBuilder)
    sp._knowledge_root = tmp_path
    assert sp._count_graph_edges() == 3
    assert sp._count_graph_nodes() == 2


@pytest.mark.parametrize("body", ["", "not json", '{"version": 1}'])
def test_an_unreadable_graph_reads_as_empty(tmp_path, body):
    # Reading as empty makes the rule FIRE, which rebuilds — the safe
    # direction. Reading as healthy would hide a corrupt file forever.
    (tmp_path / "graph.json").write_text(body, encoding="utf-8")
    sp = StateSnapshotBuilder.__new__(StateSnapshotBuilder)
    sp._knowledge_root = tmp_path
    assert sp._count_graph_edges() == 0
