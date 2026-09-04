"""The v2 graph never persisted a single edge.

Measured on prod 2026-09-04: graph.json holds 11,975 nodes and the v2
store loads ZERO edges from it, while 7,037 triples sit extracted in
memory_facts.jsonl. 0.23% of nodes have any link at all.

Two subsystems share one file. `backend/knowledge_graph` owns the legacy
`edges` DICT keyed by subject; `backend/graph` is the v2 store with
`nodes` and typed edges. `migration.py` set the terms — legacy `edges`
stays untouched, v2 edges live in `edges_v2` — and
`knowledge_graph._save()` honours them, preserving version/nodes/
edges_v2 on every write.

`store.py` did not. It read `data.get("edges")` and iterated it: on the
legacy dict that yields KEYS, `GraphEdge.from_dict("некоторая строка")`
raises, the except swallows it, and every edge the builder made is
dropped on the next load. Its save wrote `"edges": [...]`, replacing the
legacy dict with a list — which `FIRE_GRAPH_MAINTENANCE` then calls
`.items()` on.

The builder was always right: `add_fact` emits `is_about` for topics and
`mentions` for every triple. Only persistence was broken.
"""
import json

import pytest

from backend.graph.model import GraphEdge, GraphNode
from backend.graph.store import Graph


LEGACY = {
    "version": 1,
    "edges": {
        "1 usd": [{"target": "363.17 amd", "relation": "exchanged_for",
                   "note": "_memory", "weight": 0.98}],
    },
    "nodes": [{"id": "fact:aaa", "kind": "fact", "label": "A fact.",
               "weight": 0.9, "metadata": {}}],
}


@pytest.fixture()
def graph_file(tmp_path, monkeypatch):
    p = tmp_path / "knowledge" / "graph.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(LEGACY, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    return p


def _graph(_path=None):
    """`path` is a read-only property resolved from the data dir, which
    the fixture has already redirected."""
    return Graph()


def test_a_legacy_edges_dict_does_not_wipe_the_v2_edges(graph_file):
    """The failure exactly: iterating the legacy dict yields strings and
    every edge is silently dropped."""
    g = _graph(graph_file)
    g.upsert_edge(GraphEdge(source="fact:aaa", target="topic:x",
                            kind="is_about"))
    g.save()

    reloaded = _graph(graph_file)
    assert reloaded.edge_count() == 1


def test_the_legacy_dict_survives_a_v2_save(graph_file):
    """`FIRE_GRAPH_MAINTENANCE` calls .items() on it. Replacing it with a
    list is not a schema change, it is a break."""
    g = _graph(graph_file)
    g.upsert_edge(GraphEdge(source="fact:aaa", target="topic:x",
                            kind="is_about"))
    g.save()

    on_disk = json.loads(graph_file.read_text(encoding="utf-8"))
    assert isinstance(on_disk["edges"], dict)
    assert on_disk["edges"]["1 usd"][0]["target"] == "363.17 amd"
    assert isinstance(on_disk["edges_v2"], list)
    assert on_disk["edges_v2"][0]["kind"] == "is_about"


def test_a_file_written_by_the_old_store_still_loads(tmp_path, monkeypatch):
    """Back-compat: before this, v2 edges went into `edges` as a LIST.
    Those files exist and must not lose their edges to the fix."""
    p = tmp_path / "knowledge" / "graph.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "version": 1,
        "nodes": [],
        "edges": [{"source": "a", "target": "b", "kind": "is_about",
                   "weight": 1.0, "metadata": {}}],
    }), encoding="utf-8")
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    assert _graph(p).edge_count() == 1


def test_keys_this_store_does_not_own_are_preserved(graph_file):
    """Symmetry with knowledge_graph._save(), which already preserves
    version/nodes/edges_v2. Both writers keep what is not theirs."""
    data = json.loads(graph_file.read_text(encoding="utf-8"))
    data["some_other_writers_key"] = {"keep": "me"}
    graph_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    g = _graph(graph_file)
    g.upsert_node(GraphNode(id="fact:bbb", kind="fact", label="B."))
    g.save()

    on_disk = json.loads(graph_file.read_text(encoding="utf-8"))
    assert on_disk["some_other_writers_key"] == {"keep": "me"}
