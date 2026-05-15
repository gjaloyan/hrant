"""Tests for the knowledge graph (Phase 16C).

Coverage:
  - model.py: id slugifier dedup, fact_id collision resistance
  - store.py: round-trip via .save()/.clear()+reload, upsert merge
    semantics for nodes (weight max, metadata union) and edges
    (weight sum), schema version guard
  - builder.py: rebuild from memory_facts.jsonl is idempotent;
    `add_fact` produces stable ids across re-runs
  - query.py: stats top_topics by degree, neighborhood direction
    tagging, search ranks by degree
  - api/graph.py: REST surface
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.graph import (
    builder as _builder,
    model as _model,
    query as _query,
    store as _store,
)
from backend.graph.model import GraphEdge, GraphNode


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated data_dir + reset of the module-level singleton.

    We reset GRAPH *in place* rather than replacing the attribute on
    `_store`. `builder.py` does `from .store import GRAPH`, which
    creates its own binding to the singleton at import time —
    `monkeypatch.setattr(_store, 'GRAPH', new)` would update
    `_store.GRAPH` but NOT `builder.GRAPH`, and they'd silently
    diverge. Mutating the shared instance avoids that pitfall."""
    data_dir = tmp_path / "hrant"
    (data_dir / "knowledge").mkdir(parents=True)
    monkeypatch.setenv("HRANT_DATA_DIR", str(data_dir))
    # Reset the singleton: clear in-memory state, point its root
    # override at the tmp path so it doesn't pollute the dev's real
    # ~/.hrant/data/knowledge/graph.json.
    _store.GRAPH.clear()
    monkeypatch.setattr(_store.GRAPH, "_root_override", data_dir / "knowledge")
    yield data_dir
    _store.GRAPH.clear()


# ─── model.py: id helpers ───────────────────────────────────────────


def test_topic_id_dedupes_case_and_whitespace():
    assert _model.topic_id("Voice") == _model.topic_id("voice")
    assert _model.topic_id("  voice  ") == _model.topic_id("voice")
    assert _model.topic_id("voice-tts") == _model.topic_id("Voice TTS")  # both → voice_tts


def test_topic_id_drops_punctuation_but_keeps_separators():
    # Slashes / hyphens / underscores collapse to single _; punctuation drops.
    assert _model.topic_id("audio/voice").startswith("topic:")
    assert "_" in _model.topic_id("audio/voice")
    # Brackets, commas, emoji drop silently — output stays slug-safe.
    out = _model.topic_id("[draft] voice 🎙")
    assert " " not in out
    assert "[" not in out


def test_fact_id_normalises_then_hashes():
    """Same text with different whitespace/case → same id. Different
    text → different id."""
    a = _model.fact_id("User likes dark theme.")
    b = _model.fact_id("  user  likes   dark    theme.   ")
    c = _model.fact_id("User likes LIGHT theme.")
    assert a == b
    assert a != c


def test_fact_id_avoids_obvious_collisions():
    """Hash size is small (6 bytes) but at the personal-agent scale
    the birthday-paradox risk is negligible. Spot-check that 100
    distinct facts produce 100 distinct ids."""
    ids = {_model.fact_id(f"unique sentence number {i}") for i in range(100)}
    assert len(ids) == 100


# ─── store.py: persistence + concurrency-ish ───────────────────────


def test_graph_round_trips_via_disk(home):
    g = _store.GRAPH
    g.upsert_node(GraphNode(id="topic:voice", kind="topic", label="voice"))
    g.upsert_node(GraphNode(id="fact:abc", kind="fact", label="hi"))
    g.upsert_edge(GraphEdge(source="fact:abc", target="topic:voice", kind="is_about"))
    g.save()

    # Force a fresh load by clearing the in-memory state.
    g2 = _store.Graph(root=home / "knowledge")
    assert g2.node_count() == 2
    assert g2.get_node("topic:voice") is not None
    edges = list(g2.iter_edges())
    assert len(edges) == 1
    assert edges[0].source == "fact:abc"


def test_upsert_node_merges_metadata_keeps_max_weight(home):
    g = _store.GRAPH
    g.upsert_node(GraphNode(
        id="n1", kind="topic", label="x", weight=0.5,
        metadata={"a": 1},
    ))
    g.upsert_node(GraphNode(
        id="n1", kind="topic", label="x", weight=0.8,
        metadata={"b": 2},
    ))
    n = g.get_node("n1")
    assert n is not None
    assert n.weight == 0.8                   # max, not overwrite
    assert n.metadata == {"a": 1, "b": 2}    # union


def test_upsert_edge_sums_weights(home):
    g = _store.GRAPH
    e1 = GraphEdge(source="a", target="b", kind="is_about", weight=1.0)
    e2 = GraphEdge(source="a", target="b", kind="is_about", weight=2.0)
    g.upsert_edge(e1)
    g.upsert_edge(e2)
    edges = list(g.iter_edges())
    assert len(edges) == 1
    # Weight accumulated — captures "we saw this connection multiple times"
    assert edges[0].weight == 3.0


def test_upsert_edge_keeps_distinct_kinds_separate(home):
    g = _store.GRAPH
    g.upsert_edge(GraphEdge(source="a", target="b", kind="is_about"))
    g.upsert_edge(GraphEdge(source="a", target="b", kind="relates_to"))
    assert g.edge_count() == 2


def test_remove_node_drops_incident_edges(home):
    g = _store.GRAPH
    g.upsert_node(GraphNode(id="a", kind="topic", label="a"))
    g.upsert_node(GraphNode(id="b", kind="topic", label="b"))
    g.upsert_edge(GraphEdge(source="a", target="b", kind="is_about"))
    assert g.edge_count() == 1
    g.remove_node("a")
    assert g.node_count() == 1
    assert g.edge_count() == 0


def test_load_ignores_unknown_schema_version(home):
    """Forward-compat: a future graph file shouldn't crash today's
    loader. It starts empty + logs a warning."""
    p = home / "knowledge" / "graph.json"
    p.write_text(json.dumps({
        "version": 999,
        "nodes": [{"id": "x", "kind": "topic", "label": "x"}],
    }), encoding="utf-8")
    g = _store.Graph(root=home / "knowledge")
    assert g.node_count() == 0  # cautiously ignored


# ─── builder.py ─────────────────────────────────────────────────────


def _write_memory_facts(home, rows: list[dict]) -> None:
    p = home / "knowledge" / "memory_facts.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_rebuild_creates_fact_nodes_with_topic_edges(home):
    _write_memory_facts(home, [
        {"summary": "User uses Tailscale", "tags": ["network", "tools"],
         "confidence": 0.9, "category": "preference"},
        {"summary": "Prefers dark theme", "tags": ["ui"],
         "confidence": 0.85, "category": "preference"},
    ])
    stats = _builder.rebuild()
    assert stats["facts"] == 2
    assert stats["topics"] == 3   # network, tools, ui — distinct
    # Each fact has at least one is_about edge.
    is_about = [e for e in _store.GRAPH.iter_edges(kind="is_about")]
    assert len(is_about) == 3   # 2+1 topic refs


def test_rebuild_is_idempotent(home):
    """Calling rebuild() twice produces the same node + edge counts."""
    _write_memory_facts(home, [
        {"summary": "X", "tags": ["a", "b"], "confidence": 0.9},
        {"summary": "Y", "tags": ["b", "c"], "confidence": 0.9},
    ])
    _builder.rebuild()
    n1 = _store.GRAPH.node_count()
    e1 = _store.GRAPH.edge_count()
    _builder.rebuild()  # second pass
    assert _store.GRAPH.node_count() == n1
    assert _store.GRAPH.edge_count() == e1


def test_rebuild_dedups_facts_with_same_text(home):
    """Two memory_facts.jsonl rows with the same text become ONE
    graph node (fact_id hashes the normalised text)."""
    _write_memory_facts(home, [
        {"summary": "User likes coffee", "tags": ["personal"], "confidence": 0.9},
        {"summary": "User likes coffee", "tags": ["habit"], "confidence": 0.85},
    ])
    _builder.rebuild()
    fact_nodes = list(_store.GRAPH.iter_nodes(kind="fact"))
    assert len(fact_nodes) == 1
    # Both topics still attached.
    topics = list(_store.GRAPH.iter_nodes(kind="topic"))
    assert len(topics) == 2


def test_add_fact_returns_stable_id(home):
    """Same text → same fact_id, twice. Caller can use the id to
    cross-reference the fact in digests + memory_facts."""
    fid1 = _builder.add_fact(text="some claim", related_topics=["x"], confidence=0.9)
    fid2 = _builder.add_fact(text="some claim", related_topics=["y"], confidence=0.95)
    assert fid1 == fid2
    # Both topic edges added.
    topics = {n.label for n in _store.GRAPH.iter_nodes(kind="topic")}
    assert topics == {"x", "y"}


def test_add_fact_processes_triples_into_entity_edges(home):
    _builder.add_fact(
        text="claim",
        related_topics=["x"],
        confidence=0.9,
        triples=[["user", "uses", "Tailscale"]],
    )
    entities = list(_store.GRAPH.iter_nodes(kind="entity"))
    assert len(entities) == 1
    assert entities[0].label == "Tailscale"
    mentions = list(_store.GRAPH.iter_edges(kind="mentions"))
    assert len(mentions) == 1
    assert mentions[0].metadata.get("predicate") == "uses"


# ─── query.py ───────────────────────────────────────────────────────


def test_stats_reports_counts_by_kind(home):
    _write_memory_facts(home, [
        {"summary": "X", "tags": ["a"], "confidence": 0.9},
    ])
    _builder.rebuild()
    s = _query.stats()
    assert s["total_nodes"] >= 2  # 1 fact + 1 topic
    assert s["by_kind"]["fact"] == 1
    assert s["by_kind"]["topic"] == 1


def test_top_topics_ranks_by_incoming_degree(home):
    # `voice` referenced by 3 facts, `ui` by 1 → voice ranks first.
    _write_memory_facts(home, [
        {"summary": "F1", "tags": ["voice"], "confidence": 0.9},
        {"summary": "F2", "tags": ["voice"], "confidence": 0.9},
        {"summary": "F3", "tags": ["voice"], "confidence": 0.9},
        {"summary": "F4", "tags": ["ui"], "confidence": 0.9},
    ])
    _builder.rebuild()
    s = _query.stats()
    labels = [t["label"] for t in s["top_topics"]]
    assert labels[0] == "voice"
    assert "ui" in labels


def test_neighborhood_tags_direction(home):
    _write_memory_facts(home, [
        {"summary": "fact A", "tags": ["topic_x"], "confidence": 0.9},
    ])
    _builder.rebuild()
    # The fact has an outgoing is_about → topic_x; topic_x has the same
    # edge incoming.
    fact_id = next(iter(_store.GRAPH.iter_nodes(kind="fact"))).id
    nb_fact = _query.neighborhood(fact_id)
    assert nb_fact is not None
    assert all(n["direction"] == "out" for n in nb_fact["neighbors"])
    topic_id = next(iter(_store.GRAPH.iter_nodes(kind="topic"))).id
    nb_topic = _query.neighborhood(topic_id)
    assert nb_topic is not None
    assert all(n["direction"] == "in" for n in nb_topic["neighbors"])


def test_search_ranks_by_degree(home):
    """Two nodes both match the query; the one with more connections
    should rank first."""
    _write_memory_facts(home, [
        {"summary": "F1 about voice", "tags": ["voice", "tts"], "confidence": 0.9},
        {"summary": "F2 about voice", "tags": ["voice"], "confidence": 0.9},
        {"summary": "F3 about voice", "tags": ["voice"], "confidence": 0.9},
    ])
    _builder.rebuild()
    results = _query.search("voice")
    # `voice` topic has highest degree (3 incoming), should come first.
    assert results[0]["label"] == "voice"


def test_search_filter_by_kind(home):
    _write_memory_facts(home, [
        {"summary": "voice fact", "tags": ["voice"], "confidence": 0.9},
    ])
    _builder.rebuild()
    only_topics = _query.search("voice", kind="topic")
    assert all(r["kind"] == "topic" for r in only_topics)
    only_facts = _query.search("voice", kind="fact")
    assert all(r["kind"] == "fact" for r in only_facts)


def test_search_empty_query_returns_empty(home):
    _write_memory_facts(home, [{"summary": "x", "tags": ["y"], "confidence": 0.9}])
    _builder.rebuild()
    assert _query.search("") == []
    assert _query.search("   ") == []


# ─── REST API ──────────────────────────────────────────────────────


@pytest.fixture
def api_client(home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import graph as graph_api
    app = FastAPI()
    app.include_router(graph_api.router)
    return TestClient(app)


def test_api_stats_returns_empty_state(home, api_client):
    r = api_client.get("/api/kgraph/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_nodes"] == 0
    assert body["total_edges"] == 0


def test_api_rebuild_then_stats(home, api_client):
    _write_memory_facts(home, [
        {"summary": "claim", "tags": ["x"], "confidence": 0.9},
    ])
    r = api_client.post("/api/kgraph/rebuild")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    r2 = api_client.get("/api/kgraph/stats")
    body = r2.json()
    assert body["by_kind"]["fact"] == 1
    assert body["by_kind"]["topic"] == 1


def test_api_search_returns_matches(home, api_client):
    _write_memory_facts(home, [
        {"summary": "claim about tailscale", "tags": ["tailscale"], "confidence": 0.9},
    ])
    api_client.post("/api/kgraph/rebuild")
    r = api_client.get("/api/kgraph/search?q=tail")
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) >= 1
    assert any("tail" in r["label"].lower() for r in body["results"])


def test_api_node_returns_neighborhood(home, api_client):
    _write_memory_facts(home, [
        {"summary": "F1", "tags": ["topic_x"], "confidence": 0.9},
    ])
    api_client.post("/api/kgraph/rebuild")
    # Look up the topic_x node id.
    r = api_client.get("/api/kgraph/search?q=topic_x&kind=topic")
    nid = r.json()["results"][0]["id"]
    r2 = api_client.get(f"/api/kgraph/node/{nid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["node"]["id"] == nid
    assert body["neighbor_count"] >= 1


def test_api_node_404_for_unknown(home, api_client):
    r = api_client.get("/api/kgraph/node/topic:does_not_exist")
    assert r.status_code == 404


def test_api_full_graph_includes_nodes_and_edges(home, api_client):
    _write_memory_facts(home, [
        {"summary": "X", "tags": ["a", "b"], "confidence": 0.9},
    ])
    api_client.post("/api/kgraph/rebuild")
    r = api_client.get("/api/kgraph")
    body = r.json()
    assert body["node_count"] >= 3   # 1 fact + 2 topics
    assert isinstance(body["nodes"], list)
    assert isinstance(body["edges"], list)
