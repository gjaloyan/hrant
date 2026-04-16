"""Tests for backend.knowledge_graph module."""
from pathlib import Path

import pytest

from backend.knowledge_graph import (
    KnowledgeGraph,
    extract_relations_from_note_body,
    parse_entity_relations,
)


# ---- parse_entity_relations ----

def test_parse_pipe_separated():
    text = "Python GIL | causes | single-threaded bottleneck"
    result = parse_entity_relations(text)
    assert result == [("Python GIL", "causes", "single-threaded bottleneck")]


def test_parse_arrow_separated():
    text = "RS-485 -> uses -> differential signaling"
    result = parse_entity_relations(text)
    assert result == [("RS-485", "uses", "differential signaling")]


def test_parse_multiple_lines():
    text = """- CPU | has | cache
- RAM | connects_to | motherboard
- GPU -> renders -> frames"""
    result = parse_entity_relations(text)
    assert len(result) == 3
    assert result[0] == ("CPU", "has", "cache")
    assert result[2] == ("GPU", "renders", "frames")


def test_parse_ignores_bad_lines():
    text = """valid | relation | target
just some text without separators
also invalid line"""
    result = parse_entity_relations(text)
    assert len(result) == 1


def test_parse_empty():
    assert parse_entity_relations("") == []
    assert parse_entity_relations("   \n  \n") == []


# ---- extract_relations_from_note_body ----

def test_extract_wiki_links():
    body = "RS-485 uses [[differential signaling]] and connects via [[UART]]."
    triples = extract_relations_from_note_body(body, "RS-485")
    topics = [(s, r, o) for s, r, o in triples if r == "related_to"]
    assert ("rs-485", "related_to", "differential signaling") in topics
    assert ("rs-485", "related_to", "uart") in topics


def test_extract_bold_terms():
    body = "The **Global Interpreter Lock** prevents **multithreading** in CPython."
    triples = extract_relations_from_note_body(body, "Python")
    mentions = [(s, r, o) for s, r, o in triples if r == "mentions"]
    assert ("python", "mentions", "global interpreter lock") in mentions
    assert ("python", "mentions", "multithreading") in mentions


def test_extract_skips_short_bold():
    body = "A **x** and **ab** are too short. But **valid term** is ok."
    triples = extract_relations_from_note_body(body, "topic")
    mentions = [o for _, r, o in triples if r == "mentions"]
    assert "valid term" in mentions
    assert "x" not in mentions
    assert "ab" not in mentions


def test_extract_skips_question_bold():
    body = "**Q: What is this?** is a question header."
    triples = extract_relations_from_note_body(body, "topic")
    mentions = [o for _, r, o in triples if r == "mentions"]
    assert not any(o.startswith("q:") for o in mentions)


# ---- KnowledgeGraph ----

@pytest.fixture
def graph(tmp_path):
    return KnowledgeGraph(path=tmp_path / "graph.json")


def test_empty_graph(graph):
    assert graph.stats() == {"entities": 0, "edges": 0, "notes_indexed": 0}
    assert graph.entity_count() == 0
    assert graph.edge_count() == 0


def test_add_relations(graph):
    triples = [
        ("Python", "has", "GIL"),
        ("GIL", "causes", "bottleneck"),
    ]
    added = graph.add_relations(triples, source_note="python-note")
    assert added == 2
    stats = graph.stats()
    assert stats["entities"] == 3  # python, gil, bottleneck
    assert stats["edges"] > 0
    assert stats["notes_indexed"] == 1


def test_bidirectional_edges(graph):
    graph.add_relations([("A", "links", "B")], source_note="note1")
    # Forward edge
    neighbors_a = graph.get_neighbors("A")
    assert any(n["target"] == "b" for n in neighbors_a)
    # Reverse edge
    neighbors_b = graph.get_neighbors("B")
    assert any(n["target"] == "a" for n in neighbors_b)
    # Reverse edge has lower weight
    rev = [n for n in neighbors_b if n["target"] == "a"][0]
    assert rev["weight"] == 0.5


def test_no_duplicate_edges(graph):
    triples = [("X", "rel", "Y")]
    graph.add_relations(triples, source_note="n1")
    graph.add_relations(triples, source_note="n1")  # same note, same triple
    edges_x = graph.get_neighbors("X")
    forward = [e for e in edges_x if e["target"] == "y" and e["relation"] == "rel"]
    assert len(forward) == 1


def test_same_triple_different_notes(graph):
    triples = [("X", "rel", "Y")]
    graph.add_relations(triples, source_note="n1")
    graph.add_relations(triples, source_note="n2")
    edges_x = graph.get_neighbors("X")
    forward = [e for e in edges_x if e["target"] == "y" and e["relation"] == "rel"]
    assert len(forward) == 2  # one per source note


def test_remove_note(graph):
    graph.add_relations([("A", "r", "B")], source_note="note1")
    graph.add_relations([("C", "r", "D")], source_note="note2")
    graph.remove_note("note1")
    assert graph.get_neighbors("A") == []
    assert len(graph.get_neighbors("C")) > 0


def test_persistence(tmp_path):
    path = tmp_path / "graph.json"
    g1 = KnowledgeGraph(path=path)
    g1.add_relations([("X", "r", "Y")], source_note="n1")
    # Load fresh instance from same file
    g2 = KnowledgeGraph(path=path)
    assert g2.entity_count() > 0
    neighbors = g2.get_neighbors("X")
    assert any(n["target"] == "y" for n in neighbors)


def test_find_related_notes_direct(graph):
    graph.add_relations([("python", "has", "gil")], source_note="python-note")
    graph.add_relations([("javascript", "has", "event loop")], source_note="js-note")
    results = graph.find_related_notes("python")
    slugs = [slug for slug, _ in results]
    assert "python-note" in slugs


def test_find_related_notes_via_hop(graph):
    graph.add_relations([("python", "has", "gil")], source_note="python-note")
    graph.add_relations([("gil", "causes", "bottleneck")], source_note="perf-note")
    results = graph.find_related_notes("python", max_hops=2)
    slugs = [slug for slug, _ in results]
    # Should find perf-note via python -> gil -> bottleneck path
    assert "perf-note" in slugs


def test_find_related_notes_empty_query(graph):
    graph.add_relations([("A", "r", "B")], source_note="n1")
    assert graph.find_related_notes("") == []


def test_find_related_notes_no_match(graph):
    graph.add_relations([("frontend", "uses", "react")], source_note="n1")
    results = graph.find_related_notes("zzzzz")
    assert results == []


def test_normalize_case(graph):
    graph.add_relations([("Python", "has", "GIL")], source_note="n1")
    neighbors = graph.get_neighbors("PYTHON")
    assert len(neighbors) > 0
    assert neighbors[0]["target"] == "gil"


def test_skip_empty_entities(graph):
    added = graph.add_relations([("", "rel", "B"), ("A", "", "B"), ("A", "rel", "")], source_note="n1")
    assert added == 0


def test_get_neighbors_unknown_entity(graph):
    assert graph.get_neighbors("nonexistent") == []


def test_corrupted_file(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("not valid json {{{", encoding="utf-8")
    g = KnowledgeGraph(path=path)
    assert g.stats()["entities"] == 0
