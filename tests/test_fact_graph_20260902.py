"""Facts reached by their links, not by resembling the question.

Nightly consolidation builds fact nodes joined to topics and entities —
3675 facts and 21281 edges on prod — and until 2026-09-02 nothing
traversed a single one of them during a turn. The store was written every
night and read by nobody.

The vector store answers "what is similar to this sentence" and does it
well. It cannot answer "what do I know about my brother", because the
useful facts about a person rarely resemble the question asking for them.
"""
import pytest

from backend import fact_graph as fg


class _G:
    """A tiny graph: two entities, some facts hanging off them."""

    def __init__(self, nodes, hoods):
        self._nodes, self._hoods = nodes, hoods

    def search(self, q, limit=50):
        return [n for n in self._nodes if q.lower() in n["label"].lower()][:limit]

    def neighborhood(self, node_id):
        return self._hoods.get(node_id)


def _wire(monkeypatch, nodes, hoods):
    g = _G(nodes, hoods)
    import backend.graph.query as gq

    monkeypatch.setattr(gq, "search", g.search)
    monkeypatch.setattr(gq, "neighborhood", g.neighborhood)


NODES = [
    {"id": "entity:tigran", "kind": "entity", "label": "Tigran", "degree": 3},
    {"id": "fact:x", "kind": "fact", "label": "Tigran drives a Nissan", "degree": 1},
]
HOODS = {
    "entity:tigran": {
        "node": NODES[0],
        "neighbors": [
            {"node": {"kind": "fact", "label": "Tigran is the owner's brother"}},
            {"node": {"kind": "fact", "label": "Tigran drives a Nissan"}},
            {"node": {"kind": "topic", "label": "family"}},
        ],
    }
}


def test_it_returns_facts_linked_to_the_named_thing(monkeypatch):
    _wire(monkeypatch, NODES, HOODS)
    got = fg.facts_about("what do you know about Tigran")
    summaries = [g["summary"] for g in got]
    assert "Tigran is the owner's brother" in summaries
    assert "Tigran drives a Nissan" in summaries


def test_it_says_how_each_fact_was_reached(monkeypatch):
    """`via` is what makes the result explainable instead of a bare list."""
    _wire(monkeypatch, NODES, HOODS)
    got = fg.facts_about("Tigran")
    assert got and all(g["via"] == "Tigran" for g in got)
    assert all(g["source"] == "fact_graph" for g in got)


def test_non_fact_neighbours_are_left_out(monkeypatch):
    _wire(monkeypatch, NODES, HOODS)
    got = fg.facts_about("Tigran")
    assert "family" not in [g["summary"] for g in got]


def test_it_anchors_on_entities_not_on_facts(monkeypatch):
    """A fact node matching the text is what the vector store already
    returns; anchoring on it would duplicate that and traverse nothing."""
    only_facts = [{"id": "fact:y", "kind": "fact", "label": "Nissan", "degree": 5}]
    _wire(monkeypatch, only_facts, {})
    assert fg.facts_about("Nissan") == []


def test_an_isolated_entity_yields_nothing(monkeypatch):
    lonely = [{"id": "entity:z", "kind": "entity", "label": "Zed", "degree": 0}]
    _wire(monkeypatch, lonely, {})
    assert fg.facts_about("Zed") == []


@pytest.mark.parametrize("q", ["", "  ", "ab"])
def test_a_query_too_short_to_mean_anything_is_skipped(q):
    assert fg.facts_about(q) == []


def test_a_broken_graph_never_breaks_the_search(monkeypatch):
    """This rides along on every search_knowledge call. Retrieval is an
    enhancement; it must not take the search down with it."""
    import backend.graph.query as gq

    def boom(*a, **k):
        raise RuntimeError("graph.json is corrupt")

    monkeypatch.setattr(gq, "search", boom)
    assert fg.facts_about("anything at all") == []


def test_duplicates_across_anchors_are_dropped(monkeypatch):
    two = [
        {"id": "entity:a", "kind": "entity", "label": "Nissan car", "degree": 2},
        {"id": "entity:b", "kind": "entity", "label": "Nissan", "degree": 2},
    ]
    same = {"node": {"kind": "fact", "label": "The Nissan needs a decoy"}}
    hoods = {
        "entity:a": {"node": two[0], "neighbors": [same]},
        "entity:b": {"node": two[1], "neighbors": [same]},
    }
    _wire(monkeypatch, two, hoods)
    got = fg.facts_about("Nissan")
    assert len(got) == 1
