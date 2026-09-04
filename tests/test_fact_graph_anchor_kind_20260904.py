"""The traversal searched every node kind and then threw most away.

With the graph finally carrying edges (21,425 after the persistence fix,
up from 61), `facts_about("Gor")` still returned nothing. The nodes were
there: `topic:gor` with degree 4, `entity:gor` with degree 4.

`gq.search` ranks by degree across ALL kinds, and a fact whose label is a
whole sentence containing "Gor" outranks the two-letter topic node —

    search("Gor", limit=8) -> six facts, degrees 14, 14, 12, 12, 7, 7

so the entity/topic filter applied AFTERWARDS had nothing left to keep.
`gq.search` has taken a `kind` argument the whole time; asking for the
kinds it wants is the fix.
"""
from unittest.mock import patch

from backend import fact_graph


def test_anchors_are_requested_by_kind_not_filtered_afterwards():
    asked = []

    def _search(term, *, kind=None, limit=50, graph=None):
        asked.append(kind)
        if kind == "topic":
            return [{"id": "topic:gor", "kind": "topic", "label": "Gor",
                     "degree": 4}]
        return []

    def _neighborhood(node_id, **kw):
        return {"neighbors": [
            {"node": {"id": "fact:1", "kind": "fact",
                      "label": "User's name is Gor."}},
        ]}

    with patch("backend.graph.query.search", _search), \
         patch("backend.graph.query.neighborhood", _neighborhood):
        out = fact_graph.facts_about("Gor")

    assert None not in asked, "an unfiltered search lets facts crowd out anchors"
    assert {"entity", "topic"} <= set(asked)
    assert out and out[0]["summary"] == "User's name is Gor."
    assert out[0]["via"] == "Gor"


def test_a_node_with_no_links_is_still_not_an_anchor():
    """Unchanged: there is nothing to traverse from it."""
    def _search(term, *, kind=None, limit=50, graph=None):
        return [{"id": "topic:x", "kind": "topic", "label": "x", "degree": 0}]

    with patch("backend.graph.query.search", _search):
        assert fact_graph.facts_about("something") == []
