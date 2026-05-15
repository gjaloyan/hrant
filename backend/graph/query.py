"""Read-only graph queries used by the API + CLI + WebUI.

Kept thin on purpose — heavy graph algorithms live elsewhere.
What's here:
    - `stats` — counts per node kind, edge kind, top-N topics by
      degree (the topic with the most facts attached)
    - `neighborhood` — node + edges + immediate neighbours, the
      view that backs `GET /api/graph/node/{id}`
    - `search` — case-insensitive substring search across node
      labels, ranked by degree
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Optional

from .store import GRAPH, Graph
from .model import NODE_KINDS


def stats(*, graph: Optional[Graph] = None) -> dict:
    """Summary numbers for the Knowledge tab's status banner."""
    g = graph or GRAPH
    by_kind: dict[str, int] = {k: 0 for k in NODE_KINDS}
    for n in g.iter_nodes():
        by_kind[n.kind] = by_kind.get(n.kind, 0) + 1

    edge_kinds: Counter = Counter()
    for e in g.iter_edges():
        edge_kinds[e.kind] += 1

    # Top topics by degree (incoming edges). Tells the user "what
    # is the agent talking about most".
    degree: Counter = Counter()
    for e in g.iter_edges():
        degree[e.target] += 1
    top_topic_nodes = [
        g.get_node(node_id)
        for node_id, _ in degree.most_common(20)
        if g.get_node(node_id) is not None
           and g.get_node(node_id).kind == "topic"  # type: ignore[union-attr]
    ][:10]
    top_topics = [
        {"id": n.id, "label": n.label, "degree": degree[n.id]}
        for n in top_topic_nodes if n is not None
    ]

    return {
        "total_nodes": g.node_count(),
        "total_edges": g.edge_count(),
        "by_kind": by_kind,
        "edge_kinds": dict(edge_kinds),
        "top_topics": top_topics,
    }


def neighborhood(node_id: str, *, graph: Optional[Graph] = None) -> Optional[dict]:
    """Node + its neighbours. Returns None if the node doesn't
    exist. Used by the detail-pane view in the WebUI."""
    g = graph or GRAPH
    n = g.get_node(node_id)
    if n is None:
        return None
    neighbours: list[dict] = []
    for edge, other in g.neighbors(node_id):
        # Direction matters for the UI: "I am the source" vs "I am
        # the target". Tag each entry so the renderer can use a
        # different arrow direction.
        direction = "out" if edge.source == node_id else "in"
        neighbours.append({
            "edge": asdict(edge),
            "node": asdict(other),
            "direction": direction,
        })
    return {
        "node": asdict(n),
        "neighbors": neighbours,
        "neighbor_count": len(neighbours),
    }


def search(
    query: str,
    *,
    kind: Optional[str] = None,
    limit: int = 50,
    graph: Optional[Graph] = None,
) -> list[dict]:
    """Substring search across node labels. Case-insensitive.
    Results ranked by degree (most-connected first) so the most
    "central" matches surface at the top.

    `kind` filter limits to one node kind (fact/topic/skill/...).
    """
    g = graph or GRAPH
    q = (query or "").strip().lower()
    if not q:
        return []
    matches: list = []
    degree: Counter = Counter()
    for e in g.iter_edges():
        degree[e.source] += 1
        degree[e.target] += 1
    for n in g.iter_nodes(kind=kind):
        if q in n.label.lower():
            matches.append((n, degree[n.id]))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [
        {**asdict(n), "degree": deg}
        for n, deg in matches[:limit]
    ]
