"""REST endpoints for the knowledge graph (Phase 16C).

  GET  /api/kgraph             — full graph (nodes + edges)
  GET  /api/kgraph/stats       — counts per kind, top topics
  GET  /api/kgraph/search?q=…  — substring search across labels
  GET  /api/kgraph/node/{id}   — single node + neighbours
  POST /api/kgraph/rebuild     — re-derive from sources

The full-graph endpoint is fine at the personal-agent scale
(typical ~50–500 nodes), but the WebUI prefers `/stats` +
`/search` + `/node/{id}` for incremental loading — never sends
the whole graph unless the user opens the "Graph view" sub-tab.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..graph import builder as _builder, query as _query
from ..graph.store import GRAPH
from ._auth import require_owner_for_writes

router = APIRouter()


@router.get("/api/kgraph")
def get_graph():
    """Full graph. Used by the WebUI's "Graph view" SVG canvas.
    Skipped on initial tab load; only fetched when the user clicks
    "Show graph"."""
    return GRAPH.to_dict()


@router.get("/api/kgraph/stats")
def get_stats():
    """Lightweight summary — what the Knowledge tab loads on first
    render. Keeps the initial paint snappy."""
    return _query.stats()


@router.get("/api/kgraph/search")
def search(
    q: str = Query(default="", description="substring (case-insensitive)"),
    kind: Optional[str] = Query(
        default=None,
        description="filter by node kind: fact | topic | skill | project | entity",
    ),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Ranked matches by node label."""
    return {"q": q, "kind": kind, "results": _query.search(q, kind=kind, limit=limit)}


@router.get("/api/kgraph/node/{node_id:path}")
def get_node(node_id: str):
    """Node + immediate neighbourhood. Used by the WebUI detail
    pane. `:path` lets the id contain `/` (none of ours do, but
    paranoid)."""
    if ".." in node_id:
        raise HTTPException(status_code=400, detail="invalid node id")
    out = _query.neighborhood(node_id)
    if out is None:
        raise HTTPException(status_code=404, detail="node not found")
    return out


@router.post("/api/kgraph/rebuild")
def rebuild():
    """Wipe the in-memory graph and re-derive from current sources
    (memory_facts.jsonl + skills + goals.json). Synchronous — runs
    in the request thread. At <1k facts this completes in under
    a second; if it ever gets slow, move to a background task."""
    require_owner_for_writes(action="rebuilding the knowledge graph")
    stats = _builder.rebuild()
    return {"ok": True, "stats": stats}
