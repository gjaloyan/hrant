"""REST endpoints for the subagent dispatch records.

  GET    /api/subagents                 — list persisted sessions
                                          (paged, newest first)
  GET    /api/subagents/active          — currently-running sessions
                                          (cheap, no disk I/O)
  GET    /api/subagents/_/stats         — counts by status
  GET    /api/subagents/roles           — registry: name → description
                                          so the WebUI knows the
                                          dropdown contents
  GET    /api/subagents/{session_id}    — single session, full record

The Settings → Subagents tab is the primary consumer. The list /
active endpoints return full records (not slim summaries) — they're
small enough that a 500-row page stays well under 100KB.

No cancel endpoint in v1. Cooperative interrupt is on the roadmap
but the current dispatcher runs `router.call_with_tools` to
completion before returning; killing the worker mid-LLM is its own
project. Until then, the panel is display-only.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..subagents import SUBAGENT_STORE, available_roles


router = APIRouter()


@router.get("/api/subagents")
def list_subagents(
    status: Optional[str] = Query(default=None, description="filter by status"),
    role: Optional[str] = Query(default=None, description="filter by role name"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Newest first. `status` ∈ {running, completed, failed};
    `role` ∈ {researcher, coder, reviewer}. Total reflects the
    SAME filter (not the unfiltered count) so the WebUI badge
    doesn't lie when a filter is active."""
    rows = SUBAGENT_STORE.list(
        status=status, role=role, limit=limit, offset=offset,
    )
    total_filtered = sum(
        1 for _ in SUBAGENT_STORE.list(
            status=status, role=role, limit=10_000, offset=0,
        )
    )
    return {
        "total": total_filtered,
        "limit": limit,
        "offset": offset,
        "subagents": [r.to_dict() for r in rows],
    }


@router.get("/api/subagents/active")
def active_subagents():
    """Live snapshot of currently-running subagent dispatches.

    The active registry is in-memory only — when the process restarts
    it's empty, even if disk records say something was running.
    That's a feature: subagents are synchronous, so a process restart
    means the dispatch was interrupted; the persisted record stays in
    `running` state until the next finalize call (which won't happen
    after restart) but the active list reflects ground truth."""
    rows = SUBAGENT_STORE.active()
    return {
        "count": len(rows),
        "subagents": [r.to_dict() for r in rows],
    }


@router.get("/api/subagents/_/stats")
def subagent_stats():
    """Counts per status for the WebUI tab badge."""
    return SUBAGENT_STORE.stats()


@router.get("/api/subagents/roles")
def list_roles():
    """Registry — name → one-line description. The WebUI uses this to
    populate the manual-delegate dropdown (future) AND to render
    role pills with consistent colours. Keeping it server-side means
    a new role appears in the WebUI without a frontend deploy."""
    return {"roles": available_roles()}


@router.get("/api/subagents/{session_id}")
def get_subagent(session_id: str):
    """Full session record including the tool_calls list. The list
    can carry up to `role.max_iterations` records, each with a
    200-char `result_preview`, so the response is bounded."""
    sess = SUBAGENT_STORE.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="subagent session not found")
    return sess.to_dict()
