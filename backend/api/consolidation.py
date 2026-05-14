"""REST endpoints for daily memory consolidation.

  GET  /api/consolidation/status            — scheduler state + gates
  POST /api/consolidation/run?force=true&dry_run=false  — fire now
  GET  /api/consolidation/digests           — index of past digests
  GET  /api/consolidation/digests/{date}    — single digest record

Phase 16B will add:
  - POST /api/consolidation/rollback/{date}
  - POST /api/consolidation/trash/{item_id}/restore

Phase 16C:
  - GET  /api/consolidation/graph
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..consolidation import digest as _digest_mod
from ..consolidation import scheduler as _sched

router = APIRouter()


@router.get("/api/consolidation/status")
def get_status():
    """Scheduler state for the WebUI banner: last run, next eligible
    time, idle status, would-fire-now flag with reason."""
    return _sched.status()


@router.post("/api/consolidation/run")
async def run_consolidation(
    force: bool = Query(
        default=True,
        description="Bypass cooldown + idle gates (default true — the WebUI "
                    "wouldn't call this endpoint unless the user clicked "
                    "Run, which is by definition an override).",
    ),
    dry_run: bool = Query(
        default=False,
        description="Preview without writing memory_facts or profiles. The "
                    "digest itself is still persisted so the WebUI can show "
                    "what WOULD have been added.",
    ),
):
    """Manual fire from the WebUI's 'Run now' button or the CLI's
    `hrant consolidate run`. Returns the full digest synchronously
    — caller blocks until the LLM pipeline completes (~30-60s for
    a typical day)."""
    # NB: the `force` flag is mostly cosmetic — `fire_now` always
    # forces. Kept on the URL so a future "request but respect gates"
    # mode has a place to live without a breaking change.
    _ = force
    d = await _sched.fire_now(dry_run=dry_run)
    return d.to_dict()


@router.get("/api/consolidation/digests")
def list_digests(limit: int = Query(default=30, ge=1, le=365)):
    """Lightweight index, newest first. Each row is small enough
    that the WebUI can render 30 of them on the Memory Digests
    tab without a per-row follow-up fetch."""
    rows = _digest_mod.list_all()
    return {"total": len(rows), "digests": rows[:limit]}


@router.get("/api/consolidation/digests/{date_str}")
def get_digest(date_str: str):
    """Full digest record for one day (YYYY-MM-DD)."""
    # Cheap path-traversal guard — the date filename is timestamp-
    # shaped so anything with `/` or `..` is invalid.
    if "/" in date_str or "\\" in date_str or ".." in date_str:
        raise HTTPException(status_code=400, detail="invalid date")
    d = _digest_mod.read(date_str)
    if d is None:
        raise HTTPException(status_code=404, detail="digest not found")
    return d.to_dict()
