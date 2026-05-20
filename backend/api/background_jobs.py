"""REST endpoints for the background-job registry.

Phase 2 WebUI follow-up: the `start_background_job` tool stores its
state in `~/.hrant/data/jobs/background.json` and updates it from
the subprocess runner thread, but until now there was no HTTP
surface for the WebUI to read that state. The chat UI's
`TaskStatusCard` polls these endpoints every few seconds to render
a live status block (label, exit, elapsed, progress %, log tail)
for in-flight jobs.

Owner-only — same gate as `/api/chat`. Anyone with shell access
already owns the box.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..tools import background_jobs as _bg
from ._auth import require_owner_for_writes


log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/background-jobs")
async def list_jobs(status: str = "", limit: int = 20):
    """List background jobs (newest first). Optional `status` filter
    matches the registry's status values: running / done / error /
    interrupted / killed / vanished / stale."""
    require_owner_for_writes(action="reading background jobs")
    rows = _bg.list_jobs(status=(status or "").strip() or None, limit=limit)
    return {"jobs": rows}


@router.get("/api/background-jobs/{job_id}")
async def get_one(job_id: str):
    """Full record for a single job, including:
      • supervisor context (original_user_request, parent_job_id,
        retry_count, supervisor_history, supervisor_terminal)
      • watchdog context (total_units, progress_probe_cmd,
        last_heartbeat_progress, last_log_size, stale flag)
      • the stdout / stderr tail captured at the runner's
        boundary (full logs live on disk under
        ~/.hrant/data/jobs/<job_id>/{stdout,stderr}.log)

    The TaskStatusCard merges this with a separate live probe call
    (see `progress` field below)."""
    require_owner_for_writes(action="reading background jobs")
    row = _bg.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return row


@router.get("/api/background-jobs/{job_id}/progress")
async def get_progress(job_id: str):
    """Run the job's `progress_probe_cmd` once and return the
    sampled count + the implied percent. Used by the TaskStatusCard
    to refresh the progress bar without polling the full job record.
    Returns `{ done: int|null, total: int|null, pct: float|null }`."""
    require_owner_for_writes(action="reading background jobs")
    row = _bg.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    probe = row.get("progress_probe_cmd") or ""
    total = row.get("total_units") or None
    if not probe or not total:
        return {"done": None, "total": total, "pct": None}
    try:
        from .. import bg_job_watchdog as _bgw
        done = _bgw._run_probe(cmd=probe, cwd=row.get("cwd") or None)
    except Exception as e:
        log.warning("progress probe for %s failed: %s", job_id, e)
        done = None
    if done is None:
        return {"done": None, "total": total, "pct": None}
    pct = min(1.0, max(0.0, done / float(total))) if total > 0 else None
    return {"done": done, "total": total, "pct": pct}


@router.get("/api/background-jobs/{job_id}/log")
async def get_log(job_id: str, stream: str = "stdout", tail: int = 4000):
    """Return the tail of the job's stdout or stderr log. `stream`
    must be 'stdout' or 'stderr'. `tail` is the byte cap; the WebUI
    typically asks for 4 KB. Used by TaskStatusCard's expandable
    'log' section."""
    require_owner_for_writes(action="reading background jobs")
    if stream not in ("stdout", "stderr"):
        raise HTTPException(status_code=400, detail="stream must be stdout/stderr")
    row = _bg.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    log_path = _bg.STORE._job_dir(job_id) / f"{stream}.log"
    if not log_path.exists():
        # Pre-finalize jobs have no on-disk log yet — fall back to
        # the tail captured in the registry record.
        tail_text = row.get(f"{stream}_tail") or ""
        return {"tail": tail_text, "source": "registry"}
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            if size > tail:
                f.seek(size - tail)
            blob = f.read()
        text = blob.decode("utf-8", errors="replace")
    except Exception as e:
        log.warning("log read for %s failed: %s", job_id, e)
        text = ""
    return {"tail": text, "source": "disk", "size": log_path.stat().st_size}
