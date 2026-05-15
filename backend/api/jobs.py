"""REST endpoints for the durable job records.

  GET    /api/jobs                  — list with filters + pagination
  GET    /api/jobs/{id}             — single job (full record)
  POST   /api/jobs/{id}/retry       — clone as a new queued job
  POST   /api/jobs/{id}/cancel      — mark a non-terminal job cancelled
  DELETE /api/jobs/{id}             — purge the file
  GET    /api/jobs/_/stats          — counts per status (for badges)

The WebUI Jobs tab is the primary consumer. CLI (`hrant jobs ...`)
goes straight to `backend.jobs.JOBS`; it doesn't need HTTP.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .. import jobs as _jobs
from ._auth import require_owner_for_writes

router = APIRouter()


@router.get("/api/jobs")
def list_jobs(
    status: Optional[str] = Query(default=None, description="filter by status"),
    channel: Optional[str] = Query(default=None, description="filter by channel"),
    speaker_id: Optional[str] = Query(default=None, description="filter by speaker"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Paged list of jobs, newest first. Filter params are AND-ed.

    The full job record is returned for each row (not a slim
    summary) — `prompt` + `response` are usually short enough that
    this stays under 100KB even at limit=500. The WebUI uses them
    directly to render the list without a per-row /api/jobs/{id}
    follow-up."""
    rows = _jobs.JOBS.list(
        status=status,
        channel=channel,
        speaker_id=speaker_id,
        limit=limit,
        offset=offset,
    )
    # `total` reflects the same filter as `jobs[]` — otherwise the
    # WebUI badge ("Jobs (50 total)") would lie when a status filter
    # is active. Count once via the same filter, full scan is fine
    # at the <1k-jobs scale this surface targets.
    total_filtered = sum(
        1 for _ in _jobs.JOBS.list(
            status=status, channel=channel, speaker_id=speaker_id,
            limit=10_000, offset=0,
        )
    )
    return {
        "total": total_filtered,
        "total_all": _jobs.JOBS.count(),
        "limit": limit,
        "offset": offset,
        "jobs": [j.to_dict() for j in rows],
    }


@router.get("/api/jobs/_/stats")
def job_stats():
    """Counts per status. Used by the WebUI Jobs-tab badge to show
    'N interrupted needing attention' without fetching the full list."""
    return {
        s: _jobs.JOBS.count(status=s) for s in _jobs.VALID_STATUSES
    }


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@router.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    """Clone the job's prompt into a new `queued` record. Does NOT
    run it — the new job sits queued until the user re-sends the
    prompt via /api/chat or the equivalent channel. Returns the
    new job id so the WebUI can poll its status."""
    require_owner_for_writes(action="retrying a job")
    orig = _jobs.JOBS.get(job_id)
    if orig is None:
        raise HTTPException(status_code=404, detail="job not found")
    new_job = _jobs.JOBS.retry(job_id)
    if new_job is None:
        raise HTTPException(status_code=500, detail="retry failed")
    return {"new_job_id": new_job.id, "prompt": new_job.prompt, "channel": new_job.channel}


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    require_owner_for_writes(action="cancelling a job")
    job = _jobs.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in _jobs.TERMINAL_STATUSES:
        # Idempotent — already terminal. Tell the user but don't 4xx.
        return {"ok": True, "status": job.status, "note": "already terminal"}
    _jobs.JOBS.mark_cancelled(job_id)
    return {"ok": True, "status": "cancelled"}


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    require_owner_for_writes(action="deleting a job")
    ok = _jobs.JOBS.delete(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True}


@router.post("/api/jobs/_/cleanup")
def cleanup_jobs(
    max_age_days: int = Query(default=30, ge=1, le=3650),
    keep_failed: bool = Query(
        default=True,
        description="When true, failed/interrupted jobs survive regardless of age (audit log).",
    ),
):
    """Purge old jobs to bound disk usage. By default keeps everything
    failed/interrupted so the user can still retry them weeks later;
    completed/cancelled jobs are pure history and get pruned.

    Manual trigger only — there's no auto-cleanup cron yet. Run from
    `Settings → Jobs → Cleanup` or `hrant jobs cleanup --older-than 30d`."""
    require_owner_for_writes(action="cleaning up jobs")
    keep = {"failed", "interrupted"} if keep_failed else set()
    deleted = _jobs.JOBS.cleanup_old(
        max_age_seconds=max_age_days * 86400.0,
        keep_statuses=keep,
    )
    return {"ok": True, "deleted_count": len(deleted), "deleted": deleted}
