"""Append-only store pruning — Bundle B (Fix I8).

The audit found three stores that grew unbounded:
  - tools/background_jobs.py::BackgroundJobStore (background.json)
  - scheduled_messages.py (scheduled_messages.jsonl)
  - evaluator.py (eval_log.jsonl)

None of the three had a GC path. A box running benchmarks plants a
background-job row per launch; a daily reminder cron grows the
scheduled ledger forever; the evaluator writes a row per turn.
Every `_load` re-reads the whole file → multi-MB reads on every tick.

Pinned behaviour:
  - prune() drops oldest non-running / non-pending rows past max.
  - Running / pending rows are NEVER evicted regardless of cap.
  - Returns the count actually dropped.
  - Atomic rewrite (.tmp + rename) so a crash mid-prune is safe.
"""
from __future__ import annotations

import time

import pytest


# ─── BackgroundJobStore ─────────────────────────────────────────────


@pytest.fixture
def bg_store(tmp_path):
    from backend.tools.background_jobs import BackgroundJobStore, BackgroundJob
    store = BackgroundJobStore(root=tmp_path)
    return store, BackgroundJob


def _mk_bg_job(BackgroundJob, *, job_id, status, started_at):
    return BackgroundJob(
        job_id=job_id,
        label="t",
        command="echo hi",
        cwd="",
        started_at=started_at,
        finished_at=None,
        status=status,
        exit_code=0 if status == "done" else None,
        stdout_tail="",
        stderr_tail="",
        requester="webui:default",
        pid=None,
    )


def test_background_jobs_prune_drops_oldest_non_running(bg_store):
    """Past max_rows, oldest closed rows are dropped first."""
    store, BackgroundJob = bg_store

    # 5 closed jobs, ascending started_at.
    for i in range(5):
        store.add(_mk_bg_job(BackgroundJob, job_id=f"j{i}", status="done", started_at=100.0 + i))

    dropped = store.prune(max_rows=3)
    assert dropped == 2

    remaining = {j.job_id for j in store.list(limit=50)}
    # Oldest (j0, j1) dropped; newest survive.
    assert remaining == {"j2", "j3", "j4"}


def test_background_jobs_prune_keeps_running(bg_store):
    """Running rows survive prune even when they would otherwise be
    eligible for eviction by age."""
    store, BackgroundJob = bg_store

    # The running row is the OLDEST by started_at — without protection
    # it would be the first to go.
    store.add(_mk_bg_job(BackgroundJob, job_id="alive", status="running", started_at=10.0))
    for i in range(5):
        store.add(_mk_bg_job(BackgroundJob, job_id=f"d{i}", status="done", started_at=100.0 + i))

    dropped = store.prune(max_rows=3)
    # 5 closed + 1 running = 6 rows. max=3, but running stays no matter
    # what; so closed slots = 2. Drop 5-2=3.
    assert dropped == 3
    ids = {j.job_id for j in store.list(limit=50)}
    assert "alive" in ids
    # Two newest closed survive.
    assert "d4" in ids and "d3" in ids


def test_background_jobs_prune_below_cap_is_noop(bg_store):
    """When rows < max, prune does nothing and returns 0."""
    store, BackgroundJob = bg_store

    for i in range(3):
        store.add(_mk_bg_job(BackgroundJob, job_id=f"j{i}", status="done", started_at=100.0 + i))

    assert store.prune(max_rows=10) == 0
    assert len(store.list(limit=50)) == 3


def test_background_jobs_prune_keep_running_false(bg_store):
    """When keep_running=False, running rows are evictable too."""
    store, BackgroundJob = bg_store

    store.add(_mk_bg_job(BackgroundJob, job_id="alive", status="running", started_at=10.0))
    for i in range(3):
        store.add(_mk_bg_job(BackgroundJob, job_id=f"d{i}", status="done", started_at=100.0 + i))

    dropped = store.prune(max_rows=2, keep_running=False)
    # 4 rows, max 2, drop 2.
    assert dropped == 2
    ids = {j.job_id for j in store.list(limit=50)}
    # "alive" was the oldest — gone.
    assert "alive" not in ids


# ─── scheduled_messages ─────────────────────────────────────────────


@pytest.fixture
def isolated_sched(tmp_path, monkeypatch):
    from backend import scheduled_messages as _sm
    path = tmp_path / "scheduled.jsonl"
    monkeypatch.setattr(_sm, "_path", lambda: path)
    return _sm


def test_scheduled_messages_prune_drops_old_completed(isolated_sched):
    """Closed rows (sent/failed/cancelled) past max_rows are dropped
    oldest-first; pending rows are preserved."""
    sm = isolated_sched

    pending = sm.schedule(
        target_speaker="webui:default", text="future",
        due_at="2099-01-01T00:00:00Z", requested_by="webui:default",
    )

    # 5 historic rows; manually flip them to sent in order, so
    # requested_at orders chronologically.
    sent_ids = []
    for i in range(5):
        r = sm.schedule(
            target_speaker="webui:default", text=f"old-{i}",
            due_at="2026-01-01T00:00:00Z", requested_by="webui:default",
        )
        sm.mark_sent(r["id"])
        sent_ids.append(r["id"])
        # Tiny gap so requested_at sorts in insertion order.
        time.sleep(0.001)

    dropped = sm.prune(max_rows=3)
    # 6 total rows; pending always kept; max=3 → 1 pending + 2 closed
    # survive → drop 3.
    assert dropped == 3

    remaining = sm.list_all()
    statuses = {r["id"]: r["status"] for r in remaining}
    assert pending["id"] in statuses
    assert statuses[pending["id"]] == "pending"
    # 2 newest sent survive, 3 oldest dropped.
    sent_remaining = [r for r in remaining if r["status"] == "sent"]
    assert len(sent_remaining) == 2


def test_scheduled_messages_prune_below_cap_is_noop(isolated_sched):
    """When rows <= max, prune does nothing."""
    sm = isolated_sched

    for i in range(2):
        r = sm.schedule(
            target_speaker="webui:default", text=f"r{i}",
            due_at="2099-01-01T00:00:00Z", requested_by="webui:default",
        )
        sm.mark_sent(r["id"])

    assert sm.prune(max_rows=10) == 0
    assert len(sm.list_all()) == 2


def test_scheduled_messages_prune_preserves_all_pending(isolated_sched):
    """All-pending ledger: prune is a no-op even when over the cap.

    Pending rows are scheduled future work; the user is counting on
    them. Better to grow the file than silently drop scheduled
    reminders.
    """
    sm = isolated_sched

    for i in range(5):
        sm.schedule(
            target_speaker="webui:default", text=f"p{i}",
            due_at="2099-01-01T00:00:00Z", requested_by="webui:default",
        )

    dropped = sm.prune(max_rows=2)
    # No closed rows to evict; pending stays.
    assert dropped == 0
    assert len(sm.list_all()) == 5


# ─── evaluator ──────────────────────────────────────────────────────


def test_evaluator_prune_drops_oldest(tmp_path, monkeypatch):
    """eval_log.jsonl rows past max_rows are dropped oldest-first."""
    from backend.evaluator import SelfEvaluator, EvalEntry

    log_path = tmp_path / "eval_log.jsonl"
    ev = SelfEvaluator(path=log_path)

    for i in range(20):
        ev.log(EvalEntry(
            question=f"q{i}", intent="task", confidence=70,
            topics_used=[], contradictions=0, unverified=0,
            verified=1, is_chat=False, response_time_ms=10,
            ts=f"2026-06-{(i % 28) + 1:02d} 12:00:00",
        ))

    dropped = ev.prune(max_rows=5)
    assert dropped == 15
    # File should have only 5 lines.
    lines = [
        ln for ln in log_path.read_text(encoding="utf-8").split("\n")
        if ln.strip()
    ]
    assert len(lines) == 5


def test_evaluator_prune_below_cap_is_noop(tmp_path):
    """No-op when row count is below the cap."""
    from backend.evaluator import SelfEvaluator, EvalEntry

    log_path = tmp_path / "eval_log.jsonl"
    ev = SelfEvaluator(path=log_path)
    for i in range(3):
        ev.log(EvalEntry(
            question=f"q{i}", intent="task", confidence=70,
            topics_used=[], contradictions=0, unverified=0,
            verified=1, is_chat=False, response_time_ms=10,
        ))
    assert ev.prune(max_rows=10) == 0


def test_evaluator_prune_missing_file_is_noop(tmp_path):
    """When the log file doesn't exist yet, prune returns 0."""
    from backend.evaluator import SelfEvaluator

    ev = SelfEvaluator(path=tmp_path / "never_written.jsonl")
    assert ev.prune(max_rows=100) == 0
