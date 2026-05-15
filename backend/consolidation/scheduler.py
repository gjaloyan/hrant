"""Background asyncio task that fires daily consolidation.

Gating logic (in order):
  1. Cooldown: was the LAST run at least COOLDOWN_SECONDS ago?
  2. Idleness: has the agent been quiet for IDLE_THRESHOLD_SECONDS?
  3. Activity: MIN_JOBS_FOR_RUN met? (default 0 → always fires)
  4. (Phase 16B will add) Cost-cap not exceeded.

The task runs SCHEDULER_TICK_SECONDS apart (60s default), checks
the gates, fires if open. Cheap — just a timestamp comparison
plus a single `jobs.list(limit=1)` call to find latest activity.

Lifecycle:
  - `start_scheduler(app)` is called from FastAPI lifespan startup
  - The asyncio task lives on `app.state.consolidation_task`
  - `stop_scheduler(app)` is called from lifespan shutdown — it
    sets a cancel event so the loop exits cleanly before uvicorn
    tears down

Concurrency:
  Only one consolidation runs at a time. A second tick that
  triggers while the first is still in `pipeline.run()` is gated
  by `_running` so the LLM router isn't hammered twice.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from . import config, digest as _digest_mod, gather, pipeline, state

log = logging.getLogger(__name__)


_running = asyncio.Lock()


async def _fire_one(*, force: bool = False, dry_run: bool = False):
    """Run the pipeline once. Updates persisted state at the end.
    Honours the lock so two concurrent fire requests serialise
    (the second waits, doesn't skip — audit #13 comment fix).

    `force` only relaxes the early "another run in progress"
    short-circuit; it does NOT bypass the lock itself. A force
    request still queues behind whatever's running."""
    if _running.locked() and not force:
        log.info("consolidation: another run is already in progress; skipping tick")
        return None
    async with _running:
        # Audit #2: wrap the whole pipeline + digest-write + state-
        # save chain so an exception in any one of those updates
        # persisted state to status=failed. Pre-fix: only
        # pipeline.run's internal failures landed in state; an
        # exception from `digest.write` (disk full) or `state.save`
        # left `last_run_status` lying as "success" from the
        # previous run, indefinitely.
        try:
            bundle = await asyncio.to_thread(gather.gather)
            d = await asyncio.to_thread(pipeline.run, bundle=bundle, dry_run=dry_run)
            path = await asyncio.to_thread(_digest_mod.write, d)
            # Persist scheduler state for the WebUI banner.
            st = state.load()
            st.last_run_at = d.completed_at or time.time()
            st.last_run_status = d.status
            st.last_run_digest = str(path)
            st.last_run_error = d.error
            st.last_run_duration_seconds = (
                (d.completed_at or 0.0) - (d.started_at or 0.0)
            )
            st.last_run_tokens_used = d.tokens_used
            st.last_run_cost_usd_estimate = d.estimated_cost_usd
            st.last_run_speakers_seen = list(d.speakers_active)
            st.last_run_jobs_analyzed = d.turns_analyzed
            st.last_run_facts_added = sum(1 for f in d.new_facts if f.promoted)
            st.total_runs += 1
            state.save(st)
            # Audit #17: piggyback an opportunistic jobs/ cleanup
            # on each daily consolidation so the directory doesn't
            # grow unbounded. Default keeps failed/interrupted
            # forever (they're the audit log) and prunes
            # completed/cancelled jobs older than 30 days. Best
            # effort — cleanup failures must not break the run.
            try:
                from .. import jobs as _jobs
                purged = _jobs.JOBS.cleanup_old(
                    max_age_seconds=30 * 86400.0,
                    keep_statuses={"failed", "interrupted"},
                )
                if purged:
                    log.info(
                        "consolidation: opportunistic jobs cleanup removed %d",
                        len(purged),
                    )
            except Exception as e:
                log.warning("consolidation: jobs cleanup failed: %s", e)
            return d
        except Exception as e:
            log.exception("consolidation: _fire_one outer failure: %s", e)
            # Best-effort: record the failure in state. If state.save
            # itself is what's broken we just log; nothing else to do.
            try:
                st = state.load()
                st.last_run_at = time.time()
                st.last_run_status = "failed"
                st.last_run_error = f"scheduler exception: {type(e).__name__}: {e}"[:500]
                state.save(st)
            except Exception:
                log.warning("could not persist failed-run state after outer error")
            return None


def _should_fire(now: Optional[float] = None) -> tuple[bool, str]:
    """Return (should_fire, reason). `reason` always populated so
    the WebUI / log can show 'waiting because X'."""
    if now is None:
        now = time.time()
    st = state.load()
    cooldown_left = state.cooldown_remaining_seconds(st)
    if cooldown_left > 0:
        return False, f"cooldown ({int(cooldown_left)}s remaining)"
    last_active = gather.last_activity_ts()
    if last_active is not None:
        idle_for = now - last_active
        if idle_for < config.IDLE_THRESHOLD_SECONDS:
            return False, f"not idle (active {int(idle_for)}s ago)"
    # `MIN_JOBS_FOR_RUN` defaults to 1: skip truly empty days
    # (the LLM pipeline would short-circuit anyway, but skipping
    # here avoids a useless state.save round-trip).
    if config.MIN_JOBS_FOR_RUN > 0:
        # Audit #4: the prior version called `gather.gather()` on
        # every tick after cooldown ended — a full scan of jobs/
        # every 60s. For MIN=1 (the default) we only need to know
        # whether ANY job exists, which `last_activity_ts()` already
        # answered above. Only fall back to the full scan when the
        # gate requires a real count (MIN >= 2).
        if config.MIN_JOBS_FOR_RUN == 1:
            if last_active is None:
                return False, "too few turns (0 < 1)"
        else:
            bundle = gather.gather()
            if bundle.turn_count < config.MIN_JOBS_FOR_RUN:
                return (
                    False,
                    f"too few turns ({bundle.turn_count} < {config.MIN_JOBS_FOR_RUN})",
                )
    return True, "ready"


async def _scheduler_loop(cancel_event: asyncio.Event) -> None:
    """The actual asyncio task. Wakes every SCHEDULER_TICK_SECONDS,
    checks gates, fires if open. Exits when cancel_event is set."""
    log.info(
        "consolidation scheduler started "
        "(tick=%.0fs cooldown=%.0fs idle=%.0fs)",
        config.SCHEDULER_TICK_SECONDS,
        config.COOLDOWN_SECONDS,
        config.IDLE_THRESHOLD_SECONDS,
    )
    while not cancel_event.is_set():
        try:
            should, reason = _should_fire()
            if should:
                log.info("consolidation: gates open (%s), firing", reason)
                try:
                    await _fire_one()
                except Exception as e:
                    log.exception("consolidation run crashed: %s", e)
            else:
                # Debug-level only to avoid spamming the log every 60s
                # with "cooldown 86399s remaining".
                log.debug("consolidation: skip (%s)", reason)
        except Exception as e:
            log.exception("consolidation scheduler tick crashed: %s", e)
        try:
            await asyncio.wait_for(
                cancel_event.wait(),
                timeout=config.SCHEDULER_TICK_SECONDS,
            )
        except asyncio.TimeoutError:
            continue  # normal — just the tick interval expired
    log.info("consolidation scheduler stopping")


# ─── Lifespan hooks ──────────────────────────────────────────────────


async def start_scheduler(application) -> None:
    """Called from main.py:lifespan() at startup.

    Audit #11: idempotent — a second call (test reruns, lifespan
    re-entry on hot reload) cancels the previous task before
    starting a new one rather than leaking it forever."""
    prev = getattr(application.state, "consolidation_task", None)
    prev_event = getattr(application.state, "consolidation_cancel", None)
    if prev is not None and not prev.done():
        log.info("consolidation: cancelling previous scheduler task before restart")
        if prev_event is not None:
            prev_event.set()
        try:
            await asyncio.wait_for(prev, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            prev.cancel()
            try:
                await prev
            except (asyncio.CancelledError, Exception):
                pass
    cancel_event = asyncio.Event()
    application.state.consolidation_cancel = cancel_event
    application.state.consolidation_task = asyncio.create_task(
        _scheduler_loop(cancel_event),
    )


async def stop_scheduler(application) -> None:
    """Called from main.py:lifespan() at shutdown.

    Audit #20: properly await the cancellation. Previously, on
    a timeout we called `task.cancel()` but never awaited the
    cancelled task — uvicorn could tear down with the task in a
    half-cancelled state. Now we await again after cancel."""
    ev = getattr(application.state, "consolidation_cancel", None)
    task = getattr(application.state, "consolidation_task", None)
    if ev is not None:
        ev.set()
    if task is not None:
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("consolidation scheduler shutdown error: %s", e)


# ─── Manual fire (used by REST + CLI) ───────────────────────────────


async def fire_now(*, dry_run: bool = False) -> _digest_mod.Digest:
    """Public entry: run a consolidation right now, regardless of
    gates. Returns the Digest. Used by `POST /api/consolidation/run?force=true`
    and `hrant consolidate run --force`."""
    d = await _fire_one(force=True, dry_run=dry_run)
    if d is None:
        # Lock was held — wait briefly, then return whatever the
        # last run produced.
        await asyncio.sleep(0.1)
        st = state.load()
        if st.last_run_digest:
            existing = _digest_mod.read(_digest_mod.today_str())
            if existing is not None:
                return existing
        # Fallback: empty failed digest so caller sees something
        d = _digest_mod.Digest(
            date=_digest_mod.today_str(),
            started_at=time.time(),
            completed_at=time.time(),
            status="skipped",
            skip_reason="another_run_in_progress",
        )
    return d


def status() -> dict:
    """Snapshot of scheduler state for the WebUI banner."""
    st = state.load()
    should, reason = _should_fire()
    now = time.time()
    last_active = gather.last_activity_ts()
    return {
        "state": st.to_dict(),
        "would_fire_now": should,
        "gate_reason": reason,
        "now": now,
        "cooldown_remaining_seconds": state.cooldown_remaining_seconds(st),
        "idle_for_seconds": (now - last_active) if last_active is not None else None,
        "config": {
            "idle_threshold_seconds": config.IDLE_THRESHOLD_SECONDS,
            "cooldown_seconds": config.COOLDOWN_SECONDS,
            "min_jobs_for_run": config.MIN_JOBS_FOR_RUN,
            "tick_seconds": config.SCHEDULER_TICK_SECONDS,
        },
    }
