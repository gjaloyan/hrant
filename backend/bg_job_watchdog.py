"""Background-job watchdog (Phase 2 of audit T6 follow-up).

Phase 1 closed the loop on job COMPLETION: when a subprocess exits,
`job_supervisor.on_job_completed` spawns a supervisor turn that
decides retry / deliver / escalate. But Phase 1 leaves three holes
in long-running jobs:

  1. Heartbeat — for a 9-hour SWE-bench the user gets ZERO signal
     until terminal. Past behaviour pattern shows they'll type
     `status?` after a few hours to verify liveness. We want to
     send proactive "still running, 60% complete" DMs at meaningful
     milestones so the user doesn't need to ask.

  2. Vanished — the subprocess can die without firing `on_done`:
     OOM-killer, systemd shutdown, segfault that takes out
     `_runner` itself, or a job spawned via `systemd-run --user`
     that detached and the runner never saw its exit. Registry
     keeps showing `status=running` forever. Need to detect when
     the recorded PID is gone from `ps` and trigger a supervisor.

  3. Stale — the process is technically alive but stopped making
     progress: log file hasn't grown, no new report.json files,
     stuck in a flaky-network retry loop. The supervisor needs to
     decide kill-and-retry vs wait-longer.

This module owns one daemon thread that ticks every 60 seconds and
walks each running job:

  • PID liveness via `psutil.pid_exists` (falls back to `os.kill(pid, 0)`
    when psutil isn't importable, which still works on POSIX).
  • Progress probe — if the job set `progress_probe_cmd`, run it
    under cwd, parse an int from stdout, divide by `total_units`.
    Emit a passive heartbeat DM at each 30% milestone (30 / 60 / 90),
    throttled so the same milestone can't double-fire.
  • Time-fallback heartbeat — for jobs without a probe, emit one
    "still running, elapsed Xh Ym" DM every 2 hours.
  • Stale check — sample stdout.log size; if it hasn't grown in
    >30 min AND the job has been running for >10 min, open a
    supervisor turn with a synthetic "stale" message (and set
    `stale_supervisor_opened` so we don't re-fire while it's
    deciding).

All side-effects (DMs, supervisor turn) go through the same
fallback DM hook + `_run_supervisor_turn` paths Phase 1 wired up,
so nothing is duplicated.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import time
from typing import Optional

from .tools.background_jobs import BackgroundJob, STORE


log = logging.getLogger(__name__)


# Watchdog tick interval. 60s strikes a balance: fast enough to
# notice a 30% milestone before the user pings, slow enough that the
# probe subprocesses don't add measurable CPU on a 9-hour run.
_TICK_INTERVAL_SEC = 60.0

# Progress milestones (in fraction, not percent). Fire a heartbeat
# the first time the running pct crosses one of these.
_PROGRESS_MILESTONES = [0.30, 0.60, 0.90]

# Time-fallback: jobs WITHOUT a progress probe still get a passive
# liveness DM every this-many seconds so the user has some signal
# on multi-hour runs.
_TIME_FALLBACK_INTERVAL_SEC = 2 * 3600.0

# Stale-detection knobs: a job that hasn't grown its stdout.log in
# THIS-many seconds, AND has been alive at least `_STALE_MIN_AGE_SEC`,
# is considered stuck → supervisor turn.
_STALE_NO_GROWTH_SEC = 30 * 60.0
_STALE_MIN_AGE_SEC = 10 * 60.0

# Probe subprocess timeout. The probe runs every 60s on every running
# job; a slow probe must not hang the watchdog tick.
_PROBE_TIMEOUT_SEC = 10.0


def _pid_alive(pid: Optional[int]) -> bool:
    """True if the given PID is alive on this host. Returns True for
    None/0 — we can't say it's dead, so don't trigger vanished
    detection on a job that hasn't recorded its pid yet."""
    if not pid:
        return True
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        pass
    # POSIX fallback: signal 0 doesn't deliver but does the existence
    # check. Returns True on Windows because os.kill(pid, 0) raises
    # on a missing pid AND on an inaccessible-but-live pid — the
    # error is ambiguous. So on Windows without psutil we assume
    # alive (false negative is safer than false positive on
    # vanished-detection).
    if os.name != "posix":
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but isn't ours — still alive from our POV.
        return True
    except Exception:
        return True


def _run_probe(
    *, cmd: str, cwd: Optional[str],
) -> Optional[int]:
    """Run `cmd` and parse an integer from stdout. Returns None on
    any failure (non-zero exit, non-int output, timeout). Used only
    for progress probing — failures are non-fatal, the watchdog
    just skips the milestone check this tick."""
    if not (cmd or "").strip():
        return None
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=(cwd or None),
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        # Allow probes that emit `K/N` ("90/300"); take the first int.
        first = out.split()[0] if out else ""
        first = first.split("/")[0].strip()
        return int(first)
    except Exception:
        return None


def _highest_milestone_crossed(
    *, pct_now: float, pct_last: Optional[float],
) -> Optional[float]:
    """Find the highest progress milestone in `_PROGRESS_MILESTONES`
    that we've crossed since `pct_last`. Returns the milestone (e.g.
    0.30) or None if we haven't crossed anything new. Throttles to
    AT MOST ONE milestone fire per tick — if we go from 25% to 70%
    in one observation we'd otherwise burst two heartbeats."""
    if pct_last is None:
        pct_last = 0.0
    crossed = [
        m for m in _PROGRESS_MILESTONES
        if pct_last < m <= pct_now
    ]
    return max(crossed) if crossed else None


def _format_milestone_dm(
    job: BackgroundJob, *, pct: float, done: int,
) -> str:
    """Compose the passive heartbeat DM for a progress-milestone
    crossing. Telegram HTML formatting; kept short — the user wants
    to see liveness, not a wall of text."""
    elapsed = int(time.time() - (job.started_at or time.time()))
    h, rem = divmod(elapsed, 3600)
    m, _ = divmod(rem, 60)
    pct_int = int(round(pct * 100))
    return (
        f"📊 <b>Job running</b> — <code>{job.job_id}</code>\n"
        f"<b>label:</b> {job.label[:120]}\n"
        f"<b>progress:</b> {pct_int}% ({done}/{job.total_units})\n"
        f"<b>elapsed:</b> {h}h {m}m"
    )


def _format_timefallback_dm(job: BackgroundJob) -> str:
    """Composite passive DM for jobs without a progress probe.
    Carries elapsed time + the tail of the log so the user sees
    the job is making *some* output."""
    elapsed = int(time.time() - (job.started_at or time.time()))
    h, rem = divmod(elapsed, 3600)
    m, _ = divmod(rem, 60)
    tail = ""
    # Read live log tail directly from disk (the in-memory tail in
    # the job record is only updated when the runner buffers a
    # chunk — usually on exit).
    try:
        log_path = STORE._job_dir(job.job_id) / "stdout.log"
        if log_path.exists():
            data = log_path.read_bytes()
            if data:
                tail_bytes = data[-600:]
                tail = tail_bytes.decode("utf-8", errors="replace")
    except Exception:
        pass
    out = [
        f"📊 <b>Job still running</b> — <code>{job.job_id}</code>",
        f"<b>label:</b> {job.label[:120]}",
        f"<b>elapsed:</b> {h}h {m}m",
    ]
    if tail.strip():
        # Escape and tighten the tail so it doesn't blow the
        # message length budget.
        from .tg_format import escape_html  # late import — avoid cycle
        out.append(
            f"<b>stdout (tail):</b>\n<pre>{escape_html(tail[-600:])}</pre>"
        )
    return "\n".join(out)


def _send_heartbeat_dm(job: BackgroundJob, text: str) -> bool:
    """Best-effort heartbeat DM. Uses the same channel hook the
    supervisor's `complete_supervisor` tool uses. Returns True on
    success so the caller can decide whether to update
    `last_heartbeat_at`."""
    try:
        from . import channels as _ch
        cm = getattr(_ch, "CHANNELS", None)
        if cm is None:
            return False
        for _ch_obj in (getattr(cm, "channels", {}) or {}).values():
            if hasattr(_ch_obj, "_send_with_buttons") and job.original_chat_id:
                try:
                    _ch_obj._send_with_buttons(
                        job.original_chat_id, text, None,
                    )
                    return True
                except Exception as e:
                    log.warning(
                        "heartbeat DM via channel %r failed: %s",
                        type(_ch_obj).__name__, e,
                    )
        return False
    except Exception as e:
        log.warning("heartbeat DM dispatch failed: %s", e)
        return False


def _open_stale_supervisor(job: BackgroundJob, reason: str) -> None:
    """Kick the supervisor turn with a synthetic message that the
    job appears stuck. The LLM decides kill-and-retry vs wait-more.
    Marks `stale_supervisor_opened` so the next tick doesn't
    re-fire while it's deciding."""
    job.stale_supervisor_opened = True
    STORE.update(job)
    # Reuse `on_job_completed` but stamp a synthetic stderr that
    # makes the supervisor's decision tree obvious.
    synthetic = BackgroundJob.from_dict(job.to_dict())
    synthetic.status = "stale"
    synthetic.stderr_tail = (
        (synthetic.stderr_tail or "")
        + f"\n[watchdog: {reason}]"
    )
    try:
        from . import job_supervisor as _jsup
        _jsup.on_job_completed(synthetic)
    except Exception as e:
        log.warning("stale supervisor open failed: %s", e)


def _open_vanished_supervisor(job: BackgroundJob) -> None:
    """When the recorded PID disappeared without `on_done` firing
    (OOM-kill, segfault, host reboot), flip the job to `vanished`
    and let the supervisor decide retry / escalate."""
    job.status = "vanished"
    job.finished_at = time.time()
    job.exit_code = -1
    job.stderr_tail = (
        (job.stderr_tail or "")
        + f"\n[watchdog: PID {job.pid} disappeared without exit]"
    )
    STORE.update(job)
    try:
        from . import job_supervisor as _jsup
        _jsup.on_job_completed(job)
    except Exception as e:
        log.warning("vanished supervisor open failed: %s", e)


def _tick_one(job: BackgroundJob) -> None:
    """One watchdog evaluation of a single running job."""
    # 1. Liveness — if pid is recorded and gone, mark vanished.
    if job.pid and not _pid_alive(job.pid):
        log.info("watchdog: job %s PID %s vanished", job.job_id, job.pid)
        _open_vanished_supervisor(job)
        return

    now = time.time()
    age = now - (job.started_at or now)

    # 2. Progress milestone heartbeat.
    if job.progress_probe_cmd and job.total_units:
        done = _run_probe(cmd=job.progress_probe_cmd, cwd=job.cwd or None)
        if done is not None and job.total_units > 0:
            pct = min(1.0, max(0.0, done / float(job.total_units)))
            milestone = _highest_milestone_crossed(
                pct_now=pct,
                pct_last=job.last_heartbeat_progress,
            )
            if milestone is not None:
                if _send_heartbeat_dm(job, _format_milestone_dm(
                    job, pct=pct, done=done,
                )):
                    job.last_heartbeat_at = now
                    job.last_heartbeat_progress = milestone
                    STORE.update(job)
                    return

    # 3. Time-fallback heartbeat for jobs without a probe.
    if not job.progress_probe_cmd:
        last_at = job.last_heartbeat_at or job.started_at or now
        if (now - last_at) >= _TIME_FALLBACK_INTERVAL_SEC:
            if _send_heartbeat_dm(job, _format_timefallback_dm(job)):
                job.last_heartbeat_at = now
                STORE.update(job)

    # 4. Stale detection — sample stdout.log size growth. We only
    # open a stale supervisor ONCE per job; the flag clears when
    # the supervisor decides (it'll either kill-and-retry, which
    # creates a child with its own flag, or mark terminal).
    if not job.stale_supervisor_opened and age >= _STALE_MIN_AGE_SEC:
        try:
            log_path = STORE._job_dir(job.job_id) / "stdout.log"
            if log_path.exists():
                size_now = log_path.stat().st_size
                if (
                    job.last_log_size is not None
                    and size_now == job.last_log_size
                    and job.last_log_size_at is not None
                    and (now - job.last_log_size_at) >= _STALE_NO_GROWTH_SEC
                ):
                    _open_stale_supervisor(
                        job,
                        f"stdout.log unchanged for "
                        f"{int((now - job.last_log_size_at) / 60)}m "
                        f"(size {size_now} bytes)",
                    )
                    return
                # Update sample if size changed (or first sample).
                if (
                    job.last_log_size is None
                    or size_now != job.last_log_size
                ):
                    job.last_log_size = size_now
                    job.last_log_size_at = now
                    STORE.update(job)
        except Exception as e:
            log.debug("stale check for %s failed: %s", job.job_id, e)


def _gc_sweep_if_due(state: dict) -> None:
    """Audit follow-up: run the question + endpoint store GC once
    every 24h. `state` is the watchdog loop's mutable scratch dict —
    tracks `last_gc_at` so we don't sweep every minute. Per-store
    failures are swallowed; this is best-effort housekeeping."""
    last = state.get("last_gc_at") or 0.0
    if time.time() - last < 86400:  # once per day
        return
    state["last_gc_at"] = time.time()
    try:
        from .tools import ask_user as _aq
        removed_q = _aq.STORE.gc_old(older_than_days=7)
        if removed_q:
            log.info("watchdog: question store GC removed %d", removed_q)
    except Exception as e:
        log.warning("watchdog: question GC failed: %s", e)
    try:
        from . import task_endpoint as _te
        removed_e = _te.STORE.gc_old(older_than_days=14)
        if removed_e:
            log.info("watchdog: endpoint store GC removed %d", removed_e)
    except Exception as e:
        log.warning("watchdog: endpoint GC failed: %s", e)
    try:
        from .log_bus import gc_old as _lb_gc
        removed_l = _lb_gc(days=7)
        if removed_l:
            log.info("watchdog: log files GC removed %d", removed_l)
    except Exception as e:
        log.warning("watchdog: log GC failed: %s", e)
    # Audit Important #12 (2026-05-23): workspace/ holds turn
    # artifacts (~7MB each) under linear-growth pressure. The store's
    # opportunistic sweep only fires from save_outbox/save_turn —
    # idle days leave the marker stale. Run from the daily watchdog
    # tick so cleanup happens even when the agent isn't being used.
    try:
        from .workspace import get_workspace
        from .config import CONFIG
        ws_cfg = CONFIG.workspace
        get_workspace().sweep_old(
            inbox_retention_days=int(ws_cfg.get("inbox_retention_days", 90) or 0),
            outbox_retention_days=int(ws_cfg.get("outbox_retention_days", 0) or 0),
            notes_retention_days=int(ws_cfg.get("notes_retention_days", 0) or 0),
            turns_retention_days=int(ws_cfg.get("turns_retention_days", 30) or 0),
        )
    except Exception as e:
        log.warning("watchdog: workspace GC failed: %s", e)


def _watchdog_loop() -> None:
    """Daemon-thread main. Loops forever; per-job failures are
    swallowed so one bad job can't break the watchdog for the rest."""
    log.info("bg-job watchdog: started, tick interval %.0fs", _TICK_INTERVAL_SEC)
    state: dict = {"last_gc_at": 0.0}
    while True:
        try:
            jobs = STORE.list(status="running", limit=200)
        except Exception as e:
            log.warning("watchdog: list() failed: %s", e)
            jobs = []
        for job in jobs:
            # Don't re-fire on supervisor-terminal chains.
            if job.supervisor_terminal:
                continue
            try:
                _tick_one(job)
            except Exception as e:
                log.warning(
                    "watchdog: _tick_one for %s failed: %s",
                    job.job_id, e,
                )
        # Daily housekeeping — sweep stale answered questions and
        # terminal endpoints so the JSON stores don't grow forever.
        try:
            _gc_sweep_if_due(state)
        except Exception as e:
            log.warning("watchdog: GC sweep failed: %s", e)
        time.sleep(_TICK_INTERVAL_SEC)


_started = False
_lock = threading.Lock()


def start_watchdog() -> None:
    """Idempotent — safe to call multiple times. First call spawns
    the daemon thread, subsequent calls are no-ops. Invoked from
    the FastAPI lifespan startup."""
    global _started
    with _lock:
        if _started:
            return
        t = threading.Thread(
            target=_watchdog_loop,
            daemon=True,
            name="bg-job-watchdog",
        )
        t.start()
        _started = True
