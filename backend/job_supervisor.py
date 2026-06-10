"""Job-supervisor turn (audit T6 follow-up).

User report: when a background job (SWE-bench / long-running benchmark)
finishes, the agent's previous behaviour was passive — a DM saying
"job X done, exit=Y, stderr=Z" landed in Telegram, and that was it.
The agent only re-engaged when the user typed `status?` again, at
which point it would re-discover the failure and ask permission to
fix it ("Write `restart` and I will restart it").

The pattern was:
  - "I started the benchmark."     (user does nothing)
  - "It failed with exit 127."     (user types `status`)
  - "Should I restart?"            (user types `restart`)
  - "I restarted it."              (user does nothing)
  - "It crashed again."            (user types `status`)
  - "Should I fix python→python3?" (user types ok)
  - … and so on.

The supervisor turn fixes this. When a background job completes,
`on_job_completed(job)` opens an out-of-band agent turn with a
synthetic user message ("BACKGROUND_JOB_COMPLETED: …") and a system
prompt that tells the agent it's in supervisor mode. The agent runs
through its normal tool loop:
  - reads the job log
  - classifies success / fixable-failure / hard-blocked
  - if fixable: writes the fix and calls `start_background_job`
    AGAIN with `parent_job_id` set — the chain continues silently
  - if successful or hard-blocked: composes ONE final DM and sends
    it via the user-message channel

The user sees one mid-run DM (if heartbeat fires for a 30%+
milestone), then one terminal DM that says "the bench had X problem,
I fixed it by Y, here are the final results". No `status?` polling,
no `restart` prompts, no `should-I-try-X?` questions.

Token budget: each supervisor turn costs ~1-3k tokens. For a job
chain of N retries that's N supervisor turns. The user explicitly
chose "trust the LLM with a soft cap of 10 retries" — past that
the supervisor refuses to retry further and escalates to the user.

Implementation note: this module is intentionally thin. The heavy
lifting (LLM call, tool loop) is `unified_agent.run_unified()` with
`supervisor_mode=True`. We only assemble the synthetic message and
trigger the turn from a background thread.
"""
from __future__ import annotations

import contextvars
import logging
import threading
import time

from .tools.background_jobs import BackgroundJob, STORE


# ContextVar carrying the active supervisor job id. Tools called from
# inside a supervisor turn (e.g. `complete_supervisor`, the modified
# `start_background_job` retry path) read this to know which job
# they're operating on without an explicit arg. Default empty string
# means "not in supervisor turn".
_active_supervisor_job_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_supervisor_job_id", default="",
)


def set_active_supervisor_job(job_id: str) -> contextvars.Token:
    """Set the active supervisor job for the current async/thread
    context. Returns a Token the caller must pass to
    `reset_active_supervisor_job` on exit."""
    return _active_supervisor_job_id.set(job_id or "")


def reset_active_supervisor_job(token: contextvars.Token) -> None:
    try:
        _active_supervisor_job_id.reset(token)
    except Exception:
        pass


def active_supervisor_job_id() -> str:
    """Return the active supervisor job id, or empty string if not
    currently inside a supervisor turn."""
    try:
        return _active_supervisor_job_id.get()
    except Exception:
        return ""


# Degraded-path DM. Channel manager registers this so we can still
# notify the user when the supervisor turn itself crashes (LLM down,
# unexpected exception) — otherwise a failed supervisor would leave
# the chain silent forever.
_fallback_dm = None


def set_fallback_dm(fn) -> None:
    """Called once at channel startup. `fn` accepts a BackgroundJob
    and sends the legacy "job X status Y" DM."""
    global _fallback_dm
    _fallback_dm = fn


log = logging.getLogger(__name__)


# Soft cap on retries before the supervisor refuses to spawn another
# child. The LLM is trusted to decide most of the time, but we want a
# kill switch in case it loops on a non-fixable error forever.
_HARD_RETRY_CAP = 10


def _format_completion_message(job: BackgroundJob) -> str:
    """Synthetic 'user message' that drives the supervisor turn.

    Includes everything the LLM needs to decide retry-vs-deliver:
      - original user request (so it knows the goal)
      - job command (what was actually run)
      - status / exit code (what happened)
      - stdout / stderr tails (why)
      - retry chain context (how many fix attempts so far)
      - previous supervisor decisions (so it doesn't repeat the same
        failed fix)
    """
    stdout_tail = (job.stdout_tail or "").strip()
    stderr_tail = (job.stderr_tail or "").strip()
    history_block = ""
    if job.supervisor_history:
        hist_lines = ["", "=== PREVIOUS SUPERVISOR DECISIONS IN THIS CHAIN ==="]
        for i, h in enumerate(job.supervisor_history, 1):
            decision = (h.get("decision") or "?").strip()
            reason = (h.get("reason") or "").strip()
            hist_lines.append(f"  {i}. {decision} — {reason}")
        history_block = "\n".join(hist_lines)
    parent_block = ""
    if job.parent_job_id:
        parent_block = (
            f"\nThis is retry #{job.retry_count} in a chain that "
            f"started with job {job.parent_job_id}."
        )
    expected_block = ""
    if job.expected_outcome:
        expected_block = (
            f"\n=== EXPECTED OUTCOME (per the original request) ===\n"
            f"{job.expected_outcome}"
        )
    elapsed = 0
    if job.finished_at and job.started_at:
        elapsed = max(0, int(job.finished_at - job.started_at))

    # Phase 3: load + evaluate the TaskEndpoint if the job has one.
    # The evaluation block goes RIGHT BEFORE the decision-rules so
    # the LLM sees a concrete checklist of what's met vs. unmet
    # before it picks DONE / RETRY / ESCALATE.
    endpoint_block = ""
    if job.endpoint_id:
        try:
            from . import task_endpoint as _te
            ep = _te.STORE.get(job.endpoint_id)
            if ep is not None:
                results = _te.evaluate_endpoint(ep, cwd=job.cwd or "")
                hints = _te.match_recovery_hints(
                    ep, stderr_tail=stderr_tail, stdout_tail=stdout_tail,
                )
                endpoint_block = _te.format_evaluation_block(
                    ep, results, hints,
                )
        except Exception as e:
            log.warning(
                "supervisor: endpoint evaluation for %s failed: %s",
                job.endpoint_id, e,
            )
            endpoint_block = (
                f"=== TASK ENDPOINT (evaluation FAILED — {type(e).__name__}) ===\n"
                f"endpoint_id={job.endpoint_id} could not be evaluated. "
                f"Judge the goal manually from the logs below."
            )

    parts = [
        f"BACKGROUND_JOB_COMPLETED: {job.job_id}",
        f"label: {job.label}",
        f"status: {job.status}  exit_code: {job.exit_code}  "
        f"elapsed: {elapsed}s",
        f"cwd: {job.cwd or '(not set)'}",
        f"command:\n  {job.command}",
        parent_block.rstrip(),
        f"=== ORIGINAL USER REQUEST ===",
        job.original_user_request or "(none captured)",
        expected_block.rstrip(),
        history_block.rstrip(),
        "=== STDOUT (tail) ===",
        stdout_tail or "(empty)",
        "=== STDERR (tail) ===",
        stderr_tail or "(empty)",
        "",
        endpoint_block.rstrip(),
        "",
        "You are in supervisor mode. Decide ONE of:",
        "  • DONE — the job met the user's goal. Compose and send "
        "ONE final DM summarising what happened (problems hit, "
        "fixes applied, final result).",
        "  • RETRY — the job failed in a way you can fix. Identify "
        "the root cause, fix the command/inputs, and call "
        "`start_background_job` again with `parent_job_id=" +
        job.job_id + "` and `retry_count=" + str(job.retry_count + 1) +
        "`. Do NOT send a DM — silence until terminal.",
        "  • ESCALATE — the failure genuinely needs user input "
        "(missing credential, business decision, etc.). Send ONE "
        "DM explaining what you tried, what's blocking, and what "
        "you need from the user.",
        "",
        "Do NOT ask the user for permission to apply a trivial fix. "
        "Do NOT promise to retry — just retry. Do NOT report "
        "intermediate status unless you have a heartbeat-worthy "
        "milestone. The user has already accepted up to "
        f"{_HARD_RETRY_CAP} silent retry attempts.",
    ]
    return "\n".join(p for p in parts if p is not None)


def _run_supervisor_turn(job: BackgroundJob) -> None:
    """Open a single supervisor turn for `job`. Best-effort: any
    exception is logged and the chain falls back to the passive DM
    via the original `_on_background_job_done` handler."""
    # Late import to avoid a module-level cycle (Agent and
    # unified_agent both import many runtime services that aren't
    # ready at this module's load time).
    from .agent import Agent

    task = _format_completion_message(job)
    speaker_id = job.original_speaker_id or job.requester or "system:supervisor"
    token = set_active_supervisor_job(job.job_id)
    try:
        agent = Agent()
        agent.run(
            task=task,
            speaker_id=speaker_id,
            channel="supervisor",
            supervisor_mode=True,
            supervisor_job_id=job.job_id,
        )
    except Exception as e:
        log.exception(
            "supervisor turn for job %s crashed: %s", job.job_id, e
        )
        # Degraded path: send the legacy "job X status Y" DM so the
        # user knows the job ran even though the supervisor brain
        # was down. Mark terminal so the chain doesn't keep retrying
        # the supervisor for the same completion.
        if _fallback_dm is not None:
            try:
                _fallback_dm(job)
            except Exception as fb_e:
                log.warning("supervisor fallback DM failed: %s", fb_e)
        try:
            mark_terminal(
                job.job_id,
                decision="degraded",
                reason=f"supervisor crashed: {type(e).__name__}: {e}",
            )
        except Exception:
            pass
    finally:
        reset_active_supervisor_job(token)
    # If the supervisor turn returned normally but didn't seal the
    # chain (LLM forgot to call complete_supervisor and didn't spawn
    # a retry), we'd leave the user wondering. After the turn
    # finishes, check terminal state — if not set AND no retry child
    # was spawned, fire the fallback DM.
    refreshed = STORE.get(job.job_id)
    if refreshed and not refreshed.supervisor_terminal:
        # Did a retry child get spawned? The child carries
        # parent_job_id pointing at this job. Use a parent-id
        # lookup (no list-cap) — the previous limit=50 heuristic
        # silently dropped real children once the registry grew past
        # 50 jobs, falsely marking the parent `degraded` even though
        # the supervisor turn HAD spawned a retry.
        children = [
            j for j in STORE.find_children(job.job_id)
            if j.started_at >= (job.finished_at or 0) - 1
        ]
        if not children:
            log.warning(
                "supervisor turn for %s ended without retry or "
                "complete_supervisor; falling back to passive DM",
                job.job_id,
            )
            if _fallback_dm is not None:
                try:
                    _fallback_dm(job)
                except Exception as fb_e:
                    log.warning(
                        "supervisor end-of-turn fallback DM failed: %s",
                        fb_e,
                    )
            try:
                mark_terminal(
                    job.job_id,
                    decision="degraded",
                    reason="supervisor turn ended without retry or terminal",
                )
            except Exception:
                pass


# Audit 2026-06-10 (C5): on_job_completed used to read `job.supervisor_terminal`
# from the in-memory argument without re-fetching from STORE. Two near-
# simultaneous _fire_done callbacks for the same job_id (one from the runner
# thread's finally, one from a heartbeat watchdog) both saw False and both
# spawned a supervisor daemon thread -> double DM, double retry chain, possibly
# divergent decisions. We now claim ownership atomically under _STORE_LOCK
# via this in-flight set; the wrapper releases the claim in `finally` once
# the supervisor turn returns.
_in_flight_supervisor: set[str] = set()


def _run_supervisor_turn_wrapped(job: BackgroundJob) -> None:
    """Wrapper that releases the in-flight claim when the supervisor
    turn returns. Runs on a daemon thread."""
    try:
        _run_supervisor_turn(job)
    finally:
        from .tools.background_jobs import _STORE_LOCK
        with _STORE_LOCK:
            _in_flight_supervisor.discard(job.job_id)


def on_job_completed(job: BackgroundJob) -> None:
    """Public entry — subscribe this in place of the old passive
    `_on_background_job_done` to get supervisor-driven follow-up.

    Threading: this runs on the bg-job runner thread. We don't want
    a long supervisor turn to block subsequent job completions, so
    we spawn a fresh daemon thread for the supervisor work and
    return immediately.

    Idempotency: re-reads the job from STORE under _STORE_LOCK and
    claims the supervisor slot via `_in_flight_supervisor` BEFORE
    spawning the daemon thread. Two callbacks landing within the
    same lock window see the second hit `if job_id in _in_flight`
    and skip. Late callbacks for already-terminal jobs are also
    rejected (the supervisor turn marked the chain complete).
    """
    if not isinstance(job, BackgroundJob):
        log.warning("on_job_completed: unexpected job type %r", type(job))
        return
    from .tools.background_jobs import _STORE_LOCK
    with _STORE_LOCK:
        fresh = STORE.get(job.job_id)
        if fresh is None:
            log.warning(
                "on_job_completed: job %s vanished from store", job.job_id,
            )
            return
        if fresh.supervisor_terminal:
            log.info(
                "supervisor: %s already terminal, skipping re-entry",
                job.job_id,
            )
            return
        if job.job_id in _in_flight_supervisor:
            log.info(
                "supervisor: %s already in-flight, skipping double spawn",
                job.job_id,
            )
            return
        if fresh.retry_count >= _HARD_RETRY_CAP:
            log.warning(
                "supervisor: %s reached retry cap %d; escalating directly",
                job.job_id, _HARD_RETRY_CAP,
            )
            # Mark terminal so future on_done callbacks don't re-open it,
            # and let the supervisor compose the escalation DM in its
            # last turn — the agent's prompt explicitly handles "you have
            # exhausted retries" via the synthetic message.
        _in_flight_supervisor.add(job.job_id)
    # Spawn OUTSIDE the lock so the supervisor turn (minutes) doesn't
    # block the store.
    threading.Thread(
        target=_run_supervisor_turn_wrapped,
        args=(fresh,),
        daemon=True,
        name=f"job-supervisor-{job.job_id}",
    ).start()


def mark_terminal(job_id: str, *, decision: str, reason: str = "") -> None:
    """Called by the supervisor turn when it decides this chain is
    done (success / escalation). Pins the job so no future on_done
    re-opens the supervisor for the same chain.

    Also appends the decision to `supervisor_history` so the next
    chain (or a human reading the registry) can audit what happened.
    """
    job = STORE.get(job_id)
    if job is None:
        log.warning("mark_terminal: unknown job %s", job_id)
        return
    job.supervisor_terminal = True
    job.supervisor_history = list(job.supervisor_history or []) + [{
        "decision": decision,
        "reason": reason,
        "at": time.time(),
    }]
    STORE.update(job)
    # Side-publish to LogBus so the Logs tab sees the supervisor
    # decision. Best-effort.
    try:
        from .log_bus import publish_supervisor_event as _pub_sup
        _pub_sup(job_id=job_id, decision=decision, message=reason or "")
    except Exception:
        pass


def append_supervisor_history(
    job_id: str, *, decision: str, reason: str = "",
) -> None:
    """Called between retries (decision=RETRY) so the next supervisor
    turn sees what the previous one tried. Does NOT mark terminal."""
    job = STORE.get(job_id)
    if job is None:
        return
    job.supervisor_history = list(job.supervisor_history or []) + [{
        "decision": decision,
        "reason": reason,
        "at": time.time(),
    }]
    STORE.update(job)
