"""on_job_completed must spawn the supervisor turn exactly once.

Audit 2026-06-10 (C5): the previous implementation read
`job.supervisor_terminal` from the in-memory argument without
re-fetching from STORE. Two near-simultaneous `_fire_done` callbacks
(runner's finally + heartbeat watchdog, or two competing on_done
subscribers) both saw False and both spawned a supervisor daemon
thread — double DM, double retry chain, possibly divergent decisions.

The fix re-reads under _STORE_LOCK and uses a process-level
`_in_flight_supervisor` set to claim ownership atomically BEFORE
spawning the thread. The wrapper releases the claim in `finally`.
"""
from __future__ import annotations

import threading
import time

import pytest


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    """Each test gets a fresh jobs/ root so the global registry
    doesn't bleed across tests."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.tools import background_jobs as _bg
    monkeypatch.setattr(_bg.STORE, "_root_override", tmp_path / "jobs")
    yield tmp_path


def _make_running_job():
    """Materialize a fresh non-terminal job in the store."""
    from backend.tools import background_jobs as _bg
    j = _bg.BackgroundJob(
        job_id="bg-race-1",
        label="race-test",
        command="true",
        cwd="",
        started_at=time.time(),
        finished_at=time.time(),
        status="done",
        exit_code=0,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=None,
    )
    _bg.STORE.add(j)
    return j


def test_double_callback_spawns_only_one_supervisor_turn(
    isolated_jobs, monkeypatch,
):
    """Two `on_job_completed` calls for the same job within the
    contention window spawn the supervisor turn EXACTLY ONCE."""
    from backend import job_supervisor as _jsup

    spawn_count = {"n": 0}
    barrier = threading.Event()

    def _fake_run(job):
        # Hold the supervisor turn open until both callbacks have had
        # a chance to compete for the in-flight claim.
        spawn_count["n"] += 1
        barrier.wait(timeout=2.0)

    monkeypatch.setattr(_jsup, "_run_supervisor_turn", _fake_run)
    # Ensure the in-flight set is clean.
    _jsup._in_flight_supervisor.clear()

    job = _make_running_job()

    # Fire two callbacks back-to-back from the main thread (mimicking
    # two on_done subscribers racing the same job_id).
    _jsup.on_job_completed(job)
    _jsup.on_job_completed(job)

    # Allow the spawn(s) to enter _fake_run, then release the barrier.
    time.sleep(0.05)
    barrier.set()
    # Give any spawned daemon thread time to exit so the wrapper
    # releases the claim.
    time.sleep(0.1)

    assert spawn_count["n"] == 1, (
        f"expected exactly one supervisor turn spawn, got {spawn_count['n']}"
    )


def test_callback_for_already_terminal_job_does_not_spawn(
    isolated_jobs, monkeypatch,
):
    """A late callback for a chain the supervisor already closed
    must not re-open it."""
    from backend import job_supervisor as _jsup
    from backend.tools import background_jobs as _bg

    spawn_count = {"n": 0}

    def _fake_run(job):
        spawn_count["n"] += 1

    monkeypatch.setattr(_jsup, "_run_supervisor_turn", _fake_run)
    _jsup._in_flight_supervisor.clear()

    job = _make_running_job()
    # Seal the chain first.
    _jsup.mark_terminal(job.job_id, decision="done", reason="pre-sealed")
    # Now fire on_done.
    _jsup.on_job_completed(job)
    time.sleep(0.05)

    assert spawn_count["n"] == 0


def test_three_concurrent_callbacks_only_one_spawn(
    isolated_jobs, monkeypatch,
):
    """Re-audit 2026-06-10 (IMPORTANT): the two-callback test proves
    sequential same-job callbacks dedup, but the actual contention
    pattern under load is N>=3 concurrent callbacks (runner finally +
    heartbeat watchdog + on_done subscribers on the same channel).
    The in-flight claim must cover the entire spawn window, not just
    the lookup window."""
    from backend import job_supervisor as _jsup

    spawn_count = {"n": 0}
    spawn_lock = threading.Lock()
    barrier = threading.Event()

    def _slow_run(job):
        with spawn_lock:
            spawn_count["n"] += 1
        barrier.wait(timeout=2.0)

    monkeypatch.setattr(_jsup, "_run_supervisor_turn", _slow_run)
    _jsup._in_flight_supervisor.clear()

    job = _make_running_job()

    threads = [
        threading.Thread(target=_jsup.on_job_completed, args=(job,))
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    # Let spawns enter _slow_run, then release the barrier so daemons
    # can exit and the finally{} releases the claim.
    time.sleep(0.1)
    barrier.set()
    for t in threads:
        t.join(timeout=1.5)

    assert spawn_count["n"] == 1, (
        f"3-way contention must spawn EXACTLY once, got {spawn_count['n']}"
    )


def test_in_flight_set_cleared_after_run(isolated_jobs, monkeypatch):
    """After the supervisor turn returns the in-flight claim is
    released, so a LATER (legitimate) re-run for a different reason
    can proceed."""
    from backend import job_supervisor as _jsup

    def _fake_run(job):
        pass  # immediate return

    monkeypatch.setattr(_jsup, "_run_supervisor_turn", _fake_run)
    _jsup._in_flight_supervisor.clear()

    job = _make_running_job()
    _jsup.on_job_completed(job)
    # Wait for the daemon thread's `finally` to release the claim.
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if job.job_id not in _jsup._in_flight_supervisor:
            break
        time.sleep(0.02)
    assert job.job_id not in _jsup._in_flight_supervisor
