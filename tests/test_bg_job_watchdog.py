"""Tests for the background-job watchdog (audit T6 Phase 2).

The watchdog is a daemon thread that ticks every 60s and walks
running jobs to detect:
  - vanished PIDs (process gone without `on_done` firing)
  - progress milestones (heartbeat DM at 30/60/90%)
  - time-fallback liveness (DM every 2h for jobs without a probe)
  - stale state (log file hasn't grown in 30+ min)

These tests pin the per-job decision logic by calling `_tick_one`
directly on synthesized jobs. The watchdog loop itself (a daemon
thread with sleeps) is tested separately via `start_watchdog`
idempotency.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.tools import background_jobs as _bg
    monkeypatch.setattr(_bg.STORE, "_root_override", tmp_path / "jobs")
    yield tmp_path


# ─── _highest_milestone_crossed pure logic ────────────────────────


def test_milestone_cross_first_30():
    from backend.bg_job_watchdog import _highest_milestone_crossed
    assert _highest_milestone_crossed(pct_now=0.31, pct_last=None) == 0.30


def test_milestone_no_cross_below_threshold():
    from backend.bg_job_watchdog import _highest_milestone_crossed
    assert _highest_milestone_crossed(pct_now=0.20, pct_last=None) is None


def test_milestone_no_double_fire_same_threshold():
    """If last heartbeat already fired at 0.30, a 0.40 sample must
    NOT re-fire 0.30 — only the NEXT threshold (0.60) counts."""
    from backend.bg_job_watchdog import _highest_milestone_crossed
    assert _highest_milestone_crossed(pct_now=0.40, pct_last=0.30) is None


def test_milestone_jump_picks_highest_crossed():
    """If progress jumps from 25% to 70% in a single tick, the
    watchdog must coalesce — fire ONE heartbeat at 0.60 (the highest
    crossed), not two heartbeats at 0.30 and 0.60."""
    from backend.bg_job_watchdog import _highest_milestone_crossed
    assert _highest_milestone_crossed(pct_now=0.70, pct_last=0.0) == 0.60


def test_milestone_third_fires_after_second():
    from backend.bg_job_watchdog import _highest_milestone_crossed
    assert _highest_milestone_crossed(pct_now=0.95, pct_last=0.60) == 0.90


# ─── _run_probe — integer parsing ─────────────────────────────────


def test_run_probe_parses_integer_stdout():
    """A shell command printing an int — captured."""
    from backend.bg_job_watchdog import _run_probe
    # `echo 42` is portable on Windows (cmd.exe) and POSIX sh.
    assert _run_probe(cmd="echo 42", cwd=None) == 42


def test_run_probe_parses_k_over_n_form():
    """Probes may emit `90/300` shape; take the first int."""
    from backend.bg_job_watchdog import _run_probe
    assert _run_probe(cmd="echo 90/300", cwd=None) == 90


def test_run_probe_returns_none_on_garbage():
    from backend.bg_job_watchdog import _run_probe
    assert _run_probe(cmd="echo not-an-int", cwd=None) is None


def test_run_probe_returns_none_on_empty_cmd():
    from backend.bg_job_watchdog import _run_probe
    assert _run_probe(cmd="", cwd=None) is None


# ─── _pid_alive ───────────────────────────────────────────────────


def test_pid_alive_none_returns_true():
    """No recorded PID → don't trigger vanished detection."""
    from backend.bg_job_watchdog import _pid_alive
    assert _pid_alive(None) is True
    assert _pid_alive(0) is True


def test_pid_alive_self_pid_is_true():
    """The interpreter's own PID is definitely alive."""
    import os as _os
    from backend.bg_job_watchdog import _pid_alive
    assert _pid_alive(_os.getpid()) is True


def test_pid_alive_dead_pid_returns_false():
    """A PID that's definitely not allocated returns False. PID 1 is
    init on POSIX (always alive); we use a likely-free PID 999999."""
    from backend.bg_job_watchdog import _pid_alive
    # On Windows without psutil this falls through to True (the
    # POSIX fallback path is skipped). Skip the assertion on Windows
    # without psutil since the test isn't meaningful there.
    try:
        import psutil  # noqa: F401
        have_psutil = True
    except ImportError:
        have_psutil = False
    import os as _os
    if not have_psutil and _os.name != "posix":
        pytest.skip("liveness detection needs psutil on Windows")
    assert _pid_alive(999999) is False


# ─── start_watchdog is idempotent ─────────────────────────────────


def test_start_watchdog_idempotent():
    """Multiple calls don't spawn multiple threads. Important for
    reload/hot-restart flows where lifespan re-runs."""
    from backend import bg_job_watchdog
    bg_job_watchdog.start_watchdog()
    bg_job_watchdog.start_watchdog()
    bg_job_watchdog.start_watchdog()
    # Internal flag should be True; we don't inspect threads directly
    # because the thread is daemon and may have exited or be sleeping.
    assert bg_job_watchdog._started is True


# ─── _tick_one — vanished detection ───────────────────────────────


def test_tick_one_marks_vanished_when_pid_gone(
    isolated_jobs, monkeypatch
):
    """Job recorded `status=running` with a PID that no longer
    exists → watchdog flips to `vanished` and opens supervisor."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    # Synthesize a running job. Don't use start_job() — that would
    # actually spawn a real subprocess.
    job = _bg.BackgroundJob(
        job_id="bg-vanish-test",
        label="vanish-test",
        command="sleep 9999",
        cwd="",
        started_at=time.time() - 600,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=999998,  # certainly-not-allocated
    )
    _bg.STORE.add(job)

    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: False)
    opened = {"job": None}
    def _fake_open(_job):
        opened["job"] = _job
    monkeypatch.setattr(
        "backend.job_supervisor.on_job_completed", _fake_open,
    )

    _bgw._tick_one(job)
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.status == "vanished"
    assert refreshed.exit_code == -1
    assert "PID 999998 disappeared" in refreshed.stderr_tail
    assert opened["job"] is not None
    assert opened["job"].job_id == job.job_id


def test_tick_one_no_vanished_when_pid_alive(
    isolated_jobs, monkeypatch
):
    """Same job, but PID still alive — must NOT mark vanished."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    job = _bg.BackgroundJob(
        job_id="bg-alive-test",
        label="alive-test",
        command="sleep 9999",
        cwd="",
        started_at=time.time() - 120,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
    )
    _bg.STORE.add(job)
    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    opened = {"job": None}
    monkeypatch.setattr(
        "backend.job_supervisor.on_job_completed",
        lambda j: opened.__setitem__("job", j),
    )

    _bgw._tick_one(job)
    assert opened["job"] is None
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.status == "running"


# ─── _tick_one — milestone heartbeat ──────────────────────────────


def test_tick_one_fires_milestone_heartbeat(
    isolated_jobs, monkeypatch
):
    """Job with progress_probe_cmd + total_units. Probe returns 90
    out of 300 (30%). Watchdog fires the 0.30 milestone DM and
    persists last_heartbeat_progress."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    job = _bg.BackgroundJob(
        job_id="bg-milestone-test",
        label="SWE-bench Lite 300",
        command="run-bench",
        cwd="",
        started_at=time.time() - 120,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
        total_units=300,
        progress_probe_cmd="echo 90",
        original_chat_id=848732236,
    )
    _bg.STORE.add(job)
    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    sent = {"text": None}
    monkeypatch.setattr(
        _bgw, "_send_heartbeat_dm",
        lambda _job, text: sent.__setitem__("text", text) or True,
    )

    _bgw._tick_one(job)
    assert sent["text"] is not None
    assert "30%" in sent["text"]
    assert "(90/300)" in sent["text"]
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.last_heartbeat_progress == 0.30


def test_tick_one_does_not_re_fire_same_milestone(
    isolated_jobs, monkeypatch
):
    """Once the 30% milestone has fired, a subsequent tick at 45%
    must NOT re-fire 30%."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    job = _bg.BackgroundJob(
        job_id="bg-repeat-test",
        label="SWE-bench Lite 300",
        command="run-bench",
        cwd="",
        started_at=time.time() - 600,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
        total_units=300,
        progress_probe_cmd="echo 135",  # 45%
        last_heartbeat_progress=0.30,  # already fired
    )
    _bg.STORE.add(job)
    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    sent = {"called": False}
    monkeypatch.setattr(
        _bgw, "_send_heartbeat_dm",
        lambda *_a, **_k: sent.__setitem__("called", True) or True,
    )

    _bgw._tick_one(job)
    assert sent["called"] is False


# ─── _tick_one — stale detection ──────────────────────────────────


def test_tick_one_opens_stale_supervisor_when_log_unchanged(
    isolated_jobs, monkeypatch, tmp_path,
):
    """Job log size unchanged for >30 min AND elapsed >10 min →
    watchdog opens a supervisor turn with the stale signal."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    now = time.time()
    job = _bg.BackgroundJob(
        job_id="bg-stale-test",
        label="stale-test",
        command="run-something",
        cwd="",
        started_at=now - 3600,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
        last_log_size=1024,
        last_log_size_at=now - (35 * 60),  # 35 min ago
    )
    _bg.STORE.add(job)
    # Create the stdout.log file with the SAME size as last sample.
    log_dir = isolated_jobs / "jobs" / job.job_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_bytes(b"x" * 1024)

    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    opened = {"job": None}
    monkeypatch.setattr(
        "backend.job_supervisor.on_job_completed",
        lambda j: opened.__setitem__("job", j),
    )

    _bgw._tick_one(job)
    assert opened["job"] is not None
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.stale_supervisor_opened is True
    assert opened["job"].status == "stale"
    assert "unchanged" in opened["job"].stderr_tail


def test_tick_one_resets_stale_sample_on_log_growth(
    isolated_jobs, monkeypatch, tmp_path,
):
    """If the log file HAS grown since last sample, update the
    sample (don't open supervisor)."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    now = time.time()
    job = _bg.BackgroundJob(
        job_id="bg-grow-test",
        label="grow-test",
        command="run-something",
        cwd="",
        started_at=now - 3600,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
        last_log_size=1024,
        last_log_size_at=now - 600,
    )
    _bg.STORE.add(job)
    log_dir = isolated_jobs / "jobs" / job.job_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_bytes(b"x" * 2048)  # grew

    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    opened = {"job": None}
    monkeypatch.setattr(
        "backend.job_supervisor.on_job_completed",
        lambda j: opened.__setitem__("job", j),
    )

    _bgw._tick_one(job)
    assert opened["job"] is None  # not stale
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.last_log_size == 2048
    assert refreshed.stale_supervisor_opened is False


def test_tick_one_does_not_open_stale_twice(
    isolated_jobs, monkeypatch, tmp_path,
):
    """`stale_supervisor_opened=True` short-circuits the stale check
    so we don't fire a second supervisor while the first is
    deciding."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    now = time.time()
    job = _bg.BackgroundJob(
        job_id="bg-stale-twice",
        label="stale-twice",
        command="run-something",
        cwd="",
        started_at=now - 3600,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
        last_log_size=1024,
        last_log_size_at=now - (35 * 60),
        stale_supervisor_opened=True,  # already opened
    )
    _bg.STORE.add(job)
    log_dir = isolated_jobs / "jobs" / job.job_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_bytes(b"x" * 1024)

    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    fired = {"count": 0}
    monkeypatch.setattr(
        "backend.job_supervisor.on_job_completed",
        lambda j: fired.__setitem__("count", fired["count"] + 1),
    )

    _bgw._tick_one(job)
    assert fired["count"] == 0


# ─── time-fallback heartbeat for jobs without a probe ─────────────


def test_time_fallback_heartbeat_fires_after_interval(
    isolated_jobs, monkeypatch,
):
    """Job WITHOUT progress_probe_cmd. After 2 hours since last
    heartbeat, watchdog fires the time-fallback DM."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    now = time.time()
    job = _bg.BackgroundJob(
        job_id="bg-fallback-test",
        label="long job",
        command="run-something",
        cwd="",
        started_at=now - (2.5 * 3600),  # 2.5h ago
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
        total_units=None,
        progress_probe_cmd="",
        last_heartbeat_at=None,
        original_chat_id=848732236,
    )
    _bg.STORE.add(job)
    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    sent = {"text": None}
    monkeypatch.setattr(
        _bgw, "_send_heartbeat_dm",
        lambda _job, text: sent.__setitem__("text", text) or True,
    )

    _bgw._tick_one(job)
    assert sent["text"] is not None
    assert "still running" in sent["text"]
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.last_heartbeat_at is not None
    assert refreshed.last_heartbeat_at > now - 5


def test_time_fallback_throttles_within_interval(
    isolated_jobs, monkeypatch,
):
    """If last heartbeat was <2h ago, don't fire again."""
    from backend.tools import background_jobs as _bg
    from backend import bg_job_watchdog as _bgw

    now = time.time()
    job = _bg.BackgroundJob(
        job_id="bg-throttle-test",
        label="long job",
        command="run-something",
        cwd="",
        started_at=now - 3600,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=99999,
        total_units=None,
        progress_probe_cmd="",
        last_heartbeat_at=now - 1800,  # 30 min ago
    )
    _bg.STORE.add(job)
    monkeypatch.setattr(_bgw, "_pid_alive", lambda pid: True)
    sent = {"called": False}
    monkeypatch.setattr(
        _bgw, "_send_heartbeat_dm",
        lambda *_a, **_k: sent.__setitem__("called", True) or True,
    )

    _bgw._tick_one(job)
    assert sent["called"] is False
