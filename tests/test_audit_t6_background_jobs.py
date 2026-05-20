"""Tests for the May 2026 cost audit T6: resumable background jobs.

Background: blocking subprocess calls inside an agent turn (SWE-bench,
video transcode, large pip wheel builds) drove the worst-case
$1+/turn cases the audit caught. T6 adds `start_background_job` —
returns immediately, runs subprocess in a thread, DMs owner on done.

Pinned behaviour:
  - `start_job(command=...)` returns a BackgroundJob immediately
    with status='running' and a unique job_id.
  - On completion, the registry is updated (status, exit_code,
    stdout_tail, stderr_tail) AND on_done callbacks fire.
  - File-backed registry survives process restart; running jobs
    from a previous process get marked 'interrupted' via
    `mark_interrupted_on_startup()`.
  - Owner-only tool gating refuses non-owner callers on the
    builtin `start_background_job` / `list_background_jobs` /
    `get_background_job` handlers.
  - Tools are registered in `register_builtin_tools()`.
"""
from __future__ import annotations

import json
import threading
import time

import pytest


# ─── BackgroundJobStore basics ─────────────────────────────────────


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the BackgroundJobStore at tmp_path so we don't touch
    the dev box's real registry."""
    from backend.tools import background_jobs as bg
    bg.STORE._root_override = tmp_path / "jobs"
    saved = list(bg._ON_DONE)
    bg._ON_DONE.clear()
    yield bg
    bg.STORE._root_override = None
    bg._ON_DONE.clear()
    bg._ON_DONE.extend(saved)


def test_store_empty_initially(isolated_store):
    bg = isolated_store
    assert bg.list_jobs() == []


def test_start_job_returns_running_status(isolated_store):
    bg = isolated_store
    # A very short command — sleep 0 is portable enough; on Windows
    # `cmd /c exit 0` would work but `python -c "pass"` is cross-platform.
    job = bg.start_job(
        command="python -c \"pass\"",
        label="probe",
        requester="webui:default",
    )
    assert job.job_id.startswith("bg-")
    assert job.status == "running"
    assert job.label == "probe"


def test_job_completes_and_status_updates(isolated_store):
    bg = isolated_store
    job = bg.start_job(
        command="python -c \"print('hello bg')\"",
        label="echo",
        requester="webui:default",
    )
    # Wait for completion — generous timeout for CI / Windows.
    for _ in range(50):
        time.sleep(0.1)
        latest = bg.STORE.get(job.job_id)
        if latest and latest.status != "running":
            break
    latest = bg.STORE.get(job.job_id)
    assert latest is not None
    assert latest.status == "done"
    assert latest.exit_code == 0
    assert "hello bg" in latest.stdout_tail


def test_failed_command_marked_error(isolated_store):
    bg = isolated_store
    job = bg.start_job(
        command="python -c \"import sys; sys.exit(7)\"",
        label="fail7",
        requester="webui:default",
    )
    for _ in range(50):
        time.sleep(0.1)
        latest = bg.STORE.get(job.job_id)
        if latest and latest.status != "running":
            break
    latest = bg.STORE.get(job.job_id)
    assert latest.status == "error"
    assert latest.exit_code == 7


def test_on_done_callback_fires(isolated_store):
    bg = isolated_store
    seen: list = []
    bg.register_on_done(lambda j: seen.append(j.job_id))
    job = bg.start_job(
        command="python -c \"pass\"",
        label="cb",
        requester="webui:default",
    )
    for _ in range(50):
        time.sleep(0.1)
        if seen:
            break
    assert seen, "on_done callback should have fired after job completed"
    assert seen[0] == job.job_id


def test_register_on_done_is_idempotent(isolated_store):
    bg = isolated_store
    counts: list = []
    cb = lambda j: counts.append(j.job_id)
    bg.register_on_done(cb)
    bg.register_on_done(cb)
    bg.register_on_done(cb)
    bg.start_job(command="python -c \"pass\"", label="idem")
    for _ in range(50):
        time.sleep(0.1)
        if counts:
            break
    # Even though we registered 3x, callback fires ONCE.
    assert len(counts) == 1


def test_mark_interrupted_on_startup(isolated_store):
    """Simulate prior-process state: a job stuck in 'running' from a
    previous PID. mark_interrupted_on_startup() must flip it."""
    bg = isolated_store
    # Inject a fake running job by manipulating the file directly.
    fake = bg.BackgroundJob(
        job_id="bg-fakerun",
        label="ghost",
        command="(fake)",
        cwd="",
        started_at=time.time() - 600,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester="webui:default",
        pid=12345,
    )
    bg.STORE.add(fake)
    assert bg.STORE.get("bg-fakerun").status == "running"
    n = bg.STORE.mark_interrupted_on_startup()
    assert n == 1
    latest = bg.STORE.get("bg-fakerun")
    assert latest.status == "interrupted"
    assert latest.finished_at is not None


def test_start_job_rejects_empty_command(isolated_store):
    bg = isolated_store
    with pytest.raises(ValueError, match="empty"):
        bg.start_job(command="", label="x")


def test_concurrent_cap_enforced(isolated_store, monkeypatch):
    """Above _MAX_CONCURRENT running jobs, start_job refuses with a
    clear error rather than fork-bombing."""
    bg = isolated_store
    # Stuff the registry with fake running jobs.
    for i in range(bg._MAX_CONCURRENT):
        bg.STORE.add(bg.BackgroundJob(
            job_id=f"bg-fake{i}",
            label=f"fake-{i}",
            command="(fake)",
            cwd="",
            started_at=time.time(),
            finished_at=None,
            status="running",
            exit_code=None,
            stdout_tail="",
            stderr_tail="",
            requester="",
            pid=None,
        ))
    with pytest.raises(ValueError, match="already running"):
        bg.start_job(command="python -c \"pass\"", label="x")


# ─── tool wrappers — owner gating + JSON shape ─────────────────────


def test_start_background_job_tool_owner_only(isolated_store, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)
    out = builtin_tools._start_background_job_handler(
        command="python -c \"pass\"", label="probe",
    )
    data = json.loads(out)
    assert data["ok"] is False
    assert "owner-only" in data["error"]


def test_start_background_job_tool_owner_succeeds(isolated_store, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._start_background_job_handler(
        command="python -c \"pass\"", label="probe-ok",
    )
    data = json.loads(out)
    assert data["ok"] is True
    assert data["job_id"].startswith("bg-")
    # Supervisor follow-up: the note now references the supervisor
    # turn instead of the legacy "Telegram DM" string. Pin both
    # signals (DM mention is still present in the supervisor path
    # description) plus the don't-poll hint.
    note_l = data["note"].lower()
    assert "supervisor" in note_l or "dm" in note_l
    assert "poll" in note_l


def test_list_background_jobs_tool_owner_only(isolated_store, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)
    out = builtin_tools._list_background_jobs_handler()
    data = json.loads(out)
    assert data["ok"] is False
    assert "owner-only" in data["error"]


def test_get_background_job_tool_missing_id(isolated_store, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._get_background_job_handler(job_id="bg-doesnotexist")
    data = json.loads(out)
    assert data["ok"] is False
    assert "no job" in data["error"].lower()


def test_background_job_tools_registered():
    from backend import builtin_tools
    from backend.tool_registry import get_registry
    builtin_tools.register_builtin_tools()
    names = get_registry().names()
    assert "start_background_job" in names
    assert "list_background_jobs" in names
    assert "get_background_job" in names


# ─── Rule pin: _UNIFIED_RULES tells the agent to prefer this for
#     long-running tasks ──────────────────────────────────────────


def test_unified_rules_mention_background_job_for_long_tasks():
    from backend.unified_agent import _UNIFIED_RULES_CORE
    assert "start_background_job" in _UNIFIED_RULES_CORE
    # The "use INSTEAD of terminal_exec for long tasks" guidance.
    low = _UNIFIED_RULES_CORE.lower()
    assert "60 second" in low or "60 sec" in low or "60s" in low
    # Naming examples (SWE-bench / video transcode / benchmark).
    assert "swe-bench" in low or "bench" in low or "transcode" in low
