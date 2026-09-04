"""Tests for the LLM-callable `kick_supervisor` tool.

The supervisor turn normally fires automatically from the
background-job `_fire_done` callback. `kick_supervisor` exposes the
same entry point so the LLM can drive the autonomic loop
explicitly — re-open a finished job to apply a fresh fix, or kick
a job whose automatic callback was lost across a service restart.

Pins:
  - Non-owner is refused with ok=False.
  - Unknown job_id returns a 404-style error.
  - Running job is refused — supervisor will fire on its own at
    completion; manual kick would race.
  - Finished job dispatches the supervisor turn (we stub
    `on_job_completed` to avoid actually running a real LLM call).
  - Already-terminal job has its `supervisor_terminal` flag cleared
    so the supervisor can re-engage; a `kick_reopen` history entry
    is appended for the audit log.
"""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    """Fresh jobs/ root so the global registry doesn't bleed."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.tools import background_jobs as _bg
    monkeypatch.setattr(_bg.STORE, "_root_override", tmp_path / "jobs")
    yield tmp_path


@pytest.fixture
def stub_owner(monkeypatch):
    """Force the speaker to be treated as the owner so the
    permission gate passes without touching real role config."""
    monkeypatch.setattr("backend.roles.current_speaker", lambda: "webui:default")
    monkeypatch.setattr("backend.roles.is_owner", lambda _sid: True)


@pytest.fixture
def stub_non_owner(monkeypatch):
    """Speaker exists but lacks owner rights."""
    monkeypatch.setattr("backend.roles.current_speaker", lambda: "tg:guest")
    monkeypatch.setattr("backend.roles.is_owner", lambda _sid: False)


def _wait_for_terminal(_bg, job_id, deadline_s=20.0):
    """Block until the background subprocess exits.

    The deadline was two seconds and a timeout returned the job STILL
    RUNNING rather than saying so, which turned "the machine was busy"
    into an assertion failure three lines further down. That is the
    flake: green alone, green with neighbours, red once in a full run
    (2026-09-04) — and the mystery each time, because the message named
    the wrong thing.
    """
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        j = _bg.STORE.get(job_id)
        if j is not None and j.status != "running":
            return j
        time.sleep(0.05)
    j = _bg.STORE.get(job_id)
    raise AssertionError(
        "background job %s did not finish within %.0fs (status=%r) — the "
        "subprocess is slow or stuck, not the assertion below"
        % (job_id, deadline_s, getattr(j, "status", None))
    )


def test_kick_supervisor_owner_only(isolated_jobs, stub_non_owner):
    """A non-owner speaker is refused with a permission-denied error
    BEFORE any registry lookup."""
    from backend.builtin_tools import _kick_supervisor_handler
    raw = _kick_supervisor_handler(job_id="bg-anything")
    body = json.loads(raw)
    assert body["ok"] is False
    assert "owner" in body["error"]


def test_kick_supervisor_unknown_job(isolated_jobs, stub_owner):
    """Unknown job_id -> ok=False with a 404-style error."""
    from backend.builtin_tools import _kick_supervisor_handler
    raw = _kick_supervisor_handler(job_id="bg-does-not-exist")
    body = json.loads(raw)
    assert body["ok"] is False
    assert "no job" in body["error"]


def test_kick_supervisor_refuses_running(isolated_jobs, stub_owner, monkeypatch):
    """A still-running job is refused — supervisor will fire on its
    own at completion. Manual kick would race the automatic one."""
    from backend.builtin_tools import _kick_supervisor_handler
    from backend.tools import background_jobs as _bg
    # Spawn a long-sleeping subprocess so the job stays in 'running'
    # state during the test. Cross-platform sleep helper: prefer python
    # so this runs on Windows runners too.
    job = _bg.start_job(
        command="python -c \"import time; time.sleep(30)\"",
        label="sleeping",
    )
    raw = _kick_supervisor_handler(job_id=job.job_id)
    body = json.loads(raw)
    assert body["ok"] is False
    assert "still running" in body["error"]
    assert body["status"] == "running"
    # Cleanup: kill the sleeper so the test process doesn't linger.
    try:
        import os, signal
        os.kill(job.pid, signal.SIGTERM)
    except Exception:
        pass


def test_kick_supervisor_finished_dispatches_turn(
    isolated_jobs, stub_owner, monkeypatch,
):
    """Finished job (status=done) dispatches the supervisor turn.
    We stub `on_job_completed` to assert the call happened without
    actually spawning a real LLM turn."""
    from backend.builtin_tools import _kick_supervisor_handler
    from backend.tools import background_jobs as _bg
    from backend import job_supervisor as _jsup

    job = _bg.start_job(command="true", label="kick-test")
    final = _wait_for_terminal(_bg, job.job_id)
    assert final is not None
    assert final.status in ("done", "error")  # `true` -> done on POSIX

    fired: list = []
    monkeypatch.setattr(_jsup, "on_job_completed", lambda j: fired.append(j))

    raw = _kick_supervisor_handler(
        job_id=job.job_id, reason="reopening to apply fresh fix",
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["job_id"] == job.job_id
    assert len(fired) == 1
    assert fired[0].job_id == job.job_id


def test_kick_supervisor_clears_terminal_flag(
    isolated_jobs, stub_owner, monkeypatch,
):
    """A job that was previously marked supervisor_terminal must have
    its terminal flag cleared so the dispatched supervisor turn can
    actually proceed (the autoroute path skips terminal jobs)."""
    from backend.builtin_tools import _kick_supervisor_handler
    from backend.tools import background_jobs as _bg
    from backend import job_supervisor as _jsup

    job = _bg.start_job(command="true", label="reopen-test")
    _wait_for_terminal(_bg, job.job_id)

    _jsup.mark_terminal(
        job.job_id, decision="escalate", reason="initial chain blocked",
    )
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.supervisor_terminal is True

    fired: list = []
    monkeypatch.setattr(_jsup, "on_job_completed", lambda j: fired.append(j))

    raw = _kick_supervisor_handler(
        job_id=job.job_id, reason="user provided credential",
    )
    body = json.loads(raw)
    assert body["ok"] is True

    # Terminal flag cleared so the supervisor can run again.
    after = _bg.STORE.get(job.job_id)
    assert after.supervisor_terminal is False
    # History records the manual reopen for audit.
    decisions = [h.get("decision") for h in (after.supervisor_history or [])]
    assert "kick_reopen" in decisions
    # And the supervisor turn was dispatched.
    assert len(fired) == 1


def test_kick_supervisor_is_in_base_tools():
    """The tool must be in BASE_TOOLS so it's available without a
    `load_tool_bundle` dance. Job-management tools all live there."""
    from backend.tool_bundles import BASE_TOOLS
    assert "kick_supervisor" in BASE_TOOLS


def test_kick_supervisor_is_execute_class():
    """The tool must count as an execute-class action in
    endpoint_check, because kicking the supervisor is itself a
    state-changing action against the job."""
    from backend.endpoint_check import _EXECUTE_TOOLS
    assert "kick_supervisor" in _EXECUTE_TOOLS
