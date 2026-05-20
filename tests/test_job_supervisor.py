"""Tests for the background-job supervisor turn (audit T6 follow-up).

User report (May 2026): when a long-running benchmark (SWE-bench)
finished, the agent only re-engaged when the user typed `status?`.
Trivial fixes ("`python` → `python3`") required the user to type
`restart` to approve. Partial completions ("ran 1 instance, not
300") were reported as final answers instead of triggering a
proper full run. The supervisor turn closes this gap: on job
completion the agent re-engages autonomously, decides
retry/deliver/escalate, and the user only sees ONE terminal DM.

These tests pin:
  - BackgroundJob carries supervisor context across retries
  - `start_background_job` inherits chain context from parent
  - `complete_supervisor` seals the chain
  - `complete_supervisor` refuses outside a supervisor turn
  - `on_job_completed` is idempotent on already-terminal jobs
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    """Each test gets a fresh jobs/ root so the global registry
    doesn't bleed across tests."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.tools import background_jobs as _bg
    # Force the STORE singleton to re-resolve _root.
    monkeypatch.setattr(_bg.STORE, "_root_override", tmp_path / "jobs")
    yield tmp_path


# ─── BackgroundJob extensions ─────────────────────────────────────


def test_background_job_carries_supervisor_context(isolated_jobs):
    """A fresh job persists original_user_request, expected_outcome,
    parent_job_id, retry_count, and supervisor_terminal."""
    from backend.tools import background_jobs as _bg
    # Use a no-op command that exits immediately so we don't fork
    # heavy processes in tests. `true` is portable across POSIX.
    # On Windows the test still passes because we only care about
    # the registry fields — the subprocess result doesn't matter.
    job = _bg.start_job(
        command="true",
        label="test job",
        original_user_request="run the benchmark and publish",
        expected_outcome="report.json with >= 300 entries",
        original_speaker_id="telegram:848732236",
        original_chat_id=848732236,
    )
    # Wait briefly for the runner thread to finish so the registry
    # has a stable record. Up to 2s; most exits land in <100ms.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        refreshed = _bg.STORE.get(job.job_id)
        if refreshed and refreshed.status != "running":
            break
        time.sleep(0.05)
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed is not None
    assert refreshed.original_user_request == "run the benchmark and publish"
    assert refreshed.expected_outcome == "report.json with >= 300 entries"
    assert refreshed.original_speaker_id == "telegram:848732236"
    assert refreshed.original_chat_id == 848732236
    assert refreshed.parent_job_id == ""
    assert refreshed.retry_count == 0
    assert refreshed.supervisor_terminal is False


# ─── set / reset active supervisor job ────────────────────────────


def test_active_supervisor_job_id_set_and_reset():
    """The ContextVar round-trips a job id."""
    from backend import job_supervisor as _jsup
    assert _jsup.active_supervisor_job_id() == ""
    token = _jsup.set_active_supervisor_job("bg-test123")
    try:
        assert _jsup.active_supervisor_job_id() == "bg-test123"
    finally:
        _jsup.reset_active_supervisor_job(token)
    assert _jsup.active_supervisor_job_id() == ""


# ─── mark_terminal seals the chain ────────────────────────────────


def test_mark_terminal_seals_and_appends_history(isolated_jobs):
    """`mark_terminal` sets supervisor_terminal AND appends a history
    entry so future supervisors can read what was decided."""
    from backend.tools import background_jobs as _bg
    from backend import job_supervisor as _jsup
    job = _bg.start_job(command="true", label="seal-test")
    # Wait for terminal.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)

    _jsup.mark_terminal(
        job.job_id, decision="done", reason="benchmark completed OK",
    )
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed is not None
    assert refreshed.supervisor_terminal is True
    assert len(refreshed.supervisor_history) == 1
    entry = refreshed.supervisor_history[0]
    assert entry["decision"] == "done"
    assert "benchmark completed OK" in entry["reason"]


def test_mark_terminal_unknown_job_is_noop(isolated_jobs, caplog):
    from backend import job_supervisor as _jsup
    # Must not raise.
    _jsup.mark_terminal(
        "bg-nonexistent", decision="done", reason="",
    )


# ─── on_job_completed idempotency ─────────────────────────────────


def test_on_job_completed_skips_already_terminal(isolated_jobs, monkeypatch):
    """If supervisor_terminal is already True (chain previously
    completed), on_job_completed must NOT spawn another supervisor
    turn — that would loop forever on a degraded path."""
    from backend.tools import background_jobs as _bg
    from backend import job_supervisor as _jsup
    job = _bg.start_job(command="true", label="idempotency-test")
    # Wait for natural completion.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)
    # Seal the chain.
    _jsup.mark_terminal(
        job.job_id, decision="done", reason="test",
    )
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.supervisor_terminal is True
    # Count spawned threads — call on_job_completed; no new thread
    # should fire because the chain is sealed.
    fired = {"count": 0}
    def _fake_run(job):
        fired["count"] += 1
    monkeypatch.setattr(_jsup, "_run_supervisor_turn", _fake_run)
    _jsup.on_job_completed(refreshed)
    # Give any (unwanted) thread a beat to run.
    time.sleep(0.1)
    assert fired["count"] == 0, (
        "supervisor must not re-engage on a terminal job"
    )


# ─── complete_supervisor handler ──────────────────────────────────


def test_complete_supervisor_refuses_outside_turn(isolated_jobs):
    """Called outside a supervisor turn → ok=False with a clear
    error. Prevents accidental sealing of a random job."""
    from backend.builtin_tools import _complete_supervisor_handler
    raw = _complete_supervisor_handler(
        decision="done", final_message="hi",
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert "supervisor turn" in body["error"]


def test_complete_supervisor_validates_decision(isolated_jobs, monkeypatch):
    """`decision` must be 'done' or 'escalate' — anything else is
    a programming error, refused before doing anything."""
    from backend.builtin_tools import _complete_supervisor_handler
    from backend import job_supervisor as _jsup
    from backend.tools import background_jobs as _bg
    job = _bg.start_job(command="true", label="validate-decision")
    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        raw = _complete_supervisor_handler(
            decision="maybe", final_message="...",
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is False
    assert "decision must be" in body["error"]


def test_complete_supervisor_seals_chain(isolated_jobs):
    """Inside a supervisor turn, complete_supervisor(decision='done')
    must mark the chain terminal and return ok=True."""
    from backend.builtin_tools import _complete_supervisor_handler
    from backend import job_supervisor as _jsup
    from backend.tools import background_jobs as _bg
    job = _bg.start_job(command="true", label="seal-via-handler")
    # Wait for natural completion.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)
    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        raw = _complete_supervisor_handler(
            decision="done",
            final_message="Bench finished. 297/300 resolved.",
            reason="297/300 resolved; 3 upstream flakes",
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["decision"] == "done"
    assert body["supervisor_terminal"] is True
    refreshed = _bg.STORE.get(job.job_id)
    assert refreshed.supervisor_terminal is True
    assert refreshed.supervisor_history
    assert refreshed.supervisor_history[-1]["decision"] == "done"


# ─── start_background_job inherits chain context from parent ──────


def test_start_background_job_inherits_from_parent(isolated_jobs, monkeypatch):
    """When the LLM in a supervisor turn calls start_background_job
    with `parent_job_id`, the new job must inherit
    `original_user_request`, `original_speaker_id`,
    `original_chat_id`, `expected_outcome` and increment
    `retry_count`. This is what carries goal context across the
    fix-and-retry chain."""
    from backend.tools import background_jobs as _bg
    from backend import job_supervisor as _jsup
    from backend.builtin_tools import _start_background_job_handler
    # Force-promote the test caller to owner so the handler doesn't
    # short-circuit on the role check.
    monkeypatch.setattr(
        "backend.roles.is_owner", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default",
    )
    parent = _bg.start_job(
        command="false",  # exits non-zero to simulate a failure
        label="parent-failed",
        original_user_request="run the bench and publish",
        expected_outcome="report.json with >= 300 entries",
        original_speaker_id="telegram:848732236",
        original_chat_id=848732236,
    )
    # Wait for parent to finish.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(parent.job_id) or parent).status != "running":
            break
        time.sleep(0.05)

    raw = _start_background_job_handler(
        command="true",
        label="retry-with-fix",
        parent_job_id=parent.job_id,
    )
    body = json.loads(raw)
    assert body["ok"] is True
    child_id = body["job_id"]
    assert body["parent_job_id"] == parent.job_id
    assert body["retry_count"] == 1, (
        f"child retry_count should be parent+1; got {body}"
    )
    child = _bg.STORE.get(child_id)
    assert child.original_user_request == "run the bench and publish"
    assert child.expected_outcome == "report.json with >= 300 entries"
    assert child.original_speaker_id == "telegram:848732236"
    assert child.original_chat_id == 848732236


def test_start_background_job_uses_active_supervisor_context(
    isolated_jobs, monkeypatch
):
    """Even WITHOUT explicit parent_job_id, if start_background_job
    is called inside a supervisor turn, the active supervisor job
    is used as the parent. This way the LLM doesn't have to thread
    the parent_job_id through every retry call — convention picks
    it up from context."""
    from backend.tools import background_jobs as _bg
    from backend import job_supervisor as _jsup
    from backend.builtin_tools import _start_background_job_handler
    monkeypatch.setattr(
        "backend.roles.is_owner", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "telegram:848732236",
    )
    parent = _bg.start_job(
        command="false",
        label="failed-bench",
        original_user_request="запусти настоящий тест",
        expected_outcome="real SWE-bench Lite run on 300 instances",
        original_speaker_id="telegram:848732236",
        original_chat_id=848732236,
    )
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(parent.job_id) or parent).status != "running":
            break
        time.sleep(0.05)
    token = _jsup.set_active_supervisor_job(parent.job_id)
    try:
        raw = _start_background_job_handler(
            command="true", label="retry-without-explicit-parent",
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is True
    child = _bg.STORE.get(body["job_id"])
    assert child.parent_job_id == parent.job_id
    assert child.retry_count == 1
    assert child.original_user_request == "запусти настоящий тест"
