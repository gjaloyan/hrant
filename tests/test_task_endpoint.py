"""Tests for Phase 3 — TaskEndpoint Resolver.

User report (May 2026, after Phase 1+2 supervisor): "the agent
does not understand the logical endpoint of a task". The supervisor
fired on completion (Phase 1) but rubber-stamped 'done' on every
exit-0 run regardless of whether the user's actual goal was met.
Concrete prod failures:
  - "no-op baseline" with 300 empty predictions: exit 0 →
    supervisor said "Done." Goal was real eval, not no-op.
  - "wrote 300 predictions, No instances to run": exit 0 →
    supervisor empty. Goal was running the eval, not preparing it.

Phase 3 fix: TaskEndpoint definition BEFORE launch + auto-check via
shell commands on completion + complete_supervisor REFUSES 'done'
while critical criteria are unmet.
"""
from __future__ import annotations

import json
import pytest


@pytest.fixture
def isolated_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import task_endpoint as _te
    monkeypatch.setattr(
        _te.STORE, "_root_override", tmp_path / "jobs" / "endpoints",
    )
    yield tmp_path


# ─── create_endpoint validation ────────────────────────────────────


def test_create_endpoint_persists(isolated_endpoints):
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="Run benchmark",
        user_goal_verbatim="run bench",
        success_criteria=[
            {"id": "exit", "description": "exits 0", "check_cmd": "true"},
        ],
    )
    assert ep.endpoint_id.startswith("te-")
    refreshed = _te.STORE.get(ep.endpoint_id)
    assert refreshed is not None
    assert refreshed.task_summary == "Run benchmark"
    assert refreshed.user_goal_verbatim == "run bench"


def test_create_endpoint_assigns_ids_when_missing(isolated_endpoints):
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"description": "first"},
            {"description": "second"},
        ],
    )
    ids = [c["id"] for c in ep.success_criteria]
    assert ids == ["crit-0", "crit-1"]


def test_create_endpoint_dedupes_ids(isolated_endpoints):
    """Two criteria with the same explicit id must end up with
    distinct stored ids — collisions break supervisor evaluation."""
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "foo", "description": "first"},
            {"id": "foo", "description": "second"},
        ],
    )
    ids = [c["id"] for c in ep.success_criteria]
    assert ids[0] == "foo"
    assert ids[1] != "foo"


def test_create_endpoint_refuses_empty_criteria(isolated_endpoints):
    from backend import task_endpoint as _te
    with pytest.raises(ValueError, match="at least 1 criterion"):
        _te.create_endpoint(
            task_summary="t",
            user_goal_verbatim="g",
            success_criteria=[],
        )


def test_create_endpoint_refuses_over_twelve(isolated_endpoints):
    from backend import task_endpoint as _te
    with pytest.raises(ValueError, match="at most 12 entries"):
        _te.create_endpoint(
            task_summary="t",
            user_goal_verbatim="g",
            success_criteria=[
                {"description": f"crit {i}"} for i in range(13)
            ],
        )


def test_create_endpoint_refuses_criterion_without_description(
    isolated_endpoints,
):
    from backend import task_endpoint as _te
    with pytest.raises(ValueError, match="description is required"):
        _te.create_endpoint(
            task_summary="t",
            user_goal_verbatim="g",
            success_criteria=[{"id": "x"}],
        )


# ─── evaluate_endpoint ────────────────────────────────────────────


def test_evaluate_endpoint_marks_met_and_unmet(isolated_endpoints):
    """A criterion whose check_cmd exits 0 is 'met'; non-zero is
    'unmet'. Both stdout + stderr are captured (truncated)."""
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "ok", "description": "true", "check_cmd": "true"},
            {"id": "fail", "description": "false", "check_cmd": "false"},
        ],
    )
    results = _te.evaluate_endpoint(ep)
    statuses = {r.criterion_id: r.status for r in results}
    assert statuses == {"ok": "met", "fail": "unmet"}


def test_evaluate_endpoint_marks_needs_llm_when_no_check_cmd(
    isolated_endpoints,
):
    """When check_cmd is empty, the supervisor will ask the LLM to
    judge — we surface this as a distinct status."""
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "llm_judge", "description": "judge from logs"},
        ],
    )
    results = _te.evaluate_endpoint(ep)
    assert results[0].status == "needs_llm_judgment"


def test_unmet_critical_excludes_informational(isolated_endpoints):
    """Non-critical criteria don't block 'done' even when unmet."""
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "critical_fail", "description": "must met",
             "check_cmd": "false", "critical": True},
            {"id": "info_fail", "description": "nice to have",
             "check_cmd": "false", "critical": False},
        ],
    )
    results = _te.evaluate_endpoint(ep)
    blocking = _te.unmet_critical(results)
    assert [r.criterion_id for r in blocking] == ["critical_fail"]


def test_match_recovery_hints_substring(isolated_endpoints):
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[{"description": "x"}],
        failure_recovery=[
            {"trigger": "ModuleNotFoundError",
             "suggested_action": "propose_install"},
            {"trigger": "permission denied",
             "suggested_action": "check chmod"},
        ],
    )
    hits = _te.match_recovery_hints(
        ep,
        stderr_tail="ImportError\nModuleNotFoundError: 'swebench'\n",
        stdout_tail="",
    )
    actions = [h.suggested_action for h in hits]
    assert "propose_install" in actions
    assert "check chmod" not in actions


# ─── define_task_endpoint handler ─────────────────────────────────


def test_define_task_endpoint_handler_parses_json(
    isolated_endpoints, monkeypatch,
):
    from backend.builtin_tools import _define_task_endpoint_handler
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "webui:default",
    )
    raw = _define_task_endpoint_handler(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=json.dumps([
            {"description": "first", "check_cmd": "true"},
            {"description": "second"},
        ]),
        failure_recovery=json.dumps([
            {"trigger": "X", "suggested_action": "do Y"},
        ]),
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["endpoint_id"].startswith("te-")
    assert body["criteria_count"] == 2
    assert body["recovery_hints_count"] == 1


def test_define_task_endpoint_handler_rejects_invalid_json(
    isolated_endpoints, monkeypatch,
):
    from backend.builtin_tools import _define_task_endpoint_handler
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "webui:default",
    )
    raw = _define_task_endpoint_handler(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria="not-json",
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert "valid JSON" in body["error"]


# ─── start_background_job inherits endpoint_id on retry ───────────


def test_start_background_job_inherits_endpoint_from_parent(
    isolated_endpoints, monkeypatch,
):
    """A retry child must inherit the parent's endpoint_id so the
    supervisor evaluates against the SAME criteria across the chain
    (the goal didn't change because we re-attempted the job)."""
    from backend.tools import background_jobs as _bg
    from backend import task_endpoint as _te
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    monkeypatch.setattr(
        "backend.roles.is_owner", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "telegram:848732236",
    )
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[{"description": "c"}],
    )
    parent = _bg.start_job(
        command="false",
        label="parent",
        endpoint_id=ep.endpoint_id,
    )
    import time
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(parent.job_id) or parent).status != "running":
            break
        time.sleep(0.05)

    from backend.builtin_tools import _start_background_job_handler
    raw = _start_background_job_handler(
        command="true",
        label="retry",
        parent_job_id=parent.job_id,
    )
    body = json.loads(raw)
    assert body["ok"] is True
    child = _bg.STORE.get(body["job_id"])
    assert child.endpoint_id == ep.endpoint_id, (
        "child must inherit endpoint_id from parent"
    )


# ─── complete_supervisor gate ─────────────────────────────────────


def test_complete_supervisor_refuses_done_when_critical_unmet(
    isolated_endpoints, monkeypatch,
):
    """The core Phase 3 contract: code REFUSES 'done' while critical
    criteria are unmet, even if the LLM tries to mark it. This is
    what prevents 'job exit 0, must be done!' on runs that didn't
    actually meet the user's goal."""
    from backend.tools import background_jobs as _bg
    from backend import task_endpoint as _te
    from backend import job_supervisor as _jsup
    from backend.builtin_tools import _complete_supervisor_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    ep = _te.create_endpoint(
        task_summary="must produce report",
        user_goal_verbatim="run bench",
        success_criteria=[
            {"id": "report_exists", "description": "report.json exists",
             "check_cmd": "test -f /nonexistent/report.json",
             "critical": True},
        ],
    )
    job = _bg.start_job(
        command="true",
        label="test",
        endpoint_id=ep.endpoint_id,
    )
    import time
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)

    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        raw = _complete_supervisor_handler(
            decision="done",
            final_message="all good",
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is False
    assert body["error"] == "endpoint_criteria_unmet"
    assert body["unmet"]
    assert body["unmet"][0]["criterion_id"] == "report_exists"


def test_complete_supervisor_allows_done_with_explicit_override(
    isolated_endpoints, monkeypatch,
):
    """Escape hatch: if check_cmd is buggy (e.g. relative path bug),
    the LLM can pass `criteria_overrides={"<id>": "<reason>"}` to
    force 'done'. Each override must have a concrete explanation —
    the override goes into supervisor_history for audit."""
    from backend.tools import background_jobs as _bg
    from backend import task_endpoint as _te
    from backend import job_supervisor as _jsup
    from backend.builtin_tools import _complete_supervisor_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "buggy_check", "description": "report.json exists",
             "check_cmd": "test -f /nonexistent/path/report.json",
             "critical": True},
        ],
    )
    job = _bg.start_job(
        command="true",
        label="test",
        endpoint_id=ep.endpoint_id,
    )
    import time
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)

    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        raw = _complete_supervisor_handler(
            decision="done",
            final_message="all good",
            criteria_overrides=json.dumps({
                "buggy_check": "verified manually — file exists at actual path",
            }),
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["decision"] == "done"


def test_complete_supervisor_allows_done_when_no_endpoint(
    isolated_endpoints, monkeypatch,
):
    """Backward compat: jobs without an endpoint_id behave as before
    (Phase 1) — supervisor accepts 'done' on the LLM's judgement."""
    from backend.tools import background_jobs as _bg
    from backend import job_supervisor as _jsup
    from backend.builtin_tools import _complete_supervisor_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    job = _bg.start_job(command="true", label="no-endpoint")
    import time
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)

    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        raw = _complete_supervisor_handler(
            decision="done",
            final_message="ok",
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is True


def test_complete_supervisor_allows_done_when_all_met(
    isolated_endpoints, monkeypatch,
):
    """Sanity: with all check_cmds passing, 'done' goes through
    cleanly with no overrides needed."""
    from backend.tools import background_jobs as _bg
    from backend import task_endpoint as _te
    from backend import job_supervisor as _jsup
    from backend.builtin_tools import _complete_supervisor_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "always_ok", "description": "true",
             "check_cmd": "true", "critical": True},
        ],
    )
    job = _bg.start_job(
        command="true", label="ok", endpoint_id=ep.endpoint_id,
    )
    import time
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)

    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        raw = _complete_supervisor_handler(
            decision="done",
            final_message="all checks passed",
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is True


def test_complete_supervisor_escalate_bypasses_gate(
    isolated_endpoints, monkeypatch,
):
    """The gate only triggers on decision='done'. ESCALATE goes
    through even when criteria are unmet — that's the point: the
    agent is admitting it CAN'T meet them."""
    from backend.tools import background_jobs as _bg
    from backend import task_endpoint as _te
    from backend import job_supervisor as _jsup
    from backend.builtin_tools import _complete_supervisor_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "blocking", "description": "x",
             "check_cmd": "false", "critical": True},
        ],
    )
    job = _bg.start_job(
        command="false", label="esc", endpoint_id=ep.endpoint_id,
    )
    import time
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (_bg.STORE.get(job.job_id) or job).status != "running":
            break
        time.sleep(0.05)

    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        raw = _complete_supervisor_handler(
            decision="escalate",
            final_message="can't meet criterion X because…",
        )
    finally:
        _jsup.reset_active_supervisor_job(token)
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["decision"] == "escalate"


# ─── format_evaluation_block ──────────────────────────────────────


# ─── Phase 3a: prerequisites pre-flight gate ──────────────────────


def test_create_endpoint_accepts_prerequisites(isolated_endpoints):
    """Phase 3a: endpoints can carry pre-flight prerequisites
    separate from success_criteria."""
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[{"description": "out exists"}],
        prerequisites=[
            {"id": "input_present", "description": "input file",
             "check_cmd": "true"},
            {"id": "dep_installed", "description": "dep importable",
             "check_cmd": "false"},
        ],
    )
    assert len(ep.prerequisites) == 2
    refreshed = _te.STORE.get(ep.endpoint_id)
    assert refreshed is not None
    assert len(refreshed.prerequisites) == 2


def test_evaluate_prerequisites_runs_check_cmds(isolated_endpoints):
    """Same shape as evaluate_endpoint but reads from
    `prerequisites` instead of `success_criteria`."""
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[{"description": "out"}],
        prerequisites=[
            {"id": "ok", "description": "true", "check_cmd": "true"},
            {"id": "fail", "description": "false", "check_cmd": "false"},
        ],
    )
    results = _te.evaluate_prerequisites(ep)
    statuses = {r.criterion_id: r.status for r in results}
    assert statuses == {"ok": "met", "fail": "unmet"}


def test_evaluate_prerequisites_empty_when_no_prereqs(isolated_endpoints):
    """Backward compat: an endpoint without prerequisites returns
    an empty result list — the gate sees no blocking criteria and
    launches normally."""
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[{"description": "out"}],
    )
    assert _te.evaluate_prerequisites(ep) == []


def test_start_background_job_refuses_when_prerequisite_unmet(
    isolated_endpoints, monkeypatch,
):
    """The core Phase 3a contract: the pre-flight gate refuses
    `start_background_job` if a critical prerequisite is unmet.
    The agent gets a structured error explaining what to fix —
    the 'Please run bench' incident becomes impossible because
    the agent CAN'T launch a doomed job."""
    from backend.tools import background_jobs as _bg
    from backend.builtin_tools import _start_background_job_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    monkeypatch.setattr(
        "backend.roles.is_owner", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "telegram:848732236",
    )
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="bench",
        user_goal_verbatim="run bench",
        success_criteria=[{"description": "exit 0", "check_cmd": "true"}],
        prerequisites=[
            {"id": "patches_nonempty",
             "description": "non-empty patches",
             "check_cmd": "false",  # simulates "empty patches"
             "critical": True},
        ],
    )
    raw = _start_background_job_handler(
        command="echo run",
        label="blocked-bench",
        endpoint_id=ep.endpoint_id,
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert body["error"] == "prerequisites_unmet"
    assert body["endpoint_id"] == ep.endpoint_id
    assert body["unmet"]
    assert body["unmet"][0]["criterion_id"] == "patches_nonempty"
    # No job got created.
    running_jobs = _bg.list_jobs(status="running")
    assert running_jobs == []


def test_start_background_job_launches_when_prerequisites_met(
    isolated_endpoints, monkeypatch,
):
    """Sanity: with all prerequisites passing, launch proceeds
    normally."""
    from backend.tools import background_jobs as _bg
    from backend.builtin_tools import _start_background_job_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    monkeypatch.setattr(
        "backend.roles.is_owner", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "telegram:848732236",
    )
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="ok task",
        user_goal_verbatim="do it",
        success_criteria=[{"description": "x"}],
        prerequisites=[
            {"id": "input_ready", "description": "input file ready",
             "check_cmd": "true", "critical": True},
        ],
    )
    raw = _start_background_job_handler(
        command="true",
        label="ok-launch",
        endpoint_id=ep.endpoint_id,
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["job_id"].startswith("bg-")


def test_start_background_job_allows_launch_without_endpoint(
    isolated_endpoints, monkeypatch,
):
    """Backward compat: launches without `endpoint_id` skip the
    gate entirely."""
    from backend.tools import background_jobs as _bg
    from backend.builtin_tools import _start_background_job_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    monkeypatch.setattr(
        "backend.roles.is_owner", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default",
    )
    raw = _start_background_job_handler(
        command="true", label="no-endpoint",
    )
    body = json.loads(raw)
    assert body["ok"] is True


def test_non_critical_prerequisite_does_not_block_launch(
    isolated_endpoints, monkeypatch,
):
    """A prerequisite marked critical=False is informational —
    failing it does NOT refuse the launch (mirrors success
    criteria semantics for non-critical items)."""
    from backend.tools import background_jobs as _bg
    from backend.builtin_tools import _start_background_job_handler
    monkeypatch.setattr(
        _bg.STORE, "_root_override", isolated_endpoints / "jobs",
    )
    monkeypatch.setattr(
        "backend.roles.is_owner", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default",
    )
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[{"description": "x"}],
        prerequisites=[
            {"id": "nice_to_have", "description": "warn-only",
             "check_cmd": "false", "critical": False},
        ],
    )
    raw = _start_background_job_handler(
        command="true",
        label="non-critical-prereq",
        endpoint_id=ep.endpoint_id,
    )
    body = json.loads(raw)
    assert body["ok"] is True, (
        f"non-critical prerequisite must not block launch: {body}"
    )


def test_define_task_endpoint_handler_accepts_prerequisites(
    isolated_endpoints, monkeypatch,
):
    """End-to-end via the LLM-facing handler: prerequisites JSON
    arg is parsed and stored."""
    from backend.builtin_tools import _define_task_endpoint_handler
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default",
    )
    raw = _define_task_endpoint_handler(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=json.dumps([{"description": "out"}]),
        prerequisites=json.dumps([
            {"description": "input ready", "check_cmd": "true"},
        ]),
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["prerequisites_count"] == 1
    from backend import task_endpoint as _te
    ep = _te.STORE.get(body["endpoint_id"])
    assert len(ep.prerequisites) == 1


def test_format_evaluation_block_includes_status_icons(
    isolated_endpoints,
):
    from backend import task_endpoint as _te
    ep = _te.create_endpoint(
        task_summary="t",
        user_goal_verbatim="g",
        success_criteria=[
            {"id": "ok", "description": "first", "check_cmd": "true"},
            {"id": "fail", "description": "second", "check_cmd": "false"},
            {"id": "judge", "description": "third"},
        ],
    )
    results = _te.evaluate_endpoint(ep)
    block = _te.format_evaluation_block(ep, results, [])
    assert "✅" in block
    assert "❌" in block
    assert "🤔" in block
    assert "TASK ENDPOINT EVALUATION" in block
    assert "DECISION RULE" in block
