"""Scope-preserving retries — supervisor RETRY must fix HOW, not WHAT.

Audit 2026-05-28 smoke run exposed this gap: supervisor dropped
`--agent codex` → `--agent oracle` to "find any working command"
and silently benchmarked a different thing. Oracle replays gold
answers; benchmarking it is not benchmarking codex.

The fix is two layers:
  - Prompt rules in M4 + supervisor block tell the LLM to ESCALATE
    on scope changes.
  - This code-side guard in `_start_background_job_handler` refuses
    retries whose command changes a SCOPE flag (--agent / --dataset
    / --model / --task / --tasks) relative to the parent.
"""
from __future__ import annotations

import json


def test_no_scope_change_passes():
    from backend.builtin_tools import _detect_scope_change
    parent = "harbor run --agent codex --dataset terminal-bench --n-tasks 2"
    # Pure-HOW fix: added --verbose flag, no scope change.
    new = "harbor run --agent codex --dataset terminal-bench --n-tasks 2 --verbose"
    assert _detect_scope_change(parent_command=parent, new_command=new) == ""


def test_agent_change_detected():
    from backend.builtin_tools import _detect_scope_change
    parent = "harbor run --agent codex --dataset terminal-bench --n-tasks 2"
    new = "harbor run --agent oracle --dataset terminal-bench --n-tasks 2"
    diff = _detect_scope_change(parent_command=parent, new_command=new)
    assert "--agent" in diff
    assert "codex" in diff and "oracle" in diff


def test_dataset_change_detected():
    from backend.builtin_tools import _detect_scope_change
    parent = "harbor run --agent codex --dataset terminal-bench-2-1"
    new = "harbor run --agent codex --dataset terminal-bench"
    diff = _detect_scope_change(parent_command=parent, new_command=new)
    assert "--dataset" in diff


def test_removing_scope_flag_detected():
    """Going from `--agent codex` to no `--agent` silently uses
    Harbor's default — that's also scope drift."""
    from backend.builtin_tools import _detect_scope_change
    parent = "harbor run --agent codex --dataset x"
    new = "harbor run --dataset x"
    diff = _detect_scope_change(parent_command=parent, new_command=new)
    assert "--agent" in diff


def test_equals_form_recognised():
    """`--flag=value` and `--flag value` are the same."""
    from backend.builtin_tools import _detect_scope_change
    parent = "harbor run --agent=codex"
    new = "harbor run --agent codex"
    assert _detect_scope_change(parent_command=parent, new_command=new) == ""


def test_n_tasks_NOT_a_scope_flag():
    """--n-tasks is concurrency / pacing, not WHAT. Changing it is
    a HOW change. (--task / --tasks are scope; --n-tasks is not.)"""
    from backend.builtin_tools import _detect_scope_change
    parent = "harbor run --agent codex --n-tasks 2"
    new = "harbor run --agent codex --n-tasks 4"
    assert _detect_scope_change(parent_command=parent, new_command=new) == ""


def test_unparseable_command_does_not_crash():
    """Malformed shell (unbalanced quotes) shouldn't crash the
    guard — fall back to whitespace tokenization."""
    from backend.builtin_tools import _detect_scope_change
    parent = "harbor run --agent codex"
    new = 'harbor run --agent codex "open quote'  # unbalanced
    # Should NOT raise.
    diff = _detect_scope_change(parent_command=parent, new_command=new)
    assert isinstance(diff, str)


def test_full_handler_rejects_scope_change(monkeypatch):
    """End-to-end: when supervisor calls start_background_job with
    parent_job_id and the command changes --agent, the handler
    returns ok=False with `scope_change_in_retry` error."""
    from backend.builtin_tools import _start_background_job_handler
    from backend.tools import background_jobs as _bg

    fake_parent = _bg.BackgroundJob(
        job_id="bg-parent",
        label="Terminal-Bench codex 2",
        command="harbor run --agent codex --dataset tb --n-tasks 2",
        cwd="/tmp",
        started_at=0.0,
        finished_at=0.0,
        status="error",
        exit_code=1,
        stdout_tail="",
        stderr_tail="",
        requester="webui:default",
        pid=None,
    )
    monkeypatch.setattr(
        _bg.STORE, "get",
        lambda jid: fake_parent if jid == "bg-parent" else None,
    )

    out = json.loads(_start_background_job_handler(
        command="harbor run --agent oracle --dataset tb --n-tasks 2",
        label="Terminal-Bench retry",
        parent_job_id="bg-parent",
    ))
    assert out["ok"] is False
    assert out["error"] == "scope_change_in_retry"
    assert "--agent" in out["detail"]
    assert "ESCALATE" in out["detail"]


def test_full_handler_allows_scope_preserving_retry(monkeypatch):
    """A retry that only changes HOW (e.g. `source` → `.`) must
    pass through the gate. We can't run the full subprocess in
    tests; assert that the gate doesn't return scope_change_in_retry."""
    from backend.builtin_tools import _start_background_job_handler
    from backend.tools import background_jobs as _bg

    fake_parent = _bg.BackgroundJob(
        job_id="bg-parent",
        label="X",
        command="source .venv/bin/activate && harbor run --agent codex",
        cwd="/tmp", started_at=0.0, finished_at=0.0,
        status="error", exit_code=127,
        stdout_tail="", stderr_tail="source: not found",
        requester="webui:default", pid=None,
    )
    monkeypatch.setattr(
        _bg.STORE, "get",
        lambda jid: fake_parent if jid == "bg-parent" else None,
    )

    # `source` → `.` is a HOW fix, not a scope change.
    out = json.loads(_start_background_job_handler(
        command=". .venv/bin/activate && harbor run --agent codex",
        label="X retry",
        parent_job_id="bg-parent",
    ))
    # The gate doesn't reject. (Downstream may reject for prereq
    # reasons; we only care about the scope gate here.)
    assert out.get("error") != "scope_change_in_retry"
