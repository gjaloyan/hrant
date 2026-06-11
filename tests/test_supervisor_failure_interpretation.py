"""Supervisor prompt: interpret failures, don't just count them.

Behavioral audit 2026-06-11 (probe P7): a full-pytest background job
died in 20s with 95 collected / 93 collection errors (prod box lacks
dev test deps). The supervisor honestly DM'd "passed 0, errors 93"
with artifact paths — correct numbers, zero diagnosis. The user
couldn't tell real test failures from a broken environment.

These tests pin the prompt rule that fixes it.
"""
from __future__ import annotations

import time


def _job(**overrides):
    from backend.tools.background_jobs import BackgroundJob
    base = dict(
        job_id="bg-test-interp",
        label="Full repository pytest",
        command="pytest -q",
        cwd="/tmp",
        started_at=time.time() - 30,
        finished_at=time.time(),
        status="done",
        exit_code=2,
        stdout_tail="93 errors, 2 skipped",
        stderr_tail="",
        requester="webui:default",
        pid=None,
    )
    base.update(overrides)
    return BackgroundJob(**base)


def test_prompt_demands_failure_interpretation():
    """The supervisor prompt must instruct distinguishing REAL test
    failures from ENVIRONMENT failures and naming the cause."""
    from backend.job_supervisor import _format_completion_message

    msg = _format_completion_message(_job())
    low = msg.lower()
    assert "interpret failures" in low
    assert "environment" in low
    assert "collection" in low
    # The rule nudges RETRY-with-fix over raw-numbers-DONE for
    # fixable environmental causes.
    assert "prefer retry" in low


def test_prompt_keeps_existing_decision_contract():
    """The new rule must not displace the DONE/RETRY/ESCALATE
    contract the chain depends on."""
    from backend.job_supervisor import _format_completion_message

    msg = _format_completion_message(_job())
    assert "DONE" in msg
    assert "RETRY" in msg
    assert "ESCALATE" in msg
    assert "parent_job_id=bg-test-interp" in msg
