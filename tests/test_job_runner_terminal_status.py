"""Regression: a job must NEVER stay in `running` after `run_tracked`
returns or raises.

Pre-fix, a crash AFTER `mark_running` but BEFORE `mark_completed`
(e.g. the slice KeyError in `_extract_tool_calls`) left jobs stuck
in `running` forever. The Jobs tab showed phantom "in-progress" rows
that never resolved. This test pins the new guarantee:

  - agent.run raised → mark_failed
  - post-run code raised → mark_failed (NEW)
  - finally exits without a terminal status → mark_failed (NEW)
  - happy path → mark_completed
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend import job_runner, jobs
from backend.models import AgentAnswer, VerificationResult


def _ok_answer() -> AgentAnswer:
    return AgentAnswer(
        answer="ok",
        verification=VerificationResult(confidence=100),
        learned_topics=[], used_topics=[],
        project=None, is_chat=True, thinking_trace=[],
        execution_budget={
            "profile": "normal", "max_iterations": 500,
        },
    )


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    """Point the JOBS singleton at a tmp dir so we don't touch
    ~/.hrant on the dev machine."""
    monkeypatch.setattr(jobs, "JOBS", jobs.JobStore(root=tmp_path))
    yield jobs.JOBS


def test_happy_path_marks_completed(isolated_jobs):
    agent = MagicMock()
    agent.run.return_value = _ok_answer()
    answer, job_id = job_runner.run_tracked(agent, "hello")
    rec = isolated_jobs.get(job_id)
    assert rec.status == "completed"
    assert rec.response == "ok"
    assert rec.execution_budget["profile"] == "normal"
    assert answer.answer == "ok"


def test_agent_raise_marks_failed(isolated_jobs):
    agent = MagicMock()
    agent.run.side_effect = RuntimeError("LLM down")
    with pytest.raises(RuntimeError):
        job_runner.run_tracked(agent, "hello")
    rows = list(isolated_jobs.list())
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "LLM down" in (rows[0].error or "")


def test_post_run_crash_marks_failed_not_running(isolated_jobs):
    """The exact bug from the production audit: agent.run succeeded,
    `_extract_tool_calls` crashed, and the job sat in `running`
    forever. The new guard must convert that to `failed`."""
    agent = MagicMock()
    agent.run.return_value = _ok_answer()
    with patch.object(
        job_runner, "_extract_tool_calls",
        side_effect=KeyError("simulated post-run crash"),
    ):
        with pytest.raises(KeyError):
            job_runner.run_tracked(agent, "hello")
    rows = list(isolated_jobs.list())
    assert len(rows) == 1
    assert rows[0].status == "failed", (
        f"job left as {rows[0].status} after post-run crash"
    )
    assert "post-run tool-trace serialisation failed" in (rows[0].error or "")


def test_mark_completed_crash_still_marks_failed(isolated_jobs):
    """Failures inside `mark_completed` itself (rare — disk full,
    permission denied, etc.) also flip to `failed` instead of
    leaving the row in `running`."""
    agent = MagicMock()
    agent.run.return_value = _ok_answer()
    with patch.object(
        isolated_jobs, "mark_completed",
        side_effect=OSError("disk full"),
    ):
        # Patch mark_failed to use the real one so the failure flag
        # actually persists.
        with pytest.raises(OSError):
            job_runner.run_tracked(agent, "hello")
    rows = list(isolated_jobs.list())
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "mark_completed failed" in (rows[0].error or "")
