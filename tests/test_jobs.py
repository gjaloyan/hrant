"""Tests for the durable Job records — Phase 15A.

Layers covered:
  - JobStore CRUD: create / get / list / count / update / delete
  - Lifecycle transitions: queued → running → completed/failed/cancelled
  - Boot recovery: `recover_interrupted` flips `running`/`queued`
    rows to `interrupted` (the property the WebUI Jobs tab relies on)
  - Retry: clones the original into a new queued record, bumps
    retry_count, leaves the original in its terminal state
  - run_tracked: marks `running` before calling Agent.run, `completed`
    on success, `failed` on exception. Tool-call trace is extracted
    from AgentAnswer.thinking_trace and stored on the job.
  - REST endpoints: list filters, retry payload shape, cancel
    idempotency, 404 on missing id.

Storage is pointed at tmp_path via HRANT_DATA_DIR; the production
data_dir under the dev machine is never touched.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend import jobs as _jobs


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Fresh JobStore rooted at tmp_path. Also points HRANT_DATA_DIR
    at tmp_path so any code that reaches through the singleton (api,
    runner) sees the same isolated state."""
    root = tmp_path / "hrant" / "jobs"
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path / "hrant"))
    s = _jobs.JobStore(root=root)
    return s


# ─── CRUD ───────────────────────────────────────────────────────────


def test_list_skips_non_dict_json_files(store):
    """A stray top-level-list JSON in jobs/ (e.g. a misplaced index file)
    must NOT crash list() — pre-2026-05-22 a `background.json` list
    file put the consolidation scheduler into a 24h+ crash loop with
    `'list' object has no attribute 'items'`. Guard pattern: detect
    non-dict payload via `Job.from_dict` raising ValueError, log,
    skip the file, continue scanning the rest."""
    import json
    real = store.create(prompt="real job", channel="webui",
                        speaker_id="webui:default")
    bogus = store.root / "background.json"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_text(json.dumps([{"id": "bg-1"}, {"id": "bg-2"}]),
                     encoding="utf-8")
    rows = store.list()
    assert len(rows) == 1
    assert rows[0].id == real.id


def test_get_skips_non_dict_json_file(store):
    """Same defense for the single-file lookup path."""
    import json
    bogus = store.root / "weirdname.json"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert store.get("weirdname") is None


def test_count_skips_non_dict_json_files(store):
    """count(status=...) must not crash on a list-payload file."""
    import json
    store.create(prompt="A", channel="webui", speaker_id="webui:default")
    bogus = store.root / "background.json"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_text(json.dumps([1, 2]), encoding="utf-8")
    # status=None counts files (legit case where a glob hits a non-Job
    # file — keep the count cheap, don't open the file).
    # status="queued" must inspect contents; the non-dict file is skipped.
    assert store.count(status="queued") == 1


def test_job_from_dict_raises_value_error_on_non_dict():
    """Direct decoder check — non-dict must raise ValueError so callers
    can `except (ValueError, TypeError): continue` instead of bombing."""
    with pytest.raises(ValueError, match="expected dict"):
        _jobs.Job.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_create_persists_with_unique_id(store):
    j1 = store.create(prompt="hello", channel="webui", speaker_id="webui:default")
    j2 = store.create(prompt="world", channel="webui", speaker_id="webui:default")
    assert j1.id != j2.id
    # Both files on disk.
    assert (store.root / f"{j1.id}.json").exists()
    assert (store.root / f"{j2.id}.json").exists()
    # Round-trip through the JSON file.
    re = store.get(j1.id)
    assert re is not None
    assert re.prompt == "hello"
    assert re.status == "queued"


def test_get_returns_none_for_unknown(store):
    assert store.get("does-not-exist") is None


def test_list_newest_first(store):
    a = store.create(prompt="a", channel="webui", speaker_id="webui:default")
    time.sleep(0.01)
    b = store.create(prompt="b", channel="webui", speaker_id="webui:default")
    time.sleep(0.01)
    c = store.create(prompt="c", channel="webui", speaker_id="webui:default")
    ids = [j.id for j in store.list()]
    assert ids == [c.id, b.id, a.id]


def test_list_filters_by_status(store):
    a = store.create(prompt="a", channel="webui", speaker_id="webui:default")
    b = store.create(prompt="b", channel="webui", speaker_id="webui:default")
    store.mark_completed(a.id, response="answer-a")
    store.mark_failed(b.id, error="boom")
    completed = store.list(status="completed")
    failed = store.list(status="failed")
    assert [j.id for j in completed] == [a.id]
    assert [j.id for j in failed] == [b.id]


def test_list_filters_by_channel(store):
    web = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    tg = store.create(prompt="y", channel="telegram", speaker_id="telegram:111")
    assert [j.id for j in store.list(channel="webui")] == [web.id]
    assert [j.id for j in store.list(channel="telegram")] == [tg.id]


def test_list_pagination(store):
    ids = [
        store.create(prompt=f"p{i}", channel="webui", speaker_id="webui:default").id
        for i in range(5)
    ]
    # Newest first → page 1 of size 2 = last two creations reversed.
    page1 = store.list(limit=2, offset=0)
    page2 = store.list(limit=2, offset=2)
    assert [j.id for j in page1] == ids[-1:-3:-1]
    assert [j.id for j in page2] == ids[-3:-5:-1]


def test_count_total_and_by_status(store):
    a = store.create(prompt="a", channel="webui", speaker_id="webui:default")
    b = store.create(prompt="b", channel="webui", speaker_id="webui:default")
    store.mark_completed(a.id, response="ok")
    assert store.count() == 2
    assert store.count(status="completed") == 1
    assert store.count(status="queued") == 1


def test_delete_removes_file(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    assert store.delete(j.id) is True
    assert store.get(j.id) is None
    # Idempotent — deleting again returns False, doesn't raise.
    assert store.delete(j.id) is False


# ─── Lifecycle ──────────────────────────────────────────────────────


def test_mark_running_sets_started_at(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    assert j.started_at is None
    out = store.mark_running(j.id)
    assert out is not None
    assert out.status == "running"
    assert out.started_at is not None
    assert out.started_at > j.created_at - 0.001


def test_mark_completed_stores_response_and_clears_error(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    store.mark_running(j.id)
    out = store.mark_completed(j.id, response="42")
    assert out is not None
    assert out.status == "completed"
    assert out.response == "42"
    assert out.error is None
    assert out.completed_at is not None


def test_mark_completed_persists_tool_calls(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    tool_calls = [
        {"name": "search", "args_summary": "kw=hi", "ok": True},
        {"name": "write_file", "args_summary": "path=/tmp/x", "ok": False, "error": "denied"},
    ]
    store.mark_completed(j.id, response="done", tool_calls=tool_calls)
    out = store.get(j.id)
    assert out is not None
    assert len(out.tool_calls) == 2
    assert out.tool_calls[1]["error"] == "denied"


def test_mark_failed_stores_error(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    out = store.mark_failed(j.id, error="Connection refused")
    assert out is not None
    assert out.status == "failed"
    assert out.error == "Connection refused"
    assert out.completed_at is not None


def test_mark_cancelled_terminal(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    store.mark_running(j.id)
    out = store.mark_cancelled(j.id)
    assert out is not None
    assert out.status == "cancelled"
    assert out.completed_at is not None


def test_mark_interrupted_bumps_counter(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    store.mark_running(j.id)
    out = store.mark_interrupted(j.id)
    assert out is not None
    assert out.status == "interrupted"
    assert out.interrupted_count == 1
    # Doing it twice bumps again (e.g., user retries, crash hits a
    # second time — we keep the count).
    store.mark_running(j.id)
    out2 = store.mark_interrupted(j.id)
    assert out2.interrupted_count == 2


# ─── Boot recovery ──────────────────────────────────────────────────


def test_recover_interrupted_flips_running_and_queued(store):
    """The whole reason this module exists: when the server dies
    while agent.run is mid-flight, the next boot must mark those
    runs as interrupted so the WebUI Jobs tab can surface them."""
    a = store.create(prompt="a", channel="webui", speaker_id="webui:default")
    b = store.create(prompt="b", channel="webui", speaker_id="webui:default")
    c = store.create(prompt="c", channel="webui", speaker_id="webui:default")
    store.mark_running(a.id)
    # b stays queued (was created just before crash, never picked up)
    store.mark_completed(c.id, response="ok")  # terminal — leave alone

    touched = store.recover_interrupted()
    assert set(touched) == {a.id, b.id}
    assert store.get(a.id).status == "interrupted"
    assert store.get(b.id).status == "interrupted"
    assert store.get(c.id).status == "completed"  # untouched


def test_recover_interrupted_idempotent(store):
    """Second call after recovery is a no-op — running/queued are
    gone, only terminal states remain."""
    a = store.create(prompt="a", channel="webui", speaker_id="webui:default")
    store.mark_running(a.id)
    store.recover_interrupted()
    second_pass = store.recover_interrupted()
    assert second_pass == []


def test_recover_interrupted_empty_when_no_jobs(store):
    """Fresh install — no jobs dir entries. Must not raise."""
    assert store.recover_interrupted() == []


# ─── Retry ──────────────────────────────────────────────────────────


def test_retry_clones_prompt_into_new_queued_job(store):
    orig = store.create(prompt="hello", channel="telegram", speaker_id="telegram:111")
    store.mark_failed(orig.id, error="rate limited")
    new = store.retry(orig.id)
    assert new is not None
    assert new.id != orig.id
    assert new.status == "queued"
    assert new.prompt == orig.prompt
    assert new.channel == orig.channel
    assert new.speaker_id == orig.speaker_id
    assert new.retry_count == orig.retry_count + 1
    # Original is preserved in its terminal state — audit log.
    assert store.get(orig.id).status == "failed"


def test_retry_carries_reply_to(store):
    """Telegram retries need to preserve the chat_id so the answer
    can be delivered back to the right conversation."""
    orig = store.create(
        prompt="x", channel="telegram", speaker_id="telegram:111",
        reply_to={"telegram_chat_id": 123, "telegram_user_id": 456},
    )
    store.mark_interrupted(orig.id)
    new = store.retry(orig.id)
    assert new.reply_to["telegram_chat_id"] == 123


def test_retry_returns_none_for_unknown(store):
    assert store.retry("nope") is None


# ─── attempts trace (Phase B prep) ──────────────────────────────────


def test_add_attempt_appends_to_list(store):
    j = store.create(prompt="x", channel="webui", speaker_id="webui:default")
    store.add_attempt(j.id, {
        "provider_id": "anthropic-default", "model": "claude-3-5-sonnet",
        "ok": False, "error": "429", "elapsed_ms": 1200,
    })
    store.add_attempt(j.id, {
        "provider_id": "openai-default", "model": "gpt-4o",
        "ok": True, "elapsed_ms": 850,
    })
    out = store.get(j.id)
    assert len(out.attempts) == 2
    assert out.attempts[0]["error"] == "429"
    assert out.attempts[1]["ok"] is True


# ─── run_tracked wrapper ────────────────────────────────────────────


def test_run_tracked_marks_completed_on_success(monkeypatch, store):
    """run_tracked must wrap Agent.run with a job lifecycle: create
    → mark_running → call agent.run → mark_completed with the
    response + extracted tool calls."""
    monkeypatch.setattr(_jobs, "JOBS", store)
    from backend import job_runner

    fake_answer = MagicMock()
    fake_answer.answer = "the answer is 42"
    fake_answer.thinking_trace = []
    agent = MagicMock()
    agent.run.return_value = fake_answer

    res, job_id = job_runner.run_tracked(
        agent, "what is the meaning of life?",
        channel="webui", speaker_id="webui:default",
    )
    assert res is fake_answer
    job = store.get(job_id)
    assert job is not None
    assert job.status == "completed"
    assert job.response == "the answer is 42"
    agent.run.assert_called_once()


def test_run_tracked_marks_failed_on_exception(monkeypatch, store):
    monkeypatch.setattr(_jobs, "JOBS", store)
    from backend import job_runner

    agent = MagicMock()
    agent.run.side_effect = RuntimeError("LLM provider unreachable")

    with pytest.raises(RuntimeError):
        job_runner.run_tracked(
            agent, "hello",
            channel="webui", speaker_id="webui:default",
        )

    # One failed job in the store (the most recent).
    failed = store.list(status="failed")
    assert len(failed) == 1
    assert "LLM provider unreachable" in failed[0].error


def test_run_tracked_extracts_tool_calls_from_trace(monkeypatch, store):
    monkeypatch.setattr(_jobs, "JOBS", store)
    from backend import job_runner

    # Synthetic thinking_trace with two tool steps + one non-tool.
    fake_call_ok = MagicMock()
    fake_call_ok.model_dump.return_value = {
        "name": "read_file", "args_summary": "path=/etc/hosts",
    }
    fake_call_err = MagicMock()
    fake_call_err.model_dump.return_value = {
        "name": "write_file", "args_summary": "path=/x",
        "error": "permission denied",
    }
    step_ok = MagicMock(event="tool", tool_call=fake_call_ok)
    step_err = MagicMock(event="tool_error", tool_call=fake_call_err)
    step_other = MagicMock(event="think", tool_call=None)

    fake_answer = MagicMock()
    fake_answer.answer = "done"
    fake_answer.thinking_trace = [step_other, step_ok, step_err]
    agent = MagicMock()
    agent.run.return_value = fake_answer

    _, job_id = job_runner.run_tracked(
        agent, "do stuff", channel="webui", speaker_id="webui:default",
    )
    job = store.get(job_id)
    assert len(job.tool_calls) == 2
    assert job.tool_calls[0]["name"] == "read_file"
    assert job.tool_calls[0]["ok"] is True
    assert job.tool_calls[1]["name"] == "write_file"
    assert job.tool_calls[1]["ok"] is False
    assert job.tool_calls[1]["error"] == "permission denied"


# ─── REST API ───────────────────────────────────────────────────────


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    """FastAPI TestClient with the singleton JobStore pointed at
    tmp_path. We bypass the full app.lifespan startup (autonomic
    scheduler, channels) because we only exercise the jobs router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import jobs as jobs_api

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path / "hrant"))
    monkeypatch.setattr(
        _jobs, "JOBS", _jobs.JobStore(root=tmp_path / "hrant" / "jobs"),
    )

    app = FastAPI()
    app.include_router(jobs_api.router)
    return TestClient(app)


def test_api_list_returns_jobs(api_client):
    j = _jobs.JOBS.create(prompt="x", channel="webui", speaker_id="webui:default")
    r = api_client.get("/api/jobs")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert data["jobs"][0]["id"] == j.id


def test_api_list_filters_by_status(api_client):
    a = _jobs.JOBS.create(prompt="a", channel="webui", speaker_id="webui:default")
    b = _jobs.JOBS.create(prompt="b", channel="webui", speaker_id="webui:default")
    _jobs.JOBS.mark_completed(a.id, response="ok")
    _jobs.JOBS.mark_failed(b.id, error="boom")
    r = api_client.get("/api/jobs?status=failed")
    data = r.json()
    assert [j["id"] for j in data["jobs"]] == [b.id]


def test_api_get_returns_full_record(api_client):
    j = _jobs.JOBS.create(prompt="x", channel="webui", speaker_id="webui:default")
    r = api_client.get(f"/api/jobs/{j.id}")
    assert r.status_code == 200
    assert r.json()["id"] == j.id
    assert r.json()["prompt"] == "x"


def test_api_get_404(api_client):
    r = api_client.get("/api/jobs/nope")
    assert r.status_code == 404


def test_api_retry_creates_new_job(api_client):
    j = _jobs.JOBS.create(prompt="hello", channel="webui", speaker_id="webui:default")
    _jobs.JOBS.mark_failed(j.id, error="boom")
    r = api_client.post(f"/api/jobs/{j.id}/retry")
    assert r.status_code == 200
    body = r.json()
    assert body["new_job_id"] != j.id
    assert body["prompt"] == "hello"
    # New job exists in store, queued.
    new = _jobs.JOBS.get(body["new_job_id"])
    assert new is not None
    assert new.status == "queued"


def test_api_cancel_idempotent_on_terminal(api_client):
    j = _jobs.JOBS.create(prompt="x", channel="webui", speaker_id="webui:default")
    _jobs.JOBS.mark_completed(j.id, response="done")
    r = api_client.post(f"/api/jobs/{j.id}/cancel")
    assert r.status_code == 200
    assert r.json()["note"] == "already terminal"


def test_api_cancel_marks_running_job(api_client):
    j = _jobs.JOBS.create(prompt="x", channel="webui", speaker_id="webui:default")
    _jobs.JOBS.mark_running(j.id)
    r = api_client.post(f"/api/jobs/{j.id}/cancel")
    assert r.status_code == 200
    assert _jobs.JOBS.get(j.id).status == "cancelled"


def test_api_delete(api_client):
    j = _jobs.JOBS.create(prompt="x", channel="webui", speaker_id="webui:default")
    r = api_client.delete(f"/api/jobs/{j.id}")
    assert r.status_code == 200
    assert _jobs.JOBS.get(j.id) is None


def test_api_stats_returns_counts(api_client):
    a = _jobs.JOBS.create(prompt="a", channel="webui", speaker_id="webui:default")
    b = _jobs.JOBS.create(prompt="b", channel="webui", speaker_id="webui:default")
    _jobs.JOBS.mark_completed(a.id, response="ok")
    _jobs.JOBS.mark_failed(b.id, error="x")
    r = api_client.get("/api/jobs/_/stats")
    data = r.json()
    assert data["completed"] == 1
    assert data["failed"] == 1
    assert data["running"] == 0
