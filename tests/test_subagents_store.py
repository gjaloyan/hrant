"""Persistence + active registry for subagent sessions.

Pinned behaviour:
  - create() registers a new running session, writes it to disk
  - the active registry reflects running sessions in real time
  - finalize() moves the session out of active, writes the final
    state to disk, caps answer/error size, and prunes if needed
  - get/list/stats are consistent with what create/finalize wrote
  - get works for both active (in-memory) and persisted sessions
  - run_subagent end-to-end pumps the store correctly on
    success + LLM-error paths
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.subagents.store import (
    MAX_HISTORY,
    SUBAGENT_STORE,
    SubagentSession,
    SubagentStore,
)
from backend.subagents.dispatch import run_subagent


# --- store basics ------------------------------------------------------


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Brand-new store rooted at tmp_path. The default singleton is
    swapped so tests don't touch ~/.hrant."""
    store = SubagentStore(root=tmp_path)
    monkeypatch.setattr("backend.subagents.store.SUBAGENT_STORE", store)
    monkeypatch.setattr("backend.subagents.dispatch.SUBAGENT_STORE", store)
    return store


def test_create_writes_running_session_to_disk(isolated_store, tmp_path):
    s = isolated_store.create(role="researcher", task="find X", parent_job_id="abc")
    assert s.status == "running"
    assert s.role == "researcher"
    assert s.task == "find X"
    assert s.parent_job_id == "abc"
    assert s.started_at > 0
    # File on disk reflects the same state.
    p = tmp_path / f"{s.id}.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "running"
    assert data["role"] == "researcher"


def test_active_registry_tracks_running_sessions(isolated_store):
    a = isolated_store.create(role="coder", task="read agent.py")
    b = isolated_store.create(role="reviewer", task="check claim X")
    rows = isolated_store.active()
    ids = {r.id for r in rows}
    assert {a.id, b.id} <= ids
    assert all(r.status == "running" for r in rows)


def test_finalize_completed_moves_session_out_of_active(isolated_store):
    s = isolated_store.create(role="researcher", task="t")
    isolated_store.finalize(
        s.id, status="completed",
        answer="all done",
        iterations=2,
        tool_summary={"web_search": 1, "fetch_url": 1},
        elapsed_ms=1234,
    )
    assert isolated_store.active() == []
    saved = isolated_store.get(s.id)
    assert saved is not None
    assert saved.status == "completed"
    assert saved.answer == "all done"
    assert saved.iterations == 2
    assert saved.elapsed_ms == 1234
    assert saved.completed_at > 0


def test_finalize_failed_persists_error(isolated_store):
    s = isolated_store.create(role="coder", task="t")
    isolated_store.finalize(s.id, status="failed", error="LLM down")
    saved = isolated_store.get(s.id)
    assert saved.status == "failed"
    assert "LLM down" in saved.error


def test_finalize_caps_oversized_answer(isolated_store):
    """Long answers / errors must be capped so a runaway response
    can't bloat disk."""
    s = isolated_store.create(role="researcher", task="t")
    huge = "x" * 100_000
    isolated_store.finalize(s.id, status="completed", answer=huge)
    saved = isolated_store.get(s.id)
    assert 0 < len(saved.answer) <= 8000


def test_finalize_with_invalid_status_falls_back_to_failed(isolated_store):
    s = isolated_store.create(role="researcher", task="t")
    isolated_store.finalize(s.id, status="weird-status")
    assert isolated_store.get(s.id).status == "failed"


def test_finalize_returns_none_for_unknown_session(isolated_store):
    """Defensive: a double-finalize or stale session_id mustn't
    raise — returns None so the caller can log + continue."""
    res = isolated_store.finalize("does-not-exist", status="completed")
    assert res is None


def test_get_active_session_returns_live_snapshot(isolated_store):
    """Calling get() while running returns the in-memory record,
    not whatever disk would have (which is also valid here but
    not the point)."""
    s = isolated_store.create(role="coder", task="t")
    live = isolated_store.get(s.id)
    assert live is not None
    assert live.status == "running"


def test_list_returns_newest_first(isolated_store):
    a = isolated_store.create(role="researcher", task="A")
    time.sleep(0.02)
    b = isolated_store.create(role="coder", task="B")
    time.sleep(0.02)
    c = isolated_store.create(role="reviewer", task="C")
    isolated_store.finalize(a.id, status="completed")
    isolated_store.finalize(b.id, status="completed")
    isolated_store.finalize(c.id, status="completed")
    rows = isolated_store.list(limit=10)
    assert [r.id for r in rows[:3]] == [c.id, b.id, a.id]


def test_list_filters_by_status(isolated_store):
    a = isolated_store.create(role="researcher", task="A")
    b = isolated_store.create(role="researcher", task="B")
    isolated_store.finalize(a.id, status="completed")
    isolated_store.finalize(b.id, status="failed", error="x")
    only_failed = isolated_store.list(status="failed", limit=10)
    assert [r.status for r in only_failed] == ["failed"]


def test_list_filters_by_role(isolated_store):
    a = isolated_store.create(role="researcher", task="A")
    b = isolated_store.create(role="coder", task="B")
    isolated_store.finalize(a.id, status="completed")
    isolated_store.finalize(b.id, status="completed")
    only_coder = isolated_store.list(role="coder", limit=10)
    assert [r.role for r in only_coder] == ["coder"]


def test_stats_counts_per_status(isolated_store):
    """stats() distinguishes in-memory (running) from disk
    (completed / failed) records."""
    a = isolated_store.create(role="researcher", task="A")
    b = isolated_store.create(role="coder", task="B")
    c = isolated_store.create(role="reviewer", task="C")
    isolated_store.finalize(b.id, status="completed")
    isolated_store.finalize(c.id, status="failed", error="x")
    s = isolated_store.stats()
    assert s["running"] == 1
    assert s["completed"] == 1
    assert s["failed"] == 1
    assert s["total_persisted"] == 3


# --- end-to-end via run_subagent --------------------------------------


def test_run_subagent_records_completed_session_on_success(isolated_store, monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    fake_router = MagicMock()
    def _fake(*args, **kwargs):
        on_tc = kwargs.get("on_tool_call")
        if on_tc is not None:
            on_tc("web_search", {"query": "x"}, "[]", False)
        return "the answer"
    fake_router.call_with_tools.side_effect = _fake
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        res = run_subagent("researcher", "find X")
    assert res.ok
    rows = isolated_store.list(limit=10)
    assert len(rows) == 1
    sess = rows[0]
    assert sess.status == "completed"
    assert sess.role == "researcher"
    assert sess.answer == "the answer"
    assert sess.tool_summary == {"web_search": 1}
    assert sess.iterations == 1
    assert sess.elapsed_ms >= 0


def test_run_subagent_records_failed_session_on_llm_error(isolated_store, monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    from backend.llm import LLMError
    fake_router = MagicMock()
    fake_router.call_with_tools.side_effect = LLMError("provider went down")
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        res = run_subagent("researcher", "test")
    assert not res.ok
    rows = isolated_store.list(limit=10)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "provider went down" in rows[0].error


def test_run_subagent_skips_store_on_refusal(isolated_store, monkeypatch):
    """A refused dispatch (unknown role, empty task, non-owner) is
    NOT persisted — those refusals happen BEFORE we have a real
    session to track, and shouldn't pollute the history view."""
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    res = run_subagent("nonexistent-role", "x")
    assert not res.ok
    assert isolated_store.list(limit=10) == []
    assert isolated_store.active() == []


def test_run_subagent_skips_store_on_owner_refusal(isolated_store, monkeypatch):
    """Specifically: an owner-refusal must not create a session
    (otherwise a malicious guest could fill the history view with
    permission-denied rows)."""
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: False)
    fake_router = MagicMock()
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        res = run_subagent("researcher", "x")
    assert not res.ok
    assert "owner" in res.error.lower()
    assert isolated_store.list(limit=10) == []
