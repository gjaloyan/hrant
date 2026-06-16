"""Setting a step's due_at schedules a kind='check_in' row; clearing it cancels."""
from __future__ import annotations

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import config, knowledge_manager, scheduled_messages, tracker
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    importlib.reload(scheduled_messages)
    importlib.reload(tracker)
    return tracker, scheduled_messages


def test_add_step_with_due_schedules_checkin(env):
    tracker, sm = env
    t = tracker.TRACKERS.create(title="T", domain="work", steps=[])
    step = tracker.TRACKERS.add_step(t["id"], "Pay", due_at="2026-06-25T09:00:00Z",
                                     requested_by="webui:default")
    rows = [r for r in sm._read_all() if r.get("kind") == "check_in"]
    assert len(rows) == 1
    assert rows[0]["meta"]["tracker_id"] == t["id"]
    assert rows[0]["meta"]["step_id"] == step["id"]
    assert rows[0]["due_at"] == "2026-06-25T09:00:00Z"


def test_update_step_done_cancels_checkin(env):
    tracker, sm = env
    t = tracker.TRACKERS.create(title="T", domain="work", steps=[])
    step = tracker.TRACKERS.add_step(t["id"], "Pay", due_at="2026-06-25T09:00:00Z",
                                     requested_by="webui:default")
    tracker.TRACKERS.update_step(t["id"], step["id"], status="done")
    pending = [r for r in sm._read_all()
               if r.get("kind") == "check_in" and r["status"] == "pending"]
    assert pending == []
