"""Tracker store — tracker.json CRUD under knowledge/projects/<slug>/."""
from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    # KM resolves paths from HRANT_DATA_DIR at import; reload to pick up tmp.
    import importlib
    from backend import knowledge_manager, tracker
    importlib.reload(knowledge_manager)
    importlib.reload(tracker)
    return tracker.TRACKERS


def test_create_and_get(store):
    t = store.create(title="Blister tooling", domain="work",
                      steps=[{"title": "Approve drawings"}])
    assert t["id"].startswith("trk_")
    assert t["title"] == "Blister tooling"
    assert t["status"] == "active"
    assert len(t["steps"]) == 1
    assert t["steps"][0]["id"].startswith("st_")
    assert t["steps"][0]["status"] == "pending"
    got = store.get(t["id"])
    assert got["title"] == "Blister tooling"


def test_list_active_excludes_archived(store):
    a = store.create(title="A", domain="work", steps=[])
    b = store.create(title="B", domain="work", steps=[])
    store.set_status(b["id"], "archived")
    ids = [t["id"] for t in store.list(status="active")]
    assert a["id"] in ids
    assert b["id"] not in ids


def test_add_and_update_step(store):
    t = store.create(title="T", domain="work", steps=[])
    s = store.add_step(t["id"], title="Pay supplier", due_at="2026-06-25T09:00:00Z")
    assert s["title"] == "Pay supplier"
    assert s["due_at"] == "2026-06-25T09:00:00Z"
    store.update_step(t["id"], s["id"], status="done", note="paid")
    got = store.get(t["id"])
    step = next(x for x in got["steps"] if x["id"] == s["id"])
    assert step["status"] == "done"
    assert step["note"] == "paid"


def test_inbox_reminder_is_a_one_step_project(store):
    t = store.create_inbox_reminder(title="call bank",
                                    due_at="2026-06-18T11:00:00Z")
    assert t["domain"] == "inbox"
    assert len(t["steps"]) == 1
    assert t["steps"][0]["due_at"] == "2026-06-18T11:00:00Z"
    assert t["steps"][0]["check_in_kind"] == "remind"


def test_unknown_tracker_returns_none(store):
    assert store.get("trk_does_not_exist") is None
