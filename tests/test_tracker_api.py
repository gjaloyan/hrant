"""Tracker API: list trackers with steps, read one, update a step, complete."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import config, knowledge_manager, scheduled_messages, tracker
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    importlib.reload(scheduled_messages)
    importlib.reload(tracker)
    from backend.api import projects as projects_api
    importlib.reload(projects_api)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(projects_api.router)
    return TestClient(app), tracker


def test_list_and_complete(client):
    c, tracker = client
    t = tracker.TRACKERS.create(title="T", domain="work", steps=[{"title": "A"}])
    r = c.get("/api/trackers")
    assert r.status_code == 200
    assert any(x["id"] == t["id"] for x in r.json()["trackers"])
    r2 = c.post(f"/api/trackers/{t['id']}/complete")
    assert r2.status_code == 200
    assert tracker.TRACKERS.get(t["id"])["status"] == "archived"


def test_quick_add_puts_a_task_on_the_list(client):
    """Until 2026-09-01 only the agent could add anything: the owner had to
    ask for a note to himself."""
    c, tracker = client
    r = c.post("/api/todos", json={"title": "buy milk",
                                   "due_at": "2099-01-01T09:00:00Z"})
    assert r.status_code == 200, r.text
    t = r.json()["tracker"]
    assert t["domain"] == "inbox", "a task became a project"
    assert len(t["steps"]) == 1
    assert t["steps"][0]["title"] == "buy milk"
    assert t["owner"], "the task landed with no owner"

    listed = c.get("/api/trackers").json()["trackers"]
    assert any(x["id"] == t["id"] for x in listed)


def test_a_dated_task_is_armed_at_creation(client):
    """A todo whose reminder was never scheduled is a todo that never
    speaks up -- the easiest thing to forget when wiring a second caller."""
    c, tracker = client
    from backend.scheduled_messages import list_pending
    r = c.post("/api/todos", json={"title": "call the dentist",
                                   "due_at": "2099-01-01T09:00:00Z"})
    sid = r.json()["tracker"]["steps"][0]["id"]
    armed = [x for x in list_pending()
             if (x.get("meta") or {}).get("step_id") == sid]
    assert armed, "the reminder was never scheduled"


def test_an_undated_task_is_accepted_and_stays_quiet(client):
    c, tracker = client
    r = c.post("/api/todos", json={"title": "someday: learn welding"})
    assert r.status_code == 200
    assert r.json()["tracker"]["steps"][0]["due_at"] == ""


def test_an_empty_title_is_refused(client):
    c, _ = client
    assert c.post("/api/todos", json={"title": "   "}).status_code == 400


def test_the_api_does_not_serve_another_users_tasks(client, monkeypatch):
    """The HTTP layer had no scoping at all: it returned every tracker on
    disk, so a non-owner console would have listed and edited everyone's."""
    c, tracker = client
    monkeypatch.setattr("backend.roles.is_owner", lambda s: False)
    mine = tracker.TRACKERS.create(title="mine", requested_by="webui:default")
    theirs = tracker.TRACKERS.create(
        title="theirs", requested_by="telegram:2",
        steps=[{"title": "their private step"}])

    ids = {x["id"] for x in c.get("/api/trackers").json()["trackers"]}
    assert mine["id"] in ids
    assert theirs["id"] not in ids, "the API served someone else's list"

    assert c.get("/api/trackers/" + theirs["id"]).status_code == 404
    # A REAL step id of theirs — using a bogus one makes this vacuous,
    # since update_step 404s on an unknown step whether or not it checks.
    their_step = theirs["steps"][0]["id"]
    assert c.put(
        "/api/trackers/%s/steps/%s" % (theirs["id"], their_step),
        json={"status": "done"}).status_code == 404
    assert tracker.TRACKERS.get(theirs["id"])["steps"][0]["status"] != "done"
