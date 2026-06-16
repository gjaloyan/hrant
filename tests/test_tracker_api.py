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
