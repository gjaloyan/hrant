"""POST /api/scheduled-messages — Reminders menu (2026-06-12).

The chat path (schedule_message tool) parses natural-language times;
the Settings UI sends explicit ones. Exactly one of due_at /
delay_minutes; due_at must be future ISO 8601 UTC Z.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import backend.scheduled_messages as _sched
    monkeypatch.setattr(_sched, "_path", lambda: tmp_path / "sched.jsonl")
    from backend.main import app
    return TestClient(app)


def test_create_with_delay_minutes(client):
    r = client.post("/api/scheduled-messages", json={
        "text": "Check the order status",
        "delay_minutes": 2880,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    row = body["message"]
    assert row["status"] == "pending"
    assert row["text"] == "Check the order status"
    due = datetime.strptime(row["due_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc,
    )
    delta = due - datetime.now(timezone.utc)
    assert timedelta(minutes=2875) < delta < timedelta(minutes=2885)

    # And it shows up in the list.
    lst = client.get("/api/scheduled-messages?status=pending").json()
    assert any(m["id"] == row["id"] for m in lst["messages"])


def test_create_with_explicit_due_at(client):
    due = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    r = client.post("/api/scheduled-messages", json={
        "text": "Call the bank", "due_at": due,
    })
    assert r.status_code == 200
    assert r.json()["message"]["due_at"] == due


def test_validation_errors(client):
    # Missing text.
    assert client.post(
        "/api/scheduled-messages", json={"text": "  ", "delay_minutes": 5},
    ).status_code == 400
    # Neither / both time fields.
    assert client.post(
        "/api/scheduled-messages", json={"text": "x y z"},
    ).status_code == 400
    assert client.post(
        "/api/scheduled-messages",
        json={"text": "x y z", "delay_minutes": 5,
              "due_at": "2099-01-01T00:00:00Z"},
    ).status_code == 400
    # Bad format.
    assert client.post(
        "/api/scheduled-messages",
        json={"text": "x y z", "due_at": "tomorrow at noon"},
    ).status_code == 400
    # Past time.
    assert client.post(
        "/api/scheduled-messages",
        json={"text": "x y z", "due_at": "2020-01-01T00:00:00Z"},
    ).status_code == 400
    # Zero delay.
    assert client.post(
        "/api/scheduled-messages", json={"text": "x y z", "delay_minutes": 0},
    ).status_code == 400


def test_cancel_roundtrip(client):
    row = client.post("/api/scheduled-messages", json={
        "text": "cancel me please", "delay_minutes": 60,
    }).json()["message"]
    r = client.delete(f"/api/scheduled-messages/{row['id']}")
    assert r.status_code == 200
    lst = client.get("/api/scheduled-messages?status=cancelled").json()
    assert any(m["id"] == row["id"] for m in lst["messages"])
