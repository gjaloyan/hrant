"""The calendar must be readable, and readable per-person.

`schedule_message` was the only scheduling tool the model could see, so the
agent could put things on the calendar and never look at them again: no
answer to "what do I have tomorrow", no way to spot a clash before promising
a time, nothing to cancel or move. These tests pin the read side down --
including the part that matters more than the feature: one person must not
see or cancel another person's reminders.
"""
import json

import pytest

from backend import builtin_tools as bt
from backend import scheduled_messages as sm
from backend.tool_bundles import BASE_TOOLS


def _mk(store, monkeypatch, tmp_path):
    # The ledger is JSONL under the knowledge base_dir; point it at tmp.
    path = tmp_path / "scheduled_messages.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in store),
        encoding="utf-8")
    monkeypatch.setattr(sm, "_path", lambda: path)


def _row(mid, who, due, text):
    return {"id": mid, "status": "pending", "requested_by": who,
            "target_speaker": who, "due_at": due, "text": text}


def test_reachable_without_a_bundle():
    # Registered-but-invisible is the exact trap that stranded
    # schedule_message and the tracker tools: the per-turn schema is
    # BASE | loaded-bundles, so a tool outside both is never offered.
    assert "list_scheduled" in BASE_TOOLS
    assert "cancel_scheduled" in BASE_TOOLS


def test_times_come_back_in_the_owners_zone(monkeypatch, tmp_path):
    # The server runs on UTC. 09:00Z is 13:00 in Yerevan, and the answer
    # has to be given in the zone the question was asked in.
    _mk([_row("m1", "telegram:1", "2099-01-02T09:00:00Z", "dentist")],
        monkeypatch, tmp_path)
    monkeypatch.setattr(bt, "json", json)
    monkeypatch.setattr("backend.roles.current_speaker", lambda: "telegram:1")
    monkeypatch.setattr("backend.roles.is_owner", lambda s: True)
    monkeypatch.setattr("backend.settings.user_timezone", lambda: "Asia/Yerevan")

    out = json.loads(bt._list_scheduled_handler(horizon_days=100000))
    assert out["ok"] and out["count"] == 1
    assert out["timezone"] == "Asia/Yerevan"
    # 13:00, not the 09:00 sitting in the store.
    assert "13:00" in out["reminders"][0]["when"], out["reminders"][0]
    assert "09:00" not in out["reminders"][0]["when"]


def test_one_person_cannot_read_anothers(monkeypatch, tmp_path):
    _mk([_row("mine", "telegram:1", "2099-01-02T09:00:00Z", "mine"),
         _row("his", "telegram:2", "2099-01-02T10:00:00Z", "his")],
        monkeypatch, tmp_path)
    monkeypatch.setattr("backend.roles.current_speaker", lambda: "telegram:1")
    monkeypatch.setattr("backend.roles.is_owner", lambda s: False)
    monkeypatch.setattr("backend.settings.user_timezone", lambda: "Asia/Yerevan")

    out = json.loads(bt._list_scheduled_handler(horizon_days=100000))
    ids = {r["id"] for r in out["reminders"]}
    assert ids == {"mine"}, "a non-owner saw someone else's calendar"

    # And asking for everyone's is refused rather than quietly widened.
    denied = json.loads(bt._list_scheduled_handler(scope="all"))
    assert denied["ok"] is False


def test_one_person_cannot_cancel_anothers(monkeypatch, tmp_path):
    _mk([_row("his", "telegram:2", "2099-01-02T10:00:00Z", "his")],
        monkeypatch, tmp_path)
    monkeypatch.setattr("backend.roles.current_speaker", lambda: "telegram:1")
    # Not the owner: the owner may manage anyone's, which is deliberate.
    monkeypatch.setattr("backend.roles.is_owner", lambda s: False)

    out = json.loads(bt._cancel_scheduled_handler("his"))
    assert out["ok"] is False
    assert sm.list_pending()[0]["id"] == "his", "the row was cancelled anyway"


def test_the_horizon_actually_cuts(monkeypatch, tmp_path):
    _mk([_row("soon", "telegram:1", "2026-09-02T09:00:00Z", "soon"),
         _row("far", "telegram:1", "2099-01-02T09:00:00Z", "far")],
        monkeypatch, tmp_path)
    monkeypatch.setattr("backend.roles.current_speaker", lambda: "telegram:1")
    monkeypatch.setattr("backend.roles.is_owner", lambda s: True)
    monkeypatch.setattr("backend.settings.user_timezone", lambda: "Asia/Yerevan")

    out = json.loads(bt._list_scheduled_handler(horizon_days=7))
    assert {r["id"] for r in out["reminders"]} <= {"soon"}
