"""Tests for Phase 4 — scheduled-message preview + cancel button.

Pinned behaviour:
  - scheduled_messages.schedule() fires the on_message_scheduled
    callback registry.
  - register_on_message_scheduled is idempotent.
  - sched:cancel:<id> walks scheduled_messages.cancel(); only owners.
  - A nonexistent / already-delivered id surfaces an "not found"
    toast without modifying state.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_sched(tmp_path, monkeypatch):
    """Redirect scheduled_messages.json to tmp_path and reset the
    callback registry between tests."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    # Force re-resolution of the data path.
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    from backend import scheduled_messages as sm
    saved = list(sm._ON_MESSAGE_SCHEDULED)
    sm._ON_MESSAGE_SCHEDULED.clear()
    yield sm
    sm._ON_MESSAGE_SCHEDULED.clear()
    sm._ON_MESSAGE_SCHEDULED.extend(saved)


def test_schedule_fires_callback(isolated_sched):
    sm = isolated_sched
    fired: list = []
    sm.register_on_message_scheduled(lambda row: fired.append(row))
    row = sm.schedule(
        target_speaker="telegram:222",
        text="don't forget the milk",
        due_at="2026-05-18T10:00:00Z",
        requested_by="webui:default",
    )
    assert row["id"]
    assert len(fired) == 1
    assert fired[0]["text"] == "don't forget the milk"


def test_register_on_message_scheduled_idempotent(isolated_sched):
    sm = isolated_sched
    calls: list = []

    def cb(row):
        calls.append(row)

    sm.register_on_message_scheduled(cb)
    sm.register_on_message_scheduled(cb)
    sm.register_on_message_scheduled(cb)
    sm.schedule(
        target_speaker="telegram:222", text="hi",
        due_at="2026-05-18T10:00:00Z", requested_by="webui:default",
    )
    assert len(calls) == 1


def test_callback_failure_does_not_break_schedule(isolated_sched):
    sm = isolated_sched

    def bad(row):
        raise RuntimeError("subscriber kaboom")

    sm.register_on_message_scheduled(bad)
    row = sm.schedule(
        target_speaker="telegram:222", text="hi",
        due_at="2026-05-18T10:00:00Z", requested_by="webui:default",
    )
    assert row["id"]
    assert row["status"] == "pending"


def test_sched_cancel_callback_cancels_pending_message(isolated_sched):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    row = isolated_sched.schedule(
        target_speaker="telegram:222", text="hi",
        due_at="2026-05-18T10:00:00Z", requested_by="webui:default",
    )
    res = tg_interactive.dispatch_callback(
        f"sched:cancel:{row['id']}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "Cancelled" in (res.edited_text or "")
    # Underlying state — message is cancelled.
    all_rows = isolated_sched._read_all()
    r = next(x for x in all_rows if x["id"] == row["id"])
    assert r["status"] == "cancelled"


def test_sched_cancel_refuses_non_owner(isolated_sched):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:222", "trusted")
    row = isolated_sched.schedule(
        target_speaker="telegram:333", text="hi",
        due_at="2026-05-18T10:00:00Z", requested_by="webui:default",
    )
    res = tg_interactive.dispatch_callback(
        f"sched:cancel:{row['id']}",
        ctx={"clicker_speaker_id": "telegram:222"},
    )
    assert res.ok is False
    # Reworded 2026-08-31. The rule is no longer "only the owner of the bot"
    # but "only the owner OF THIS REMINDER" — a trusted user may cancel their
    # own, and telegram:222 is neither requester nor recipient here.
    assert "not your reminder" in (res.toast or "").lower()
    r = next(x for x in isolated_sched._read_all() if x["id"] == row["id"])
    assert r["status"] == "pending"  # unchanged


def test_sched_cancel_missing_id_returns_error(isolated_sched):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    res = tg_interactive.dispatch_callback(
        "sched:cancel:NOSUCH",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is False
    assert "not found" in (res.toast or "").lower() or "already" in (res.toast or "").lower()
