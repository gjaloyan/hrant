"""Scheduled-message restart resilience — Bundle B (Fix I9).

Audit finding: a row could be Telegram-accepted (or WebUI-appended)
BEFORE `mark_sent` was called. If the process crashed in between
— OOM kill, hrant.service restart, hardware power cut — the row
stayed in `pending` and the next FIRE_SCHEDULED_MESSAGES tick
re-delivered the same message. The user got duplicate messages
silently.

Fix shape:
  - `mark_delivering(id)` flips a row from `pending` to `delivering`
    BEFORE the transport call.
  - `deliver()` calls it before the Telegram send / WebUI append.
  - `due_now()` only returns `pending` rows, so `delivering` rows
    are naturally skipped — even if the recovery hook hasn't fired
    yet.
  - `recover_stuck_deliveries()` (called at FastAPI lifespan start)
    flips every leftover `delivering` row to `failed` with the
    reason "interrupted by restart" — surfaces the truth to the
    owner instead of silently re-sending.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_sched(tmp_path, monkeypatch):
    """Point scheduled_messages at a tmp ledger file."""
    from backend import scheduled_messages as _sm
    path = tmp_path / "scheduled.jsonl"
    monkeypatch.setattr(_sm, "_path", lambda: path)
    return _sm, path


def test_mark_delivering_persists_status(isolated_sched):
    """A pending row flipped via mark_delivering ends up with
    status='delivering' on disk."""
    sm, path = isolated_sched

    row = sm.schedule(
        target_speaker="webui:default",
        text="hi",
        due_at="2026-06-10T10:00:00Z",
        requested_by="webui:default",
    )

    sm.mark_delivering(row["id"])

    rows = sm.list_all()
    assert len(rows) == 1
    assert rows[0]["status"] == "delivering"
    assert rows[0]["id"] == row["id"]


def test_recover_stuck_deliveries_flips_to_failed(isolated_sched):
    """recover_stuck_deliveries finds every delivering row and
    marks it failed with the interrupted-by-restart reason."""
    sm, path = isolated_sched

    a = sm.schedule(
        target_speaker="webui:default", text="a",
        due_at="2026-06-10T10:00:00Z", requested_by="webui:default",
    )
    b = sm.schedule(
        target_speaker="webui:default", text="b",
        due_at="2026-06-10T10:00:00Z", requested_by="webui:default",
    )
    c = sm.schedule(
        target_speaker="webui:default", text="c",
        due_at="2026-06-10T10:00:00Z", requested_by="webui:default",
    )

    # Two rows were mid-delivery at crash time; one stayed pending.
    sm.mark_delivering(a["id"])
    sm.mark_delivering(b["id"])

    # Simulate the next service startup.
    n = sm.recover_stuck_deliveries()
    assert n == 2

    rows = {r["id"]: r for r in sm.list_all()}
    assert rows[a["id"]]["status"] == "failed"
    assert rows[a["id"]]["last_error"] == "interrupted by restart"
    assert rows[b["id"]]["status"] == "failed"
    assert rows[b["id"]]["last_error"] == "interrupted by restart"
    # The pending one is untouched.
    assert rows[c["id"]]["status"] == "pending"
    assert rows[c["id"]]["last_error"] == ""


def test_recover_stuck_deliveries_no_op_when_clean(isolated_sched):
    """If there are no delivering rows, recover returns 0 and
    doesn't touch the ledger."""
    sm, path = isolated_sched

    sm.schedule(
        target_speaker="webui:default", text="hi",
        due_at="2026-06-10T10:00:00Z", requested_by="webui:default",
    )
    n = sm.recover_stuck_deliveries()
    assert n == 0
    assert sm.list_all()[0]["status"] == "pending"


def test_deliver_due_skips_delivering_status(isolated_sched):
    """due_now() / list_pending() must skip delivering rows so a
    tick that fires before the recovery hook doesn't accidentally
    redeliver them."""
    sm, path = isolated_sched

    # Use a past `due_at` so due_now() actually picks rows up.
    past_due = "2020-01-01T00:00:00Z"
    pending = sm.schedule(
        target_speaker="webui:default", text="pending",
        due_at=past_due, requested_by="webui:default",
    )
    delivering = sm.schedule(
        target_speaker="webui:default", text="delivering",
        due_at=past_due, requested_by="webui:default",
    )
    sm.mark_delivering(delivering["id"])

    due = sm.due_now()
    ids = [r["id"] for r in due]
    assert pending["id"] in ids
    assert delivering["id"] not in ids


def test_deliver_marks_delivering_before_transport(isolated_sched, monkeypatch):
    """deliver() flips the row to delivering BEFORE calling the
    transport (here a fake WebUI append) — so a crash during the
    transport leaves the row in delivering state."""
    sm, path = isolated_sched

    row = sm.schedule(
        target_speaker="webui:default", text="hi",
        due_at="2026-06-10T10:00:00Z", requested_by="webui:default",
    )

    # When the conversation append fires, inspect the on-disk status.
    seen_status: dict = {}

    class _CapturingConv:
        def add_turn(self, *a, **kw):
            current = sm.list_all()
            for r in current:
                if r["id"] == row["id"]:
                    seen_status["during_transport"] = r["status"]

    monkeypatch.setattr("backend.conversation.CONVERSATION", _CapturingConv())

    ok, err = sm.deliver(row)
    assert ok is True
    # The transport saw the row in delivering state.
    assert seen_status["during_transport"] == "delivering"
    # After successful send it's flipped to sent.
    final = next(r for r in sm.list_all() if r["id"] == row["id"])
    assert final["status"] == "sent"
