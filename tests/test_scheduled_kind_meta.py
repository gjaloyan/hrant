"""schedule() must carry an optional kind + meta so the tick can route
check-ins to the agent instead of a static send."""
from __future__ import annotations

import pytest


@pytest.fixture
def sched(tmp_path, monkeypatch):
    """An ISOLATED ledger — see the note in test_checkin_routing.py.

    Setting HRANT_DATA_DIR and reloading isolates nothing: `_path()` resolves
    against CONFIG at call time, so these tests appended to the developer's
    real ledger on every run."""
    from backend import scheduled_messages as sm
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG.knowledge, "base_dir", str(tmp_path))
    return sm


def test_schedule_defaults_to_message_kind(sched):
    row = sched.schedule(target_speaker="webui:default", text="hi",
                         due_at="2026-06-25T09:00:00Z", requested_by="webui:default")
    assert row["kind"] == "message"
    assert row["meta"] == {}


def test_schedule_records_kind_and_meta(sched):
    row = sched.schedule(
        target_speaker="webui:default", text="", due_at="2026-06-25T09:00:00Z",
        requested_by="webui:default", kind="check_in",
        meta={"tracker_id": "trk_1", "step_id": "st_1", "check_in_kind": "ask_status"},
    )
    assert row["kind"] == "check_in"
    assert row["meta"]["tracker_id"] == "trk_1"
