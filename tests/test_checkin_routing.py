"""deliver_due() must route kind=='check_in' rows to the agent-wake path
and NOT to static deliver(); normal rows still go to deliver()."""
from __future__ import annotations

import pytest


@pytest.fixture
def sched(tmp_path, monkeypatch):
    """An ISOLATED ledger.

    This fixture used to set HRANT_DATA_DIR and reload the module, which
    isolated nothing: `_path()` resolves against `CONFIG.knowledge["base_dir"]`
    at call time, not against the environment at import time. So every run of
    this test appended to the developer's real ledger
    (~/.hrant/data/knowledge/scheduled_messages.jsonl — 343 rows by
    2026-08-10), and `due_now()` returned that accumulated junk alongside the
    two rows the test scheduled. It passed or failed depending on what was
    lying in a shared file, which is how it stayed green for months while
    asserting nothing, then went red when new test files shifted the order.
    """
    from backend import scheduled_messages as sm
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG.knowledge, "base_dir", str(tmp_path))
    return sm


def test_checkin_row_wakes_agent_not_deliver(sched, monkeypatch):
    delivered, woken = [], []
    monkeypatch.setattr(sched, "deliver", lambda row: (delivered.append(row["id"]) or (True, "")))
    import backend.tracker_checkin as tc
    monkeypatch.setattr(tc, "run_check_in", lambda row: woken.append(row["id"]))

    sched.schedule(target_speaker="webui:default", text="status?",
                   due_at="2000-01-01T00:00:00Z", requested_by="webui:default",
                   kind="check_in", meta={"tracker_id": "trk_1", "step_id": "st_1"})
    sched.schedule(target_speaker="webui:default", text="plain",
                   due_at="2000-01-01T00:00:00Z", requested_by="webui:default")

    summary = sched.deliver_due()
    assert len(woken) == 1              # the check-in row woke the agent
    assert len(delivered) == 1          # the plain message went to deliver()
