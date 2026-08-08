"""Section 4 of the 2026-08-08 audit: the goal loop and subagent collection.

The goal loop had no attempt bookkeeping at all. A failed subtask stayed
`pending` with no record, and Phase 2 always picked `pending[0]`, so the
driver re-dispatched the identical builder subagent every cycle forever.

Measured on prod: goal 7fb4280130 ("Learn: TensorFlow checkpoint binary format
parsing and GPT-2 architecture implementation in C") had 52 byte-identical
subagent sessions for ONE subtask between 2026-07-06 and 2026-07-10 — median
gap 7263s, every one failed with "empty answer from subagent", iterations=0.
Four days of builder tool-loops on one subtask while subtasks 4-6 sat
untouched, and because the lever only ever takes the highest-priority goal,
that one goal starved all the others.
"""
from __future__ import annotations

from backend.goals import Goal


def _goal_with(n=3):
    g = Goal(description="learn a thing")
    for i in range(n):
        g.add_subtask(f"subtask {i + 1}")
    return g


def test_a_failed_subtask_records_the_attempt():
    g = _goal_with()
    assert g.note_subtask_attempt(0, error="empty answer from subagent") == 1
    assert g.subtasks[0]["attempts"] == 1
    assert "empty answer" in g.subtasks[0]["last_error"]
    assert g.subtasks[0]["status"] == "pending"      # still retryable


def test_a_subtask_is_blocked_after_the_attempt_limit():
    g = _goal_with()
    for _ in range(Goal.MAX_SUBTASK_ATTEMPTS):
        g.note_subtask_attempt(0, error="empty answer from subagent")
    assert g.subtasks[0]["status"] == "blocked"
    # and it is no longer picked up as pending work
    pending = [s for s in g.subtasks if s.get("status") == "pending"]
    assert g.subtasks[0] not in pending
    assert len(pending) == 2


def test_the_driver_picks_the_least_attempted_subtask():
    """The starvation half: subtasks behind a failing one were never tried."""
    g = _goal_with()
    g.note_subtask_attempt(0, error="boom")
    g.note_subtask_attempt(0, error="boom")

    pending = [(i, st) for i, st in enumerate(g.subtasks)
               if st.get("status") == "pending"]
    idx, _ = min(pending, key=lambda p: int(p[1].get("attempts") or 0))
    assert idx == 1, "a twice-failed subtask must not keep winning the pick"


def test_fifty_two_identical_dispatches_can_no_longer_happen():
    """The prod incident, compressed: a permanently failing subtask must stop
    consuming cycles instead of running until someone notices four days on."""
    g = _goal_with(n=2)
    dispatched = []
    for _ in range(20):
        pending = [(i, st) for i, st in enumerate(g.subtasks)
                   if st.get("status") == "pending"]
        if not pending:
            break
        idx, _st = min(pending, key=lambda p: int(p[1].get("attempts") or 0))
        dispatched.append(idx)
        g.note_subtask_attempt(idx, error="empty answer from subagent")

    assert len(dispatched) == 2 * Goal.MAX_SUBTASK_ATTEMPTS
    assert dispatched.count(0) == Goal.MAX_SUBTASK_ATTEMPTS
    assert dispatched.count(1) == Goal.MAX_SUBTASK_ATTEMPTS, \
        "the second subtask must get its turn, not starve behind the first"
    assert all(s["status"] == "blocked" for s in g.subtasks)


def test_a_successful_subtask_still_completes_normally():
    g = _goal_with()
    g.complete_subtask(0, result="built it")
    assert g.subtasks[0]["status"] == "done"
    assert g.subtasks[0]["result"] == "built it"


def test_note_attempt_on_a_bogus_index_is_harmless():
    g = _goal_with()
    assert g.note_subtask_attempt(99, error="x") == 0
    assert all(s["status"] == "pending" for s in g.subtasks)


# ── collecting subagent results ───────────────────────────────────────

def test_check_subagents_reports_what_it_did_not_show(monkeypatch, tmp_path):
    """8 dispatched, all completed, 6 collected — and no signal that two were
    dropped, so the parent would re-dispatch and duplicate side effects."""
    import json
    import backend.builtin_tools as bt
    from backend.subagents.store import SUBAGENT_STORE

    class _S:
        def __init__(self, i):
            self.id, self.role, self.status = f"s{i}", "builder", "completed"
            self.task, self.answer, self.error = f"task {i}", "done", ""
            self.created_at = 1000 + i

    made = [_S(i) for i in range(12)]
    monkeypatch.setattr(SUBAGENT_STORE, "list", lambda limit=6: made[:limit])

    out = json.loads(bt._check_subagents_handler())
    assert out["ok"] is True
    assert out["truncated"] == 0 or "not shown" in out["note"]
    assert out["returned"] == len(out["sessions"])


def test_running_sessions_are_collected_before_old_history(monkeypatch):
    """Ordering is mtime over ALL history, so an old session rewritten by the
    reaper could push an in-flight one out of the window."""
    import json
    import backend.builtin_tools as bt
    from backend.subagents.store import SUBAGENT_STORE

    class _S:
        def __init__(self, i, status):
            self.id, self.role, self.status = f"s{i}", "builder", status
            self.task, self.answer, self.error = f"task {i}", "", ""
            self.created_at = 1_000_000_000 + i

    made = [_S(i, "completed") for i in range(10)] + [_S(99, "running")]
    monkeypatch.setattr(SUBAGENT_STORE, "list", lambda limit=6: made[:limit])
    monkeypatch.setattr(SUBAGENT_STORE, "_write", lambda s: None)

    out = json.loads(bt._check_subagents_handler())
    ids = [s["session_id"] for s in out["sessions"]]
    assert "s99" in ids, "an in-flight session must never be crowded out"
