"""The task list must be private, and it must carry a task to the end.

Two defects found 2026-09-01 while answering "does the agent have a todo
list and calendar":

  * `create` recorded no owner and `list` returned every tracker on disk,
    so a second user would have been shown the owner's whole list. The
    owner's rule for reminders -- "notifications need to work isolated for
    each user" -- applies here too.
  * A check-in fired exactly once. Unanswered, the step sat at "pending"
    forever and nothing ever asked again, so the list could not do the one
    thing a list is for.
"""
import json

import pytest

from backend import builtin_tools as bt
from backend import follow_up as fu
from backend import tracker as tk
from backend import tracker_checkin as tc


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(tk, "data_dir", lambda: tmp_path)
    (tmp_path / "knowledge" / "projects").mkdir(parents=True, exist_ok=True)
    return tk.TrackerStore()


def _no_scheduling(monkeypatch):
    """The ledger is exercised by its own tests; keep these on the store."""
    monkeypatch.setattr(tk.TrackerStore, "_schedule_check_in",
                        lambda *a, **k: None)


# --- isolation ----------------------------------------------------------

def test_a_tracker_records_who_made_it(store):
    t = store.create(title="mine", requested_by="telegram:1")
    assert t.get("owner") == "telegram:1"


def test_one_user_does_not_see_anothers_list(store, monkeypatch):
    monkeypatch.setattr("backend.roles.is_owner", lambda s: False)
    store.create(title="his groceries", requested_by="telegram:1")
    store.create(title="her taxes", requested_by="telegram:2")

    titles = {t["title"] for t in store.list("all", requested_by="telegram:1")}
    assert titles == {"his groceries"}, "a user saw someone else's task list"


def test_the_owner_still_sees_everything(store, monkeypatch):
    monkeypatch.setattr("backend.roles.is_owner", lambda s: s == "telegram:9")
    store.create(title="a", requested_by="telegram:1")
    store.create(title="b", requested_by="telegram:2")
    assert len(store.list("all", requested_by="telegram:9")) == 2


def test_old_ownerless_trackers_are_not_public(store, monkeypatch):
    # Rows written before the owner field existed must fall to the owner,
    # not to everyone -- "unowned means public" is the leak itself.
    monkeypatch.setattr("backend.roles.is_owner", lambda s: s == "telegram:9")
    legacy = {"id": "trk_old", "title": "legacy", "status": "active",
              "steps": [], "created_at": "2026-01-01T00:00:00Z"}
    assert tk.may_access(legacy, "telegram:9") is True
    assert tk.may_access(legacy, "telegram:2") is False


def test_get_hides_existence_not_just_content(store, monkeypatch):
    monkeypatch.setattr("backend.roles.is_owner", lambda s: False)
    monkeypatch.setattr("backend.roles.current_speaker", lambda: "telegram:2")
    monkeypatch.setattr("backend.tracker.TRACKERS", store)
    t = store.create(title="private", requested_by="telegram:1")
    out = json.loads(bt._get_tracker_handler(t["id"]))
    assert out["ok"] is False
    assert "private" not in json.dumps(out)


# --- carrying a task to the end -----------------------------------------

def test_an_unanswered_step_is_raised_again(store, monkeypatch):
    _no_scheduling(monkeypatch)
    t = store.create(title="todo", requested_by="telegram:1",
                     steps=[{"title": "buy medicine",
                             "due_at": "2026-09-01T09:00:00Z"}])
    sid = t["steps"][0]["id"]

    step = store.arm_follow_up(t["id"], sid, requested_by="telegram:1")
    assert step is not None, "the step was dropped after one reminder"
    assert step["nudges"] == 1
    assert step["next_check_at"], "nothing was armed -- it will never ask again"


def test_it_gives_up_instead_of_nagging_forever(store, monkeypatch):
    _no_scheduling(monkeypatch)
    t = store.create(title="todo", requested_by="telegram:1",
                     steps=[{"title": "x", "due_at": "2026-09-01T09:00:00Z"}])
    sid = t["steps"][0]["id"]

    armed = [store.arm_follow_up(t["id"], sid) for _ in range(len(fu.BACKOFF_HOURS))]
    assert armed[-1] is None, "follow-ups never run out"
    store.park_stalled(t["id"], sid)
    step = store.get(t["id"])["steps"][0]
    assert step["status"] == "stalled"
    assert step["next_check_at"] == ""


def test_finishing_a_task_silences_it(store, monkeypatch):
    _no_scheduling(monkeypatch)
    t = store.create(title="todo", requested_by="telegram:1",
                     steps=[{"title": "x", "due_at": "2026-09-01T09:00:00Z"}])
    sid = t["steps"][0]["id"]
    store.arm_follow_up(t["id"], sid)

    store.update_step(t["id"], sid, status="done")
    step = store.get(t["id"])["steps"][0]
    assert step["next_check_at"] == "", "a finished task would be raised again"


def test_a_closed_step_wakes_nobody(store, monkeypatch):
    # run_check_in must not start an agent turn for something already done.
    ran = []
    monkeypatch.setattr("backend.tracker.TRACKERS", store)
    monkeypatch.setattr(tc, "run_check_in", tc.run_check_in)
    t = store.create(title="todo", requested_by="telegram:1",
                     steps=[{"title": "x", "due_at": "2026-09-01T09:00:00Z"}])
    sid = t["steps"][0]["id"]
    store.update_step(t["id"], sid, status="done")

    import backend.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent",
                        lambda *a, **k: ran.append(1) or pytest.fail("woke up"))
    tc.run_check_in({"id": "r1", "target_speaker": "telegram:1",
                     "meta": {"tracker_id": t["id"], "step_id": sid}})
    assert not ran


def test_the_next_one_is_armed_even_if_the_turn_explodes(store, monkeypatch):
    # A nudge scheduled only on the happy path is fire-once-and-forget again.
    _no_scheduling(monkeypatch)
    monkeypatch.setattr("backend.tracker.TRACKERS", store)
    t = store.create(title="todo", requested_by="telegram:1",
                     steps=[{"title": "x", "due_at": "2026-09-01T09:00:00Z"}])
    sid = t["steps"][0]["id"]

    import backend.agent as agent_mod

    def _boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(agent_mod, "Agent", _boom)
    tc.run_check_in({"id": "r1", "target_speaker": "telegram:1",
                     "meta": {"tracker_id": t["id"], "step_id": sid}})

    step = store.get(t["id"])["steps"][0]
    assert step["nudges"] == 1, "the follow-up was lost when the turn failed"
    assert step["next_check_at"]
