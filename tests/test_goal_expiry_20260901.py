"""An unreviewed suggestion is not a failure.

The Goals board read "509 failed" against 5 completed, which looks like an
agent that cannot finish anything. Every one of those 509 carried the same
note: "Auto-archived as stale: pending human approval >14 days". They were
never attempted -- they were improvement proposals waiting on the owner,
and the stale-archiver filed them under `failed`.

Measured on prod 2026-09-01: 509 expired, 100 still waiting, 5 approved.
"""
import importlib
import json

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, goals
    importlib.reload(config)
    importlib.reload(goals)
    return goals


STALE = ("Auto-archived as stale: pending human approval >14 days. "
         "Re-propose via meta_learner if still relevant.")


def _write(mod, rows):
    mod.GOALS.path.parent.mkdir(parents=True, exist_ok=True)
    mod.GOALS.path.write_text(
        json.dumps({"goals": rows, "interaction_count": 0}), encoding="utf-8")
    mod.GOALS._load()


def _goal(gid, status, notes):
    return {"id": gid, "description": "d" + gid, "priority": 5,
            "goal_type": "improvement", "status": status,
            "subtasks": [], "created": "2026-06-01 00:00:00",
            "progress_notes": notes, "context": "", "source": "meta_learner",
            "completed": ""}


def test_never_attempted_is_expired_not_failed(store):
    _write(store, [_goal("a", "failed", [STALE])])
    assert store.GOALS.stats()["expired"] == 1
    assert store.GOALS.stats()["failed"] == 0


def test_a_real_failure_stays_failed(store):
    # The migration must not sweep genuine breakage into the quiet bucket.
    _write(store, [_goal("b", "failed", ["[failed] tests did not pass"])])
    assert store.GOALS.stats()["failed"] == 1
    assert store.GOALS.stats()["expired"] == 0


def test_the_two_are_counted_apart(store):
    _write(store, [
        _goal("a", "failed", [STALE]),
        _goal("b", "failed", [STALE]),
        _goal("c", "failed", ["[failed] provider returned garbage"]),
        _goal("d", "active", []),
        _goal("e", "completed", []),
    ])
    s = store.GOALS.stats()
    assert (s["expired"], s["failed"], s["active"], s["completed"]) == (2, 1, 1, 1)


def test_the_migration_is_written_back(store):
    _write(store, [_goal("a", "failed", [STALE])])
    on_disk = json.loads(store.GOALS.path.read_text(encoding="utf-8"))
    assert on_disk["goals"][0]["status"] == "expired", (
        "reclassified in memory only — the next process would redo it")


def test_expire_says_what_happened(store):
    _write(store, [_goal("a", "active", [])])
    g = store.GOALS._goals[0]
    g.expire("nobody looked at it")
    assert g.status == "expired"
    assert any("expired" in n for n in g.progress_notes)


HYGIENE = ("[failed] auto-archived: stale >14d without execution "
           "(hygiene sweep); regenerate if still relevant")


def test_the_hygiene_sweep_also_expires(store):
    """437 of the 509 came from THIS sweep, not the approval archiver — the
    first fix covered only 72 of them."""
    _write(store, [_goal("h", "active", [])])
    store.GOALS._goals[0].goal_type = "improvement"
    store.GOALS._goals[0].created = "2020-01-01 00:00:00"
    assert store.GOALS.archive_stale(days=14) == 1
    assert store.GOALS._goals[0].status == "expired"


def test_the_migration_covers_the_hygiene_marker(store):
    _write(store, [_goal("h", "failed", [HYGIENE])])
    assert store.GOALS.stats()["expired"] == 1


def test_the_archiver_itself_expires_rather_than_fails(store, monkeypatch):
    """Pin the source, not just the repair.

    The migration would quietly re-fix anything the lever mislabels, which
    is exactly the shape of bug that survives for months. This asserts the
    lever writes the right status in the first place.
    """
    from backend.autonomic.levers import goal_executor as ge

    _write(store, [{
        "id": "z", "description": "Improve prompt: something",
        "priority": 5, "goal_type": "improvement", "status": "active",
        "subtasks": [{"description": "Owner to approve or reject the proposal",
                      "status": "pending"}],
        "created": "2020-01-01 00:00:00", "progress_notes": [],
        "context": "", "source": "meta_learner", "completed": "",
    }])
    lever = ge.FIRE_GOAL_EXECUTOR()
    lever.run({"stale_days": 14}, {})

    g = store.GOALS._goals[0]
    assert g.status == "expired", (
        "an unattempted proposal was filed as a failure again")
    # One archiver, one sentence. There used to be a second branch further
    # down this lever writing a different note for the same outcome; it was
    # unreachable, because the sweep above shares its cutoff.
    notes = " ".join(g.progress_notes)
    assert "hygiene sweep" in notes
    assert "pending human approval" not in notes
