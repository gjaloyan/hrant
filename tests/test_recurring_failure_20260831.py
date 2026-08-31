"""A tool failing the same way across turns is a bug the agent owns.

`unified_agent` already says so, in a comment above its failure counter:
"never once considered that the defect was in its own handler, which it
has always-on tools to repair. Nothing told it that a tool failing the
same way repeatedly is a BUG IT OWNS rather than an environment it must
route around."

The counter that was meant to say it lives inside `run_unified` and resets
every turn, so it only fires on three failures IN ONE TURN.

Measured 2026-08-31. The brother's reminder was refused five times across
four turns — one, one, two, one. The per-turn threshold was never reached;
the agent apologised in Armenian each time and the defect sat in a handler
it could have read and proposed a patch for.

Recurrence is therefore counted persistently, by DISTINCT TURNS, on a
signature of the failure rather than its exact text.
"""
import pytest

from backend import recurring_failures as rf


REFUSAL = "refused: trusted users may only schedule messages to the owner"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "rf.json")
    yield


# ── the measured case ───────────────────────────────────────────────

def test_the_brothers_refusal_would_now_be_caught():
    """Five failures across four turns: 1, 1, 2, 1."""
    seen = []
    for turn, times in (("t1", 1), ("t2", 1), ("t3", 2), ("t4", 1)):
        for _ in range(times):
            seen.append(rf.note("schedule_message", REFUSAL, turn_id=turn))
    assert max(seen) == 4
    assert rf.RECURRENCE_THRESHOLD in seen, (
        "the threshold must be crossed somewhere in that sequence")


def test_retries_inside_one_turn_count_once():
    """A tool retried five times in one turn is one piece of evidence.
    Counting them separately would fire on a single flaky moment."""
    for _ in range(5):
        n = rf.note("schedule_message", REFUSAL, turn_id="same-turn")
    assert n == 1


def test_the_threshold_needs_three_separate_turns():
    assert rf.note("t", "boom", turn_id="a") == 1
    assert rf.note("t", "boom", turn_id="b") == 2
    assert rf.note("t", "boom", turn_id="c") == 3


# ── the signature ───────────────────────────────────────────────────

def test_changing_ids_do_not_split_one_bug():
    """"no pending message with id a1b2c3" and "...d4e5f6" are one defect."""
    a = rf.signature("sched", "no pending message with id a1b2c3d4")
    b = rf.signature("sched", "no pending message with id 99887766")
    assert a == b


def test_changing_paths_and_targets_do_not_split_one_bug():
    a = rf.signature("fetch", "could not resolve target 'Тигран'")
    b = rf.signature("fetch", "could not resolve target 'Ashot'")
    assert a == b


def test_genuinely_different_failures_stay_apart():
    """Over-collapsing would blame one bug for another's evidence."""
    a = rf.signature("schedule_message", REFUSAL)
    b = rf.signature("schedule_message", "could not resolve target")
    assert a != b


def test_the_same_message_from_different_tools_stays_apart():
    assert rf.signature("a", "boom") != rf.signature("b", "boom")


# ── housekeeping must not distort the count ─────────────────────────

def test_a_fixed_failure_can_be_forgotten():
    rf.note("t", "boom", turn_id="a")
    rf.note("t", "boom", turn_id="b")
    rf.clear("t", "boom")
    assert rf.note("t", "boom", turn_id="c") == 1


def test_stale_signatures_stop_counting(monkeypatch):
    """A bug fixed weeks ago must not resurface as evidence against the
    code that replaced it."""
    import time
    rf.note("t", "boom", turn_id="old")
    data = rf._load()
    for v in data.values():
        v["last_seen"] = time.time() - rf.MAX_AGE_SECONDS - 10
    rf._save(data)
    assert rf.note("t", "boom", turn_id="new") == 1


def test_a_burst_of_one_offs_cannot_evict_a_live_problem():
    for t in ("a", "b"):
        rf.note("real", REFUSAL, turn_id=t)
    for i in range(rf._MAX_ENTRIES + 20):
        rf.note(f"noise{i}", "x", turn_id="n")
    assert rf.note("real", REFUSAL, turn_id="c") == 3


def test_missing_identifiers_are_ignored_not_counted():
    assert rf.note("", "boom", turn_id="a") == 0
    assert rf.note("t", "boom", turn_id="") == 0


def test_a_broken_store_does_not_raise(tmp_path, monkeypatch):
    p = tmp_path / "rf.json"
    p.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(rf, "_path", lambda: p)
    assert rf.note("t", "boom", turn_id="a") == 1


# ── what the agent is told ──────────────────────────────────────────

def test_the_marker_says_the_bug_is_the_agents_own():
    m = rf.marker("schedule_message", REFUSAL, 3)
    assert "OLDER THAN THIS TURN" in m
    assert "defect in code you can read and change" in m


def test_the_marker_names_the_tools_that_can_fix_it():
    m = rf.marker("schedule_message", REFUSAL, 3)
    assert "propose_self_modification" in m


def test_the_marker_does_not_assume_the_refusal_is_wrong():
    """Some refusals are correct. Telling it to patch unconditionally
    would trade a stuck agent for a destructive one."""
    m = rf.marker("schedule_message", REFUSAL, 3).lower()
    assert "decide whether the refusal is correct" in m
    assert "if it is, say so plainly" in m


def test_the_marker_carries_the_failure_text():
    assert "trusted users" in rf.marker("schedule_message", REFUSAL, 3)


# ── wiring ──────────────────────────────────────────────────────────

def test_the_turn_loop_records_cross_turn_failures():
    import inspect
    from backend import unified_agent as ua
    src = inspect.getsource(ua)
    assert "recurring_failures as _rf" in src
    assert "_rf.RECURRENCE_THRESHOLD" in src


def test_the_turn_identifier_cannot_raise_on_a_tool_turn():
    """`turn_id` is assigned on the fast chat path only; referencing it
    here would NameError on exactly the turns that use tools."""
    import inspect
    from backend import unified_agent as ua
    src = inspect.getsource(ua)
    assert 'turn_id=str(job_id or "") or f"t{_run_started_at:.0f}"' in src
