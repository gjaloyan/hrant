"""Two turns running at once must not share one accounting bucket.

From the GPT-6 Astra audit, 2026-09-05, finding 2. `TOKENS` is one
`TokenTracker` per process and the per-request totals were plain
attributes on it, so `reset_request()` from a turn starting mid-flight
zeroed the running total of the turn already in progress. The auditor's
interleaving: A records 100, B starts and records 200, A records 50 —
and A then reads 250 as its own spend, with B's calls in its breakdown.
The lock made each update atomic; it never made an update belong to
anyone.

The process-wide totals were always right. This is an attribution bug,
which is worse than it sounds: `request_usage()` is what per-turn cost
telemetry and any budget gate read.
"""
from __future__ import annotations

import threading

import pytest

from backend import llm as _llm


@pytest.fixture()
def tracker():
    return _llm.TokenTracker()


def _record(tracker, n_in, n_out=0):
    tracker.record(
        task_type="complex_solving", model="m", provider="p",
        usage={"input_tokens": n_in, "output_tokens": n_out},
        duration_ms=1,
    )


def test_interleaved_turns_keep_their_own_totals(tracker):
    """The audit's exact sequence, driven deterministically."""
    a_started, b_recorded, a_done = (threading.Event() for _ in range(3))
    seen: dict = {}

    def turn_a():
        tracker.reset_request()
        _record(tracker, 100)
        a_started.set()
        assert b_recorded.wait(5)
        _record(tracker, 50)
        seen["a"] = tracker.request_usage()
        a_done.set()

    def turn_b():
        assert a_started.wait(5)
        tracker.reset_request()          # this used to zero A's total
        _record(tracker, 200)
        b_recorded.set()
        assert a_done.wait(5)
        seen["b"] = tracker.request_usage()

    ta, tb = threading.Thread(target=turn_a), threading.Thread(target=turn_b)
    ta.start(); tb.start(); ta.join(5); tb.join(5)

    assert seen["a"]["input_tokens"] == 150, "A must not absorb B's spend"
    assert seen["b"]["input_tokens"] == 200
    assert seen["a"]["llm_calls"] == 2 and seen["b"]["llm_calls"] == 1
    # And the process-wide view still sees every call.
    assert len(tracker.recent_calls(limit=50)) == 3


def test_the_breakdown_does_not_mix_turns(tracker):
    """`request_calls_since` feeds per-iteration telemetry into the turn
    artifact; it must not hand a turn someone else's records."""
    started, other_done = threading.Event(), threading.Event()
    seen: dict = {}

    def turn_a():
        tracker.reset_request()
        start = tracker.request_calls_count()
        _record(tracker, 10)
        started.set()
        assert other_done.wait(5)
        seen["a"] = tracker.request_calls_since(start)

    def turn_b():
        assert started.wait(5)
        tracker.reset_request()
        for _ in range(4):
            _record(tracker, 7)
        other_done.set()

    ta, tb = threading.Thread(target=turn_a), threading.Thread(target=turn_b)
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    assert len(seen["a"]) == 1
    assert seen["a"][0]["input_tokens"] == 10


def test_a_nested_run_folds_its_spend_into_the_parent(tracker):
    """The rule the audit asked us to make explicit: a child agent's
    tokens belong to the turn that delegated to it. The child still
    reports only its own while it runs."""
    parent_token = tracker.reset_request()
    _record(tracker, 100)

    child_token = tracker.reset_request()
    _record(tracker, 30)
    assert tracker.request_usage()["input_tokens"] == 30, "child sees only its own"
    tracker.end_request(child_token)

    assert tracker.request_usage()["input_tokens"] == 130, "parent absorbs the child"
    assert tracker.request_usage()["llm_calls"] == 2
    tracker.end_request(parent_token)


def test_recording_outside_any_turn_touches_no_bucket(tracker):
    """An autonomic tick has no turn. Its tokens count process-wide and
    are attributed to nobody, instead of landing in whichever turn
    happens to be open."""
    token = tracker.reset_request()
    _record(tracker, 100)

    done = threading.Event()

    def tick():                       # own context, never resets
        _record(tracker, 999)
        done.set()

    t = threading.Thread(target=tick); t.start(); done.wait(5); t.join(5)

    assert tracker.request_usage()["input_tokens"] == 100
    assert len(tracker.recent_calls(limit=50)) == 2
    tracker.end_request(token)
