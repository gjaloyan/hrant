"""Context compressor tests — Phase 2.B.2.

Pinned behaviour:
  - Cheap budget check (needs_compaction) returns False when the
    history is under threshold OR shorter than head+tail+min-middle.
  - Compaction is a no-op when it would only span < MIN_MIDDLE_TURNS.
  - Head + tail turns are NEVER touched.
  - Iterative re-summarisation: a prior summary turn in the middle
    is folded into the new summariser input rather than competing.
  - Compaction replaces the middle band with a single summary turn
    that's marked `is_summary=True` so future passes can find it.
  - Failure cooldown: after an LLM error, subsequent maybe_compact
    calls return without firing for FAILURE_COOLDOWN_SECONDS.
  - In-flight guard: parallel calls for the same speaker return
    early on the second one.
  - Other-speaker turns in the same CONVERSATION buffer are NOT
    touched by another speaker's compaction.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from backend import context_compressor as cc


# --- helpers ----------------------------------------------------------


def _turn(user, answer, *, speaker_id="test:speaker", is_summary=False, intent=""):
    """Build one turn dict matching the conversation shape."""
    out = {
        "ts": "2026-05-16 10:00:00",
        "user": user,
        "answer": answer,
        "speaker_id": speaker_id,
        "channel": "webui",
        "intent": intent,
    }
    if is_summary:
        out["is_summary"] = True
        out["intent"] = cc.SUMMARY_INTENT
    return out


@pytest.fixture
def stub_conversation(monkeypatch):
    """Replace CONVERSATION's _turns + recent() so tests are
    isolated from disk and other tests' state.

    Tests assign to `stub_conversation["turns"]` to seed the buffer;
    a property-style proxy on the dict syncs the assignment back
    onto `cc.CONVERSATION._turns` so the compactor (which reads
    `CONVERSATION._turns` directly) sees the same data."""

    class _State(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            if k == "turns":
                # Mirror onto the live singleton so direct reads of
                # _turns inside the compactor pick up the latest
                # seeded state.
                cc.CONVERSATION._turns = list(v)

    state = _State()
    state["turns"] = []

    def _recent(n, *, channel=None, speaker_id=None):
        live = cc.CONVERSATION._turns
        if speaker_id is None:
            return list(live)[-n:]
        return [t for t in live if t.get("speaker_id") == speaker_id][-n:]

    def _save():
        pass

    monkeypatch.setattr(cc.CONVERSATION, "recent", _recent)
    monkeypatch.setattr(cc.CONVERSATION, "_save", _save)
    monkeypatch.setattr(cc.CONVERSATION, "_turns", [])
    # Also clear any cooldown / in-flight state between tests.
    monkeypatch.setattr(cc, "_last_failure_ts", 0.0)
    cc._in_flight.clear()
    return state


# --- budget check -----------------------------------------------------


def test_needs_compaction_false_when_history_short(stub_conversation):
    """Under HEAD+TAIL+MIN_MIDDLE turns → False regardless of size."""
    stub_conversation["turns"] = [
        _turn("hi", "hello") for _ in range(3)
    ]
    assert cc.needs_compaction(speaker_id="test:speaker") is False


def test_needs_compaction_false_when_total_chars_small(stub_conversation):
    """Long history but each turn tiny → still under threshold."""
    stub_conversation["turns"] = [
        _turn(f"q{i}", f"a{i}") for i in range(30)
    ]
    assert cc.needs_compaction(speaker_id="test:speaker") is False


def test_needs_compaction_true_above_budget(stub_conversation):
    """Enough turns AND enough total chars to pass threshold."""
    big = "x" * 5000  # 5KB per turn
    stub_conversation["turns"] = [
        _turn(big, big) for _ in range(15)  # 150KB total
    ]
    assert cc.needs_compaction(speaker_id="test:speaker") is True


# --- maybe_compact no-op paths ---------------------------------------


def test_maybe_compact_returns_under_threshold(stub_conversation):
    stub_conversation["turns"] = [_turn("q", "a") for _ in range(5)]
    stats = cc.maybe_compact(speaker_id="test:speaker")
    assert stats.fired is False
    assert "under threshold" in stats.reason or "too short" in stats.reason


def test_maybe_compact_needs_speaker_id():
    stats = cc.maybe_compact()  # no speaker_id, no force
    assert stats.fired is False


def test_force_runs_even_when_under_threshold(stub_conversation):
    """Force flag bypasses the budget check (used by tests + a
    future 'compact now' WebUI button)."""
    stub_conversation["turns"] = (
        [_turn("setup-A", "ok") for _ in range(2)]   # head
        + [_turn("middle-i", "ack") for _ in range(5)]  # middle
        + [_turn("tail-i", "tail") for _ in range(6)]   # tail
    )
    fake_router = MagicMock()
    fake_router.call.return_value = "## Summary\nfoo"
    with patch("backend.context_compressor.maybe_compact",
               wraps=cc.maybe_compact):  # spy
        with patch("backend.llm.router", return_value=fake_router):
            stats = cc.maybe_compact(speaker_id="test:speaker", force=True)
    assert stats.fired is True
    assert stats.turns_compacted == 5


# --- happy path: compaction runs, structure preserved ---------------


def test_compaction_replaces_middle_band(stub_conversation):
    """After compaction the buffer should hold: head + 1 summary + tail."""
    stub_conversation["turns"] = (
        [_turn("setup-A", "setup-ack") for _ in range(2)]
        + [_turn("middle-q", "middle-ack") for _ in range(5)]
        + [_turn("tail-q", "tail-ack") for _ in range(6)]
    )
    fake_router = MagicMock()
    fake_router.call.return_value = (
        "## What the user wants\nstuff\n## Resolved\n- a\n## Pending\n(none)"
    )
    with patch("backend.llm.router", return_value=fake_router):
        stats = cc.maybe_compact(speaker_id="test:speaker", force=True)
    assert stats.fired
    # Buffer now: head(2) + summary(1) + tail(6) = 9
    buf = cc.CONVERSATION._turns
    speaker_turns = [t for t in buf if t.get("speaker_id") == "test:speaker"]
    assert len(speaker_turns) == 9
    # Summary turn carries the marker.
    summaries = [t for t in speaker_turns if t.get("is_summary")]
    assert len(summaries) == 1
    assert "## What the user wants" in summaries[0]["answer"]
    assert cc.SUMMARY_PREFIX in summaries[0]["answer"]


def test_compaction_preserves_head_turns_intact(stub_conversation):
    """The first HEAD_TURNS turns must survive byte-for-byte."""
    head_turns = [_turn(f"setup-{i}", f"ack-{i}") for i in range(2)]
    middle_turns = [_turn("mid-q", "mid-ack") for _ in range(5)]
    tail_turns = [_turn(f"tail-{i}", f"t-{i}") for i in range(6)]
    stub_conversation["turns"] = head_turns + middle_turns + tail_turns
    fake_router = MagicMock()
    fake_router.call.return_value = "summary body"
    with patch("backend.llm.router", return_value=fake_router):
        cc.maybe_compact(speaker_id="test:speaker", force=True)
    buf_for_speaker = [
        t for t in cc.CONVERSATION._turns
        if t.get("speaker_id") == "test:speaker"
    ]
    for i, original in enumerate(head_turns):
        assert buf_for_speaker[i]["user"] == original["user"]
        assert buf_for_speaker[i]["answer"] == original["answer"]


def test_compaction_preserves_tail_turns_intact(stub_conversation):
    head_turns = [_turn(f"setup-{i}", f"ack-{i}") for i in range(2)]
    middle_turns = [_turn("mid", "mid-ack") for _ in range(5)]
    tail_turns = [_turn(f"tail-{i}", f"t-{i}") for i in range(6)]
    stub_conversation["turns"] = head_turns + middle_turns + tail_turns
    fake_router = MagicMock()
    fake_router.call.return_value = "summary body"
    with patch("backend.llm.router", return_value=fake_router):
        cc.maybe_compact(speaker_id="test:speaker", force=True)
    buf_for_speaker = [
        t for t in cc.CONVERSATION._turns
        if t.get("speaker_id") == "test:speaker"
    ]
    # Last len(tail_turns) should match the tail bytes-for-bytes.
    final_tail = buf_for_speaker[-len(tail_turns):]
    for i, original in enumerate(tail_turns):
        assert final_tail[i]["user"] == original["user"]
        assert final_tail[i]["answer"] == original["answer"]


def test_other_speakers_turns_not_touched(stub_conversation):
    """A compaction for `test:A` must not modify turns belonging
    to `test:B` in the same buffer."""
    head = [_turn("h1", "a", speaker_id="test:A"), _turn("h2", "a", speaker_id="test:A")]
    middle = [_turn("m", "a", speaker_id="test:A") for _ in range(5)]
    tail = [_turn("t", "a", speaker_id="test:A") for _ in range(6)]
    other = [_turn(f"other-{i}", "ans", speaker_id="test:B") for i in range(3)]
    stub_conversation["turns"] = head + middle + tail + other

    fake_router = MagicMock()
    fake_router.call.return_value = "summary body"
    with patch("backend.llm.router", return_value=fake_router):
        cc.maybe_compact(speaker_id="test:A", force=True)
    others_post = [
        t for t in cc.CONVERSATION._turns
        if t.get("speaker_id") == "test:B"
    ]
    assert len(others_post) == 3
    for i, orig in enumerate(other):
        assert others_post[i]["user"] == orig["user"]


# --- iterative re-summarisation --------------------------------------


def test_prior_summary_is_folded_into_new_one(stub_conversation):
    """When the middle already contains an `is_summary` turn (from
    an earlier compaction), the new summariser call receives BOTH
    the old summary AND the new turns as input — so the resulting
    summary preserves earlier facts."""
    head = [_turn("setup", "ok") for _ in range(2)]
    # Prior summary lives in middle:
    prior_summary = _turn(
        cc.SUMMARY_TURN_USER,
        "OLD SUMMARY BODY — user wanted X",
        is_summary=True,
    )
    middle = [prior_summary] + [_turn(f"m-{i}", "ack") for i in range(5)]
    tail = [_turn(f"t-{i}", "tail") for i in range(6)]
    stub_conversation["turns"] = head + middle + tail

    captured: dict = {}
    fake_router = MagicMock()

    def _call(task_type, system, user, **kw):
        captured["user_prompt"] = user
        return "## Updated summary"
    fake_router.call.side_effect = _call

    with patch("backend.llm.router", return_value=fake_router):
        cc.maybe_compact(speaker_id="test:speaker", force=True)

    assert "OLD SUMMARY BODY" in captured.get("user_prompt", "")
    assert "PRIOR COMPACTION SUMMARY" in captured.get("user_prompt", "")


# --- error paths ------------------------------------------------------


def test_llm_error_marks_cooldown(stub_conversation, monkeypatch):
    """A failed LLM call should set the failure-cooldown so the
    next call returns early without re-firing."""
    monkeypatch.setattr(cc, "_last_failure_ts", 0.0)
    head = [_turn("setup", "ok") for _ in range(2)]
    middle = [_turn(f"m-{i}", "ack") for i in range(5)]
    tail = [_turn(f"t-{i}", "tail") for i in range(6)]
    stub_conversation["turns"] = head + middle + tail

    from backend.llm import LLMError
    fake_router = MagicMock()
    fake_router.call.side_effect = LLMError("provider down")
    with patch("backend.llm.router", return_value=fake_router):
        stats1 = cc.maybe_compact(speaker_id="test:speaker", force=True)
    assert stats1.fired is False
    assert "LLM error" in stats1.reason
    # Failure should have been recorded.
    assert cc._last_failure_ts > 0.0
    # Next call WITHOUT force should respect the cooldown.
    stats2 = cc.maybe_compact(speaker_id="test:speaker")
    assert stats2.fired is False
    assert "cooldown" in stats2.reason.lower()


def test_empty_summary_marked_as_failure(stub_conversation, monkeypatch):
    monkeypatch.setattr(cc, "_last_failure_ts", 0.0)
    head = [_turn("setup", "ok") for _ in range(2)]
    middle = [_turn(f"m-{i}", "ack") for i in range(5)]
    tail = [_turn(f"t-{i}", "tail") for i in range(6)]
    stub_conversation["turns"] = head + middle + tail

    fake_router = MagicMock()
    fake_router.call.return_value = "   "  # whitespace only
    with patch("backend.llm.router", return_value=fake_router):
        stats = cc.maybe_compact(speaker_id="test:speaker", force=True)
    assert stats.fired is False
    assert "empty summary" in stats.reason


# --- in-flight guard --------------------------------------------------


def test_in_flight_guard_short_circuits_concurrent_calls(stub_conversation, monkeypatch):
    """If another thread already entered `_begin_compaction` for
    this speaker, the second call returns early."""
    monkeypatch.setattr(cc, "_last_failure_ts", 0.0)
    head = [_turn("setup", "ok") for _ in range(2)]
    middle = [_turn(f"m-{i}", "ack") for i in range(5)]
    tail = [_turn(f"t-{i}", "tail") for i in range(6)]
    stub_conversation["turns"] = head + middle + tail

    # Manually mark in-flight.
    cc._in_flight["test:speaker"] = True
    try:
        stats = cc.maybe_compact(speaker_id="test:speaker", force=True)
        assert stats.fired is False
        assert "in flight" in stats.reason
    finally:
        cc._in_flight.pop("test:speaker", None)
