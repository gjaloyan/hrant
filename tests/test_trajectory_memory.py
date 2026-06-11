"""Trajectory memory — case-based reasoning over past turns (AGI #2).

The workspace persists a full artifact per turn; trajectory memory
indexes the qualifying ones (successful, multi-tool, non-chat) by
task-text embedding and retrieves "how did I solve similar tasks"
as a PAST EXPERIENCE prompt block on future turns.
"""
from __future__ import annotations

import json
import time

import pytest


# Deterministic fake embeddings: map known phrases to fixed vectors so
# cosine similarity is controllable without a real embedder.
_VECS = {
    "run the terminal benchmark suite on two tasks": [1.0, 0.0, 0.0],
    "please run terminal-bench smoke with 2 tasks": [0.97, 0.24, 0.0],
    "convert the family video to a vertical reel": [0.0, 1.0, 0.0],
    "completely unrelated question about cooking pasta": [0.0, 0.0, 1.0],
}


class _FakeEmbedder:
    def status(self):
        return {"backend": "fake", "dim": 3, "model": "fake-3d"}

    def embed(self, text):
        return _VECS.get(text)


class _DisabledEmbedder:
    def status(self):
        return {"backend": "disabled", "last_error": "off"}

    def embed(self, text):  # pragma: no cover — must not be reached
        raise AssertionError("embed called while disabled")


def _artifact(task, *, is_chat=False, confidence=85, tools=None,
              answer="Launched the job; 2/2 tasks passed."):
    trace = []
    for name in (tools if tools is not None
                 else ["read_file", "terminal_exec", "start_background_job"]):
        trace.append({"event": "tool", "tool_call": {"name": name, "args": {}}})
    return {
        "ts": "2026-06-09 12:00:00",
        "user": task,
        "answer": answer,
        "is_chat": is_chat,
        "confidence": confidence,
        "n_tool_calls": len(trace),
        "thinking_trace": trace,
    }


@pytest.fixture
def traj(tmp_path, monkeypatch):
    """Isolated trajectory module: tmp store, tmp turns dir, fake
    embedder."""
    from backend import trajectory_memory as tm

    monkeypatch.setattr(tm, "EMBEDDER", _FakeEmbedder())
    monkeypatch.setattr(tm, "_store_path", lambda: tmp_path / "traj.json")
    turns = tmp_path / "turns"
    turns.mkdir()
    monkeypatch.setattr(tm, "_turns_dir", lambda: turns)
    tm._new_store_for_test()
    return tm, turns


# ─── Qualification gates ──────────────────────────────────────────


def test_chat_turn_does_not_qualify(traj):
    tm, _ = traj
    ok, reason = tm.qualifies(_artifact("x" * 30, is_chat=True))
    assert ok is False and reason == "chat-turn"


def test_low_confidence_does_not_qualify(traj):
    tm, _ = traj
    ok, reason = tm.qualifies(_artifact("x" * 30, confidence=50))
    assert ok is False and "low-confidence" in reason


def test_single_tool_does_not_qualify(traj):
    tm, _ = traj
    ok, reason = tm.qualifies(_artifact("x" * 30, tools=["read_file"]))
    assert ok is False and "too-few-tools" in reason


def test_good_turn_qualifies(traj):
    tm, _ = traj
    ok, reason = tm.qualifies(_artifact("x" * 30))
    assert ok is True and reason == "ok"


def test_tool_sequence_collapses_consecutive_duplicates(traj):
    tm, _ = traj
    art = _artifact(
        "x" * 30,
        tools=["read_file", "read_file", "grep", "read_file", "terminal_exec"],
    )
    assert tm.tool_sequence(art) == [
        "read_file", "grep", "read_file", "terminal_exec",
    ]


# ─── Index + recall roundtrip ─────────────────────────────────────


def test_index_and_recall_similar(traj):
    tm, turns = traj
    task_a = "run the terminal benchmark suite on two tasks"
    art = _artifact(task_a)
    (turns / "t1.json").write_text(json.dumps(art), encoding="utf-8")

    assert tm.index_turn("t1", art) is True

    hits = tm.recall_similar("please run terminal-bench smoke with 2 tasks")
    assert len(hits) == 1
    assert hits[0]["turn_id"] == "t1"
    assert hits[0]["tool_seq"] == [
        "read_file", "terminal_exec", "start_background_job",
    ]
    assert hits[0]["score"] >= 0.9


def test_unrelated_query_below_floor_returns_nothing(traj):
    tm, turns = traj
    task_a = "run the terminal benchmark suite on two tasks"
    art = _artifact(task_a)
    (turns / "t1.json").write_text(json.dumps(art), encoding="utf-8")
    tm.index_turn("t1", art)

    hits = tm.recall_similar("completely unrelated question about cooking pasta")
    assert hits == []


def test_block_renders_tools_and_outcome(traj):
    tm, turns = traj
    task_a = "run the terminal benchmark suite on two tasks"
    art = _artifact(task_a)
    (turns / "t1.json").write_text(json.dumps(art), encoding="utf-8")
    tm.index_turn("t1", art)

    block = tm.past_experience_block(
        "please run terminal-bench smoke with 2 tasks",
    )
    assert "# PAST EXPERIENCE" in block
    assert "read_file -> terminal_exec -> start_background_job" in block
    assert "2/2 tasks passed" in block
    assert "2026-06-09" in block


def test_missing_artifact_self_heals_index(traj):
    """A hit whose artifact was swept by retention is dropped from
    the index and excluded from results."""
    tm, turns = traj
    task_a = "run the terminal benchmark suite on two tasks"
    art = _artifact(task_a)
    (turns / "t1.json").write_text(json.dumps(art), encoding="utf-8")
    tm.index_turn("t1", art)
    (turns / "t1.json").unlink()  # retention sweep

    hits = tm.recall_similar("please run terminal-bench smoke with 2 tasks")
    assert hits == []
    assert tm.get_store().has("t1") is False, "stale index entry must be removed"


def test_duplicate_index_is_noop(traj):
    tm, turns = traj
    task_a = "run the terminal benchmark suite on two tasks"
    art = _artifact(task_a)
    (turns / "t1.json").write_text(json.dumps(art), encoding="utf-8")
    assert tm.index_turn("t1", art) is True
    assert tm.index_turn("t1", art) is False  # already indexed
    assert tm.get_store().count() == 1


# ─── Backfill ─────────────────────────────────────────────────────


def test_backfill_indexes_qualifying_skips_rest(traj):
    tm, turns = traj
    good = _artifact("run the terminal benchmark suite on two tasks")
    chat = _artifact("convert the family video to a vertical reel",
                     is_chat=True)
    (turns / "good.json").write_text(json.dumps(good), encoding="utf-8")
    (turns / "chat.json").write_text(json.dumps(chat), encoding="utf-8")
    (turns / "broken.json").write_text("{not json", encoding="utf-8")

    stats = tm.backfill()
    assert stats["ok"] is True
    assert stats["indexed"] == 1
    assert tm.get_store().has("good")
    assert not tm.get_store().has("chat")

    # Second run: everything already handled.
    stats2 = tm.backfill()
    assert stats2["indexed"] == 0


# ─── Degradation ──────────────────────────────────────────────────


def test_disabled_embedder_degrades_everywhere(traj, monkeypatch):
    tm, turns = traj
    monkeypatch.setattr(tm, "EMBEDDER", _DisabledEmbedder())

    art = _artifact("run the terminal benchmark suite on two tasks")
    assert tm.index_turn("t1", art) is False
    assert tm.recall_similar("please run terminal-bench smoke with 2 tasks") == []
    assert tm.past_experience_block("please run terminal-bench smoke with 2 tasks") == ""
    stats = tm.backfill()
    assert stats["ok"] is False


def test_embedder_model_change_wipes_incompatible_store(traj, monkeypatch):
    """Vectors from different embedder models don't share a space —
    a model change must wipe and restamp, not mix."""
    tm, turns = traj
    task_a = "run the terminal benchmark suite on two tasks"
    art = _artifact(task_a)
    (turns / "t1.json").write_text(json.dumps(art), encoding="utf-8")
    tm.index_turn("t1", art)
    assert tm.get_store().count() == 1

    class _NewModel(_FakeEmbedder):
        def status(self):
            return {"backend": "fake", "dim": 3, "model": "fake-3d-v2"}

    monkeypatch.setattr(tm, "EMBEDDER", _NewModel())
    art2 = _artifact("convert the family video to a vertical reel")
    (turns / "t2.json").write_text(json.dumps(art2), encoding="utf-8")
    tm.index_turn("t2", art2)

    store = tm.get_store()
    assert store.has("t2")
    assert not store.has("t1"), "old-model vectors must be wiped"
