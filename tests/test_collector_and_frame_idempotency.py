"""Battery findings (2026-07-06), fixed structurally:

A. Background builders often finish AFTER the parent's turn — a collector
   check-in must be scheduled automatically so results get integrated.
B. The agent re-framed the same project every continuation round — a fresh
   same-slug frame is returned as-is, not recreated.
"""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, knowledge_manager
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    import backend.scheduled_messages as sched_mod
    importlib.reload(sched_mod)
    import backend.builtin_tools as bt
    importlib.reload(bt)
    monkeypatch.setattr(bt, "_check_owner",
                        lambda *a, **k: (False, "webui:default"))
    return bt, sched_mod


def test_background_delegate_schedules_collector_checkin(tools, monkeypatch):
    bt, sched_mod = tools

    def fake_run_subagent(role, task, *, depth=0, speaker_id=None,
                          on_session=None, **kw):
        if on_session:
            on_session("sess-xyz")

        class _R:
            ok = True
        return _R()

    monkeypatch.setattr("backend.subagents.run_subagent", fake_run_subagent)
    out = json.loads(bt._delegate_handler("builder", "build the cart module",
                                          background=True))
    assert out["ok"] is True
    rows = sched_mod.list_pending()
    checkins = [r for r in rows if r.get("kind") == "check_in"
                and r.get("meta", {}).get("subagent_session") == "sess-xyz"]
    assert len(checkins) == 1
    assert "check_subagents" in checkins[0]["text"]


def test_reframing_same_title_returns_existing_frame(tools):
    bt, _ = tools
    first = json.loads(bt._frame_problem_handler(
        title="Survey platform",
        components=[{"name": "auth", "mvp": True}]))
    second = json.loads(bt._frame_problem_handler(
        title="Survey platform",
        components=[{"name": "something-else", "mvp": True}]))
    assert second["ok"] is True
    assert second["frame_id"] == first["frame_id"]      # same frame, not new
    assert "already framed" in second["note"].lower()
    # original components preserved (not overwritten by the re-frame)
    assert second["frame"]["components"][0]["name"] == "auth"


def test_different_title_still_creates_new_frame(tools):
    bt, _ = tools
    a = json.loads(bt._frame_problem_handler(
        title="Project A", components=[{"name": "x", "mvp": True}]))
    b = json.loads(bt._frame_problem_handler(
        title="Project B", components=[{"name": "y", "mvp": True}]))
    assert a["frame_id"] != b["frame_id"]
