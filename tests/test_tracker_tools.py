"""create_tracker proposes steps from recalled experience when steps are
omitted; tools round-trip through TRACKERS."""
from __future__ import annotations

import json
import pytest


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import knowledge_manager, config, scheduled_messages, tracker, builtin_tools
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    importlib.reload(scheduled_messages)
    importlib.reload(tracker)
    importlib.reload(builtin_tools)
    # owner gate: make _check_owner pass (refuse=False, speaker)
    monkeypatch.setattr(builtin_tools, "_check_owner", lambda *a, **k: (False, "webui:default"))
    return builtin_tools


def test_create_tracker_explicit_steps(tools):
    out = json.loads(tools._create_tracker_handler(
        title="Tooling", domain="work",
        steps=[{"title": "Approve drawings"}, {"title": "Pay"}]))
    assert out["ok"] is True
    assert len(out["tracker"]["steps"]) == 2


def test_create_tracker_idempotent_on_active_title(tools):
    """A second create_tracker with the same active title returns the existing
    tracker instead of a duplicate. The model sometimes re-issues the call in
    one turn (2026-06-19 audit: 2 calls -> 2 same-title trackers)."""
    from backend.tracker import TRACKERS
    a = json.loads(tools._create_tracker_handler(
        title="Launch landing", steps=[{"title": "draft"}]))
    b = json.loads(tools._create_tracker_handler(
        title="  launch  LANDING ", steps=[{"title": "draft"}]))  # same modulo case/space
    assert a["ok"] is True and b["ok"] is True
    assert a["tracker"]["id"] == b["tracker"]["id"]      # returned the existing one
    assert "already exists" in (b.get("note") or "")
    assert len(TRACKERS.list(status="active")) == 1       # no duplicate created


def test_create_tracker_recalls_steps_when_omitted(tools, monkeypatch):
    monkeypatch.setattr(
        "backend.trajectory_memory.recall_similar",
        lambda task, limit=2: [{"steps": ["design", "approve", "ship"]}],
    )
    out = json.loads(tools._create_tracker_handler(title="Tooling from China",
                                                   domain="work", steps=None))
    titles = [s["title"] for s in out["tracker"]["steps"]]
    assert "design" in titles and "ship" in titles


def test_list_and_update(tools):
    json.loads(tools._create_tracker_handler(title="T", domain="work",
                                             steps=[{"title": "A"}]))
    listed = json.loads(tools._list_trackers_handler())
    assert listed["count"] == 1
    tid = listed["trackers"][0]["id"]
    sid = listed["trackers"][0]["steps"][0]["id"]
    upd = json.loads(tools._update_step_handler(tracker_id=tid, step_id=sid, status="done"))
    assert upd["ok"] is True
