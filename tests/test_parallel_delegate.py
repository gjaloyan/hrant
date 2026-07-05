"""Parallel delegation: delegate(background=true) returns a session ticket
immediately (subagent runs on a thread); check_subagents collects results.
"""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config
    importlib.reload(config)
    import backend.subagents.store as store_mod
    importlib.reload(store_mod)
    import backend.builtin_tools as bt
    monkeypatch.setattr("backend.subagents.store.SUBAGENT_STORE",
                        store_mod.SUBAGENT_STORE, raising=False)
    return bt, store_mod


def test_background_delegate_returns_ticket(tools, monkeypatch):
    bt, store_mod = tools

    def fake_run_subagent(role, task, *, depth=0, speaker_id=None,
                          on_session=None, **kw):
        if on_session:
            on_session("sess-abc123")

        class _R:
            ok = True
        return _R()

    monkeypatch.setattr("backend.subagents.run_subagent", fake_run_subagent)
    out = json.loads(bt._delegate_handler("builder", "do a thing",
                                          background=True))
    assert out["ok"] is True and out["background"] is True
    assert out["session_id"] == "sess-abc123"
    assert "check_subagents" in out["note"]


def test_foreground_delegate_unchanged(tools, monkeypatch):
    bt, _ = tools

    class _R:
        ok = True; role = "builder"; task = "t"; answer = "done"
        tool_summary = {}; iterations = 1; elapsed_ms = 5; error = ""

    monkeypatch.setattr("backend.subagents.run_subagent",
                        lambda *a, **k: _R())
    out = json.loads(bt._delegate_handler("builder", "t"))
    assert out["ok"] is True and out["answer"] == "done"
    assert "background" not in out


def test_check_subagents_lists_sessions(tools):
    bt, store_mod = tools
    s = store_mod.SUBAGENT_STORE.create(role="builder", task="build cart")
    store_mod.SUBAGENT_STORE.finalize(s.id, status="completed",
                                      answer="cart built + verified")
    out = json.loads(bt._check_subagents_handler())
    assert out["ok"] is True
    ids = [x["session_id"] for x in out["sessions"]]
    assert s.id in ids
    one = json.loads(bt._check_subagents_handler(session_id=s.id))
    assert one["sessions"][0]["status"] == "completed"
    assert "cart built" in one["sessions"][0]["answer"]


def test_check_subagents_registered_and_base():
    import backend.builtin_tools as bt
    bt.register_builtin_tools()
    from backend.tool_registry import REGISTRY
    from backend.tool_bundles import BASE_TOOLS
    assert "check_subagents" in REGISTRY.names()
    assert "check_subagents" in BASE_TOOLS
