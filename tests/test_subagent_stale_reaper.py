"""check_subagents reaps ghost sessions: a builder whose thread died with its
parent process stays 'running' forever (battery finding 2026-07-06 — Phase-2
builder stuck >1h). Reaped to 'stale' on sight so the agent re-dispatches
instead of waiting on a ghost.
"""
from __future__ import annotations

import importlib
import json
import time

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config
    importlib.reload(config)
    import backend.subagents.store as store_mod
    importlib.reload(store_mod)
    import backend.builtin_tools as bt
    importlib.reload(bt)
    return bt, store_mod


def test_old_running_session_reaped_to_stale(env):
    bt, store_mod = env
    s = store_mod.SUBAGENT_STORE.create(role="builder", task="ghost build",
                                        parent_speaker="webui:default")
    s.status = "running"
    s.created_at = time.time() - 2 * 3600          # 2h ago
    store_mod.SUBAGENT_STORE._write(s)

    out = json.loads(bt._check_subagents_handler(s.id))
    assert out["sessions"][0]["status"] == "stale"
    # persisted, not just displayed
    assert store_mod.SUBAGENT_STORE.get(s.id).status == "stale"


def test_fresh_running_session_untouched(env):
    bt, store_mod = env
    s = store_mod.SUBAGENT_STORE.create(role="builder", task="live build",
                                        parent_speaker="webui:default")
    s.status = "running"
    store_mod.SUBAGENT_STORE._write(s)

    out = json.loads(bt._check_subagents_handler(s.id))
    assert out["sessions"][0]["status"] == "running"
    assert store_mod.SUBAGENT_STORE.get(s.id).status == "running"


def test_completed_session_untouched(env):
    bt, store_mod = env
    s = store_mod.SUBAGENT_STORE.create(role="builder", task="done build",
                                        parent_speaker="webui:default")
    s.status = "completed"
    s.created_at = time.time() - 2 * 3600
    store_mod.SUBAGENT_STORE._write(s)

    out = json.loads(bt._check_subagents_handler(s.id))
    assert out["sessions"][0]["status"] == "completed"
