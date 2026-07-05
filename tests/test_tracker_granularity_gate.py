"""Granularity gate: a fresh many-component frame must not collapse into a
handful of tracker mega-steps (one step per component is the contract)."""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, knowledge_manager, builtin_tools
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    importlib.reload(builtin_tools)
    monkeypatch.setattr(builtin_tools, "_check_owner",
                        lambda *a, **k: (False, "webui:default"))
    return builtin_tools


def _write_frame(bt, n_components):
    json.loads(bt._frame_problem_handler(
        title="Big system",
        components=[{"name": f"c{i}", "mvp": True} for i in range(n_components)],
    ))


def test_megasteps_blocked_after_big_frame(tools):
    _write_frame(tools, 16)
    out = json.loads(tools._create_tracker_handler(
        title="Big system",
        steps=[{"title": "backend"}, {"title": "frontend"},
               {"title": "launch"}, {"title": "verify"}]))
    assert out["ok"] is False
    assert "mega-step" in out["error"]


def test_granular_steps_pass(tools):
    _write_frame(tools, 16)
    out = json.loads(tools._create_tracker_handler(
        title="Big system",
        steps=[{"title": f"step {i}"} for i in range(12)]))
    assert out["ok"] is True


def test_small_frame_not_gated(tools):
    _write_frame(tools, 5)
    out = json.loads(tools._create_tracker_handler(
        title="Small thing", steps=[{"title": "one"}]))
    assert out["ok"] is True


def test_no_frame_not_gated(tools):
    out = json.loads(tools._create_tracker_handler(
        title="No frame", steps=[{"title": "a"}, {"title": "b"}]))
    assert out["ok"] is True
