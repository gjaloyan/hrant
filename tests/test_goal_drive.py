"""FIRE_GOAL_DRIVE closes the dead goal loop: user/learning goals get
decomposed, then executed one subtask per tick via a builder subagent, and
completed when the last subtask lands.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config
    importlib.reload(config)
    import backend.goals as goals_mod
    importlib.reload(goals_mod)
    import backend.autonomic.levers.goal_drive as gd
    importlib.reload(gd)
    monkeypatch.setattr(gd, "GOALS", goals_mod.GOALS)
    return goals_mod, gd


def _fire(gd):
    lever = gd.FIRE_GOAL_DRIVE()
    return lever.run({}, {})


def test_skips_when_no_driveable_goals(env):
    goals_mod, gd = env
    rep = _fire(gd)
    assert rep.status.value == "skipped"
    assert "no_driveable" in rep.reason


def test_decomposes_bare_goal_first(env, monkeypatch):
    goals_mod, gd = env
    g = goals_mod.GOALS.add("Learn FastAPI auth best practices",
                            goal_type="learning", priority=8)

    class _R:
        @staticmethod
        def call_json(*a, **k):
            return {"subtasks": ["study docs", "write summary note"]}

    monkeypatch.setattr("backend.llm.router", lambda: _R())
    rep = _fire(gd)
    assert "decomposed" in rep.reason
    fresh = goals_mod.GOALS.get(g.id)
    assert len(fresh.subtasks) == 2
    assert all(st["status"] == "pending" for st in fresh.subtasks)


def test_runs_one_subtask_via_builder_and_completes(env, monkeypatch):
    goals_mod, gd = env
    g = goals_mod.GOALS.add("Do X", goal_type="user", priority=9)
    g.add_subtask("only step")
    goals_mod.GOALS._save()

    calls = {}

    class _Res:
        ok = True
        answer = "did the step, verified"

    def fake_run_subagent(role, task, **kw):
        calls["role"], calls["task"] = role, task
        return _Res()

    monkeypatch.setattr("backend.subagents.run_subagent", fake_run_subagent)
    rep = _fire(gd)
    assert calls["role"] == "builder"
    assert "only step" in calls["task"]
    fresh = goals_mod.GOALS.get(g.id)
    assert fresh.subtasks[0]["status"] == "done"
    assert fresh.status == "completed"          # last subtask -> goal completed
    assert "left0" in rep.reason


def test_improvement_goals_left_to_goal_executor(env):
    goals_mod, gd = env
    goals_mod.GOALS.add("improve module X", goal_type="improvement")
    rep = _fire(gd)
    assert rep.status.value == "skipped"
