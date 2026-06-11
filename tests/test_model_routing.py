"""Per-task model routing — cheap tasks on cheap models.

The Router consults model_routing.json per call: routed task types
go DIRECT to the configured (provider, model); anything unrouted —
and any routed call that fails — uses the active pin unchanged.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def routing(tmp_path, monkeypatch):
    from backend import model_routing as mr
    monkeypatch.setattr(mr, "_config_path", lambda: tmp_path / "mr.json")
    mr.load_config(force=True)  # prime cache from empty
    yield mr
    mr.load_config(force=True)


# ─── Config layer ─────────────────────────────────────────────────


def test_default_disabled_empty(routing):
    cfg = routing.load_config(force=True)
    assert cfg == {"enabled": False, "routing": {}}
    assert routing.route_for("classification") is None


def test_save_and_route(routing):
    routing.save_config({
        "enabled": True,
        "routing": {
            "Classification": {
                "provider_id": "openrouter-x",
                "model": "qwen/qwen3.6-35b-a3b",
            },
            "garbage": "not-a-dict",
            "empty": {"provider_id": "", "model": "m"},
        },
    })
    # Task-type keys normalize to lowercase; junk entries dropped.
    assert routing.route_for("classification") == (
        "openrouter-x", "qwen/qwen3.6-35b-a3b",
    )
    assert routing.route_for("CLASSIFICATION") == (
        "openrouter-x", "qwen/qwen3.6-35b-a3b",
    )
    assert routing.route_for("garbage") is None
    assert routing.route_for("empty") is None
    assert routing.route_for("complex_solving") is None


def test_disabled_table_routes_nothing(routing):
    routing.save_config({
        "enabled": False,
        "routing": {
            "classification": {"provider_id": "p", "model": "m"},
        },
    })
    assert routing.route_for("classification") is None


def test_unreadable_file_degrades_to_disabled(routing, tmp_path):
    (tmp_path / "mr.json").write_text("{broken", encoding="utf-8")
    cfg = routing.load_config(force=True)
    assert cfg["enabled"] is False


# ─── Router integration ───────────────────────────────────────────


class _FakeLLM:
    def __init__(self, reply="routed-reply", fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = 0

    def complete(self, system, user, **kw):
        self.calls += 1
        if self.fail:
            from backend.llm import LLMError
            raise LLMError("routed model down")
        return self.reply


@pytest.fixture
def router_with_route(routing, monkeypatch):
    """A Router whose CLASSIFICATION is routed to a fake LLM and whose
    active pin is another fake."""
    from backend.llm import DualModelRouter as Router
    import backend.llm as llm_mod

    routing.save_config({
        "enabled": True,
        "routing": {"classification": {
            "provider_id": "openrouter-x", "model": "qwen/test",
        }},
    })

    routed_llm = _FakeLLM("routed-reply")
    active_llm = _FakeLLM("active-reply")

    r = Router.__new__(Router)  # skip heavy __init__
    r._routed_llms = {}
    r._active_llm = active_llm
    r._active_cfg_hash = "active:fake"
    r.state = {"last_reason": ""}
    monkeypatch.setattr(r, "_save_state", lambda: None, raising=False)
    monkeypatch.setattr(
        r, "_track_active_model_call", lambda **kw: None, raising=False,
    )
    monkeypatch.setattr(r, "_get_active_llm", lambda: active_llm, raising=False)
    monkeypatch.setattr(
        r, "_call_with_failover_chain",
        lambda primary_fn, fallback_factory: (primary_fn(), "", "", False),
        raising=False,
    )
    monkeypatch.setattr(llm_mod, "_active_provider_chain", lambda tt: [])

    import backend.failover as _fo
    monkeypatch.setattr(
        _fo, "resolve_entry_cfg",
        lambda pid, model: {"provider_id": pid, "model": model},
    )
    monkeypatch.setattr(llm_mod, "create_llm", lambda cfg: routed_llm)

    return r, routed_llm, active_llm


def test_routed_task_uses_routed_model(router_with_route):
    from backend.llm import TaskType
    r, routed_llm, active_llm = router_with_route

    out = r.call(TaskType.CLASSIFICATION, "sys", "user")
    assert out == "routed-reply"
    assert routed_llm.calls == 1
    assert active_llm.calls == 0
    assert "task-routed" in r.state["last_reason"]


def test_unrouted_task_uses_active(router_with_route):
    from backend.llm import TaskType
    r, routed_llm, active_llm = router_with_route

    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "user")
    assert out == "active-reply"
    assert routed_llm.calls == 0
    assert active_llm.calls == 1


def test_routed_failure_falls_back_to_active(router_with_route):
    from backend.llm import TaskType
    r, routed_llm, active_llm = router_with_route
    routed_llm.fail = True

    out = r.call(TaskType.CLASSIFICATION, "sys", "user")
    assert out == "active-reply"
    assert routed_llm.calls == 1   # tried
    assert active_llm.calls == 1   # rescued


def test_unresolvable_route_falls_back(router_with_route, monkeypatch):
    from backend.llm import TaskType
    import backend.failover as _fo
    r, routed_llm, active_llm = router_with_route
    monkeypatch.setattr(_fo, "resolve_entry_cfg", lambda pid, m: None)

    out = r.call(TaskType.CLASSIFICATION, "sys", "user")
    assert out == "active-reply"
    assert routed_llm.calls == 0
