import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule, default_rules
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
from backend.autonomic.safety import SafetyGate
from backend.autonomic.state import StateSnapshotBuilder
from backend.autonomic.tick import make_real_tick
from backend.autonomic.types import LeverReport


@pytest.fixture(autouse=True)
def _reg():
    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
    yield
    clear_registry()


def _build_tick(tmp_path: Path, rules=None):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    lever_log = tmp_path / "lever_log.jsonl"
    tick_log = tmp_path / "tick_log.jsonl"
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=lever_log,
    )
    engine = Layer0Engine(rules=rules if rules is not None else default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )
    return tick, lever_log


def test_three_d06_ticks_fire_in_order(tmp_path: Path):
    d06_only = [
        LayerZeroRule(name="model_eval_tick", predicate=lambda s: True,
                      lever="FIRE_MODEL_EVAL", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="session_archive_tick", predicate=lambda s: True,
                      lever="FIRE_SESSION_ARCHIVE", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="cost_audit_tick", predicate=lambda s: True,
                      lever="FIRE_COST_AUDIT", params={}, cooldown_seconds=3600.0),
    ]

    model_eval_log = tmp_path / "model_eval_log.jsonl"
    cost_log = tmp_path / "cost_audit_log.jsonl"
    sessions_path = tmp_path / "sessions.json"
    history_dir = tmp_path / "_history"
    router_state = tmp_path / "router_state.json"

    sessions_path.write_text(json.dumps({
        "current_id": "x",
        "sessions": [{
            "id": "old1",
            "started": "2020-01-01 00:00:00",
            "ended": "2024-01-01 00:00:00",
            "title": "old",
            "archived": False,
            "consolidated": True,
            "turns": [{"ts": "x", "user": "hi", "answer": "hi", "intent": "chat"}],
        }],
    }), encoding="utf-8")

    router_state.write_text(json.dumps({
        "date": "2026-04-18", "api_calls_today": 3, "api_cost_today": 0.01,
        "model_b_calls_today": 0, "total_a_calls": 100, "total_b_calls": 0,
        "last_reason": "ok",
    }), encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path, rules=d06_only)

    from backend.autonomic.levers.model_eval import FIRE_MODEL_EVAL
    from backend.autonomic.levers.session_archive import FIRE_SESSION_ARCHIVE
    from backend.autonomic.levers.cost_audit import FIRE_COST_AUDIT

    orig_me = FIRE_MODEL_EVAL.run
    orig_sa = FIRE_SESSION_ARCHIVE.run
    orig_ca = FIRE_COST_AUDIT.run

    def me_wrap(self, params, context):
        p = dict(params); p.setdefault("log_path", str(model_eval_log))
        return orig_me(self, p, context)

    def sa_wrap(self, params, context):
        p = dict(params)
        p.setdefault("sessions_path", str(sessions_path))
        p.setdefault("history_dir", str(history_dir))
        return orig_sa(self, p, context)

    def ca_wrap(self, params, context):
        p = dict(params)
        p.setdefault("router_state_path", str(router_state))
        p.setdefault("log_path", str(cost_log))
        return orig_ca(self, p, context)

    fake_eval = MagicMock()
    fake_eval.daily_report.return_value = {
        "date": "2026-04-17", "total_interactions": 2, "tasks": 1, "chats": 1,
        "avg_confidence": 85, "total_contradictions": 0, "total_unverified": 0,
        "topics_used": [], "by_intent": {}, "low_confidence_count": 0, "high_confidence_count": 0,
    }
    fake_eval.detect_regression.return_value = []
    fake_eval.suggest_priorities.return_value = []

    with patch.object(FIRE_MODEL_EVAL, "run", me_wrap), \
         patch.object(FIRE_SESSION_ARCHIVE, "run", sa_wrap), \
         patch.object(FIRE_COST_AUDIT, "run", ca_wrap), \
         patch("backend.autonomic.levers.model_eval.EVALUATOR", fake_eval):
        tick()
        tick()
        tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_MODEL_EVAL", "FIRE_SESSION_ARCHIVE", "FIRE_COST_AUDIT"]


def test_reactive_rule_preempts_d06_scheduled(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path)
    tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_d06_logs_receive_one_line_each_per_tick(tmp_path: Path):
    d06_only = [
        LayerZeroRule(name="model_eval_tick", predicate=lambda s: True,
                      lever="FIRE_MODEL_EVAL", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="cost_audit_tick", predicate=lambda s: True,
                      lever="FIRE_COST_AUDIT", params={}, cooldown_seconds=3600.0),
    ]
    model_eval_log = tmp_path / "model_eval_log.jsonl"
    cost_log = tmp_path / "cost_audit_log.jsonl"
    router_state = tmp_path / "router_state.json"
    router_state.write_text(json.dumps({
        "date": "2026-04-18", "api_calls_today": 3, "api_cost_today": 0.01,
        "total_a_calls": 100, "total_b_calls": 0, "last_reason": "ok",
    }), encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path, rules=d06_only)

    from backend.autonomic.levers.model_eval import FIRE_MODEL_EVAL
    from backend.autonomic.levers.cost_audit import FIRE_COST_AUDIT

    orig_me = FIRE_MODEL_EVAL.run
    orig_ca = FIRE_COST_AUDIT.run

    def me_wrap(self, params, context):
        p = dict(params); p.setdefault("log_path", str(model_eval_log))
        return orig_me(self, p, context)

    def ca_wrap(self, params, context):
        p = dict(params)
        p.setdefault("router_state_path", str(router_state))
        p.setdefault("log_path", str(cost_log))
        return orig_ca(self, p, context)

    fake_eval = MagicMock()
    fake_eval.daily_report.return_value = {"total_interactions": 3, "date": "2026-04-17",
                                           "avg_confidence": 80}
    fake_eval.detect_regression.return_value = []
    fake_eval.suggest_priorities.return_value = []

    with patch.object(FIRE_MODEL_EVAL, "run", me_wrap), \
         patch.object(FIRE_COST_AUDIT, "run", ca_wrap), \
         patch("backend.autonomic.levers.model_eval.EVALUATOR", fake_eval):
        tick()
        tick()

    assert model_eval_log.read_text(encoding="utf-8").count("\n") == 1
    assert cost_log.read_text(encoding="utf-8").count("\n") == 1
