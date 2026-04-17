import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
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
    yield
    clear_registry()


def _builder(tmp_path: Path) -> StateSnapshotBuilder:
    return StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=tmp_path / "lever_log.jsonl",
    )


def test_tick_idle_writes_to_tick_log(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl")
    tick_log = tmp_path / "tick_log.jsonl"
    engine = Layer0Engine(rules=[])
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
    )
    tick()
    lines = tick_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["lever"] is None
    assert entry["reason"] == "idle_no_rules_matched"


def test_tick_fires_lever_and_writes_both_logs(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 10}) + "\n",
        encoding="utf-8",
    )
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl")
    tick_log = tmp_path / "tick_log.jsonl"
    rule = LayerZeroRule(
        name="errors_present",
        predicate=lambda s: len(s.recent_errors) > 0,
        lever="FIRE_ERROR_TRIAGE",
        params={},
    )
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=Layer0Engine(rules=[rule]),
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
    )
    tick()
    tick_lines = tick_log.read_text(encoding="utf-8").splitlines()
    assert len(tick_lines) == 1
    assert json.loads(tick_lines[0])["lever"] == "FIRE_ERROR_TRIAGE"
    lever_lines = (tmp_path / "lever_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lever_lines) == 1
    report = LeverReport.from_jsonl(lever_lines[0])
    assert report.lever == "FIRE_ERROR_TRIAGE"


def test_tick_unknown_lever_in_rule_is_logged_but_not_fatal(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl")
    tick_log = tmp_path / "tick_log.jsonl"
    rule = LayerZeroRule(
        name="nonsense",
        predicate=lambda s: True,
        lever="DOES_NOT_EXIST",
        params={},
    )
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=Layer0Engine(rules=[rule]),
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
    )
    tick()
    entry = json.loads(tick_log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["lever"] == "DOES_NOT_EXIST"
    assert entry.get("executed") is False
    assert "unknown_lever" in entry.get("note", "")


def test_tick_event_bus_receives_tick_completed(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("tick.completed", received.append)
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl", event_bus=bus)
    tick_log = tmp_path / "tick_log.jsonl"
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=Layer0Engine(rules=[]),
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )
    tick()
    assert len(received) == 1
    assert received[0]["source"] == "L0_reflex"
