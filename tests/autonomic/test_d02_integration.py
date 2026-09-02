import asyncio
import json
from pathlib import Path

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.kill_switch import KillSwitch
from backend.autonomic.layer0 import Layer0Engine, default_rules
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_default_immune_levers,
)
from backend.autonomic.safety import SafetyGate
from backend.autonomic.scheduler import AutonomicScheduler
from backend.autonomic.state import StateSnapshotBuilder
from backend.autonomic.tick import make_real_tick
from backend.autonomic.types import LeverReport


@pytest.fixture(autouse=True)
def _reg():
    clear_registry()
    register_default_immune_levers()
    yield
    clear_registry()


@pytest.mark.asyncio
async def test_end_to_end_error_triggers_triage_lever(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": "boom 1", "confidence": 10}) + "\n"
        + json.dumps({"ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": "boom 2", "confidence": 50}) + "\n",
        encoding="utf-8",
    )
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    ks = KillSwitch(ks_path)

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
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )

    received_levers: list[dict] = []
    bus.subscribe("lever.executed", received_levers.append)

    sched = AutonomicScheduler(ks, on_tick=tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()

    lever_lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lever_lines) >= 1
    report = LeverReport.from_jsonl(lever_lines[0])
    assert report.lever == "FIRE_ERROR_TRIAGE"
    assert report.outcome["total"] == 2
    assert any(e.get("lever") == "FIRE_ERROR_TRIAGE" for e in received_levers)


@pytest.mark.asyncio
async def test_kill_switch_disabled_means_no_ticks_execute(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("false")
    ks = KillSwitch(ks_path)

    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    lever_log = tmp_path / "lever_log.jsonl"
    tick_log = tmp_path / "tick_log.jsonl"
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=lever_log,
    )
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
    )

    sched = AutonomicScheduler(ks, on_tick=tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()

    assert not lever_log.exists() or lever_log.read_text(encoding="utf-8") == ""
    assert not tick_log.exists() or tick_log.read_text(encoding="utf-8") == ""
