import asyncio
import json
from pathlib import Path

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.kill_switch import KillSwitch
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_lever,
)
from backend.autonomic.levers.noop_green_tick import NoopGreenTick
from backend.autonomic.levers.noop_yellow_demand import NoopYellowDemand
from backend.autonomic.safety import SafetyDecision, SafetyGate
from backend.autonomic.scheduler import AutonomicScheduler
from backend.autonomic.types import LeverReport, LeverStatus


@pytest.fixture(autouse=True)
def reset_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.mark.asyncio
async def test_green_lever_runs_and_appends_report(tmp_path: Path):
    register_lever(NoopGreenTick)
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)

    def run_one_tick():
        lever = LeverRegistry.instance().get("NOOP_GREEN_TICK")
        assert lever is not None
        if gate.evaluate(lever, {}) == SafetyDecision.ALLOW:
            report = lever.run({}, {})
            with lever_log.open("a", encoding="utf-8") as f:
                f.write(report.to_jsonl())

    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    sched = AutonomicScheduler(KillSwitch(ks_path), on_tick=run_one_tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.15)
    await sched.stop()

    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    report = LeverReport.from_jsonl(lines[0])
    assert report.lever == "NOOP_GREEN_TICK"
    assert report.status == LeverStatus.SUCCESS
    assert pending.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_yellow_lever_queues_to_pending(tmp_path: Path):
    register_lever(NoopYellowDemand)
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    lever_log = tmp_path / "lever_log.jsonl"
    lever_log.touch()

    def run_one_tick():
        lever = LeverRegistry.instance().get("NOOP_YELLOW_DEMAND")
        assert lever is not None
        decision = gate.evaluate(lever, {"reason": "test"})
        assert decision == SafetyDecision.QUEUE_FOR_APPROVAL

    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    sched = AutonomicScheduler(KillSwitch(ks_path), on_tick=run_one_tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.15)
    await sched.stop()

    assert lever_log.read_text(encoding="utf-8") == ""
    pending_lines = pending.read_text(encoding="utf-8").splitlines()
    assert len(pending_lines) >= 2
    entry = json.loads(pending_lines[0])
    assert entry["lever"] == "NOOP_YELLOW_DEMAND"
    assert entry["params"] == {"reason": "test"}


def test_event_bus_coordinates_tick_and_subscriber():
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("lever.executed", received.append)
    bus.publish("lever.executed", {"lever": "X", "status": "success"})
    assert received == [{"lever": "X", "status": "success"}]
