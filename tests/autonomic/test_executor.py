import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.lever import Lever
from backend.autonomic.safety import SafetyGate
from backend.autonomic.types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)


def _snapshot() -> StateSnapshot:
    return StateSnapshot(
        taken_at=datetime.now(timezone.utc),
        uptime_seconds=0.0,
        disk_free_gb=100.0,
        memory_free_gb=8.0,
        cpu_load_1m=0.0,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )


class _GreenStub(Lever):
    name = "GREEN_STUB"
    category = LeverCategory.META
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        now = utcnow()
        return LeverReport(
            lever=self.name,
            params=params,
            started_at=now,
            finished_at=now,
            status=LeverStatus.SUCCESS,
            outcome={"ok": True},
            reason="stub",
        )


class _YellowStub(_GreenStub):
    name = "YELLOW_STUB"
    safety = LeverSafety.YELLOW


class _RaisingStub(_GreenStub):
    name = "RAISING_STUB"

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        raise RuntimeError("simulated")


class _PreconditionsFalseStub(_GreenStub):
    name = "PRECONDITIONS_FALSE_STUB"

    def preconditions(self, state: StateSnapshot) -> bool:
        return False


def test_green_lever_executes_and_writes_to_log(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_GreenStub(), {"foo": "bar"}, _snapshot())
    assert report is not None
    assert report.status == LeverStatus.SUCCESS
    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = LeverReport.from_jsonl(lines[0])
    assert saved.lever == "GREEN_STUB"
    assert saved.params == {"foo": "bar"}


def test_yellow_lever_is_queued_not_executed(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_YellowStub(), {}, _snapshot())
    assert report is None
    assert not lever_log.exists() or lever_log.read_text(encoding="utf-8") == ""
    pending_lines = pending.read_text(encoding="utf-8").splitlines()
    assert len(pending_lines) == 1
    entry = json.loads(pending_lines[0])
    assert entry["lever"] == "YELLOW_STUB"


def test_preconditions_false_short_circuits(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_PreconditionsFalseStub(), {}, _snapshot())
    assert report is not None
    assert report.status == LeverStatus.SKIPPED
    assert "preconditions" in report.reason.lower()
    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_raising_lever_logs_failure_report(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_RaisingStub(), {}, _snapshot())
    assert report is not None
    assert report.status == LeverStatus.FAILURE
    assert "simulated" in report.reason
    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_event_bus_receives_lever_executed(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("lever.executed", received.append)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    execu.execute(_GreenStub(), {}, _snapshot())
    assert len(received) == 1
    assert received[0]["lever"] == "GREEN_STUB"
    assert received[0]["status"] == "success"


def test_bypass_safety_runs_yellow_lever_directly(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)

    report = execu.execute(_YellowStub(), {"z": 9}, _snapshot(), bypass_safety=True)

    assert report is not None
    assert report.status == LeverStatus.SUCCESS
    # Pending file untouched — gate bypassed
    assert pending.read_text(encoding="utf-8") == ""
    # Lever log has the execution
    assert lever_log.read_text(encoding="utf-8").count("\n") == 1


def test_yellow_default_still_queues_when_not_bypassed(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)

    report = execu.execute(_YellowStub(), {"z": 9}, _snapshot())

    assert report is None  # queued, not executed
    assert "id" in pending.read_text(encoding="utf-8")
