from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.autonomic.types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    TickDecisionSource,
)


def test_cost_defaults_to_zero():
    c = Cost()
    assert c.tokens_in == 0
    assert c.tokens_out == 0
    assert c.seconds == 0.0
    assert c.usd == 0.0


def test_cost_addition():
    a = Cost(tokens_in=10, tokens_out=5, seconds=0.5, usd=0.001)
    b = Cost(tokens_in=20, tokens_out=10, seconds=1.0, usd=0.002)
    total = a + b
    assert total.tokens_in == 30
    assert total.tokens_out == 15
    assert total.seconds == 1.5
    assert total.usd == pytest.approx(0.003)


def test_lever_safety_values():
    assert LeverSafety.GREEN.value == "green"
    assert LeverSafety.YELLOW.value == "yellow"
    assert LeverSafety.RED.value == "red"


def test_lever_category_values():
    assert LeverCategory.AUTONOMIC.value == "autonomic"
    assert LeverCategory.TELEMETRY.value == "telemetry"
    assert LeverCategory.IMMUNE.value == "immune"
    assert LeverCategory.BODY.value == "body"
    assert LeverCategory.META.value == "meta"


def test_lever_status_values():
    assert LeverStatus.SUCCESS.value == "success"
    assert LeverStatus.FAILURE.value == "failure"
    assert LeverStatus.SKIPPED.value == "skipped"
    assert LeverStatus.ESCALATED.value == "escalated"
    assert LeverStatus.BLOCKED_BY_SAFETY.value == "blocked_by_safety"


def test_lever_report_to_jsonl_roundtrip():
    started = datetime(2026, 4, 16, 12, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 16, 12, 0, 1, tzinfo=timezone.utc)
    report = LeverReport(
        lever="FIRE_TEST",
        params={"x": 1},
        started_at=started,
        finished_at=finished,
        status=LeverStatus.SUCCESS,
        outcome={"done": True},
        cost=Cost(seconds=1.0),
        reason="unit test",
        follow_ups=["noop"],
    )
    line = report.to_jsonl()
    assert line.endswith("\n")
    restored = LeverReport.from_jsonl(line)
    assert restored.lever == "FIRE_TEST"
    assert restored.params == {"x": 1}
    assert restored.status == LeverStatus.SUCCESS
    assert restored.outcome == {"done": True}
    assert restored.cost.seconds == 1.0
    assert restored.follow_ups == ["noop"]


def test_state_snapshot_minimal():
    snap = StateSnapshot(
        taken_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        uptime_seconds=60.0,
        disk_free_gb=100.0,
        memory_free_gb=10.0,
        cpu_load_1m=0.5,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )
    assert snap.uptime_seconds == 60.0
    assert snap.pending_approvals == 0


def test_tick_decision_source_values():
    assert TickDecisionSource.L0_REFLEX.value == "L0_reflex"
    assert TickDecisionSource.L0_IMMUNE.value == "L0_immune"
    assert TickDecisionSource.L1_ROUTER.value == "L1_router"
    assert TickDecisionSource.L2_DIAGNOSER.value == "L2_diagnoser"
    assert TickDecisionSource.L3_ESCALATION.value == "L3_escalation"
