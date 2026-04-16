from datetime import datetime, timezone

import pytest

from backend.autonomic.lever import Lever
from backend.autonomic.types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)


class ToyLever(Lever):
    name = "TOY"
    category = LeverCategory.META
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.01)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict, context: dict) -> LeverReport:
        started = utcnow()
        return LeverReport(
            lever=self.name,
            params=params,
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"hit": True},
            reason="toy",
        )


def _snap() -> StateSnapshot:
    return StateSnapshot(
        taken_at=utcnow(),
        uptime_seconds=0.0,
        disk_free_gb=10.0,
        memory_free_gb=1.0,
        cpu_load_1m=0.0,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )


def test_toy_lever_has_required_attributes():
    lever = ToyLever()
    assert lever.name == "TOY"
    assert lever.category == LeverCategory.META
    assert lever.safety == LeverSafety.GREEN


def test_toy_lever_runs_and_returns_report():
    lever = ToyLever()
    report = lever.run({"x": 1}, {})
    assert report.lever == "TOY"
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome == {"hit": True}


def test_toy_lever_preconditions_true():
    lever = ToyLever()
    assert lever.preconditions(_snap()) is True


def test_lever_rollback_default_noop():
    lever = ToyLever()
    report = lever.run({}, {})
    lever.rollback(report)


def test_incomplete_lever_cannot_instantiate():
    class BrokenLever(Lever):
        pass

    with pytest.raises(TypeError):
        BrokenLever()
