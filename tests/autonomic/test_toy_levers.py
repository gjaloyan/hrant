from backend.autonomic.levers.noop_green_tick import NoopGreenTick
from backend.autonomic.levers.noop_yellow_demand import NoopYellowDemand
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, utcnow, StateSnapshot


def _snap() -> StateSnapshot:
    return StateSnapshot(
        taken_at=utcnow(), uptime_seconds=0, disk_free_gb=10, memory_free_gb=1,
        cpu_load_1m=0, last_run={}, recent_errors=[], pending_approvals=0,
        kb_notes_count=0, kb_graph_nodes=0,
    )


def test_noop_green_tick_metadata():
    lever = NoopGreenTick()
    assert lever.name == "NOOP_GREEN_TICK"
    assert lever.category == LeverCategory.META
    assert lever.safety == LeverSafety.GREEN


def test_noop_green_tick_runs():
    lever = NoopGreenTick()
    assert lever.preconditions(_snap()) is True
    report = lever.run({}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome == {"ticked": True}


def test_noop_yellow_demand_metadata():
    lever = NoopYellowDemand()
    assert lever.name == "NOOP_YELLOW_DEMAND"
    assert lever.safety == LeverSafety.YELLOW


def test_noop_yellow_demand_runs():
    lever = NoopYellowDemand()
    report = lever.run({"reason": "test"}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome == {"demanded": True, "reason": "test"}
