from datetime import datetime, timezone

from backend.autonomic.levers.server_health import FIRE_SERVER_HEALTH
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, StateSnapshot


def _snapshot(**overrides) -> StateSnapshot:
    base = dict(
        taken_at=datetime.now(timezone.utc),
        uptime_seconds=10.0,
        disk_free_gb=100.0,
        memory_free_gb=8.0,
        cpu_load_1m=0.5,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )
    base.update(overrides)
    return StateSnapshot(**base)


def test_server_health_metadata():
    lever = FIRE_SERVER_HEALTH()
    assert lever.name == "FIRE_SERVER_HEALTH"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_server_health_healthy_system_has_no_issues():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(disk_free_gb=100.0, memory_free_gb=8.0, cpu_load_1m=0.5)
    report = lever.run({}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["issues"] == []
    assert report.outcome["disk_free_gb"] == 100.0


def test_server_health_flags_low_disk():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(disk_free_gb=0.5)
    report = lever.run({"disk_min_gb": 1.0}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    issues = report.outcome["issues"]
    assert any("disk" in i for i in issues)


def test_server_health_flags_low_memory():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(memory_free_gb=0.2)
    report = lever.run({"memory_min_gb": 0.5}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    issues = report.outcome["issues"]
    assert any("memory" in i for i in issues)


def test_server_health_flags_high_cpu():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(cpu_load_1m=10.0)
    report = lever.run({"cpu_max_load": 4.0}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    issues = report.outcome["issues"]
    assert any("cpu" in i for i in issues)


def test_server_health_no_state_falls_back_to_live_reading():
    lever = FIRE_SERVER_HEALTH()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SUCCESS
    assert "disk_free_gb" in report.outcome
    assert "memory_free_gb" in report.outcome


def test_server_health_preconditions_always_true():
    lever = FIRE_SERVER_HEALTH()
    assert lever.preconditions(_snapshot()) is True
