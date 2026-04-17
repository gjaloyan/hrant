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


from backend.autonomic.levers.error_triage import FIRE_ERROR_TRIAGE


def test_error_triage_metadata():
    lever = FIRE_ERROR_TRIAGE()
    assert lever.name == "FIRE_ERROR_TRIAGE"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN


def test_error_triage_empty_snapshot_returns_zero_counts():
    lever = FIRE_ERROR_TRIAGE()
    state = _snapshot(recent_errors=[])
    report = lever.run({}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total"] == 0
    assert report.outcome["by_severity"] == {}


def test_error_triage_classifies_by_confidence():
    lever = FIRE_ERROR_TRIAGE()
    state = _snapshot(recent_errors=[
        {"confidence": 20, "question": "q1"},
        {"confidence": 45, "question": "q2"},
        {"confidence": 75, "question": "q3"},
        {"confidence": 10, "question": "q4"},
    ])
    report = lever.run({}, {"state": state})
    assert report.outcome["total"] == 4
    by_sev = report.outcome["by_severity"]
    assert by_sev.get("critical", 0) == 2
    assert by_sev.get("warn", 0) == 1
    assert by_sev.get("info", 0) == 1


def test_error_triage_uses_explicit_severity_when_provided():
    lever = FIRE_ERROR_TRIAGE()
    state = _snapshot(recent_errors=[
        {"severity": "critical", "message": "boom"},
        {"severity": "warn", "message": "meh"},
        {"severity": "info", "message": "fine"},
    ])
    report = lever.run({}, {"state": state})
    assert report.outcome["by_severity"] == {"critical": 1, "warn": 1, "info": 1}


def test_error_triage_preconditions_requires_errors():
    lever = FIRE_ERROR_TRIAGE()
    assert lever.preconditions(_snapshot(recent_errors=[])) is False
    assert lever.preconditions(_snapshot(recent_errors=[{"message": "x"}])) is True


from backend.autonomic.levers.service_repair import FIRE_SERVICE_REPAIR


def test_service_repair_metadata():
    lever = FIRE_SERVICE_REPAIR()
    assert lever.name == "FIRE_SERVICE_REPAIR"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN


def test_service_repair_rejects_service_not_in_whitelist():
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "rm_rf_please"}, {})
    assert report.status == LeverStatus.BLOCKED_BY_SAFETY
    assert "whitelist" in report.reason


def test_service_repair_on_unsupported_platform_returns_skipped(monkeypatch):
    import backend.autonomic.levers.service_repair as mod
    monkeypatch.setattr(mod, "_PLATFORM_SUPPORTED", False)
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "ollama"}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "platform" in report.reason.lower()


def test_service_repair_runs_subprocess_on_supported_platform(monkeypatch):
    import backend.autonomic.levers.service_repair as mod
    calls: list[list[str]] = []

    class _Result:
        def __init__(self, rc: int = 0, out: str = "active (running)"):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "status":
            return _Result(rc=0, out="active (running)")
        return _Result(rc=0, out="")

    monkeypatch.setattr(mod, "_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "ollama", "max_attempts": 1}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["service"] == "ollama"
    assert report.outcome["final_status_active"] is True
    assert any(c[:2] == ["systemctl", "restart"] for c in calls)


def test_service_repair_failure_escalates(monkeypatch):
    import backend.autonomic.levers.service_repair as mod

    class _Result:
        def __init__(self, rc: int, out: str = ""):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        return _Result(rc=3, out="inactive (failed)")

    monkeypatch.setattr(mod, "_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "ollama", "max_attempts": 1}, {})
    assert report.status == LeverStatus.ESCALATED
    assert report.outcome["final_status_active"] is False
