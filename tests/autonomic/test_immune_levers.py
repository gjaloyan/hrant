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
    # "ollama" left the whitelist on 2026-08-10, and the whitelist check runs
    # before the platform check — use an allowed unit or this asserts the
    # wrong gate.
    report = lever.run({"service": "hrant"}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "platform" in report.reason.lower()


def _fake_systemctl(monkeypatch, mod, *, rc=0, active="active", moved=True):
    """Drive the lever through its contract as of 2026-08-10.

    The two tests this replaces mocked `subprocess.run`, passed
    service="ollama" as a param, and verified by grepping `systemctl status`
    for "active (running)". All three are gone: ollama left the whitelist (it
    is a name collision between a healthy system unit and a crash-looping
    user one), the unit now comes from the tick state, and a repair is proven
    by a MOVED ActiveEnterTimestamp — because a polkit-denied restart against
    an already-running unit used to be logged as a successful repair.
    """
    calls = []
    stamps = iter(["A", "B" if moved else "A"])

    class _R:
        def __init__(self, code, out=""):
            self.returncode, self.stdout, self.stderr = code, out, ""

    def fake(manager, *args, timeout=30.0):
        calls.append((manager, args[0]))
        if args[0] == "restart":
            return _R(rc)
        if args[0] == "is-active":
            return _R(0, active)
        if args[0] == "show":
            return _R(0, next(stamps))
        return _R(0)

    monkeypatch.setattr(mod, "_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(mod, "_systemctl", fake)
    return calls


class _FailedState:
    def __init__(self, failed):
        self.failed_services = failed


def test_service_repair_restarts_a_failed_whitelisted_unit(monkeypatch):
    import backend.autonomic.levers.service_repair as mod
    calls = _fake_systemctl(monkeypatch, mod)
    report = FIRE_SERVICE_REPAIR().run(
        {"max_attempts": 1},
        {"state": _FailedState(["user:lightrag.service"])})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["service"] == "lightrag"
    assert ("user", "restart") in calls


def test_service_repair_escalates_when_the_restart_is_refused(monkeypatch):
    import backend.autonomic.levers.service_repair as mod
    _fake_systemctl(monkeypatch, mod, rc=1, active="active")
    report = FIRE_SERVICE_REPAIR().run(
        {"max_attempts": 1},
        {"state": _FailedState(["user:lightrag.service"])})
    assert report.status != LeverStatus.SUCCESS


import json
from pathlib import Path

from backend.autonomic.levers.self_heal import FIRE_SELF_HEAL


def _write_seed_sigs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "id": "heal_test_v1",
            "pattern": {"source": "error_log", "msg_regex": "boom"},
            "severity": "warn",
            "fix_lever": "FIRE_SERVER_HEALTH",
            "fix_params": {"verbose": True},
            "observed_count": 0,
            "success_rate": None,
        }) + "\n",
        encoding="utf-8",
    )


def test_self_heal_metadata():
    lever = FIRE_SELF_HEAL()
    assert lever.name == "FIRE_SELF_HEAL"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN


def test_self_heal_without_signature_id_is_skipped(tmp_path: Path):
    sig_path = tmp_path / "sig.jsonl"
    _write_seed_sigs(sig_path)
    lever = FIRE_SELF_HEAL()
    report = lever.run({"signatures_path": str(sig_path)}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "signature_id" in report.reason


def test_self_heal_unknown_signature_is_skipped(tmp_path: Path):
    sig_path = tmp_path / "sig.jsonl"
    _write_seed_sigs(sig_path)
    lever = FIRE_SELF_HEAL()
    report = lever.run(
        {"signature_id": "nope", "signatures_path": str(sig_path)},
        {},
    )
    assert report.status == LeverStatus.SKIPPED
    assert "unknown_signature" in report.reason


def test_self_heal_returns_fix_plan_without_executing(tmp_path: Path):
    sig_path = tmp_path / "sig.jsonl"
    _write_seed_sigs(sig_path)
    lever = FIRE_SELF_HEAL()
    report = lever.run(
        {"signature_id": "heal_test_v1", "signatures_path": str(sig_path)},
        {},
    )
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["signature_id"] == "heal_test_v1"
    assert report.outcome["fix_lever"] == "FIRE_SERVER_HEALTH"
    assert report.outcome["fix_params"] == {"verbose": True}
    assert report.follow_ups == ["FIRE_SERVER_HEALTH"]


def test_self_heal_preconditions_always_true():
    lever = FIRE_SELF_HEAL()
    assert lever.preconditions(_snapshot()) is True
