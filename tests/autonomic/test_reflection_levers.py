import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.autonomic.levers.self_reflection import FIRE_SELF_REFLECTION
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


def _stats(total=0, avg_severity=0.0, patterns_count=0, by_domain=None, by_cause=None):
    return {
        "total_failures": total,
        "by_root_cause": by_cause or {},
        "by_domain": by_domain or {},
        "avg_severity": avg_severity,
        "patterns_count": patterns_count,
        "patterns": [],
    }


def test_self_reflection_metadata():
    lever = FIRE_SELF_REFLECTION()
    assert lever.name == "FIRE_SELF_REFLECTION"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_self_reflection_preconditions_true():
    lever = FIRE_SELF_REFLECTION()
    assert lever.preconditions(_snapshot()) is True


def test_self_reflection_skips_when_not_enough_failures(tmp_path: Path):
    fake_meta = MagicMock()
    fake_meta.stats.return_value = _stats(total=2)
    log_path = tmp_path / "self_reflection_log.jsonl"

    lever = FIRE_SELF_REFLECTION()
    with patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "insufficient_failures"
    assert not log_path.exists()
    fake_meta.extract_patterns.assert_not_called()


def test_self_reflection_writes_snapshot_when_enough_data(tmp_path: Path):
    fake_meta = MagicMock()
    fake_meta.stats.return_value = _stats(
        total=8, avg_severity=6.5, patterns_count=2,
        by_domain={"python": 5, "databases": 3},
        by_cause={"missing_context": 4, "wrong_tool": 3, "unknown": 1},
    )
    fake_meta.extract_patterns.return_value = [
        {"pattern": "DB queries without schema context", "priority": 8, "frequency": 3,
         "suggested_fix": "Load schema first"},
        {"pattern": "Python async misunderstanding", "priority": 6, "frequency": 2,
         "suggested_fix": "Study asyncio basics"},
    ]
    log_path = tmp_path / "self_reflection_log.jsonl"

    lever = FIRE_SELF_REFLECTION()
    with patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total_failures"] == 8
    assert report.outcome["avg_severity"] == 6.5
    assert report.outcome["patterns_count"] == 2
    fake_meta.extract_patterns.assert_called_once()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["total_failures"] == 8
    assert entry["by_domain"]["python"] == 5
    assert len(entry["patterns"]) == 2
    assert entry["patterns"][0]["priority"] == 8


def test_self_reflection_tolerates_extract_patterns_exception(tmp_path: Path):
    fake_meta = MagicMock()
    fake_meta.stats.return_value = _stats(total=5, avg_severity=5.0, patterns_count=0)
    fake_meta.extract_patterns.side_effect = RuntimeError("cortex timeout")
    log_path = tmp_path / "self_reflection_log.jsonl"

    lever = FIRE_SELF_REFLECTION()
    with patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.FAILURE
    assert "reflect_failed" in report.reason
    assert not log_path.exists()
