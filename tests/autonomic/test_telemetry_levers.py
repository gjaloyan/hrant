import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.autonomic.levers.model_eval import FIRE_MODEL_EVAL
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


def _report(total=0, avg_conf=0, intents=None):
    return {
        "date": "2026-04-17",
        "total_interactions": total,
        "tasks": max(0, total - 1),
        "chats": min(1, total),
        "avg_confidence": avg_conf,
        "total_contradictions": 0,
        "total_unverified": 0,
        "topics_used": [],
        "by_intent": intents or {},
        "low_confidence_count": 0,
        "high_confidence_count": 0,
    }


def test_model_eval_metadata():
    lever = FIRE_MODEL_EVAL()
    assert lever.name == "FIRE_MODEL_EVAL"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_model_eval_preconditions_true():
    lever = FIRE_MODEL_EVAL()
    assert lever.preconditions(_snapshot()) is True


def test_model_eval_skips_when_no_entries(tmp_path: Path):
    fake_eval = MagicMock()
    fake_eval.daily_report.return_value = _report(total=0)
    log_path = tmp_path / "model_eval_log.jsonl"

    lever = FIRE_MODEL_EVAL()
    with patch("backend.autonomic.levers.model_eval.EVALUATOR", fake_eval):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_eval_entries"
    assert not log_path.exists()


def test_model_eval_writes_snapshot_with_regressions_and_priorities(tmp_path: Path):
    fake_eval = MagicMock()
    fake_eval.daily_report.return_value = _report(total=4, avg_conf=82, intents={"task": 3, "chat": 1})
    fake_eval.detect_regression.return_value = [
        {"domain": "python", "this_week_avg": 70.0, "last_week_avg": 88.0, "drop": 18.0, "sample_size": 5},
    ]
    fake_eval.suggest_priorities.return_value = [
        {"topic": "python", "reason": "low_confidence", "priority": 8},
    ]
    log_path = tmp_path / "model_eval_log.jsonl"

    lever = FIRE_MODEL_EVAL()
    with patch("backend.autonomic.levers.model_eval.EVALUATOR", fake_eval):
        report = lever.run({"log_path": str(log_path), "target_date": "2026-04-17"}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total"] == 4
    assert report.outcome["regressions_count"] == 1
    assert report.outcome["priorities_count"] == 1

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["date"] == "2026-04-17"
    assert entry["daily_report"]["total_interactions"] == 4
    assert entry["regressions"][0]["domain"] == "python"
    assert entry["priorities"][0]["topic"] == "python"


def test_model_eval_tolerates_regression_exception(tmp_path: Path):
    fake_eval = MagicMock()
    fake_eval.daily_report.return_value = _report(total=2)
    fake_eval.detect_regression.side_effect = RuntimeError("boom")
    fake_eval.suggest_priorities.return_value = []
    log_path = tmp_path / "model_eval_log.jsonl"

    lever = FIRE_MODEL_EVAL()
    with patch("backend.autonomic.levers.model_eval.EVALUATOR", fake_eval):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.SUCCESS
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["regressions"] == []
    assert entry["priorities"] == []
