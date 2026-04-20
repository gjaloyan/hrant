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


from backend.autonomic.levers.finetune_qc import FIRE_FINETUNE_QC


def _pair_json(
    user_text: str = "what is python?",
    assistant_text: str = "A programming language.",
    confidence: int = 85,
    category: str = "factual_qa",
    boosted: bool = False,
    verified: bool = True,
    sources: list[str] | None = None,
) -> str:
    obj = {
        "id": f"id_{hash(user_text) & 0xfff}",
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text * 3},
        ],
        "metadata": {
            "source_notes": sources or ["python"],
            "confidence": confidence,
            "project": None,
            "timestamp": "2026-04-20T12:00:00",
            "verified": verified,
            "category": category,
            "boosted": boosted,
            "original_wrong_answer": None,
        },
    }
    return json.dumps(obj)


def _legacy_json() -> str:
    return json.dumps({
        "instruction": "legacy question",
        "response": "legacy answer",
        "sources": ["x"],
        "confidence": 80,
        "timestamp": "2026-04-07T10:51:30",
    })


def test_finetune_qc_metadata():
    lever = FIRE_FINETUNE_QC()
    assert lever.name == "FIRE_FINETUNE_QC"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_finetune_qc_skips_when_queue_missing(tmp_path: Path):
    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(tmp_path / "nope.jsonl"),
        "log_path": str(tmp_path / "finetune_qc_log.jsonl"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_valid_pairs"


def test_finetune_qc_skips_when_only_legacy(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    queue.write_text(_legacy_json() + "\n" + _legacy_json() + "\n", encoding="utf-8")

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(tmp_path / "finetune_qc_log.jsonl"),
    }, {})

    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_valid_pairs"
    assert report.outcome["legacy_entries"] == 2


def test_finetune_qc_scores_and_writes_snapshot(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    queue.write_text(
        "\n".join([
            _pair_json(user_text="q1", confidence=90, category="factual_qa"),
            _pair_json(user_text="q2", confidence=90, category="correction", boosted=True),
            _pair_json(user_text="q3", confidence=50, category="other"),
        ]) + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(log_path),
    }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total"] == 3
    assert report.outcome["legacy_entries"] == 0
    assert report.outcome["avg_score"] > 0

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["total"] == 3
    assert entry["low"] + entry["medium"] + entry["high"] == 3
    assert entry["by_category"]["factual_qa"] == 1
    assert entry["by_category"]["correction"] == 1
    assert entry["boosted"] == 1
    assert entry["verified"] == 3


def test_finetune_qc_counts_legacy_alongside_valid(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    queue.write_text(
        "\n".join([
            _legacy_json(),
            _pair_json(user_text="q1"),
            _legacy_json(),
            _pair_json(user_text="q2"),
        ]) + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(log_path),
    }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total"] == 2
    assert report.outcome["legacy_entries"] == 2


def test_finetune_qc_curated_count_applies_min_score_and_dedup(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    queue.write_text(
        "\n".join([
            _pair_json(user_text="how to install python", confidence=95, category="correction"),
            _pair_json(user_text="what is asyncio", confidence=95, category="procedure"),
            _pair_json(user_text="explain decorators", confidence=95, category="factual_qa"),
            _pair_json(user_text="how to install python", confidence=95, category="correction"),  # dup
            _pair_json(user_text="random q", confidence=10, category="other", verified=False, sources=[]),  # low
        ]) + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(log_path),
    }, {})

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["total"] == 5
    assert entry["curated"] == 3


def test_finetune_qc_never_mutates_queue(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    original = _pair_json(user_text="q1") + "\n" + _legacy_json() + "\n"
    queue.write_text(original, encoding="utf-8")
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    lever.run({"queue_path": str(queue), "log_path": str(log_path)}, {})

    assert queue.read_text(encoding="utf-8") == original
