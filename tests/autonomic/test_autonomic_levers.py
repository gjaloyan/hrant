import json
from datetime import datetime, timezone
from pathlib import Path

from backend.autonomic.levers.integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
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


def test_integrity_heartbeat_metadata():
    lever = FIRE_INTEGRITY_HEARTBEAT()
    assert lever.name == "FIRE_INTEGRITY_HEARTBEAT"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_integrity_heartbeat_empty_knowledge_is_clean(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["orphan_files"] == []
    assert report.outcome["dead_entries"] == []
    assert report.outcome["index_count"] == 0
    assert report.outcome["file_count"] == 0


def test_integrity_heartbeat_detects_orphan_file(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    notes_dir = tmp_path / "profession"
    notes_dir.mkdir()
    (notes_dir / "python.md").write_text("# Python\n", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == ["profession/python.md"]
    assert report.outcome["dead_entries"] == []
    assert "drift" in report.reason


def test_integrity_heartbeat_detects_dead_entry(tmp_path: Path):
    index = {"profession/ghost.md": {"topic": "ghost"}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["dead_entries"] == ["profession/ghost.md"]


def test_integrity_heartbeat_excludes_system_dirs(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    for excluded in ("_history", "autonomic", "immune", "identity"):
        d = tmp_path / excluded
        d.mkdir()
        (d / "note.md").write_text("x", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == []
    assert report.outcome["file_count"] == 0


def test_integrity_heartbeat_reports_ok_when_matched(tmp_path: Path):
    index = {"profession/python.md": {"topic": "python"}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "profession").mkdir()
    (tmp_path / "profession" / "python.md").write_text("# Python\n", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == []
    assert report.outcome["dead_entries"] == []
    assert report.reason == "integrity_ok"


def test_integrity_heartbeat_preconditions_always_true():
    lever = FIRE_INTEGRITY_HEARTBEAT()
    assert lever.preconditions(_snapshot()) is True


from unittest.mock import patch

from backend.autonomic.levers.goal_propose import FIRE_GOAL_PROPOSE


def test_goal_propose_metadata():
    lever = FIRE_GOAL_PROPOSE()
    assert lever.name == "FIRE_GOAL_PROPOSE"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_goal_propose_skips_when_gaps_file_missing(tmp_path: Path):
    lever = FIRE_GOAL_PROPOSE()
    report = lever.run({"gaps_path": str(tmp_path / "missing.json")}, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_goal_propose_skips_when_gaps_file_empty(tmp_path: Path):
    gaps_path = tmp_path / "gaps.json"
    gaps_path.write_text("{}", encoding="utf-8")
    lever = FIRE_GOAL_PROPOSE()
    report = lever.run({"gaps_path": str(gaps_path)}, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_goal_propose_delegates_to_goals_manager(tmp_path: Path):
    gaps_path = tmp_path / "gaps.json"
    gaps_path.write_text(json.dumps({
        "python_async": {"topic": "python_async", "count": 3, "last": "2026-04-15"},
        "rust_ownership": {"topic": "rust_ownership", "count": 2, "last": "2026-04-16"},
    }), encoding="utf-8")

    captured: dict = {}

    class _FakeGoal:
        def __init__(self, description: str):
            self.description = description
            self.id = "id_" + description[:5]

    def _fake_suggest(gaps, max_goals=3):
        captured["gaps"] = gaps
        captured["max_goals"] = max_goals
        return [_FakeGoal(f"Learn about: {g['topic']}") for g in gaps[:max_goals]]

    lever = FIRE_GOAL_PROPOSE()
    with patch("backend.autonomic.levers.goal_propose.GOALS") as mock_goals:
        mock_goals.suggest_from_gaps.side_effect = _fake_suggest
        report = lever.run({"gaps_path": str(gaps_path), "max_goals": 2}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["proposed"] == 2
    assert report.outcome["gap_count"] == 2
    assert captured["max_goals"] == 2
    assert {g["topic"] for g in captured["gaps"]} == {"python_async", "rust_ownership"}


def test_goal_propose_preconditions_true():
    lever = FIRE_GOAL_PROPOSE()
    assert lever.preconditions(_snapshot()) is True
