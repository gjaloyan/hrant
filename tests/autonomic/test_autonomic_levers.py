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
