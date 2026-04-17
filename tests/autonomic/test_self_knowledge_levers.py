import json
from datetime import datetime, timezone
from pathlib import Path

from backend.autonomic.levers.capability_scan import FIRE_CAPABILITY_SCAN
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


def test_capability_scan_metadata():
    lever = FIRE_CAPABILITY_SCAN()
    assert lever.name == "FIRE_CAPABILITY_SCAN"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_capability_scan_preconditions_true():
    lever = FIRE_CAPABILITY_SCAN()
    assert lever.preconditions(_snapshot()) is True


def test_capability_scan_writes_tools_notes(tmp_path: Path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("", encoding="utf-8")
    (tools_dir / "web_search.py").write_text(
        '"""Web search tool — Tavily or DuckDuckGo."""\n\ndef search(q: str) -> list: return []\n',
        encoding="utf-8",
    )
    (tools_dir / "file_reader.py").write_text(
        '"""File reader tool."""\n\ndef read(p: str) -> str: return ""\n',
        encoding="utf-8",
    )
    self_root = tmp_path / "knowledge_self"

    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tools_dir),
        "skills_dir": str(tmp_path / "skills"),
        "channels_path": str(tmp_path / "channels.json"),
        "self_root": str(self_root),
    }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["tools_written"] == 2

    web_note = self_root / "tools" / "web_search.md"
    assert web_note.exists()
    content = web_note.read_text(encoding="utf-8")
    assert "Web search tool" in content
    assert "kind: tool" in content


def test_capability_scan_tools_handles_missing_docstring(tmp_path: Path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "bare.py").write_text("def f(): pass\n", encoding="utf-8")
    self_root = tmp_path / "knowledge_self"

    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tools_dir),
        "skills_dir": str(tmp_path / "skills"),
        "channels_path": str(tmp_path / "channels.json"),
        "self_root": str(self_root),
    }, {})

    note = self_root / "tools" / "bare.md"
    assert note.exists()
    assert "(no docstring)" in note.read_text(encoding="utf-8")
