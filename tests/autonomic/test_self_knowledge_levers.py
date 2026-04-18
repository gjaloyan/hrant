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


def test_capability_scan_writes_skills_notes(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    calc_dir = skills_dir / "calc"
    calc_dir.mkdir()
    (calc_dir / "SKILL.md").write_text("# calc\nSafe arithmetic evaluator.\n", encoding="utf-8")
    (calc_dir / "handler.py").write_text("# handler", encoding="utf-8")
    bare_dir = skills_dir / "no_skill_md"
    bare_dir.mkdir()
    (bare_dir / "handler.py").write_text("# handler", encoding="utf-8")
    self_root = tmp_path / "knowledge_self"

    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tmp_path / "tools"),
        "skills_dir": str(skills_dir),
        "channels_path": str(tmp_path / "channels.json"),
        "self_root": str(self_root),
    }, {})

    assert report.outcome["skills_written"] == 2
    calc_note = (self_root / "skills" / "calc.md").read_text(encoding="utf-8")
    assert "Safe arithmetic evaluator" in calc_note
    assert "kind: skill" in calc_note
    bare_note = (self_root / "skills" / "no_skill_md.md").read_text(encoding="utf-8")
    assert "(no SKILL.md)" in bare_note


def test_capability_scan_skills_missing_dir_is_zero(tmp_path: Path):
    self_root = tmp_path / "knowledge_self"
    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tmp_path / "tools"),
        "skills_dir": str(tmp_path / "skills_does_not_exist"),
        "channels_path": str(tmp_path / "channels.json"),
        "self_root": str(self_root),
    }, {})
    assert report.outcome["skills_written"] == 0


def test_capability_scan_writes_channels_summary(tmp_path: Path):
    channels_path = tmp_path / "channels.json"
    channels_path.write_text(json.dumps({
        "channels": [
            {"id": "telegram", "type": "telegram", "enabled": True, "auto_start": True, "status": "running"},
            {"id": "slack", "type": "slack", "enabled": False, "auto_start": False, "status": "stopped"},
        ],
    }), encoding="utf-8")
    self_root = tmp_path / "knowledge_self"

    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tmp_path / "tools"),
        "skills_dir": str(tmp_path / "skills"),
        "channels_path": str(channels_path),
        "self_root": str(self_root),
    }, {})

    assert report.outcome["mcp_written"] is True
    content = (self_root / "mcp_servers" / "channels.md").read_text(encoding="utf-8")
    assert "| telegram | telegram | True | True | running |" in content
    assert "| slack | slack | False | False | stopped |" in content


def test_capability_scan_channels_missing_file_is_false(tmp_path: Path):
    self_root = tmp_path / "knowledge_self"
    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tmp_path / "tools"),
        "skills_dir": str(tmp_path / "skills"),
        "channels_path": str(tmp_path / "nope.json"),
        "self_root": str(self_root),
    }, {})
    assert report.outcome["mcp_written"] is False


def test_capability_scan_writes_server_inventory(tmp_path: Path):
    self_root = tmp_path / "knowledge_self"
    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tmp_path / "tools"),
        "skills_dir": str(tmp_path / "skills"),
        "channels_path": str(tmp_path / "channels.json"),
        "self_root": str(self_root),
    }, {})
    assert report.outcome["server_written"] is True
    content = (self_root / "server_inventory.md").read_text(encoding="utf-8")
    assert "platform:" in content
    assert "cpu_count_physical:" in content
    assert "memory_total_gb:" in content


def test_capability_scan_reason_reports_total_artifacts(tmp_path: Path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "t.py").write_text('"""tool t."""\n', encoding="utf-8")
    self_root = tmp_path / "knowledge_self"

    lever = FIRE_CAPABILITY_SCAN()
    report = lever.run({
        "tools_dir": str(tools_dir),
        "skills_dir": str(tmp_path / "skills"),
        "channels_path": str(tmp_path / "channels.json"),
        "self_root": str(self_root),
    }, {})
    assert "scanned:2_artifacts" == report.reason
