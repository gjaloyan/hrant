import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.autonomic.levers.capability_scan import FIRE_CAPABILITY_SCAN
from backend.autonomic.levers.self_study import FIRE_SELF_STUDY
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, StateSnapshot, utcnow


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


def _fake_study_response() -> dict:
    return {
        "purpose": "Routes tick decisions through Layer 0 rules.",
        "public_interface": [
            {"name": "Layer0Engine", "kind": "class", "one_line": "Evaluates rules in order."},
            {"name": "default_rules", "kind": "function", "one_line": "Returns the seeded rule list."},
        ],
        "dependencies": ["backend.autonomic.types"],
        "notes": "Cooldown fall-through preserves single-rule semantics.",
    }


def _seed_backend(root: Path, module_names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    for name in module_names:
        (root / f"{name}.py").write_text(f'"""Module {name}."""\n\nX = 1\n', encoding="utf-8")


def test_self_study_metadata():
    lever = FIRE_SELF_STUDY()
    assert lever.name == "FIRE_SELF_STUDY"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_self_study_preconditions_true():
    lever = FIRE_SELF_STUDY()
    assert lever.preconditions(_snapshot()) is True


def test_self_study_picks_new_modules_first_and_caps_at_max(tmp_path: Path):
    backend_root = tmp_path / "backend"
    _seed_backend(backend_root, ["a", "b", "c", "d", "e"])
    self_root = tmp_path / "knowledge_self"

    lever = FIRE_SELF_STUDY()
    with patch("backend.autonomic.levers.self_study.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_study_response()
        report = lever.run({
            "backend_root": str(backend_root),
            "self_root": str(self_root),
            "max_modules": 3,
        }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["modules_processed"] == 3
    assert report.outcome["modules_total"] == 5
    notes = sorted((self_root / "modules").glob("*.md"))
    assert len(notes) == 3


def test_self_study_prefers_stale_over_fresh(tmp_path: Path):
    backend_root = tmp_path / "backend"
    _seed_backend(backend_root, ["fresh", "stale"])
    self_root = tmp_path / "knowledge_self"
    modules_dir = self_root / "modules"
    modules_dir.mkdir(parents=True)

    fresh_mtime_iso = datetime.fromtimestamp((backend_root / "fresh.py").stat().st_mtime, timezone.utc).isoformat()
    (modules_dir / "fresh.md").write_text(
        f"---\nmodule: backend/fresh.py\ncategory: self\nkind: module\nupdated: {utcnow().isoformat()}\nsource_mtime: {fresh_mtime_iso}\nloc: 1\n---\n\n# fresh\n",
        encoding="utf-8",
    )
    (modules_dir / "stale.md").write_text(
        "---\nmodule: backend/stale.py\ncategory: self\nkind: module\nupdated: 2020-01-01T00:00:00+00:00\nsource_mtime: 2020-01-01T00:00:00+00:00\nloc: 1\n---\n\n# stale\n",
        encoding="utf-8",
    )

    lever = FIRE_SELF_STUDY()
    with patch("backend.autonomic.levers.self_study.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_study_response()
        report = lever.run({
            "backend_root": str(backend_root),
            "self_root": str(self_root),
            "max_modules": 1,
        }, {})

    assert report.outcome["modules_processed"] == 1
    stale_after = (modules_dir / "stale.md").read_text(encoding="utf-8")
    assert "Routes tick decisions" in stale_after


def test_self_study_skips_pycache_tests_venv(tmp_path: Path):
    backend_root = tmp_path / "backend"
    backend_root.mkdir()
    (backend_root / "__init__.py").write_text("", encoding="utf-8")
    (backend_root / "real.py").write_text('"""real."""\n', encoding="utf-8")
    for skip_dir in ("__pycache__", "tests", ".venv"):
        d = backend_root / skip_dir
        d.mkdir()
        (d / "junk.py").write_text('"""junk."""\n', encoding="utf-8")
    self_root = tmp_path / "knowledge_self"

    lever = FIRE_SELF_STUDY()
    with patch("backend.autonomic.levers.self_study.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_study_response()
        report = lever.run({
            "backend_root": str(backend_root),
            "self_root": str(self_root),
            "max_modules": 5,
        }, {})

    assert report.outcome["modules_total"] == 1
    assert report.outcome["modules_processed"] == 1


def test_self_study_cortex_failure_skips_that_module(tmp_path: Path):
    backend_root = tmp_path / "backend"
    _seed_backend(backend_root, ["a", "b"])
    self_root = tmp_path / "knowledge_self"

    call_count = {"n": 0}

    def flaky_call(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("cortex timeout")
        return _fake_study_response()

    lever = FIRE_SELF_STUDY()
    with patch("backend.autonomic.levers.self_study.router") as mock_router:
        mock_router.return_value.call_json.side_effect = flaky_call
        report = lever.run({
            "backend_root": str(backend_root),
            "self_root": str(self_root),
            "max_modules": 2,
        }, {})

    assert report.outcome["modules_processed"] == 1
    assert report.outcome["modules_skipped"] == 1


def test_self_study_truncates_large_files(tmp_path: Path):
    backend_root = tmp_path / "backend"
    backend_root.mkdir()
    (backend_root / "__init__.py").write_text("", encoding="utf-8")
    big = "\n".join(f"x = {i}" for i in range(3500))
    (backend_root / "big.py").write_text(f'"""big module."""\n{big}\n', encoding="utf-8")
    self_root = tmp_path / "knowledge_self"

    captured: dict = {}

    def capture_call(task_type, system, user, **kw):
        captured["user"] = user
        return _fake_study_response()

    lever = FIRE_SELF_STUDY()
    with patch("backend.autonomic.levers.self_study.router") as mock_router:
        mock_router.return_value.call_json.side_effect = capture_call
        lever.run({
            "backend_root": str(backend_root),
            "self_root": str(self_root),
            "max_modules": 1,
        }, {})

    assert "[truncated]" in captured["user"]
