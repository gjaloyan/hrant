# D-04 — Self-knowledge cohort (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `FIRE_CAPABILITY_SCAN` (python, 6h cadence) and `FIRE_SELF_STUDY` (claude-delegate, daily, max 3 modules/tick), seed the `knowledge/self/` directory structure, extend `default_rules()` from 7 to 9, and integration-test the full tick chain.

**Architecture:** Each lever is a single file in `backend/autonomic/levers/` subclassing `Lever`. `FIRE_CAPABILITY_SCAN` does pure filesystem + psutil walks and writes markdown snapshots. `FIRE_SELF_STUDY` selects up to 3 priority-ordered backend modules per tick, sends each to the cortex via `router().call_json`, and writes one markdown note per module. Both register through the existing `register_default_autonomic_levers()` function — no new registration entry point.

**Tech Stack:** Python 3.11+, existing autonomic contracts (`Lever`, `LayerZeroRule`, `Layer0Engine`, `LeverExecutor`), existing `backend.llm.router` / `TaskType.TASK_ANALYSIS`, `psutil`, `platform`, pytest.

**Parent spec:** [docs/superpowers/specs/2026-04-18-d-04-self-knowledge-design.md](../specs/2026-04-18-d-04-self-knowledge-design.md)

---

## File Structure

**New files (4):**

```
backend/autonomic/levers/
├── capability_scan.py           # FIRE_CAPABILITY_SCAN (~180 lines)
└── self_study.py                # FIRE_SELF_STUDY (~160 lines)

tests/autonomic/
├── test_self_knowledge_levers.py    # 14 unit tests
└── test_d04_integration.py          # 3 integration tests
```

**Modified files (3):**

- `backend/autonomic/layer0.py` — `default_rules()` grows from 7 to 9 rules.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers()` registers 2 more levers.
- `README.md` — mention 2 new levers + `knowledge/self/`.

**Seed files (runtime):**

- `knowledge/self/.gitkeep` — empty sentinel committed so the dir exists. All subdirectories (`modules/`, `tools/`, `skills/`, `mcp_servers/`) are created at runtime by the levers via `path.parent.mkdir(parents=True, exist_ok=True)`.

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `api.py`, `startup.py`, `backend/main.py`, existing D-01..D-03 levers.

---

## Task 1: Seed `knowledge/self/` directory

**Files:**
- Create: `knowledge/self/.gitkeep`

- [ ] **Step 1: Create empty sentinel**

```bash
mkdir -p knowledge/self
touch knowledge/self/.gitkeep
```

- [ ] **Step 2: Verify directory exists**

Run: `ls knowledge/self/`
Expected: `.gitkeep` listed.

- [ ] **Step 3: Commit**

```bash
git add knowledge/self/.gitkeep
git commit -m "chore(self-knowledge): seed knowledge/self/ directory"
```

---

## Task 2: FIRE_CAPABILITY_SCAN — tools subpass

This lever has four independent subpasses (tools / skills / mcp / server_inventory). We build it one subpass at a time to keep commits focused. Each subpass gets its own failing test → implementation → passing.

**Files:**
- Create: `backend/autonomic/levers/capability_scan.py`
- Test: `tests/autonomic/test_self_knowledge_levers.py`

- [ ] **Step 1: Write failing tests for metadata + tools subpass**

Create `tests/autonomic/test_self_knowledge_levers.py`:

```python
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
        "skills_dir": str(tmp_path / "skills"),  # empty / missing dir OK
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.capability_scan'`.

- [ ] **Step 3: Scaffold `capability_scan.py` with metadata + tools subpass**

Create `backend/autonomic/levers/capability_scan.py`:

```python
"""FIRE_CAPABILITY_SCAN — inventory tools, skills, channels, and server into knowledge/self/."""
from __future__ import annotations

import ast
import json
import logging
import platform
from datetime import date
from pathlib import Path
from typing import Any

import psutil

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

log = logging.getLogger(__name__)

DEFAULT_TOOLS_DIR = Path("backend/tools")
DEFAULT_SKILLS_DIR = Path("backend/skills")
DEFAULT_CHANNELS_PATH = Path("knowledge/channels.json")
DEFAULT_SELF_ROOT = Path("knowledge/self")


class FIRE_CAPABILITY_SCAN(Lever):
    name = "FIRE_CAPABILITY_SCAN"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.5)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        tools_dir = Path(params.get("tools_dir") or DEFAULT_TOOLS_DIR)
        skills_dir = Path(params.get("skills_dir") or DEFAULT_SKILLS_DIR)
        channels_path = Path(params.get("channels_path") or DEFAULT_CHANNELS_PATH)
        self_root = Path(params.get("self_root") or DEFAULT_SELF_ROOT)

        tools_written = self._scan_tools(tools_dir, self_root / "tools")
        skills_written = 0
        mcp_written = False
        server_written = False

        total = tools_written + skills_written + (1 if mcp_written else 0) + (1 if server_written else 0)
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "tools_written": tools_written,
                "skills_written": skills_written,
                "mcp_written": mcp_written,
                "server_written": server_written,
            },
            reason=f"scanned:{total}_artifacts",
        )

    def _scan_tools(self, tools_dir: Path, out_dir: Path) -> int:
        if not tools_dir.exists():
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for py in sorted(tools_dir.glob("*.py")):
            if py.name == "__init__.py":
                continue
            try:
                src = py.read_text(encoding="utf-8")
            except OSError as exc:
                log.warning("capability_scan: could not read %s: %s", py, exc)
                continue
            doc = _module_docstring(src)
            funcs = _top_level_functions(src)
            note_path = out_dir / f"{py.stem}.md"
            note_path.write_text(
                _render_tool_note(py.name, doc, funcs, py.stat().st_mtime),
                encoding="utf-8",
            )
            written += 1
        return written


def _module_docstring(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "(parse error)"
    doc = ast.get_docstring(tree)
    return doc or "(no docstring)"


def _top_level_functions(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            out.append(node.name)
    return out


def _render_tool_note(file_name: str, doc: str, funcs: list[str], mtime: float) -> str:
    from datetime import datetime, timezone
    mtime_iso = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
    updated_iso = utcnow().isoformat()
    lines = [
        "---",
        f"module: backend/tools/{file_name}",
        "category: self",
        "kind: tool",
        f"updated: {updated_iso}",
        f"source_mtime: {mtime_iso}",
        "---",
        "",
        f"# backend/tools/{file_name}",
        "",
        "## Purpose",
        doc,
        "",
        "## Top-level functions",
    ]
    if funcs:
        lines.extend(f"- `{f}`" for f in funcs)
    else:
        lines.append("(none)")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v`

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/capability_scan.py tests/autonomic/test_self_knowledge_levers.py
git commit -m "feat(autonomic): FIRE_CAPABILITY_SCAN scaffold + tools subpass"
```

---

## Task 3: FIRE_CAPABILITY_SCAN — skills subpass

**Files:**
- Modify: `backend/autonomic/levers/capability_scan.py`
- Test: extend `tests/autonomic/test_self_knowledge_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_self_knowledge_levers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v -k skills`

Expected: FAIL with `assert 0 == 2`.

- [ ] **Step 3: Add skills subpass to `capability_scan.py`**

Replace the `run()` and add `_scan_skills()`:

```python
    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        tools_dir = Path(params.get("tools_dir") or DEFAULT_TOOLS_DIR)
        skills_dir = Path(params.get("skills_dir") or DEFAULT_SKILLS_DIR)
        channels_path = Path(params.get("channels_path") or DEFAULT_CHANNELS_PATH)
        self_root = Path(params.get("self_root") or DEFAULT_SELF_ROOT)

        tools_written = self._scan_tools(tools_dir, self_root / "tools")
        skills_written = self._scan_skills(skills_dir, self_root / "skills")
        mcp_written = False
        server_written = False

        total = tools_written + skills_written + (1 if mcp_written else 0) + (1 if server_written else 0)
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "tools_written": tools_written,
                "skills_written": skills_written,
                "mcp_written": mcp_written,
                "server_written": server_written,
            },
            reason=f"scanned:{total}_artifacts",
        )

    def _scan_skills(self, skills_dir: Path, out_dir: Path) -> int:
        if not skills_dir.exists():
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            skill_md = entry / "SKILL.md"
            description = skill_md.read_text(encoding="utf-8") if skill_md.exists() else "(no SKILL.md)"
            files = sorted(p.name for p in entry.iterdir() if p.is_file())
            note_path = out_dir / f"{entry.name}.md"
            note_path.write_text(
                _render_skill_note(entry.name, description, files),
                encoding="utf-8",
            )
            written += 1
        return written
```

Add the `_render_skill_note` helper at module scope (below `_render_tool_note`):

```python
def _render_skill_note(name: str, description: str, files: list[str]) -> str:
    updated_iso = utcnow().isoformat()
    lines = [
        "---",
        f"skill: {name}",
        "category: self",
        "kind: skill",
        f"updated: {updated_iso}",
        f"file_count: {len(files)}",
        "---",
        "",
        f"# skill: {name}",
        "",
        "## Description",
        description.strip(),
        "",
        "## Files",
    ]
    if files:
        lines.extend(f"- `{f}`" for f in files)
    else:
        lines.append("(empty)")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v`

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/capability_scan.py tests/autonomic/test_self_knowledge_levers.py
git commit -m "feat(autonomic): CAPABILITY_SCAN skills subpass"
```

---

## Task 4: FIRE_CAPABILITY_SCAN — channels + server_inventory subpasses

**Files:**
- Modify: `backend/autonomic/levers/capability_scan.py`
- Test: extend `tests/autonomic/test_self_knowledge_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_self_knowledge_levers.py`:

```python
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
    assert "cpu_count:" in content
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
    # 1 tool + 0 skills + 0 mcp + 1 server = 2
    assert "scanned:2_artifacts" == report.reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v -k "channels or server or reason_reports"`

Expected: FAIL — mcp_written stays False, server_written stays False.

- [ ] **Step 3: Add channels + server subpasses to `capability_scan.py`**

Replace the `run()` method with:

```python
    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        tools_dir = Path(params.get("tools_dir") or DEFAULT_TOOLS_DIR)
        skills_dir = Path(params.get("skills_dir") or DEFAULT_SKILLS_DIR)
        channels_path = Path(params.get("channels_path") or DEFAULT_CHANNELS_PATH)
        self_root = Path(params.get("self_root") or DEFAULT_SELF_ROOT)

        tools_written = self._scan_tools(tools_dir, self_root / "tools")
        skills_written = self._scan_skills(skills_dir, self_root / "skills")
        mcp_written = self._scan_channels(channels_path, self_root / "mcp_servers")
        server_written = self._scan_server(self_root / "server_inventory.md")

        total = tools_written + skills_written + (1 if mcp_written else 0) + (1 if server_written else 0)
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "tools_written": tools_written,
                "skills_written": skills_written,
                "mcp_written": mcp_written,
                "server_written": server_written,
            },
            reason=f"scanned:{total}_artifacts",
        )

    def _scan_channels(self, channels_path: Path, out_dir: Path) -> bool:
        if not channels_path.exists():
            return False
        try:
            data = json.loads(channels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        channels = data.get("channels", []) if isinstance(data, dict) else []
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "---",
            "category: self",
            "kind: mcp_channels",
            f"updated: {utcnow().isoformat()}",
            f"channel_count: {len(channels)}",
            "---",
            "",
            "# MCP / channels inventory",
            "",
            "| id | type | enabled | auto_start | status |",
            "| --- | --- | --- | --- | --- |",
        ]
        for ch in channels:
            lines.append(
                f"| {ch.get('id','?')} | {ch.get('type','?')} | {ch.get('enabled','?')} | {ch.get('auto_start','?')} | {ch.get('status','?')} |"
            )
        lines.append("")
        (out_dir / "channels.md").write_text("\n".join(lines), encoding="utf-8")
        return True

    def _scan_server(self, out_path: Path) -> bool:
        try:
            vm = psutil.virtual_memory()
            disk = psutil.disk_usage(".")
            boot_iso = datetime.fromtimestamp(psutil.boot_time(), timezone.utc).isoformat()
            info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "machine": platform.machine(),
                "cpu_count_physical": psutil.cpu_count(logical=False) or 0,
                "cpu_count_logical": psutil.cpu_count(logical=True) or 0,
                "memory_total_gb": round(vm.total / (1024 ** 3), 2),
                "memory_available_gb": round(vm.available / (1024 ** 3), 2),
                "disk_total_gb": round(disk.total / (1024 ** 3), 2),
                "disk_free_gb": round(disk.free / (1024 ** 3), 2),
                "boot_time": boot_iso,
            }
        except Exception as exc:
            log.warning("capability_scan: server inventory failed: %s", exc)
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "---",
            "category: self",
            "kind: server_inventory",
            f"updated: {utcnow().isoformat()}",
            f"host: {platform.node()}",
            "---",
            "",
            "# Server inventory",
            "",
        ]
        for k, v in info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return True
```

Add the missing imports at the top of the file:

```python
from datetime import datetime, timezone
```

(Add `datetime, timezone` to the existing `from datetime import date` line: `from datetime import date, datetime, timezone`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v`

Expected: 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/capability_scan.py tests/autonomic/test_self_knowledge_levers.py
git commit -m "feat(autonomic): CAPABILITY_SCAN channels + server inventory subpasses"
```

---

## Task 5: FIRE_SELF_STUDY

**Files:**
- Create: `backend/autonomic/levers/self_study.py`
- Test: extend `tests/autonomic/test_self_knowledge_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_self_knowledge_levers.py`:

```python
from unittest.mock import patch

from backend.autonomic.levers.self_study import FIRE_SELF_STUDY


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

    fresh_mtime = datetime.fromtimestamp((backend_root / "fresh.py").stat().st_mtime, timezone.utc).isoformat()
    (modules_dir / "fresh.md").write_text(
        f"---\nmodule: backend/fresh.py\ncategory: self\nkind: module\nupdated: {utcnow_iso_helper()}\nsource_mtime: {fresh_mtime}\nloc: 1\n---\n\n# fresh\n",
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


# Helper for the stale-priority test
def utcnow_iso_helper() -> str:
    return utcnow().isoformat()


# Bring `utcnow` and `datetime, timezone` into scope at module top (if not already)
from backend.autonomic.types import utcnow  # noqa: E402 (placed here to show intent; prefer top-of-file in real code)
from datetime import datetime, timezone  # noqa: E402
```

**Note:** the two imports at the bottom (`utcnow`, `datetime, timezone`) should be moved to the top of the test file once the file is loaded. Leaving them at the bottom in the plan keeps the test code local and readable; engineer moves them when committing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v -k self_study`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.self_study'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/self_study.py`**

```python
"""FIRE_SELF_STUDY — generate/refresh knowledge/self/modules/*.md via cortex."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

from backend.llm import router, TaskType

log = logging.getLogger(__name__)

DEFAULT_BACKEND_ROOT = Path("backend")
DEFAULT_SELF_ROOT = Path("knowledge/self")
MAX_LINES = 3000
SKIP_DIR_NAMES = {"__pycache__", "tests", ".venv", "node_modules"}

SELF_STUDY_SYSTEM = """You are the self-knowledge module of an AI agent. You read one Python module
from the agent's own source tree and produce a short structured description.

Return strictly JSON:
{
  "purpose": "one-paragraph summary of what this module is for",
  "public_interface": [
    {"name": "<class or function name>", "kind": "class|function|constant", "one_line": "<short description>"}
  ],
  "dependencies": ["<internal module>", ...],
  "notes": "optional observations about complexity, invariants, gotchas — 2-3 sentences"
}

Rules:
- Skip __dunder__ helpers unless they define the public API.
- "dependencies" lists imports from the same project (backend.*), not stdlib or third-party.
- Max 12 entries in public_interface.
- No speculation about what the module might do — describe what it does."""


class FIRE_SELF_STUDY(Lever):
    name = "FIRE_SELF_STUDY"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=45.0, tokens_in=6000, tokens_out=1500)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        backend_root = Path(params.get("backend_root") or DEFAULT_BACKEND_ROOT)
        self_root = Path(params.get("self_root") or DEFAULT_SELF_ROOT)
        max_modules = int(params.get("max_modules", 3))
        modules_dir = self_root / "modules"
        modules_dir.mkdir(parents=True, exist_ok=True)

        candidates = _find_modules(backend_root)
        total = len(candidates)
        targets = _select_targets(candidates, modules_dir, backend_root, max_modules)

        processed = 0
        skipped = 0
        for module_path in targets:
            try:
                self._study_one(module_path, backend_root, modules_dir)
                processed += 1
            except Exception as exc:
                log.warning("self_study: failed %s: %s", module_path, exc)
                skipped += 1

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "modules_processed": processed,
                "modules_skipped": skipped,
                "modules_total": total,
            },
            reason=f"studied_{processed}_modules",
        )

    def _study_one(self, module_path: Path, backend_root: Path, modules_dir: Path) -> None:
        rel = module_path.relative_to(backend_root)
        slug = _slug_from_relpath(rel)

        src = module_path.read_text(encoding="utf-8")
        lines = src.splitlines()
        truncated = False
        if len(lines) > MAX_LINES:
            src = "\n".join(lines[:MAX_LINES]) + "\n# [truncated]"
            truncated = True
        loc = min(len(lines), MAX_LINES)

        data = router().call_json(
            TaskType.TASK_ANALYSIS,
            SELF_STUDY_SYSTEM,
            src,
            max_tokens=1500,
            temperature=0.2,
        )

        mtime_iso = datetime.fromtimestamp(module_path.stat().st_mtime, timezone.utc).isoformat()
        note = _render_module_note(rel.as_posix(), loc, mtime_iso, data, truncated)
        (modules_dir / f"{slug}.md").write_text(note, encoding="utf-8")


def _find_modules(backend_root: Path) -> list[Path]:
    if not backend_root.exists():
        return []
    out: list[Path] = []
    for py in backend_root.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        parts = py.relative_to(backend_root).parts
        if any(p in SKIP_DIR_NAMES for p in parts):
            continue
        out.append(py)
    return sorted(out)


def _slug_from_relpath(rel: Path) -> str:
    stem_parts = list(rel.with_suffix("").parts)
    return "__".join(stem_parts)


_MTIME_RE = re.compile(r"^source_mtime:\s*(.+)$", re.MULTILINE)
_UPDATED_RE = re.compile(r"^updated:\s*(.+)$", re.MULTILINE)


def _select_targets(
    candidates: list[Path],
    modules_dir: Path,
    backend_root: Path,
    max_modules: int,
) -> list[Path]:
    if not candidates:
        return []
    new_mods: list[Path] = []
    stale_mods: list[tuple[Path, float]] = []
    old_mods: list[tuple[Path, str]] = []

    for module_path in candidates:
        rel = module_path.relative_to(backend_root)
        slug = _slug_from_relpath(rel)
        note_path = modules_dir / f"{slug}.md"
        if not note_path.exists():
            new_mods.append(module_path)
            continue
        try:
            header = note_path.read_text(encoding="utf-8")[:2000]
        except OSError:
            new_mods.append(module_path)
            continue
        mtime_match = _MTIME_RE.search(header)
        updated_match = _UPDATED_RE.search(header)
        note_mtime_iso = mtime_match.group(1).strip() if mtime_match else ""
        updated_iso = updated_match.group(1).strip() if updated_match else ""
        file_mtime = module_path.stat().st_mtime
        try:
            note_mtime_ts = datetime.fromisoformat(note_mtime_iso).timestamp() if note_mtime_iso else 0.0
        except ValueError:
            note_mtime_ts = 0.0
        if note_mtime_ts < file_mtime:
            stale_mods.append((module_path, file_mtime - note_mtime_ts))
        else:
            old_mods.append((module_path, updated_iso))

    stale_mods.sort(key=lambda t: t[1], reverse=True)
    old_mods.sort(key=lambda t: t[1])

    ordered = new_mods + [p for p, _ in stale_mods] + [p for p, _ in old_mods]
    return ordered[:max_modules]


def _render_module_note(
    module_relpath: str,
    loc: int,
    mtime_iso: str,
    data: dict[str, Any],
    truncated: bool,
) -> str:
    updated_iso = utcnow().isoformat()
    purpose = str(data.get("purpose", "")).strip() or "(no purpose returned)"
    notes = str(data.get("notes", "")).strip()
    deps = [str(d) for d in (data.get("dependencies") or [])]
    interface = data.get("public_interface") or []

    lines = [
        "---",
        f"module: backend/{module_relpath}",
        "category: self",
        "kind: module",
        f"updated: {updated_iso}",
        f"source_mtime: {mtime_iso}",
        f"loc: {loc}",
        f"truncated: {str(truncated).lower()}",
        "---",
        "",
        f"# backend/{module_relpath}",
        "",
        "## Purpose",
        purpose,
        "",
        "## Public interface",
    ]
    if interface:
        for item in interface[:12]:
            name = str(item.get("name", ""))
            kind = str(item.get("kind", ""))
            one_line = str(item.get("one_line", "")).strip()
            lines.append(f"- `{name}` ({kind}) — {one_line}")
    else:
        lines.append("(none)")
    lines.extend([
        "",
        "## Dependencies",
    ])
    if deps:
        lines.extend(f"- {d}" for d in deps)
    else:
        lines.append("(none)")
    lines.extend([
        "",
        "## Notes",
        notes if notes else "(none)",
        "",
    ])
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_self_knowledge_levers.py -v`

Expected: 17 tests PASS (10 from CAPABILITY_SCAN + 7 from SELF_STUDY).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/self_study.py tests/autonomic/test_self_knowledge_levers.py
git commit -m "feat(autonomic): FIRE_SELF_STUDY lever with priority-ordered module processing"
```

---

## Task 6: Register autonomic levers + extend default_rules()

**Files:**
- Modify: `backend/autonomic/levers/__init__.py`
- Modify: `backend/autonomic/layer0.py`
- Test: extend `tests/autonomic/test_registry.py`
- Test: extend `tests/autonomic/test_layer0.py`

- [ ] **Step 1: Append failing registry test**

Append to `tests/autonomic/test_registry.py`:

```python
def test_autonomic_levers_include_d04_cohort():
    from backend.autonomic.levers import register_default_autonomic_levers
    clear_registry()
    register_default_autonomic_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    # D-03 levers
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    # D-04 levers
    assert "FIRE_CAPABILITY_SCAN" in names
    assert "FIRE_SELF_STUDY" in names
    clear_registry()
```

- [ ] **Step 2: Append failing layer0 tests**

Append to `tests/autonomic/test_layer0.py`:

```python
def test_default_rules_has_nine_rules_after_d04():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    assert len(rules) == 9


def test_default_rules_d04_scheduled_rules_at_end():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    names_tail = [r.name for r in rules[-2:]]
    assert names_tail == ["capability_scan_tick", "self_study_tick"]


def test_default_rules_d04_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["capability_scan_tick"].lever == "FIRE_CAPABILITY_SCAN"
    assert rules["capability_scan_tick"].cooldown_seconds == 21600.0
    assert rules["self_study_tick"].lever == "FIRE_SELF_STUDY"
    assert rules["self_study_tick"].cooldown_seconds == 86400.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py tests/autonomic/test_layer0.py -v -k "d04 or nine"`

Expected: FAIL — names missing + only 7 rules.

- [ ] **Step 4: Extend `register_default_autonomic_levers()` in `backend/autonomic/levers/__init__.py`**

Replace the function body with:

```python
def register_default_autonomic_levers() -> None:
    from .capability_scan import FIRE_CAPABILITY_SCAN
    from .goal_propose import FIRE_GOAL_PROPOSE
    from .integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from .memory_consolidation import FIRE_MEMORY_CONSOLIDATION
    from .self_study import FIRE_SELF_STUDY
    register_lever(FIRE_INTEGRITY_HEARTBEAT)
    register_lever(FIRE_GOAL_PROPOSE)
    register_lever(FIRE_MEMORY_CONSOLIDATION)
    register_lever(FIRE_CAPABILITY_SCAN)
    register_lever(FIRE_SELF_STUDY)
```

- [ ] **Step 5: Extend `default_rules()` in `backend/autonomic/layer0.py`**

Append two rules inside the existing `default_rules()` list, after the `consolidation_tick` rule:

```python
        LayerZeroRule(
            name="capability_scan_tick",
            predicate=lambda s: True,
            lever="FIRE_CAPABILITY_SCAN",
            params={},
            cooldown_seconds=21600.0,
        ),
        LayerZeroRule(
            name="self_study_tick",
            predicate=lambda s: True,
            lever="FIRE_SELF_STUDY",
            params={},
            cooldown_seconds=86400.0,
        ),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py tests/autonomic/test_layer0.py tests/autonomic/test_startup_hook.py -v`

Expected: all pass. The existing `test_build_scheduler_registers_all_d03_levers` still passes (it checks D-01..D-03 levers).

- [ ] **Step 7: Smoke-test FastAPI app import**

Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"`

Expected: prints `Self-Learning Agent` with no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/autonomic/levers/__init__.py backend/autonomic/layer0.py tests/autonomic/test_registry.py tests/autonomic/test_layer0.py
git commit -m "feat(autonomic): register D-04 levers and extend default_rules() to 9"
```

---

## Task 7: End-to-end D-04 integration test

**Files:**
- Create: `tests/autonomic/test_d04_integration.py`

- [ ] **Step 1: Write integration tests**

Create `tests/autonomic/test_d04_integration.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.layer0 import Layer0Engine, default_rules
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
from backend.autonomic.safety import SafetyGate
from backend.autonomic.state import StateSnapshotBuilder
from backend.autonomic.tick import make_real_tick
from backend.autonomic.types import LeverReport


@pytest.fixture(autouse=True)
def _reg():
    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
    yield
    clear_registry()


def _fake_study() -> dict:
    return {
        "purpose": "test module purpose",
        "public_interface": [{"name": "X", "kind": "constant", "one_line": "ok"}],
        "dependencies": [],
        "notes": "ok",
    }


def _build_tick(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    lever_log = tmp_path / "lever_log.jsonl"
    tick_log = tmp_path / "tick_log.jsonl"
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=lever_log,
    )
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )
    return tick, lever_log


def _isolate_paths(tmp_path: Path):
    """Return patch helpers for capability_scan & self_study that inject tmp paths."""
    from backend.autonomic.levers.capability_scan import FIRE_CAPABILITY_SCAN
    from backend.autonomic.levers.self_study import FIRE_SELF_STUDY

    backend_root = tmp_path / "fake_backend"
    tools_dir = backend_root / "tools"
    skills_dir = backend_root / "skills"
    tools_dir.mkdir(parents=True)
    skills_dir.mkdir(parents=True)
    (backend_root / "__init__.py").write_text("", encoding="utf-8")
    (backend_root / "sample.py").write_text('"""sample module."""\n', encoding="utf-8")
    (tools_dir / "my_tool.py").write_text('"""my tool."""\n', encoding="utf-8")
    channels_path = tmp_path / "channels.json"
    channels_path.write_text(json.dumps({"channels": []}), encoding="utf-8")
    self_root = tmp_path / "self"

    orig_scan = FIRE_CAPABILITY_SCAN.run
    orig_study = FIRE_SELF_STUDY.run

    def scan_wrap(self, params, context):
        p = dict(params)
        p.setdefault("tools_dir", str(tools_dir))
        p.setdefault("skills_dir", str(skills_dir))
        p.setdefault("channels_path", str(channels_path))
        p.setdefault("self_root", str(self_root))
        return orig_scan(self, p, context)

    def study_wrap(self, params, context):
        p = dict(params)
        p.setdefault("backend_root", str(backend_root))
        p.setdefault("self_root", str(self_root))
        p.setdefault("max_modules", 3)
        return orig_study(self, p, context)

    return FIRE_CAPABILITY_SCAN, FIRE_SELF_STUDY, scan_wrap, study_wrap, self_root


def test_two_ticks_fire_capability_scan_then_self_study(tmp_path: Path):
    tick, lever_log = _build_tick(tmp_path)
    scan_cls, study_cls, scan_wrap, study_wrap, self_root = _isolate_paths(tmp_path)

    # Pre-fire the three D-03 scheduled rules' levers so only D-04 rules are fresh
    # on subsequent ticks. Simplest way: seed sessions.json with consolidated=True,
    # empty gaps.json, and an index.json. integrity_tick and goal_propose_tick
    # will still match+fire on tick 1 (we want D-04 rules on tick 4-5). Instead,
    # use a patched engine where only D-04 scheduled rules exist.
    from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule
    d04_only = [
        LayerZeroRule(name="capability_scan_tick", predicate=lambda s: True,
                      lever="FIRE_CAPABILITY_SCAN", params={}, cooldown_seconds=21600.0),
        LayerZeroRule(name="self_study_tick", predicate=lambda s: True,
                      lever="FIRE_SELF_STUDY", params={}, cooldown_seconds=86400.0),
    ]
    # Rebuild tick with D-04-only engine
    from backend.autonomic.events import EventBus
    from backend.autonomic.executor import LeverExecutor
    from backend.autonomic.safety import SafetyGate
    from backend.autonomic.state import StateSnapshotBuilder
    from backend.autonomic.tick import make_real_tick
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=lever_log,
    )
    engine2 = Layer0Engine(rules=d04_only)
    tick = make_real_tick(
        builder=builder, engine=engine2, registry=LeverRegistry.instance(),
        executor=execu, tick_log_path=tmp_path / "tick_log.jsonl", event_bus=bus,
    )

    with patch.object(scan_cls, "run", scan_wrap), \
         patch.object(study_cls, "run", study_wrap), \
         patch("backend.autonomic.levers.self_study.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_study()
        tick()
        tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_CAPABILITY_SCAN", "FIRE_SELF_STUDY"]

    # Capability scan wrote files
    assert (self_root / "server_inventory.md").exists()
    assert (self_root / "tools" / "my_tool.md").exists()
    # Self-study wrote at least one module note
    module_notes = list((self_root / "modules").glob("*.md"))
    assert len(module_notes) >= 1


def test_reactive_rule_preempts_d04_scheduled(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path)
    tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_capability_scan_produces_all_four_artifacts_in_end_to_end(tmp_path: Path):
    """Smoke: CAPABILITY_SCAN alone writes tools + skills + channels + server inventory."""
    scan_cls, _study_cls, scan_wrap, _study_wrap, self_root = _isolate_paths(tmp_path)

    tick, lever_log = _build_tick(tmp_path)
    from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule
    scan_only = [
        LayerZeroRule(name="capability_scan_tick", predicate=lambda s: True,
                      lever="FIRE_CAPABILITY_SCAN", params={}, cooldown_seconds=21600.0),
    ]
    from backend.autonomic.events import EventBus
    from backend.autonomic.executor import LeverExecutor
    from backend.autonomic.safety import SafetyGate
    from backend.autonomic.state import StateSnapshotBuilder
    from backend.autonomic.tick import make_real_tick
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=lever_log,
    )
    tick = make_real_tick(
        builder=builder,
        engine=Layer0Engine(rules=scan_only),
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tmp_path / "tick_log.jsonl",
        event_bus=bus,
    )

    with patch.object(scan_cls, "run", scan_wrap):
        tick()

    assert (self_root / "tools").exists()
    assert (self_root / "server_inventory.md").exists()
    assert (self_root / "mcp_servers" / "channels.md").exists()
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_d04_integration.py -v`

Expected: 3 tests PASS.

- [ ] **Step 3: Run full autonomic suite**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q`

Expected: ~159 tests PASS (139 from D-03 + 20 new).

- [ ] **Step 4: Commit**

```bash
git add tests/autonomic/test_d04_integration.py
git commit -m "test(autonomic): D-04 end-to-end — capability_scan + self_study tick chain"
```

---

## Task 8: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Extend the autonomic subsection**

In `README.md`, the existing `_Autonomic levers (D-03, scheduled self-maintenance):_` block lists three levers. Add a new cohort block right below it:

```markdown
_Self-knowledge levers (D-04):_
- `FIRE_CAPABILITY_SCAN` — every 6h, inventories `backend/tools/`, `backend/skills/`, `knowledge/channels.json`, and the host via psutil into `knowledge/self/` (green, python).
- `FIRE_SELF_STUDY` — daily, reads up to 3 priority-ordered `backend/**/*.py` modules per tick and writes one markdown note per module to `knowledge/self/modules/` via cortex (green, claude).
```

Also append to the **Paths:** list:

```markdown
- Self-knowledge: `knowledge/self/modules/`, `knowledge/self/tools/`, `knowledge/self/skills/`, `knowledge/self/mcp_servers/`, `knowledge/self/server_inventory.md` (written by D-04 levers).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README lists D-04 self-knowledge levers + paths"
```

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q` — all pass (~159 tests).
- [ ] Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"` — prints `Self-Learning Agent`.
- [ ] `GET /api/autonomic/status` — `registered_levers` list includes all 9 lever names.
- [ ] Start FastAPI: `uvicorn backend.main:app --reload`. After ~30s, check `knowledge/autonomic/tick_log.jsonl` — `FIRE_CAPABILITY_SCAN` should have fired once and populated `knowledge/self/` with `server_inventory.md` plus `tools/`, `skills/`, `mcp_servers/` subdirs.
- [ ] After 24h+ of runtime (or with `AUTONOMIC_TICK_SECONDS=0.1` for smoke): `knowledge/self/modules/` starts receiving module notes, up to 3 per daily tick.

If all pass, D-04 is done. Proceed to D-05 (`FIRE_NOTE_CURATION` + `FIRE_GRAPH_MAINTENANCE` + `backend/background.py` retirement).

---

## Out of scope for D-04

Explicitly NOT in this plan (belongs to later D plans):

- `FIRE_NOTE_CURATION` — D-05.
- `FIRE_GRAPH_MAINTENANCE` — D-05.
- Retirement of `backend/background.py` — D-05.
- bge-m3 embedding index over `knowledge/self/modules/` + `hybrid_searcher` integration — D-05 or D-06.
- `FIRE_MODEL_EVAL`, `FIRE_SESSION_ARCHIVE`, `FIRE_COST_AUDIT`, `FIRE_SELF_REFLECTION`, `FIRE_FINETUNE_QC`, `FIRE_GAP_DETECTION` — D-06.
- OS-level Linux inventory (`dpkg -l`, `systemctl list-units`, `ollama list`, `ss -tlnp`) — D-07 body cohort, after Linux deploy.
- Auto-generated `knowledge/self/architecture_overview.md` — manual for now.
- `FIRE_TOOL_INSTALL` (yellow), AutonomicPanel frontend — D-07.
