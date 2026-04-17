"""FIRE_CAPABILITY_SCAN — inventory tools, skills, channels, and server into knowledge/self/."""
from __future__ import annotations

import ast
import json
import logging
import platform
from datetime import date, datetime, timezone
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
