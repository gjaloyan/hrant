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
  "notes": "optional observations about complexity, invariants, gotchas - 2-3 sentences"
}

Rules:
- Skip __dunder__ helpers unless they define the public API.
- "dependencies" lists imports from the same project (backend.*), not stdlib or third-party.
- Max 12 entries in public_interface.
- No speculation about what the module might do - describe what it does."""


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
            lines.append(f"- `{name}` ({kind}) - {one_line}")
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
