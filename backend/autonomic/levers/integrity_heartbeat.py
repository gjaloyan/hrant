"""FIRE_INTEGRITY_HEARTBEAT — read-only integrity check against knowledge/index.json."""
from __future__ import annotations

import json
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

DEFAULT_KNOWLEDGE_ROOT = Path("knowledge")
EXCLUDED_DIRS = {"_history", "autonomic", "immune", "identity"}


class FIRE_INTEGRITY_HEARTBEAT(Lever):
    name = "FIRE_INTEGRITY_HEARTBEAT"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.1)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        root_param = params.get("knowledge_root")
        root = Path(root_param) if root_param else DEFAULT_KNOWLEDGE_ROOT

        index_path = root / "index.json"
        index: dict[str, Any] = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if not isinstance(index, dict):
                    index = {}
            except json.JSONDecodeError:
                index = {}

        files_on_disk: set[str] = set()
        if root.exists():
            for md in root.rglob("*.md"):
                rel = md.relative_to(root)
                if rel.parts and rel.parts[0] in EXCLUDED_DIRS:
                    continue
                files_on_disk.add(rel.as_posix())

        index_keys = set(index.keys())
        orphan_files = sorted(files_on_disk - index_keys)
        dead_entries = sorted(index_keys - files_on_disk)

        issues = len(orphan_files) + len(dead_entries)
        reason = "integrity_ok" if issues == 0 else f"integrity_drift:{issues}_issues"

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "index_count": len(index_keys),
                "file_count": len(files_on_disk),
                "orphan_files": orphan_files,
                "dead_entries": dead_entries,
            },
            reason=reason,
        )
