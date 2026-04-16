"""Build a StateSnapshot from live system + filesystem readings."""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from .types import LeverReport, StateSnapshot

_APP_STARTED_MONOTONIC = time.monotonic()


class StateSnapshotBuilder:
    def __init__(
        self,
        knowledge_root: Path,
        error_log_path: Path,
        pending_approvals_path: Path,
        lever_log_path: Path,
        recent_errors_limit: int = 10,
    ) -> None:
        self._knowledge_root = knowledge_root
        self._error_log_path = error_log_path
        self._pending_approvals_path = pending_approvals_path
        self._lever_log_path = lever_log_path
        self._recent_errors_limit = recent_errors_limit

    def build(self) -> StateSnapshot:
        return StateSnapshot(
            taken_at=datetime.now(timezone.utc),
            uptime_seconds=time.monotonic() - _APP_STARTED_MONOTONIC,
            disk_free_gb=self._disk_free_gb(),
            memory_free_gb=self._memory_free_gb(),
            cpu_load_1m=self._cpu_load(),
            last_run=self._last_run_by_lever(),
            recent_errors=self._recent_errors(),
            pending_approvals=self._pending_count(),
            kb_notes_count=self._count_notes(),
            kb_graph_nodes=self._count_graph_nodes(),
        )

    def _disk_free_gb(self) -> float:
        target = str(self._knowledge_root.resolve()) if self._knowledge_root.exists() else "."
        usage = psutil.disk_usage(target)
        return usage.free / (1024 ** 3)

    def _memory_free_gb(self) -> float:
        return psutil.virtual_memory().available / (1024 ** 3)

    def _cpu_load(self) -> float:
        try:
            load1, _, _ = psutil.getloadavg()
            return float(load1)
        except (AttributeError, OSError):
            return psutil.cpu_percent(interval=None) / 100.0

    def _pending_count(self) -> int:
        if not self._pending_approvals_path.exists():
            return 0
        return sum(
            1
            for line in self._pending_approvals_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def _recent_errors(self) -> list[dict[str, Any]]:
        if not self._error_log_path.exists():
            return []
        tail: deque[dict[str, Any]] = deque(maxlen=self._recent_errors_limit)
        with self._error_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tail.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(tail)

    def _last_run_by_lever(self) -> dict[str, datetime]:
        if not self._lever_log_path.exists():
            return {}
        last: dict[str, datetime] = {}
        with self._lever_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    report = LeverReport.from_jsonl(line)
                except Exception:
                    continue
                prev = last.get(report.lever)
                if prev is None or report.finished_at > prev:
                    last[report.lever] = report.finished_at
        return last

    def _count_notes(self) -> int:
        if not self._knowledge_root.exists():
            return 0
        return sum(1 for p in self._knowledge_root.rglob("*.md"))

    def _count_graph_nodes(self) -> int:
        graph_path = self._knowledge_root / "graph.json"
        if not graph_path.exists():
            return 0
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0
        if isinstance(data, dict) and "nodes" in data:
            return len(data["nodes"])
        return 0
