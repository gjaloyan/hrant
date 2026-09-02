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


# How far back "recent" reaches. Long enough that a burst between two ticks
# is still seen together, short enough that the `errors_present` reflex can
# switch off again once things are quiet.
RECENT_ERROR_WINDOW_SECONDS = 3600.0

# The writers stamp local time as "YYYY-MM-DD HH:MM:SS"; older rows also
# appear with an ISO "T" separator.
_ERROR_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _entry_epoch(entry: dict[str, Any]) -> float | None:
    """When an error-log row was written, or None if it cannot be dated."""
    raw = entry.get("ts") or entry.get("timestamp")
    if not isinstance(raw, (str, int, float)):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = raw.strip()
    for fmt in _ERROR_TS_FORMATS:
        try:
            return datetime.strptime(text[:19], fmt).timestamp()
        except ValueError:
            continue
    try:
        # Anything else with an offset, e.g. "...+04:00".
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


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
            kb_graph_edges=self._count_graph_edges(),
            failed_services=self._failed_services(),
            unconsolidated_sessions=self._unconsolidated_sessions(),
        )

    def _unconsolidated_sessions(self) -> int:
        """Ended sessions not yet folded into memory.

        Cheap and defensive: a missing or unreadable sessions.json reads as
        zero, i.e. "nothing to do", so a bad file can never make the agent
        run an LLM consolidation pass on every tick.
        """
        import json as _json
        p = self._knowledge_root / "sessions.json"
        try:
            blob = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return 0
        try:
            return sum(
                1 for s in (blob.get("sessions") or [])
                if not s.get("consolidated") and s.get("turns")
            )
        except Exception:
            return 0

    def _failed_services(self) -> list[str]:
        """Units in the `failed` state, in both systemd managers.

        `failed` means systemd has EXHAUSTED its own retries and stopped —
        the one service condition nothing else on the box recovers from. A
        crash-looping unit is deliberately NOT reported: it shows as
        `activating (auto-restart)`, systemd is on it, and a lever restart
        would only add contention (prod has four of those right now, one at
        174k restarts).

        Never raises and never blocks a tick: a missing systemctl, a slow
        manager or a non-Linux host all yield an empty list, which reads as
        "nothing to repair".
        """
        import subprocess as _sp
        import sys as _sys
        if not _sys.platform.startswith("linux"):
            return []
        out: list[str] = []
        for manager, args in (("user", ["--user"]), ("system", [])):
            try:
                r = _sp.run(
                    ["systemctl", *args, "list-units", "--state=failed",
                     "--no-legend", "--plain", "--no-pager"],
                    capture_output=True, text=True, timeout=5,
                )
            except Exception:
                continue
            if r.returncode != 0:
                continue
            for line in (r.stdout or "").splitlines():
                unit = line.split()[0] if line.split() else ""
                if unit.endswith(".service"):
                    out.append(f"{manager}:{unit}")
        return out

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
        """The faults from the last `RECENT_ERROR_WINDOW_SECONDS`.

        This used to be the last ten lines of the file whatever their age.
        error_log.jsonl is append-only and also collects low-confidence
        answers, so one such answer left `errors_present` true forever:
        measured on prod 2026-09-02 the newest entry was eight hours old
        and FIRE_ERROR_TRIAGE had re-run every 120 seconds since, taking
        about a quarter of the tick budget to re-read the same ten rows.

        Both callers -- the `errors_present` rule and the triage lever --
        are asking "is something wrong right now", so the window is the
        answer they were already assuming.
        """
        if not self._error_log_path.exists():
            return []
        cutoff = time.time() - RECENT_ERROR_WINDOW_SECONDS
        tail: deque[dict[str, Any]] = deque(maxlen=self._recent_errors_limit)
        with self._error_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                when = _entry_epoch(entry)
                # An entry nothing can date cannot be shown to be recent.
                # Trusting it is exactly what armed the rule permanently.
                if when is None or when < cutoff:
                    continue
                tail.append(entry)
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

    def _count_graph_edges(self) -> int:
        """Entities carrying at least one link.

        Same file as the node count, different key, because the two halves
        are written by different subsystems and can be healthy or empty
        independently.
        """
        graph_path = self._knowledge_root / "graph.json"
        if not graph_path.exists():
            return 0
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0
        edges = data.get("edges") if isinstance(data, dict) else None
        return len(edges) if hasattr(edges, "__len__") else 0

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
