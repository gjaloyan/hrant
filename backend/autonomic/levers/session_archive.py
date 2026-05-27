"""FIRE_SESSION_ARCHIVE — move old consolidated sessions to knowledge/_history/."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..lever import Lever, resolve_knowledge_path
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

DEFAULT_SESSIONS_PATH = Path("knowledge/sessions.json")
DEFAULT_HISTORY_DIR = Path("knowledge/_history")
SESSION_ARCHIVE_DAYS = 30


class FIRE_SESSION_ARCHIVE(Lever):
    name = "FIRE_SESSION_ARCHIVE"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.3)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        sessions_path = resolve_knowledge_path(params.get("sessions_path") or DEFAULT_SESSIONS_PATH)
        history_dir = resolve_knowledge_path(params.get("history_dir") or DEFAULT_HISTORY_DIR)
        max_per_tick = int(params.get("max_per_tick", 10))
        cutoff_days = int(params.get("cutoff_days", SESSION_ARCHIVE_DAYS))

        if not sessions_path.exists():
            return self._skip(params, started, "no_old_sessions")

        try:
            blob = json.loads(sessions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._skip(params, started, "no_old_sessions")
        if not isinstance(blob, dict):
            return self._skip(params, started, "no_old_sessions")

        sessions = blob.get("sessions", [])
        current_id = blob.get("current_id")
        cutoff = datetime.now() - timedelta(days=cutoff_days)

        candidates: list[dict] = []
        for s in sessions:
            if not isinstance(s, dict):
                continue
            if s.get("archived"):
                continue
            if not s.get("consolidated"):
                continue
            if s.get("id") == current_id:
                continue
            ended_str = str(s.get("ended", ""))
            try:
                ended_dt = datetime.strptime(ended_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if ended_dt >= cutoff:
                continue
            candidates.append(s)

        if not candidates:
            return self._skip(params, started, "no_old_sessions")

        candidates.sort(key=lambda s: s.get("ended", ""))
        targets = candidates[:max_per_tick]

        history_dir.mkdir(parents=True, exist_ok=True)
        archived_ids: set[str] = set()
        for s in targets:
            sid = str(s.get("id", ""))
            if not sid:
                continue
            out_path = history_dir / f"{sid}.json"
            out_path.write_text(
                json.dumps(s, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            archived_ids.add(sid)

        remaining = [s for s in sessions if s.get("id") not in archived_ids]
        blob["sessions"] = remaining
        tmp = sessions_path.with_suffix(sessions_path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(sessions_path)

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "archived": len(archived_ids),
                "remaining_active": len(remaining),
                "cutoff_date": cutoff.strftime("%Y-%m-%d"),
            },
            reason=f"archived_{len(archived_ids)}_sessions",
        )

    def _skip(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
