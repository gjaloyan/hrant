"""FIRE_GAP_DETECTION — daily aggregate of knowledge/gaps.json."""
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

DEFAULT_GAPS_PATH = Path("knowledge/gaps.json")
DEFAULT_LOG_PATH = Path("knowledge/autonomic/gap_detection_log.jsonl")
STALE_DAYS = 30
ACTIONABLE_THRESHOLD = 2
HOT_LIMIT = 5


class FIRE_GAP_DETECTION(Lever):
    name = "FIRE_GAP_DETECTION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.1)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        gaps_path = resolve_knowledge_path(params.get("gaps_path") or DEFAULT_GAPS_PATH)
        log_path = resolve_knowledge_path(params.get("log_path") or DEFAULT_LOG_PATH)
        actionable_threshold = int(params.get("actionable_threshold", ACTIONABLE_THRESHOLD))

        if not gaps_path.exists():
            return self._skip(params, started, "no_gaps")
        try:
            data = json.loads(gaps_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._skip(params, started, "no_gaps")
        if not isinstance(data, dict) or not data:
            return self._skip(params, started, "no_gaps")

        cutoff = datetime.now() - timedelta(days=STALE_DAYS)
        total = 0
        actionable = 0
        stale = 0
        entries: list[dict] = []
        for slug, entry in data.items():
            if not isinstance(entry, dict):
                continue
            total += 1
            count = int(entry.get("count", 0) or 0)
            if count >= actionable_threshold:
                actionable += 1
            last_str = str(entry.get("last", ""))
            try:
                last_dt = datetime.strptime(last_str, "%Y-%m-%d %H:%M")
                if last_dt < cutoff:
                    stale += 1
            except ValueError:
                pass
            entries.append({
                "topic": str(entry.get("topic", slug)),
                "count": count,
                "last": last_str,
            })

        entries.sort(key=lambda e: e["count"], reverse=True)
        hot = entries[:HOT_LIMIT]

        snapshot = {
            "ts": utcnow().isoformat(),
            "total": total,
            "actionable": actionable,
            "stale": stale,
            "hot": hot,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "total_gaps": total,
                "actionable_gaps": actionable,
                "stale_gaps": stale,
                "hot_count": len(hot),
            },
            reason=f"detected_{total}_gaps",
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
