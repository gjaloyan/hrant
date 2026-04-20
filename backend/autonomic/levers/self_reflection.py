"""FIRE_SELF_REFLECTION — nightly failure-pattern extraction via META_LEARNER."""
from __future__ import annotations

import json
import logging
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

from backend.meta_learner import META_LEARNER

log = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("knowledge/autonomic/self_reflection_log.jsonl")
MIN_FAILURES = 3


class FIRE_SELF_REFLECTION(Lever):
    name = "FIRE_SELF_REFLECTION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=30.0, tokens_in=2000, tokens_out=800)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        log_path = Path(params.get("log_path") or DEFAULT_LOG_PATH)

        try:
            stats = META_LEARNER.stats()
        except Exception as exc:
            log.warning("self_reflection: stats failed: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={},
                reason=f"reflect_failed:stats:{exc}",
            )

        total_failures = int(stats.get("total_failures", 0) or 0)
        if total_failures < MIN_FAILURES:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"total_failures": total_failures},
                reason="insufficient_failures",
            )

        try:
            patterns = META_LEARNER.extract_patterns() or []
        except Exception as exc:
            log.warning("self_reflection: extract_patterns failed: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"total_failures": total_failures},
                reason=f"reflect_failed:extract:{exc}",
            )

        snapshot = {
            "ts": utcnow().isoformat(),
            "total_failures": total_failures,
            "by_root_cause": stats.get("by_root_cause", {}),
            "by_domain": stats.get("by_domain", {}),
            "avg_severity": stats.get("avg_severity", 0.0),
            "patterns_count": len(patterns),
            "patterns": patterns,
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
                "total_failures": total_failures,
                "avg_severity": stats.get("avg_severity", 0.0),
                "patterns_count": len(patterns),
            },
            reason=f"reflected_on_{total_failures}_failures",
        )
