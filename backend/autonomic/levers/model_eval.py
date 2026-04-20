"""FIRE_MODEL_EVAL — daily aggregation of eval_log.jsonl into model_eval_log.jsonl."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
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

from backend.evaluator import EVALUATOR

log = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("knowledge/autonomic/model_eval_log.jsonl")


class FIRE_MODEL_EVAL(Lever):
    name = "FIRE_MODEL_EVAL"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.2)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        log_path = Path(params.get("log_path") or DEFAULT_LOG_PATH)
        target_date = params.get("target_date") or (
            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        )

        try:
            daily = EVALUATOR.daily_report(target_date)
        except Exception as exc:
            log.warning("model_eval: daily_report failed: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"date": target_date},
                reason="eval_log_read_failed",
            )

        total = int(daily.get("total_interactions", 0) or 0)
        if total == 0:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"date": target_date},
                reason="no_eval_entries",
            )

        try:
            regressions = EVALUATOR.detect_regression() or []
        except Exception as exc:
            log.warning("model_eval: detect_regression failed: %s", exc)
            regressions = []

        try:
            priorities = EVALUATOR.suggest_priorities() or []
        except Exception as exc:
            log.warning("model_eval: suggest_priorities failed: %s", exc)
            priorities = []

        snapshot = {
            "ts": utcnow().isoformat(),
            "date": target_date,
            "daily_report": daily,
            "regressions": regressions,
            "priorities": priorities,
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
                "date": target_date,
                "total": total,
                "avg_confidence": daily.get("avg_confidence", 0),
                "regressions_count": len(regressions),
                "priorities_count": len(priorities),
            },
            reason=f"evaluated_{total}_entries",
        )
