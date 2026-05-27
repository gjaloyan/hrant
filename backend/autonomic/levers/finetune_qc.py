"""FIRE_FINETUNE_QC — daily audit of finetune_queue.jsonl scoring distribution."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

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

from backend.finetune_curator import FinetuneDataCurator
from backend.models import FinetunePair

log = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = Path("knowledge/finetune_queue.jsonl")
DEFAULT_LOG_PATH = Path("knowledge/autonomic/finetune_qc_log.jsonl")


class FIRE_FINETUNE_QC(Lever):
    name = "FIRE_FINETUNE_QC"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.5)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        queue_path = resolve_knowledge_path(params.get("queue_path") or DEFAULT_QUEUE_PATH)
        log_path = resolve_knowledge_path(params.get("log_path") or DEFAULT_LOG_PATH)

        if not queue_path.exists():
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"total": 0, "legacy_entries": 0},
                reason="no_valid_pairs",
            )

        pairs: list[FinetunePair] = []
        legacy_entries = 0
        for raw in queue_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                pairs.append(FinetunePair.model_validate_json(line))
            except (ValidationError, ValueError):
                legacy_entries += 1

        if not pairs:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"total": 0, "legacy_entries": legacy_entries},
                reason="no_valid_pairs",
            )

        curator = FinetuneDataCurator()
        scored = curator.score_all(pairs)
        low = sum(1 for s in scored if s.score < 0.5)
        medium = sum(1 for s in scored if 0.5 <= s.score < 0.7)
        high = sum(1 for s in scored if s.score >= 0.7)
        avg_score = sum(s.score for s in scored) / len(scored) if scored else 0.0

        by_category: Counter[str] = Counter()
        boosted = 0
        verified = 0
        for p in pairs:
            by_category[p.metadata.category] += 1
            if p.metadata.boosted:
                boosted += 1
            if p.metadata.verified:
                verified += 1

        curated = curator.curate(pairs)

        snapshot = {
            "ts": utcnow().isoformat(),
            "total": len(pairs),
            "legacy_entries": legacy_entries,
            "low": low,
            "medium": medium,
            "high": high,
            "curated": len(curated),
            "boosted": boosted,
            "verified": verified,
            "avg_score": round(avg_score, 3),
            "by_category": dict(by_category),
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
                "total": len(pairs),
                "curated": len(curated),
                "avg_score": round(avg_score, 3),
                "legacy_entries": legacy_entries,
            },
            reason=f"qc_{len(pairs)}_pairs",
        )
