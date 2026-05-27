"""FIRE_FACT_EMBEDDING_BACKFILL — keep memory_facts.jsonl
embeddings in sync with the file.

Audit 2026-05-27: 1453 facts in memory_facts.jsonl, 0 with
vectors. The new FACT_VECTOR_STORE (backend/fact_search.py) is
populated by this lever. Runs daily by default; the consolidation
pipeline ALSO embeds new facts on the spot for fresh ones, so the
lever mostly catches drift after embedder outages or schema
swaps.
"""
from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


class FIRE_FACT_EMBEDDING_BACKFILL(Lever):
    name = "FIRE_FACT_EMBEDDING_BACKFILL"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "local"
    estimated_cost = Cost(seconds=30.0, tokens_in=0, tokens_out=0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        from backend import fact_search as fs

        try:
            missing = fs.count_unembedded_facts()
        except Exception as exc:
            return self._skip(params, started, f"coverage_check_failed: {exc}")

        force = bool(params.get("force"))
        if missing == 0 and not force:
            return self._skip(
                params, started,
                f"nothing_to_backfill (store={fs.get_store().count()})",
            )

        try:
            stats = fs.backfill_fact_embeddings(force=force)
        except Exception as exc:
            log.warning("fact_embedding_backfill: raised: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"exception": str(exc)},
                reason=f"backfill_failed: {exc}",
            )

        if not stats.get("ok"):
            return self._skip(
                params, started,
                stats.get("reason", "embedder_unavailable"),
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "missing_before": missing,
                "embedded": stats.get("embedded", 0),
                "skipped": stats.get("skipped", 0),
                "errors": stats.get("errors", 0),
                "total_unique": stats.get("total_unique", 0),
                "backend": stats.get("backend"),
                "model": stats.get("model"),
                "forced": force,
            },
            reason=f"embedded_{stats.get('embedded')}_facts",
        )

    def _skip(
        self, params: dict[str, Any], started, reason: str,
    ) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
