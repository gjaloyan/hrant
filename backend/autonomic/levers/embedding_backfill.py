"""FIRE_EMBEDDING_BACKFILL — keep note embeddings in sync with the
note inventory.

Symptom this addresses (2026-05-27 audit): 13 vector_store items
vs 16 notes on prod; the gap grew silently when llama-cpp was
unavailable during note creation. The lever runs periodically,
checks `embedding_backfill.missing_count()`, and embeds the gap.
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


class FIRE_EMBEDDING_BACKFILL(Lever):
    name = "FIRE_EMBEDDING_BACKFILL"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "local"  # no LLM calls — embedder is a local/HTTP path
    estimated_cost = Cost(seconds=30.0, tokens_in=0, tokens_out=0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()

        # Late import — the lever loads at startup but EMBEDDER /
        # backfill_embeddings pull in httpx + heavy paths, and we
        # only want them on actual fire.
        from backend.embedder import EMBEDDER
        from backend import embedding_backfill as eb_mod

        status = EMBEDDER.status()
        backend = status.get("backend")
        if backend in (None, "disabled"):
            return self._skip(
                params, started,
                f"embedder_disabled: last_error={status.get('last_error')!r}",
            )

        try:
            coverage = eb_mod.missing_count()
        except Exception as exc:
            return self._skip(params, started, f"missing_count failed: {exc}")

        missing = int(coverage.get("missing") or 0)
        stale = bool(coverage.get("stale_store"))

        if missing == 0 and not stale:
            return self._skip(
                params, started,
                f"nothing_to_backfill (embedded={coverage.get('embedded')})",
            )

        # Stale store (dim/model mismatch) → force re-embed; otherwise
        # just fill the gap.
        force = bool(params.get("force") or stale)

        try:
            stats = eb_mod.backfill_embeddings(force=force)
        except Exception as exc:
            log.warning("embedding_backfill: backfill_embeddings raised: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"exception": str(exc)},
                reason=f"backfill_failed: {exc}",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "missing_before": missing,
                "embedded": int(stats.get("embedded") or 0),
                "skipped": int(stats.get("skipped") or 0),
                "errors": int(stats.get("errors") or 0),
                "total": int(stats.get("total") or 0),
                "backend": stats.get("backend"),
                "model": stats.get("model"),
                "forced": force,
            },
            reason=(
                f"embedded_{stats.get('embedded')}_of_{stats.get('total')}"
                + (" (forced)" if force else "")
            ),
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
