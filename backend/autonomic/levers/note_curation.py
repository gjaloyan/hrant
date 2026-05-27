"""FIRE_NOTE_CURATION — refresh stale/low-confidence notes via learn_topic."""
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

from backend.note_creator import learn_topic

log = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("knowledge/index.json")
STALE_DAYS = 30
HOT_ACCESS_THRESHOLD = 5
EXCLUDED_CATEGORIES = {"personal", "projects"}


class FIRE_NOTE_CURATION(Lever):
    name = "FIRE_NOTE_CURATION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=60.0, tokens_in=3000, tokens_out=2000)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        index_path = resolve_knowledge_path(params.get("index_path") or DEFAULT_INDEX_PATH)
        max_per_tick = int(params.get("max_per_tick", 2))

        if not index_path.exists():
            return self._skip(params, started, "no_stale_notes")
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._skip(params, started, "no_stale_notes")
        if not isinstance(idx, dict) or not idx:
            return self._skip(params, started, "no_stale_notes")

        candidates = self._find_candidates(idx)
        if not candidates:
            return self._skip(params, started, "no_stale_notes")

        refreshed = 0
        skipped = 0
        errors = 0
        for entry in candidates[:max_per_tick]:
            topic = entry.get("topic", "")
            category = entry.get("category", "profession")
            if not topic:
                skipped += 1
                continue
            try:
                learn_topic(topic=topic, depth="quick", category=category)
                refreshed += 1
            except Exception as exc:
                log.warning("note_curation: learn_topic failed for %r: %s", topic, exc)
                errors += 1

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "candidates": len(candidates),
                "refreshed": refreshed,
                "skipped": skipped,
                "errors": errors,
            },
            reason=f"curated_{refreshed}_notes",
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

    def _find_candidates(self, idx: dict[str, dict]) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=STALE_DAYS)
        out: list[dict] = []
        for slug, entry in idx.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("category") in EXCLUDED_CATEGORIES:
                continue
            confidence = str(entry.get("confidence", "verified")).lower()
            if confidence in ("partial", "unverified"):
                out.append(dict(entry))
                continue
            updated_str = str(entry.get("updated", ""))
            access_count = int(entry.get("access_count", 0) or 0)
            if access_count < HOT_ACCESS_THRESHOLD:
                continue
            try:
                updated_dt = datetime.strptime(updated_str, "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if updated_dt < cutoff:
                out.append(dict(entry))

        def key(e: dict) -> tuple:
            conf = str(e.get("confidence", "verified")).lower()
            conf_rank = 0 if conf in ("partial", "unverified") else 1
            updated = str(e.get("updated", "9999-12-31 23:59"))
            neg_access = -int(e.get("access_count", 0) or 0)
            return (conf_rank, updated, neg_access)
        out.sort(key=key)
        return out
