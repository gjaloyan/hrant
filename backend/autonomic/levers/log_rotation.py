"""FIRE_LOG_ROTATION — trim autonomic logs to `RETENTION_DAYS`
of entries.

Audit 2026-05-27 prod state:
  - lever_log.jsonl: 28.8 MB, 26 273 lines, ~2 200 entries/day
    (most are SKIPPED no-op records).
  - tick_log.jsonl: 7.8 MB, 36 488 lines.

Linear growth: 100+ MB/month untouched. The lever runs daily
(cooldown 86400) and rewrites each log keeping only the trailing
`RETENTION_DAYS` of entries. Malformed lines (no JSON / no
timestamp) are kept conservatively — better to log noise than to
silently drop data we can't classify.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
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

log = logging.getLogger(__name__)


RETENTION_DAYS = 7

# Field names that may hold the timestamp in a log line. lever_log
# entries use `started_at`; tick_log entries use `ts`. We probe both.
_TS_FIELDS = ("started_at", "ts", "finished_at", "timestamp")


def _knowledge_dir() -> Path:
    """Indirection so tests can pin a tmp path."""
    from backend import paths
    return paths.knowledge_dir()


def _parse_ts(s: str) -> datetime | None:
    """Parse an ISO timestamp from a log line. Returns None for
    anything we can't parse — caller treats None as "can't prove
    stale, keep the line"."""
    if not isinstance(s, str) or not s:
        return None
    # Strip any "+00:00" / "Z" suffix variations for fromisoformat.
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _is_stale(line: str, cutoff: datetime) -> bool:
    """True if the line has a parseable timestamp AND that timestamp
    is before the cutoff. Malformed lines return False (keep)."""
    try:
        data = json.loads(line)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    for field in _TS_FIELDS:
        ts = _parse_ts(data.get(field, ""))
        if ts is None:
            continue
        # Compare in UTC if both have tz, else naive-vs-naive.
        if ts.tzinfo is not None and cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        if cutoff.tzinfo is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts < cutoff
    return False  # no ts field found — keep


def _rotate_one(path: Path, cutoff: datetime) -> dict[str, int]:
    """Rewrite `path` keeping only lines that are NOT _is_stale.
    Returns {kept, dropped} for telemetry."""
    if not path.exists():
        return {"kept": 0, "dropped": 0}
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("log_rotation: read %s failed: %s", path, exc)
        return {"kept": -1, "dropped": -1}
    keep: list[str] = []
    dropped = 0
    for line in original.splitlines():
        if not line.strip():
            continue  # blank lines disappear
        if _is_stale(line, cutoff):
            dropped += 1
        else:
            keep.append(line)
    new_body = "\n".join(keep) + ("\n" if keep else "")
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(new_body, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("log_rotation: write %s failed: %s", path, exc)
        return {"kept": -1, "dropped": -1}
    return {"kept": len(keep), "dropped": dropped}


class FIRE_LOG_ROTATION(Lever):
    name = "FIRE_LOG_ROTATION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "local"
    estimated_cost = Cost(seconds=2.0, tokens_in=0, tokens_out=0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        retention_days = int(
            params.get("retention_days") or RETENTION_DAYS
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        autonomic_dir = _knowledge_dir() / "autonomic"
        lever_log = autonomic_dir / "lever_log.jsonl"
        tick_log = autonomic_dir / "tick_log.jsonl"

        if not lever_log.exists() and not tick_log.exists():
            return self._skip(params, started, "no_logs_present")

        lever_stats = _rotate_one(lever_log, cutoff)
        tick_stats = _rotate_one(tick_log, cutoff)

        total_dropped = max(0, lever_stats["dropped"]) + max(0, tick_stats["dropped"])
        if total_dropped == 0:
            return self._skip(
                params, started,
                f"already_compact (retention={retention_days}d)",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "lever_log": lever_stats,
                "tick_log": tick_stats,
                "retention_days": retention_days,
            },
            reason=f"dropped_{total_dropped}_stale_lines",
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
