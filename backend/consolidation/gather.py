"""Pull the last N hours of activity into a single structured
bundle ready for the LLM pipeline.

What "activity" means for daily consolidation:
  - Jobs from `~/.hrant/data/jobs/` whose `created_at` is in window
  - Per-job: prompt, response, tool_calls trace, attempts trace,
    channel, speaker_id, status
  - Last activity timestamp across ALL jobs (used by the scheduler
    to detect idleness)

We deliberately do NOT pull `conversation.json` directly here —
the Job records already contain the prompts + responses with
richer metadata (tool calls, failover attempts). Conversation
files are just per-speaker chronological logs and carry less info.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .. import jobs as _jobs
from . import config

log = logging.getLogger(__name__)


@dataclass
class ActivityBundle:
    """Everything the pipeline needs for one consolidation run."""

    window_start_ts: float
    window_end_ts: float
    jobs: list[_jobs.Job] = field(default_factory=list)

    # Derived stats — cheap to compute, useful for the digest report.
    @property
    def turn_count(self) -> int:
        return len(self.jobs)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for j in self.jobs:
            if j.speaker_id and j.speaker_id not in seen:
                seen.append(j.speaker_id)
        return seen

    @property
    def channels(self) -> list[str]:
        seen: list[str] = []
        for j in self.jobs:
            if j.channel and j.channel not in seen:
                seen.append(j.channel)
        return seen

    @property
    def failed_count(self) -> int:
        return sum(1 for j in self.jobs if j.status == "failed")

    @property
    def interrupted_count(self) -> int:
        return sum(1 for j in self.jobs if j.status == "interrupted")

    @property
    def completed_count(self) -> int:
        return sum(1 for j in self.jobs if j.status == "completed")


def last_activity_ts() -> Optional[float]:
    """Return the most-recent `created_at` across all jobs, or None
    if no jobs exist. Used by the scheduler to detect idleness:
    `time.time() - last_activity_ts() > IDLE_THRESHOLD_SECONDS`."""
    rows = _jobs.JOBS.list(limit=1, offset=0)
    if not rows:
        return None
    return rows[0].created_at


def is_idle(threshold_seconds: float = None) -> bool:  # type: ignore[assignment]
    """True iff the last job is older than `threshold_seconds` (or
    no jobs at all). Default threshold = config.IDLE_THRESHOLD_SECONDS."""
    if threshold_seconds is None:
        threshold_seconds = config.IDLE_THRESHOLD_SECONDS
    last = last_activity_ts()
    if last is None:
        return True
    return (time.time() - last) >= threshold_seconds


def gather(
    *,
    window_seconds: float = None,  # type: ignore[assignment]
    now: Optional[float] = None,
) -> ActivityBundle:
    """Collect jobs created within the last `window_seconds`.

    The window slides — we don't try to align to calendar days
    because the consolidation fires whenever idle, which might be
    early morning, late night, or mid-afternoon. A rolling window
    means each digest covers the period since the last one without
    gaps or overlaps.
    """
    if window_seconds is None:
        window_seconds = config.GATHER_WINDOW_SECONDS
    if now is None:
        now = time.time()
    start = now - window_seconds

    # `list(limit=10000)` scans the whole jobs dir — fine at our
    # scale (~1k jobs ceiling) and avoids needing an index.
    all_recent = _jobs.JOBS.list(limit=10_000, offset=0)
    in_window = [j for j in all_recent if start <= j.created_at <= now]

    log.info(
        "consolidation.gather: %d job(s) in window %.0fs",
        len(in_window), window_seconds,
    )
    return ActivityBundle(
        window_start_ts=start,
        window_end_ts=now,
        jobs=in_window,
    )
