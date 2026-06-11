"""Read-side of daily consolidation — the "wake up remembering
yesterday" half of the sleep cycle.

Audit 2026-06-11 found the consolidation pipeline (Phase 16A-16C)
writes narratives / open threads / lessons to memory_digests/ but
NOTHING reads them back at cognition time — only the WebUI banner
and CLI. The agent woke up amnesiac: no episodic yesterday, no
unfinished business, no lessons from failed turns. Human sleep
consolidation only matters because you wake up WITH the memories.

`yesterday_block()` renders the latest digest (narrative + open
threads + lessons) plus the most recent self-reflection failure
patterns into one compact system-prompt block (~300-600 tokens).
`unified_agent.run_unified` injects it every turn, fast path
included.

Cheap by construction:
  - pure disk reads, no LLM calls;
  - cached per calendar day (the digest only changes once a day),
    so the per-turn cost after the first render is a dict lookup.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

from . import digest as _digest_mod

log = logging.getLogger(__name__)


# How many days back to look for the most recent digest. After a
# week-long outage stale narratives stop being "yesterday" in any
# useful sense — better to show nothing.
_LOOKBACK_DAYS = 7

_NARRATIVE_CAP = 700        # chars of narrative included in the block
_MAX_THREADS = 5
_MAX_LESSONS = 5
_MAX_ROOT_CAUSES = 3

_cache_lock = threading.Lock()
# (today_str, rendered_block) — rebuilt when the calendar day flips.
_cache: "tuple[str, str] | None" = None


def clear_cache() -> None:
    """Test helper / manual invalidation after a forced consolidation."""
    global _cache
    with _cache_lock:
        _cache = None


def _latest_digest(now: Optional[float] = None) -> "Optional[_digest_mod.Digest]":
    """Most recent successful digest within the lookback window,
    scanning backwards from today. Today's own digest counts —
    consolidation may have fired this morning."""
    base = now if now is not None else time.time()
    for days_back in range(0, _LOOKBACK_DAYS + 1):
        date_str = _digest_mod.today_str(base - days_back * 86400.0)
        d = _digest_mod.read(date_str)
        if d is None:
            continue
        if d.status not in ("success", "partial"):
            continue
        if d.skip_reason == "no_activity":
            continue
        return d
    return None


def _self_reflection_line() -> str:
    """One line summarizing the latest self-reflection failure
    aggregation, e.g. 'Known failure patterns (96 analyzed):
    tool_misuse 40, hallucination 35, wrong_reasoning 16.' Empty
    string when the log is missing/unreadable."""
    try:
        from .. import paths
        p = paths.knowledge_dir() / "autonomic" / "self_reflection_log.jsonl"
        if not p.exists():
            return ""
        last_line = ""
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return ""
        row = json.loads(last_line)
        causes = row.get("by_root_cause") or {}
        if not isinstance(causes, dict) or not causes:
            return ""
        top = sorted(causes.items(), key=lambda kv: -int(kv[1] or 0))
        top = top[:_MAX_ROOT_CAUSES]
        total = row.get("total_failures")
        parts = ", ".join(f"{name} {count}" for name, count in top)
        prefix = (
            f"Known failure patterns ({total} analyzed): "
            if total else "Known failure patterns: "
        )
        return prefix + parts + "."
    except Exception as e:
        log.debug("self_reflection_line failed: %s", e)
        return ""


def _render(d: "_digest_mod.Digest") -> str:
    lines: list[str] = [
        f"# YESTERDAY (consolidated memory, {d.date})",
    ]
    narrative = (d.narrative or "").strip()
    if narrative:
        if len(narrative) > _NARRATIVE_CAP:
            narrative = narrative[:_NARRATIVE_CAP].rstrip() + "…"
        lines.append(narrative)
    threads = [t for t in (d.open_threads or []) if t][:_MAX_THREADS]
    if threads:
        lines.append("")
        lines.append("Open threads (unresolved — pick up if relevant):")
        for t in threads:
            lines.append(f"- {t}")
    lessons = [l for l in (getattr(d, "lessons", None) or []) if l][:_MAX_LESSONS]
    if lessons:
        lines.append("")
        lines.append("Lessons from yesterday's failures (APPLY these):")
        for l in lessons:
            lines.append(f"- {l}")
    refl = _self_reflection_line()
    if refl:
        lines.append("")
        lines.append(refl)
    return "\n".join(lines)


def yesterday_block(now: Optional[float] = None) -> str:
    """The wake-up context block, or "" when no usable digest exists.
    Never raises — a broken digest store must not break the turn."""
    global _cache
    try:
        today = _digest_mod.today_str(now if now is not None else time.time())
        with _cache_lock:
            if _cache is not None and _cache[0] == today:
                return _cache[1]
        d = _latest_digest(now)
        block = _render(d) if d is not None else ""
        with _cache_lock:
            _cache = (today, block)
        return block
    except Exception as e:
        log.debug("yesterday_block failed: %s", e)
        return ""
