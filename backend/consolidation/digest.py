"""The Digest record — what one consolidation run produces.

Stored at `~/.hrant/data/knowledge/memory_digests/<YYYY-MM-DD>.json`.

Schema is forward-compatible: unknown keys are ignored on read,
defaults fill in on add. Phase 16B will add `pruned` + `restore_ref`
fields without breaking 16A digests.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

log = logging.getLogger(__name__)


@dataclass
class DigestFact:
    """One durable fact extracted from a day's activity. Mirrors
    the `memory_facts.jsonl` row shape but with extra UI metadata
    (whether it was actually promoted, what topics it relates to)."""

    text: str
    related_topics: list[str] = field(default_factory=list)
    confidence: float = 0.0
    category: str = "general"
    promoted: bool = False                # was it actually appended to memory_facts.jsonl?
    reason_if_skipped: Optional[str] = None  # e.g. "low_confidence", "duplicate"


@dataclass
class ProfileUpdate:
    """A change applied to one user-profile markdown file. Speaker
    id distinguishes the global `user.md` (speaker `webui:default`)
    from per-Telegram-user files."""

    speaker_id: str
    profile_path: str         # absolute path (string for JSON portability)
    appended_text: str        # what was added at the bottom of the file
    pre_size_bytes: int = 0   # for forensics / future rollback


@dataclass
class Digest:
    """One day's consolidation output. Written verbatim to disk and
    served as-is over the API + WebUI."""

    date: str                            # YYYY-MM-DD
    started_at: float
    completed_at: float = 0.0

    # Scope of the run
    window_start_ts: float = 0.0
    window_end_ts: float = 0.0
    speakers_active: list[str] = field(default_factory=list)
    channels_active: list[str] = field(default_factory=list)

    # Counters
    turns_analyzed: int = 0
    jobs_failed: int = 0
    jobs_interrupted: int = 0

    # Narrative + extraction outputs
    narrative: str = ""
    new_facts: list[DigestFact] = field(default_factory=list)
    profile_updates: list[ProfileUpdate] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)
    links_added: list[dict] = field(default_factory=list)

    # Budget / status
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    status: str = "in_progress"          # in_progress | success | failed | skipped
    skip_reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Digest":
        # Tolerant decoder; mirrors what jobs.Job.from_dict does.
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict = {}
        for k, v in data.items():
            if k not in valid:
                continue
            if k == "new_facts":
                kwargs[k] = [
                    DigestFact(**{kk: vv for kk, vv in row.items() if kk in DigestFact.__dataclass_fields__})  # type: ignore[attr-defined]
                    for row in v if isinstance(row, dict)
                ]
            elif k == "profile_updates":
                kwargs[k] = [
                    ProfileUpdate(**{kk: vv for kk, vv in row.items() if kk in ProfileUpdate.__dataclass_fields__})  # type: ignore[attr-defined]
                    for row in v if isinstance(row, dict)
                ]
            else:
                kwargs[k] = v
        return cls(**kwargs)


# ─── Storage ──────────────────────────────────────────────────────────


_lock = threading.RLock()


def write(digest: Digest) -> Path:
    """Atomic-ish write. Returns the path written."""
    p = config.digest_path_for(digest.date)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(digest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)
    log.info("consolidation: digest written %s", p)
    return p


def read(date_str: str) -> Optional[Digest]:
    p = config.digest_path_for(date_str)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("digest %s unreadable (%s)", p, e)
        return None
    return Digest.from_dict(data)


def list_all() -> list[dict]:
    """Lightweight index of every digest on disk, newest first.

    Returns a list of `{date, status, narrative_preview, new_facts_count,
    speakers_active, completed_at}` dicts — small enough that the WebUI
    can show 30 of them on a single page without per-row API calls."""
    d = config.digests_dir()
    if not d.exists():
        return []
    rows: list[dict] = []
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "date": data.get("date") or p.stem,
            "status": data.get("status") or "unknown",
            "narrative_preview": (data.get("narrative") or "")[:200],
            "new_facts_count": len(data.get("new_facts") or []),
            "open_threads_count": len(data.get("open_threads") or []),
            "turns_analyzed": int(data.get("turns_analyzed") or 0),
            "speakers_active": data.get("speakers_active") or [],
            "completed_at": float(data.get("completed_at") or 0.0),
            "tokens_used": int(data.get("tokens_used") or 0),
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def today_str(now: Optional[float] = None) -> str:
    """YYYY-MM-DD for the local-time interpretation of `now`. The
    digest date is what the user would call "today" on their wall
    clock — not UTC midnight, which would surprise users."""
    ts = now if now is not None else time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
