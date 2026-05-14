"""Persisted scheduler state — when did we last consolidate, was
it ok, what's the next eligible time. Survives restarts so the 24h
gate isn't reset every time the server reboots.

Storage: `~/.hrant/data/knowledge/consolidation_state.json`
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import config

log = logging.getLogger(__name__)


@dataclass
class ConsolidationState:
    """One persisted row tracking the scheduler's bookkeeping."""

    last_run_at: float = 0.0           # unix ts of last completed run
    last_run_status: str = "never"     # "success" | "failed" | "skipped" | "never"
    last_run_digest: Optional[str] = None    # path of last written digest
    last_run_error: Optional[str] = None     # short failure reason
    last_run_duration_seconds: float = 0.0
    last_run_tokens_used: int = 0
    last_run_cost_usd_estimate: float = 0.0
    total_runs: int = 0
    # For metadata only — Phase 16A doesn't change behaviour based
    # on these but they make the WebUI status banner more useful.
    last_run_speakers_seen: list[str] = field(default_factory=list)
    last_run_jobs_analyzed: int = 0
    last_run_facts_added: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConsolidationState":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


_lock = threading.RLock()


def load() -> ConsolidationState:
    """Return the last persisted state, or a default if absent."""
    p = config.state_path()
    if not p.exists():
        return ConsolidationState()
    with _lock:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("consolidation_state.json unreadable (%s); using defaults", e)
            return ConsolidationState()
        return ConsolidationState.from_dict(data)


def save(state: ConsolidationState) -> ConsolidationState:
    """Atomic-ish write. Survives a crash mid-flush."""
    with _lock:
        p = config.state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)
    return state


def next_eligible_at(state: Optional[ConsolidationState] = None) -> float:
    """Unix ts after which the cooldown is satisfied. The actual
    fire time depends on idleness too — this is the FLOOR, not the
    fire time."""
    if state is None:
        state = load()
    if state.last_run_at <= 0:
        return time.time()  # never run → eligible now
    return state.last_run_at + config.COOLDOWN_SECONDS


def cooldown_remaining_seconds(state: Optional[ConsolidationState] = None) -> float:
    """Seconds until cooldown is done. 0 (or negative) means past
    the gate — the scheduler may still wait for idleness."""
    return max(0.0, next_eligible_at(state) - time.time())
