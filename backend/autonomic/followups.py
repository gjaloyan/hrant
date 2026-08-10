"""Durable follow-up queue — the link that made `LeverReport.follow_ups` real.

`follow_ups` has been a field on every lever report since D-01. It is written,
serialised to JSONL, and returned by the API. Nothing has ever read it. So a
lever could say "and now run FIRE_SERVICE_REPAIR" and be heard by no one —
which is precisely why FIRE_SELF_HEAL, whose entire job is to name a fix
lever, sat unreachable while looking implemented.

This is that missing link: a small durable queue the tick drains with priority
before consulting Layer 0. Follow-ups are reactions to something that already
happened, so they outrank the periodic table by design.

Two guards, because a queue that dispatches repairs is also a way to build a
loop:

  * MAX_DEPTH — the queue refuses to grow past a handful of entries. A storm
    that outruns the drain rate is a bug, and burying it under a thousand
    queued repairs makes it unreadable.
  * Duplicate suppression — the same (lever, signature) pair cannot be queued
    twice while one is still waiting.

Attempt caps and per-signature cooldowns live in `immune.FireLog`: they are
immune policy, not queue mechanics.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = Path("knowledge/autonomic/followups.json")

# Deep enough for a real chain (match -> plan -> fix), shallow enough that a
# runaway producer is visible immediately instead of silently absorbed.
MAX_DEPTH = 8


@dataclass
class FollowUp:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    lever: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    signature_id: str = ""      # set when this came from an immune match
    origin: str = ""            # the lever that queued it
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FollowUp":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


class FollowUpQueue:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def _resolve(self) -> Path:
        if self._path is not None:
            return self._path
        from .lever import resolve_knowledge_path
        return resolve_knowledge_path(DEFAULT_QUEUE_PATH)

    def _load(self) -> list[FollowUp]:
        p = self._resolve()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        out = []
        for d in raw:
            try:
                out.append(FollowUp.from_dict(d))
            except TypeError:
                continue
        return out

    def _save(self, items: list[FollowUp]) -> None:
        p = self._resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps([i.to_dict() for i in items],
                                    ensure_ascii=False, indent=1),
                         encoding="utf-8")
        except OSError as exc:
            log.warning("followups: could not persist queue: %s", exc)

    # ── the queue ───────────────────────────────────────────────────

    def push(self, lever: str, params: dict[str, Any] | None = None, *,
             reason: str = "", signature_id: str = "",
             origin: str = "") -> "FollowUp | None":
        """Queue a lever to run on an upcoming tick. None when refused."""
        if not lever:
            return None
        items = self._load()
        if len(items) >= MAX_DEPTH:
            log.warning("followups: queue full (%d), dropping %s",
                        len(items), lever)
            return None
        for existing in items:
            if existing.lever == lever and \
                    existing.signature_id == signature_id:
                return None
        fu = FollowUp(lever=lever, params=dict(params or {}), reason=reason,
                      signature_id=signature_id, origin=origin)
        items.append(fu)
        self._save(items)
        return fu

    def pop(self) -> "FollowUp | None":
        """Take the oldest entry. FIFO: a chain must run in order."""
        items = self._load()
        if not items:
            return None
        head = items[0]
        self._save(items[1:])
        return head

    def peek_all(self) -> list[FollowUp]:
        return self._load()

    def depth(self) -> int:
        return len(self._load())

    def clear(self) -> int:
        n = len(self._load())
        self._save([])
        return n


FOLLOWUPS = FollowUpQueue()
