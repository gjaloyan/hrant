"""File-backed store for subagent sessions + in-memory active registry.

Mirrors the pattern in `backend/jobs.py` for parent agent turns:
  - one JSON file per session under `~/.hrant/data/subagents/`
  - atomic write via `.tmp` rename so a crash mid-write doesn't
    corrupt the file
  - thread-safe singleton (`SUBAGENT_STORE`) used by the dispatcher

In addition to persisted history, the store also tracks ACTIVELY
running subagents in a process-local registry. The WebUI's Subagents
tab polls `/api/subagents/active` to render a live list; persisted
records are for the history view.

Session record fields:
  id              short hex slug (12 chars)
  status          "running" | "completed" | "failed"
  role            researcher / coder / reviewer
  task            the task string passed to delegate(...)
  parent_job_id   if dispatched inside a parent agent's turn, the
                  Job.id from backend/jobs.py — lets the WebUI
                  deep-link from the parent's history row to the
                  child's session detail
  parent_speaker  speaker_id that owned the parent turn
  created_at      time.time() at dispatch
  started_at      time.time() right before the LLM call
  completed_at    time.time() when run_subagent returned
  answer          the LLM's final answer text
  error           non-empty when status == "failed"
  iterations      sum of tool-call counts
  tool_summary    {name: count} dict
  tool_calls      per-call records (name, args, result_preview, is_error)
  elapsed_ms      end-to-end duration

Retention: history is bounded to MAX_HISTORY by the writer — when
the directory exceeds the cap, the oldest records are pruned. We
don't keep an unbounded log; the SubagentSession payload is small
but ten thousand records on a long-running agent would still slow
the list endpoint.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .. import paths


log = logging.getLogger(__name__)


# Cap on persisted history rows. Anything past this gets pruned at
# write time (oldest by `created_at` first). Bumped above the jobs
# cap because subagent records are usually smaller (no full prompt
# / answer body, just the role + summary).
MAX_HISTORY = 500


@dataclass
class SubagentSession:
    """One delegate-tool dispatch, persisted to disk."""

    id: str = ""
    status: str = "running"        # "running" | "completed" | "failed"
    role: str = ""
    task: str = ""
    parent_job_id: str = ""
    parent_speaker: str = ""
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    answer: str = ""
    error: str = ""
    iterations: int = 0
    tool_summary: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SubagentSession":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


class SubagentStore:
    """File-backed subagent session DB + in-memory active registry.

    Methods are thread-safe via an RLock. The dispatcher creates a
    session on entry, marks running once the LLM call starts, and
    flushes the final state (completed / failed) when it returns.
    """

    def __init__(self, root: Optional[Path] = None):
        self._root_override = root
        self._lock = threading.RLock()
        # Active registry — session_id → SubagentSession snapshot.
        # Mirrors the persisted record but is cheaper to read for
        # the live `/api/subagents/active` poll.
        self._active: dict[str, SubagentSession] = {}

    # ─── Path resolution (lazy to respect HRANT_DATA_DIR overrides) ───

    @property
    def root(self) -> Path:
        if self._root_override is not None:
            return self._root_override
        return paths.data_dir(require=False) / "subagents"

    def _ensure_root(self) -> Path:
        r = self.root
        r.mkdir(parents=True, exist_ok=True)
        return r

    def _path(self, session_id: str) -> Path:
        return self._ensure_root() / f"{session_id}.json"

    # ─── Internal write helper ──────────────────────────────────────

    def _write(self, sess: SubagentSession) -> None:
        p = self._path(sess.id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(sess.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)

    # ─── Pruning ────────────────────────────────────────────────────

    def _prune_if_needed(self) -> None:
        """Drop the oldest records when the directory exceeds
        MAX_HISTORY. Best-effort — failures here just log and
        continue; persistence is more important than an exact cap."""
        try:
            files = sorted(
                self._ensure_root().glob("*.json"),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError as e:
            log.debug("subagent store prune scan failed: %s", e)
            return
        excess = len(files) - MAX_HISTORY
        if excess <= 0:
            return
        for f in files[:excess]:
            try:
                f.unlink()
            except OSError as e:
                log.debug("subagent store prune unlink failed for %s: %s", f, e)

    # ─── Lifecycle (called by dispatch.run_subagent) ───────────────

    def create(
        self,
        *,
        role: str,
        task: str,
        parent_job_id: str = "",
        parent_speaker: str = "",
    ) -> SubagentSession:
        """Start a new session. Status is `running` from the moment
        of creation (we don't have a queued state — dispatch is
        synchronous). Returns the persisted session."""
        with self._lock:
            now = time.time()
            sess = SubagentSession(
                id=uuid.uuid4().hex[:12],
                status="running",
                role=role,
                task=task[:2000],   # cap so a 50KB task doesn't bloat disk
                parent_job_id=parent_job_id,
                parent_speaker=parent_speaker,
                created_at=now,
                started_at=now,
            )
            self._active[sess.id] = sess
            self._write(sess)
            return sess

    def finalize(
        self,
        session_id: str,
        *,
        status: str,
        answer: str = "",
        error: str = "",
        iterations: int = 0,
        tool_summary: Optional[dict[str, int]] = None,
        tool_calls: Optional[list[dict]] = None,
        elapsed_ms: int = 0,
    ) -> Optional[SubagentSession]:
        """Mark the session done. `status` must be `completed` or
        `failed`; anything else is coerced to `failed`. Returns the
        final session record (or None if the session was missing —
        defensive against double-finalize)."""
        if status not in ("completed", "failed"):
            status = "failed"
        with self._lock:
            sess = self._active.pop(session_id, None)
            if sess is None:
                # Re-read from disk in case the in-memory registry
                # was cleared (process restart between start and
                # finalize, very unlikely but cheap to handle).
                sess = self.get(session_id)
                if sess is None:
                    return None
            sess.status = status
            sess.completed_at = time.time()
            sess.elapsed_ms = int(elapsed_ms)
            sess.iterations = int(iterations)
            sess.tool_summary = dict(tool_summary or {})
            sess.tool_calls = list(tool_calls or [])
            # Cap answer/error so a runaway response can't bloat disk.
            sess.answer = (answer or "")[:8000]
            sess.error = (error or "")[:1000]
            self._write(sess)
            self._prune_if_needed()
            return sess

    # ─── Read ───────────────────────────────────────────────────────

    def get(self, session_id: str) -> Optional[SubagentSession]:
        with self._lock:
            if session_id in self._active:
                return self._active[session_id]
            p = self._path(session_id)
            if not p.exists():
                return None
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("subagents/%s.json unreadable (%s)", session_id, e)
                return None
            return SubagentSession.from_dict(data)

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        role: Optional[str] = None,
    ) -> list[SubagentSession]:
        """Newest first. Filters are exact-match on status / role."""
        with self._lock:
            files = sorted(
                self._ensure_root().glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        out: list[SubagentSession] = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            sess = SubagentSession.from_dict(data)
            if status and sess.status != status:
                continue
            if role and sess.role != role:
                continue
            out.append(sess)
        return out[offset : offset + max(1, limit)]

    def active(self) -> list[SubagentSession]:
        """Currently-running subagents — cheap, no disk I/O."""
        with self._lock:
            return list(self._active.values())

    def stats(self) -> dict[str, int]:
        """Summary for the API stats endpoint."""
        with self._lock:
            running = len(self._active)
        files = list(self._ensure_root().glob("*.json"))
        completed = 0
        failed = 0
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            st = data.get("status")
            if st == "completed":
                completed += 1
            elif st == "failed":
                failed += 1
        return {
            "running": running,
            "completed": completed,
            "failed": failed,
            "total_persisted": len(files),
        }


# Module-level singleton — same convention as JOBS in backend/jobs.py.
SUBAGENT_STORE = SubagentStore()
