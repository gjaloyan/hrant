"""Durable job records — every user turn gets an id, persisted to
disk, survives crashes.

Why this exists:
  - User asks a question. Agent starts thinking. Server crashes
    (OOM, kill, host reboot). Before this module, the request was
    lost — no record the user ever asked, no way to retry.
  - Same for LLM provider downtime, network blips, tool exceptions.
  - With jobs/, every turn is a row on disk: WebUI shows a "Jobs"
    tab with status + retry buttons, CLI has `hrant jobs list/show
    /retry/cancel`, and on next boot we mark anything stuck in
    `running` as `interrupted` so the user can retry it.

States:
  queued       — created, not yet picked up
  running      — agent.run currently executing
  completed    — finished cleanly with a response
  failed       — finished with an error (LLM down, tool crash, ...)
  interrupted  — was `running` when the server died; needs retry
  cancelled    — user explicitly cancelled

Storage layout:
  ~/.hrant/data/jobs/
    <job_id>.json      — one file per job, full state
  Flat layout: <1k jobs is fine for a personal agent; if it grows,
  partition by month later. Each file is small (~1-5KB).

Concurrency:
  Single-process FastAPI is the common case. An RLock per JobStore
  protects against the rare overlap (two SSE chat requests landing
  in quick succession). For multi-worker uvicorn deployments add
  flock around _write — not done here because nobody runs that.

Phase A scope (this commit):
  CRUD + recover_interrupted + list/get + tool-call trace storage.

Phase B preview (next commit — failover):
  `attempts[]` field already in the schema. Each failover attempt
  on the same job appends an entry — same Job record, multiple
  provider tries.
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

from . import paths

log = logging.getLogger(__name__)


# State strings — kept as plain strings (not Enum) so JSON round-trip
# is trivial and tests / API callers can compare without imports.
JobStatus = str


VALID_STATUSES: tuple[str, ...] = (
    "queued",
    "running",
    "completed",
    "failed",
    "interrupted",
    "cancelled",
)


# Statuses we consider "terminal" — `recover_interrupted` won't touch
# them, retry is a no-op until user explicitly retries.
TERMINAL_STATUSES: frozenset[str] = frozenset({
    "completed", "failed", "interrupted", "cancelled",
})


@dataclass
class Job:
    """One user turn, persisted. See module docstring for state
    machine. Fields are kept stable across versions because they
    appear in the API surface + WebUI."""

    id: str
    status: JobStatus = "queued"

    # Source
    channel: str = "webui"              # "webui" | "telegram" | "voice" | ...
    speaker_id: str = "webui:default"   # phase 10 partition key
    session_id: Optional[str] = None    # link to sessions.json entry (when set)

    # Content
    prompt: str = ""
    response: Optional[str] = None
    error: Optional[str] = None

    # Timing (unix seconds; floats so sub-second is visible in the UI)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Routing context — how to deliver the answer cross-restart.
    # E.g. {"telegram_chat_id": 123, "telegram_message_id": 456}.
    # Empty dict for WebUI (UI fetches via API instead).
    reply_to: dict = field(default_factory=dict)

    # Tool-call trace (lightweight — populated from
    # AgentAnswer.thinking_trace at completion time). Each entry:
    # {name, args_summary, ok, error?, elapsed_ms}
    tool_calls: list[dict] = field(default_factory=list)

    # Provider attempts — populated by the failover layer (Phase B).
    # Each entry: {provider_id, model, ok, error?, elapsed_ms, started_at}
    attempts: list[dict] = field(default_factory=list)

    # Counters
    retry_count: int = 0
    interrupted_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        # Tolerant decoder — drop unknown keys so old jobs round-trip
        # forward when we add fields, and fill in missing keys with
        # defaults so newly-added fields work on jobs created before
        # the schema bump. Raises ValueError on a non-dict payload so
        # the caller can skip the file instead of crashing — the
        # consolidation scheduler spent 24h+ in a 60-second crash loop
        # because a stray `background.json` (top-level list) made
        # `data.items()` blow up here (2026-05-22 audit).
        if not isinstance(data, dict):
            raise ValueError(
                f"Job.from_dict expected dict, got {type(data).__name__}"
            )
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


def _publish_job_transition(
    job_id: str, prev_status: str, new_status: str, *, error: str = "",
) -> None:
    """Side-publish a job state transition to the LogBus so the WebUI
    Logs tab sees it. Lazy import to avoid module-load cycles, and
    best-effort — never block or crash a job update on a logging
    concern."""
    try:
        from .log_bus import publish_job_event as _pub
        _pub(
            job_id=job_id, new_status=new_status,
            prev_status=prev_status, error=error,
        )
    except Exception:
        pass


class JobStore:
    """File-backed job database. One JSON file per job.

    Methods are thread-safe via an RLock — concurrent SSE chat
    requests in the same FastAPI process can each create / update
    jobs without corrupting state.
    """

    def __init__(self, root: Optional[Path] = None):
        self._root_override = root
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        """Lazy data-dir resolution so tests that set HRANT_DATA_DIR
        AFTER importing this module still see the override."""
        if self._root_override is not None:
            return self._root_override
        return paths.data_dir(require=False) / "jobs"

    def _ensure_root(self) -> Path:
        r = self.root
        r.mkdir(parents=True, exist_ok=True)
        return r

    def _path(self, job_id: str) -> Path:
        return self._ensure_root() / f"{job_id}.json"

    def _write(self, job: Job) -> None:
        job.updated_at = time.time()
        p = self._path(job.id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Atomic-ish replace — survives a crash mid-write (the .tmp
        # is orphaned but the original .json stays valid).
        tmp.replace(p)

    # ─── Create ──────────────────────────────────────────────────

    def create(
        self,
        *,
        prompt: str,
        channel: str = "webui",
        speaker_id: str = "webui:default",
        reply_to: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> Job:
        """Create a fresh job in `queued` state. ID is a 12-char hex
        slug — short enough to fit in a URL and stay readable, long
        enough to avoid collisions at our scale."""
        with self._lock:
            job = Job(
                id=uuid.uuid4().hex[:12],
                status="queued",
                channel=channel,
                speaker_id=speaker_id,
                prompt=prompt,
                reply_to=reply_to or {},
                session_id=session_id,
            )
            self._write(job)
            return job

    # ─── Read ────────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            p = self._path(job_id)
            if not p.exists():
                return None
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("jobs/%s.json unreadable (%s); skipping", job_id, e)
                return None
            try:
                return Job.from_dict(data)
            except (ValueError, TypeError) as e:
                log.warning("jobs/%s.json malformed (%s); skipping", job_id, e)
                return None

    def list(
        self,
        *,
        status: Optional[str] = None,
        channel: Optional[str] = None,
        speaker_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        """Return jobs sorted newest-first. Filters are AND-ed.

        Cheap implementation: read every file. At <1k jobs (the
        realistic ceiling for a personal agent) this is under 50ms.
        If it ever becomes a problem, add a JSONL index."""
        with self._lock:
            root = self.root
            if not root.exists():
                return []
            out: list[Job] = []
            for p in root.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # The background-jobs store (background.json) is a top-level
                # list sharing this dir — skip non-job-dict files silently
                # rather than logging "malformed; skipping" on every scan.
                if not isinstance(data, dict):
                    continue
                try:
                    job = Job.from_dict(data)
                except (ValueError, TypeError) as e:
                    log.warning(
                        "jobs/%s.json malformed (%s); skipping", p.name, e,
                    )
                    continue
                if status and job.status != status:
                    continue
                if channel and job.channel != channel:
                    continue
                if speaker_id and job.speaker_id != speaker_id:
                    continue
                out.append(job)
            out.sort(key=lambda j: j.created_at, reverse=True)
            return out[offset:offset + limit]

    def count(self, *, status: Optional[str] = None) -> int:
        """Cheap count — scans files. Used by the WebUI badge."""
        with self._lock:
            root = self.root
            if not root.exists():
                return 0
            n = 0
            for p in root.glob("*.json"):
                if status is None:
                    n += 1
                    continue
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("status") == status:
                    n += 1
            return n

    # ─── Update: lifecycle transitions ──────────────────────────

    def _update(self, job_id: str, **changes: Any) -> Optional[Job]:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            for k, v in changes.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            self._write(job)
            return job

    def mark_running(self, job_id: str) -> Optional[Job]:
        prev = self.get(job_id)
        prev_status = prev.status if prev else ""
        out = self._update(
            job_id,
            status="running",
            started_at=time.time(),
        )
        _publish_job_transition(job_id, prev_status, "running")
        return out

    def mark_completed(
        self,
        job_id: str,
        *,
        response: str,
        tool_calls: Optional[list[dict]] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            prev_status = job.status
            job.status = "completed"
            job.response = response
            job.completed_at = time.time()
            if tool_calls:
                # Replace rather than append — tool_calls is the
                # trace from the run that just finished, not an
                # accumulating list across retries (retries get a
                # new job record).
                job.tool_calls = tool_calls
            job.error = None
            self._write(job)
        _publish_job_transition(job_id, prev_status, "completed")
        return job

    def mark_failed(self, job_id: str, *, error: str) -> Optional[Job]:
        prev = self.get(job_id)
        prev_status = prev.status if prev else ""
        out = self._update(
            job_id,
            status="failed",
            error=error,
            completed_at=time.time(),
        )
        _publish_job_transition(job_id, prev_status, "failed", error=error)
        return out

    def mark_cancelled(self, job_id: str) -> Optional[Job]:
        prev = self.get(job_id)
        prev_status = prev.status if prev else ""
        out = self._update(
            job_id,
            status="cancelled",
            completed_at=time.time(),
        )
        _publish_job_transition(job_id, prev_status, "cancelled")
        return out

    def mark_interrupted(self, job_id: str) -> Optional[Job]:
        """Used by the boot recovery hook. Bumps interrupted_count
        so the WebUI can warn 'this turn died N times'."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            prev_status = job.status
            job.status = "interrupted"
            job.error = (job.error or "") + (
                "" if not (job.error or "") else "\n"
            ) + "Server was interrupted mid-run."
            job.interrupted_count += 1
            job.completed_at = time.time()
            self._write(job)
        _publish_job_transition(job_id, prev_status, "interrupted")
        return job

    # ─── Attempts / tool-call append (Phase B prep) ─────────────

    def add_attempt(self, job_id: str, attempt: dict) -> Optional[Job]:
        """Append a provider attempt to the job's trace. Used by the
        failover layer (Phase B) — each retry on a different provider
        appends one entry here, so the user can see in the WebUI
        which providers were tried + why each failed."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            job.attempts.append(dict(attempt))
            self._write(job)
            return job

    # ─── Retry: clone + reset ───────────────────────────────────

    def retry(self, job_id: str) -> Optional[Job]:
        """Create a NEW job with the same prompt/channel/speaker_id
        and a bumped retry_count. The original job stays in its
        terminal state — failed/interrupted records are kept for
        audit. Returns the new job (caller is expected to run it)."""
        with self._lock:
            orig = self.get(job_id)
            if orig is None:
                return None
            new = self.create(
                prompt=orig.prompt,
                channel=orig.channel,
                speaker_id=orig.speaker_id,
                reply_to=orig.reply_to,
                session_id=orig.session_id,
            )
            new.retry_count = orig.retry_count + 1
            self._write(new)
            return new

    # ─── Delete ─────────────────────────────────────────────────

    def delete(self, job_id: str) -> bool:
        with self._lock:
            p = self._path(job_id)
            if not p.exists():
                return False
            p.unlink()
            return True

    # ─── Retention ──────────────────────────────────────────────

    def cleanup_old(
        self,
        *,
        max_age_seconds: float,
        keep_statuses: Optional[set[str]] = None,
    ) -> list[str]:
        """Delete jobs whose `created_at` is older than `max_age_seconds`.

        `keep_statuses`: statuses that survive cleanup regardless of
        age. Default keeps `failed` and `interrupted` because those
        carry actionable info — the user may want to retry them
        even months later. `completed`/`cancelled` jobs are pure
        history and get pruned by age.

        Returns the list of job IDs deleted."""
        keep = keep_statuses if keep_statuses is not None else {
            "failed", "interrupted",
        }
        cutoff = time.time() - max_age_seconds
        with self._lock:
            root = self.root
            if not root.exists():
                return []
            deleted: list[str] = []
            for p in root.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # The background-jobs store (background.json) is a top-level
                # list sharing this dir; skip any non-job-dict file instead of
                # blowing up on `data.get` (was: "consolidation: jobs cleanup
                # failed: 'list' object has no attribute 'get'").
                if not isinstance(data, dict):
                    continue
                if data.get("status") in keep:
                    continue
                if (data.get("created_at") or 0) >= cutoff:
                    continue
                try:
                    p.unlink()
                    deleted.append(data.get("id") or p.stem)
                except OSError as e:
                    log.warning("failed to delete %s: %s", p, e)
            if deleted:
                log.info("jobs.cleanup_old: removed %d job(s)", len(deleted))
            return deleted

    # ─── Boot recovery ─────────────────────────────────────────

    def recover_interrupted(self) -> list[str]:
        """Scan every job; transition anything `running` or `queued`
        to `interrupted`. Called from FastAPI startup BEFORE
        accepting requests, so we know there are no in-flight
        agent.run calls — anything that says `running` died with
        the previous process.

        Returns the list of job IDs touched."""
        with self._lock:
            touched: list[str] = []
            root = self.root
            if not root.exists():
                return touched
            for p in root.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # Audit follow-up: legacy / corrupted JSON files can
                # land here as lists or scalars and `data.get(...)`
                # would throw AttributeError, which surfaced as a
                # noisy WARNING on every boot (12 occurrences over 3
                # days in prod logs). Skip non-dict entries cleanly.
                if not isinstance(data, dict):
                    log.debug(
                        "recover_interrupted: skipping non-dict job "
                        "file %s (type=%s)", p.name, type(data).__name__,
                    )
                    continue
                if data.get("status") in ("running", "queued"):
                    try:
                        job = Job.from_dict(data)
                    except Exception as e:
                        log.debug(
                            "recover_interrupted: bad shape in %s: %s",
                            p.name, e,
                        )
                        continue
                    self.mark_interrupted(job.id)
                    touched.append(job.id)
            if touched:
                log.info(
                    "recovered %d interrupted job(s): %s",
                    len(touched), ", ".join(touched[:5]),
                )
            return touched


# Module-level singleton so call sites read `from .jobs import JOBS`.
JOBS = JobStore()
