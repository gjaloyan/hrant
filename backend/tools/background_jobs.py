"""Resumable background jobs (audit T6).

Background: the May 2026 cost audit found that long-running tasks
(SWE-bench, large benchmarks, multi-hour video transcodes) blocked
the agent's turn for the entire duration. The agent kept polling
status inside the same turn → 20+ tool calls of "check progress" →
332 k tokens / $1.10 per turn for a single benchmark run.

Pattern fix: spawn the subprocess in a background thread, return a
job_id immediately, free up the turn. When the subprocess finishes,
a callback fires that DMs the owner with the result + a MEDIA: line
for any output file. The agent doesn't sit blocked.

Persistence note: the registry survives `hrant.service` restart
(file-backed JSON) BUT the actual subprocess does NOT — when the
service restarts, in-flight jobs are marked `interrupted` and the
caller must resubmit. v2 will use `systemd-run --user --unit=...`
on Linux for true restart-survival; v1 keeps the implementation
short and platform-independent.

Owner-only: callers (the `start_background_job` builtin tool) gate
on the role check. Anyone with shell access already owns the box.
"""
from __future__ import annotations

import json
import logging
import secrets
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .. import paths


log = logging.getLogger(__name__)


# Tail length captured into the registry (and thus shown to the LLM
# on get_background_job). Full stdout/stderr is written to disk if
# the job-id directory exists — see _job_dir().
_TAIL_BYTES = 4000

# Cap on the number of concurrently running background jobs. Above
# this, `start_background_job` refuses with a clear error so the
# host doesn't accidentally fork-bomb itself.
_MAX_CONCURRENT = 10


@dataclass
class BackgroundJob:
    """One subprocess we're tracking."""
    job_id: str
    label: str
    command: str
    cwd: str
    started_at: float
    finished_at: Optional[float]
    status: str          # "running" / "done" / "error" / "interrupted" / "killed"
    exit_code: Optional[int]
    stdout_tail: str
    stderr_tail: str
    requester: str
    pid: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundJob":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


_STORE_LOCK = threading.RLock()


class BackgroundJobStore:
    """File-backed registry of background jobs. JSON serialization,
    atomic .tmp rename so concurrent writers don't tear.

    Persistence layout under data_dir:
      ~/.hrant/data/jobs/background.json     ← the registry
      ~/.hrant/data/jobs/<job_id>/stdout.log ← per-job full output
      ~/.hrant/data/jobs/<job_id>/stderr.log
    """

    REGISTRY_FILENAME = "background.json"

    def __init__(self, root: Optional[Path] = None):
        self._root_override = root

    @property
    def _root(self) -> Path:
        if self._root_override is not None:
            return self._root_override
        return paths.data_dir(require=False) / "jobs"

    @property
    def _registry_path(self) -> Path:
        return self._root / self.REGISTRY_FILENAME

    def _job_dir(self, job_id: str) -> Path:
        return self._root / job_id

    def _load(self) -> list[dict]:
        p = self._registry_path
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8")) or []
        except Exception as e:
            log.warning("background-jobs registry load failed: %s", e)
            return []

    def _save(self, items: list[dict]) -> None:
        p = self._registry_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)

    def list(self, *, status: Optional[str] = None, limit: int = 50) -> list[BackgroundJob]:
        with _STORE_LOCK:
            rows = self._load()
            jobs_ = [BackgroundJob.from_dict(r) for r in rows]
        if status:
            jobs_ = [j for j in jobs_ if j.status == status]
        # Newest first.
        jobs_.sort(key=lambda j: j.started_at, reverse=True)
        return jobs_[:limit]

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        with _STORE_LOCK:
            for r in self._load():
                if r.get("job_id") == job_id:
                    return BackgroundJob.from_dict(r)
            return None

    def add(self, job: BackgroundJob) -> None:
        with _STORE_LOCK:
            items = self._load()
            items.append(job.to_dict())
            self._save(items)

    def update(self, job: BackgroundJob) -> None:
        """Replace the entry with matching job_id. No-op if not found
        (should never happen — caller adds before updating)."""
        with _STORE_LOCK:
            items = self._load()
            for i, r in enumerate(items):
                if r.get("job_id") == job.job_id:
                    items[i] = job.to_dict()
                    self._save(items)
                    return

    def mark_interrupted_on_startup(self) -> int:
        """Called from the service's lifespan/startup. Any job left
        in `running` state from a previous process must have died
        when the service stopped — flip them to 'interrupted' so the
        registry doesn't lie. Returns the count flipped."""
        with _STORE_LOCK:
            items = self._load()
            n = 0
            for r in items:
                if r.get("status") == "running":
                    r["status"] = "interrupted"
                    r["finished_at"] = time.time()
                    r["stderr_tail"] = (
                        (r.get("stderr_tail") or "")
                        + "\n[hrant.service restarted; subprocess died with parent]"
                    )
                    n += 1
            if n:
                self._save(items)
            return n

    def running_count(self) -> int:
        with _STORE_LOCK:
            return sum(
                1 for r in self._load() if r.get("status") == "running"
            )


STORE = BackgroundJobStore()


# ─── on-done subscribers ───────────────────────────────────────────


_ON_DONE: list = []


def register_on_done(fn) -> None:
    """Telegram bridge subscribes here to DM the owner on completion."""
    if fn not in _ON_DONE:
        _ON_DONE.append(fn)


def _fire_done(job: BackgroundJob) -> None:
    for fn in list(_ON_DONE):
        try:
            fn(job)
        except Exception as e:
            log.warning("background-job done callback %s failed: %s", fn, e)


# ─── start a job ───────────────────────────────────────────────────


def _write_log(path: Path, content: bytes) -> None:
    """Best-effort log write. Failure here doesn't kill the job."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except Exception as e:
        log.warning("background-job log write %s failed: %s", path, e)


def _tail_bytes(data: bytes) -> str:
    """Last `_TAIL_BYTES` bytes of stdout/stderr, decoded as utf-8
    with replacement so binary noise doesn't break JSON encoding."""
    if not data:
        return ""
    tail = data[-_TAIL_BYTES:]
    return tail.decode("utf-8", errors="replace")


def start_job(
    *,
    command: str,
    label: str = "",
    cwd: Optional[str] = None,
    requester: str = "",
    timeout_seconds: float = 3 * 3600.0,
) -> BackgroundJob:
    """Spawn `command` in a background thread; return immediately
    with the BackgroundJob record (status='running'). On completion
    the record gets updated and `on_done` callbacks fire.

    `command` is a shell string (passed to subprocess via shell=True)
    — the caller is responsible for shell-quoting. Use shlex.quote()
    on dynamic arguments if needed.

    `timeout_seconds` is a watchdog cap. The default 3h is generous
    for SWE-bench runs; long-running training jobs need an explicit
    higher value.
    """
    if not (command or "").strip():
        raise ValueError("background job: command is empty")
    if STORE.running_count() >= _MAX_CONCURRENT:
        raise ValueError(
            f"background job: {_MAX_CONCURRENT} jobs already running — "
            f"wait for some to finish or kill via the registry"
        )

    job_id = "bg-" + secrets.token_hex(6)
    now = time.time()
    job = BackgroundJob(
        job_id=job_id,
        label=(label or "").strip()[:120] or "(unlabeled)",
        command=command,
        cwd=(cwd or "").strip(),
        started_at=now,
        finished_at=None,
        status="running",
        exit_code=None,
        stdout_tail="",
        stderr_tail="",
        requester=(requester or "").strip(),
        pid=None,
    )
    STORE.add(job)

    def _runner() -> None:
        nonlocal job
        proc: Optional[subprocess.Popen] = None
        stdout_buf = b""
        stderr_buf = b""
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=(cwd or None),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Record PID so the registry shows what to kill via OS.
            job.pid = proc.pid
            STORE.update(job)
            try:
                stdout_buf, stderr_buf = proc.communicate(timeout=timeout_seconds)
                job.exit_code = proc.returncode
                job.status = "done" if proc.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    stdout_buf, stderr_buf = proc.communicate(timeout=10)
                except Exception:
                    pass
                job.exit_code = -1
                job.status = "killed"
                stderr_buf = (stderr_buf or b"") + (
                    f"\n[background-job watchdog: killed after "
                    f"{timeout_seconds}s]".encode("utf-8")
                )
        except FileNotFoundError as e:
            job.exit_code = -1
            job.status = "error"
            stderr_buf = f"{e}\n".encode("utf-8")
        except Exception as e:
            log.exception("background job %s crashed in runner", job_id)
            job.exit_code = -1
            job.status = "error"
            stderr_buf = f"{type(e).__name__}: {e}\n".encode("utf-8")
        finally:
            job.finished_at = time.time()
            job.stdout_tail = _tail_bytes(stdout_buf)
            job.stderr_tail = _tail_bytes(stderr_buf)
            # Persist full logs to disk for later inspection if needed.
            jd = STORE._job_dir(job_id)
            _write_log(jd / "stdout.log", stdout_buf)
            _write_log(jd / "stderr.log", stderr_buf)
            STORE.update(job)
            _fire_done(job)

    t = threading.Thread(target=_runner, daemon=True, name=f"bgjob-{job_id}")
    t.start()
    return job


# ─── public lookups (for the builtin tools) ────────────────────────


def list_jobs(*, status: Optional[str] = None, limit: int = 20) -> list[dict]:
    """Public lookup, returns dicts ready to JSON-encode."""
    return [j.to_dict() for j in STORE.list(status=status, limit=limit)]


def get_job(job_id: str) -> Optional[dict]:
    j = STORE.get(job_id)
    return j.to_dict() if j else None
