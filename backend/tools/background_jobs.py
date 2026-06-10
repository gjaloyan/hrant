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
import os
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
    """One subprocess we're tracking.

    Supervisor-turn fields (the bottom block) carry context the agent
    needs when it re-engages after the subprocess finishes. Without
    them, the on-completion turn would have no idea WHY the job was
    launched or how many fix attempts came before — and would either
    repeat the same broken command or give up at the first retry.
    """
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

    # ── supervisor-turn context (audit T6 follow-up) ─────────────
    # The user request that ultimately spawned this job. Survives
    # retry chains: a child job carries the SAME original_user_request
    # as its parent, so the supervisor turn knows the user-facing goal
    # no matter how many fix-and-retry rounds have happened.
    original_user_request: str = ""
    # Speaker / chat the original request came from. Lets the final
    # DM land in the right Telegram conversation.
    original_speaker_id: str = ""
    original_chat_id: Optional[int] = None
    # Plain-English description of what counts as a successful
    # outcome. Used by the supervisor turn to decide "done vs. did
    # half the work and needs another retry". Empty = supervisor
    # infers from the user request + stdout.
    expected_outcome: str = ""
    # Chain telemetry: parent job id (when this run is a fix attempt)
    # and retry count (1 for first attempt, N for Nth fix). The cap
    # is a soft signal — the LLM-driven supervisor decides when to
    # escalate, but we surface the count so it can self-limit.
    parent_job_id: str = ""
    retry_count: int = 0
    # Audit trail of supervisor decisions across the chain. Each entry
    # captures "what the supervisor decided, why, and what it did
    # next" so future turns (and post-mortems) can read the history
    # without re-running the LLM.
    supervisor_history: list = field(default_factory=list)
    # Heartbeat throttle: when the heartbeat lever last fired for
    # this job and the progress percent it reported. Avoids
    # spamming the user with `still running, 31%`, `still running,
    # 32%`, …
    last_heartbeat_at: Optional[float] = None
    last_heartbeat_progress: Optional[float] = None
    # When `True`, supervisor decided this chain is terminal — the
    # final DM has been sent (or escalation surfaced) and no further
    # retry will fire. Pinned so a late-arriving on_done doesn't
    # accidentally re-open the supervisor.
    supervisor_terminal: bool = False
    # ── Phase 2 watchdog hooks ───────────────────────────────────
    # Total work units the job is expected to complete (e.g. 300 for
    # "SWE-bench Lite 300"). When set together with
    # `progress_probe_cmd`, the watchdog computes pct = done / total
    # and fires a heartbeat DM at each 30% milestone (30 / 60 / 90).
    # Unset = no measurable progress; watchdog falls back to a
    # time-based heartbeat every 2 hours so the user knows the job
    # is still alive on long runs.
    total_units: Optional[int] = None
    # Shell command that prints an integer (the "done" count) to
    # stdout. Run by the watchdog under the job's `cwd`. Stderr is
    # ignored; non-zero exit or non-integer output → probe skipped.
    # Example: `find logs/run_evaluation -name report.json | wc -l`
    # for a SWE-bench run.
    progress_probe_cmd: str = ""
    # Last sampled size/mtime of the job's stdout.log. Used by the
    # stale-detection watchdog: when the file hasn't grown in a
    # long time the watchdog opens a supervisor turn so the LLM can
    # decide whether the run is stuck.
    last_log_size: Optional[int] = None
    last_log_size_at: Optional[float] = None
    # Stale flag — set by the watchdog when it opens a supervisor
    # turn for an apparently-stuck job, so subsequent watchdog ticks
    # don't re-fire the same supervisor while the previous one is
    # still deciding.
    stale_supervisor_opened: bool = False
    # Phase 3: TaskEndpoint id. When non-empty, the supervisor turn
    # loads the endpoint, auto-runs each criterion's check_cmd, and
    # `complete_supervisor(decision='done')` is REFUSED while
    # critical criteria are unmet. Inherited from parent on retry —
    # the chain shares one endpoint definition.
    endpoint_id: str = ""

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

    def prune(self, max_rows: int = 2000, keep_running: bool = True) -> int:
        """I8: trim oldest non-running rows past `max_rows`.

        Background: background.json was append-only with no GC. A box
        running benchmarks plants a row per launch — over months the
        registry grows to thousands of entries, each carrying a stdout
        tail. Every `load` reads + parses the whole file.

        `keep_running` (default True) protects in-flight jobs from
        eviction regardless of the cap; closed jobs (done / error /
        killed / interrupted) compete for the remaining slots,
        oldest-first. Returns the number of rows dropped. Not auto-
        fired from anywhere yet — exposed for a future autonomic
        FIRE_STALE_PROPOSALS-style lever.
        """
        with _STORE_LOCK:
            items = self._load()
            if len(items) <= max_rows:
                return 0
            if keep_running:
                running = [r for r in items if r.get("status") == "running"]
                closed = [r for r in items if r.get("status") != "running"]
            else:
                running = []
                closed = list(items)
            closed.sort(key=lambda r: r.get("started_at") or 0)
            keep_closed = max(0, max_rows - len(running))
            if keep_closed >= len(closed):
                return 0
            kept_closed = closed[-keep_closed:] if keep_closed > 0 else []
            new_items = running + kept_closed
            new_items.sort(key=lambda r: r.get("started_at") or 0)
            dropped = len(items) - len(new_items)
            self._save(new_items)
            return dropped

    def find_children(self, parent_job_id: str) -> list[BackgroundJob]:
        """Return ALL jobs whose `parent_job_id` matches. Walks the
        full registry (no `limit` cap) — the previous supervisor
        end-of-turn check used `list(limit=50)` and silently dropped
        retry children whose parent had aged past the 50-most-recent
        window, causing the chain to be marked `degraded` even
        though a real retry had been spawned.

        Newest-first ordering so the immediate retry child surfaces
        before older same-parent attempts (the supervisor only cares
        whether ANY child was spawned, but the ordering helps audits)."""
        if not (parent_job_id or "").strip():
            return []
        with _STORE_LOCK:
            rows = self._load()
        children = [
            BackgroundJob.from_dict(r) for r in rows
            if (r.get("parent_job_id") or "") == parent_job_id
        ]
        children.sort(key=lambda j: j.started_at, reverse=True)
        return children


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


def _collect_provider_env() -> dict[str, str]:
    """Build a {VAR: key} dict from every enabled configured provider
    whose ``key_env_default`` is known. The supervisor and any
    LLM-spawned bench / SDK subprocess inherit these via env so they
    can authenticate the SAME way the agent authenticates itself —
    without the agent having to manually `export OPENROUTER_API_KEY=...`
    before every `harbor run`.

    Only fills variables that are NOT already in the current process
    environment (we never overwrite an operator-set value). Empty /
    missing keys are skipped. Safe-by-default: if `providers.get_api_key`
    raises (corrupt config, missing file), we silently return an empty
    dict — the bg-job still launches, just without the auto-inject.
    """
    out: dict[str, str] = {}
    try:
        from .. import providers as _p
    except Exception:
        return out
    try:
        configured = _p.get_providers()
        types = getattr(_p, "PROVIDER_TYPES", {}) or {}
    except Exception:
        return out
    for prov in configured or []:
        try:
            if not prov.get("enabled", True):
                continue
            t = prov.get("type") or ""
            env_name = (prov.get("api_key_env") or "").strip()
            if not env_name:
                env_name = ((types.get(t) or {}).get("key_env_default") or "").strip()
            if not env_name:
                continue
            if env_name in os.environ and os.environ[env_name].strip():
                continue
            key = _p.get_api_key(prov) or ""
            if not key.strip():
                continue
            out[env_name] = key
        except Exception:
            continue
    return out


def start_job(
    *,
    command: str,
    label: str = "",
    cwd: Optional[str] = None,
    requester: str = "",
    timeout_seconds: float = 3 * 3600.0,
    original_user_request: str = "",
    original_speaker_id: str = "",
    original_chat_id: Optional[int] = None,
    expected_outcome: str = "",
    parent_job_id: str = "",
    retry_count: int = 0,
    total_units: Optional[int] = None,
    progress_probe_cmd: str = "",
    endpoint_id: str = "",
    extra_env: Optional[dict[str, str]] = None,
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

    The `original_*`/`expected_outcome`/`parent_job_id`/`retry_count`
    fields wire up supervisor-turn context (audit T6 follow-up).
    Pass them through unmodified when retrying a failed job so the
    on-completion supervisor sees the full chain.
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
        original_user_request=(original_user_request or "").strip(),
        original_speaker_id=(original_speaker_id or "").strip(),
        original_chat_id=original_chat_id,
        expected_outcome=(expected_outcome or "").strip(),
        parent_job_id=(parent_job_id or "").strip(),
        retry_count=int(retry_count or 0),
        total_units=int(total_units) if total_units else None,
        progress_probe_cmd=(progress_probe_cmd or "").strip(),
        endpoint_id=(endpoint_id or "").strip(),
    )
    STORE.add(job)

    # Prefer bash on POSIX so commands using bash features
    # (`set -o pipefail`, `[[ ]]`, here-strings, process substitution,
    # arrays) work out of the box. Default `shell=True` on Linux runs
    # `/bin/sh -c "..."` which on Debian/Ubuntu is `dash` — the LLM
    # writes bash-isms idiomatically and they would silently fail under
    # dash (e.g. `Illegal option -o pipefail`, exit 2 before any work).
    # Falls back to the default shell when `/bin/bash` is missing
    # (alpine, minimal containers).
    _shell_exe: Optional[str] = None
    try:
        _candidate = "/bin/bash"
        if os.path.exists(_candidate):
            _shell_exe = _candidate
    except Exception:
        _shell_exe = None

    # Compose subprocess env. Always inherit the current process env
    # (so PATH, HOME, etc. work). Layer on auto-collected provider
    # keys (OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, …)
    # so bench / SDK subprocesses authenticate the SAME way the
    # agent does, without manual `export` ceremonies in the command.
    # Caller-provided `extra_env` wins last (explicit > auto).
    _auto_env = _collect_provider_env()
    _job_env: dict[str, str] = dict(os.environ)
    _job_env.update(_auto_env)
    if extra_env:
        _job_env.update({str(k): str(v) for k, v in extra_env.items()})
    if _auto_env:
        log.info(
            "background-job %s: auto-injected provider env vars: %s",
            job_id, ", ".join(sorted(_auto_env.keys())),
        )

    def _runner() -> None:
        nonlocal job
        proc: Optional[subprocess.Popen] = None
        stdout_buf = b""
        stderr_buf = b""
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                executable=_shell_exe,
                cwd=(cwd or None),
                env=_job_env,
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
