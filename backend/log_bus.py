"""Unified live log bus.

Single sink for Python logging output, tool calls, job state changes,
supervisor decisions, and agent progress events. The WebUI Logs tab
subscribes via SSE; the agent's existing chat stream is unaffected.

Spec: docs/superpowers/specs/2026-05-21-logs-tab-design.md
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


log = logging.getLogger(__name__)


VALID_LEVELS = ("debug", "info", "warning", "error", "critical")
VALID_SOURCES = ("python", "tool", "job", "supervisor", "agent")


@dataclass
class LogEvent:
    """One event on the bus. Same shape for every source — the `source`
    field is both the filter key and the color code in the UI."""
    ts: float
    level: str
    source: str
    logger: str
    message: str
    meta: dict = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _logs_dir() -> Path:
    """Resolve the logs directory. Lazy import of `paths` avoids a
    circular import on module load."""
    try:
        from . import paths
        return paths.data_dir(require=False) / "logs"
    except Exception:
        return Path("/tmp/_hrant_logs_devstub")


def _current_log_file() -> Path:
    return _logs_dir() / f"agent-{datetime.now().strftime('%Y%m%d')}.jsonl"


def _write_jsonl_line(event_dict: dict) -> None:
    """Append one JSON line to the day's file. Best-effort: failures
    are swallowed by the caller so the in-memory ring keeps working
    even if disk is hostile."""
    path = _current_log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")


def gc_old(*, days: int = 7) -> int:
    """Delete daily JSONL files older than `days`. Matches `agent-
    YYYYMMDD.jsonl` strictly — other files in the dir survive (e.g.
    a README or a manual dump from the WebUI download endpoint).
    Returns the count of files removed. Called from the bg_job
    watchdog's daily sweep."""
    root = _logs_dir()
    if not root.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=max(0, int(days)))
    cutoff_stamp = cutoff.strftime("%Y%m%d")
    removed = 0
    for p in root.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith("agent-") and name.endswith(".jsonl")):
            continue
        stamp = name[len("agent-"):-len(".jsonl")]
        if not (stamp.isdigit() and len(stamp) == 8):
            continue
        if stamp < cutoff_stamp:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


class LogBus:
    """In-memory ring buffer + subscription fan-out.

    Thread-safe for publishers (any thread can call `publish`); the
    subscriber API is asyncio-native (each subscriber owns an
    `asyncio.Queue` it reads from).
    """

    def __init__(self, maxlen: int = 20_000):
        self._ring: collections.deque[dict] = collections.deque(maxlen=maxlen)
        self._lock = threading.RLock()
        # Subscribers — list of (loop, queue) pairs. The loop is the
        # asyncio loop that owns the queue, so a publisher running on
        # a worker thread can hand the event back via `call_soon_threadsafe`.
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []

    def publish(self, event: "LogEvent") -> None:
        if event.level not in VALID_LEVELS:
            event.level = "info"
        if event.source not in VALID_SOURCES:
            event.source = "python"
        d = event.to_dict()
        with self._lock:
            self._ring.append(d)
            subs = list(self._subs)
        # File persistence — best-effort, never crash on disk error.
        try:
            _write_jsonl_line(d)
        except Exception:
            pass
        # Fan out OUTSIDE the lock so a slow subscriber doesn't block
        # publishers. Each subscriber's queue has its own backpressure.
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_safe_put_nowait, q, d)
            except RuntimeError:
                # Loop is closed — drop and let unsubscribe sweep it.
                pass

    def subscribe(self, *, maxsize: int = 1000) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append((loop, q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(l, s) for (l, s) in self._subs if s is not q]

    def tail(
        self,
        *,
        level: "str | list[str] | None" = None,
        source: "str | list[str] | None" = None,
        search: str = "",
        limit: int = 500,
        before_ts: float = 0.0,
    ) -> list[dict]:
        with self._lock:
            rows = list(self._ring)
        level_set = _coerce_to_set(level)
        source_set = _coerce_to_set(source)
        s = (search or "").strip().lower()
        out: list[dict] = []
        for r in rows:
            if level_set and r.get("level") not in level_set:
                continue
            if source_set and r.get("source") not in source_set:
                continue
            if before_ts and float(r.get("ts") or 0.0) >= before_ts:
                continue
            if s:
                hay = (
                    (r.get("message") or "")
                    + " " + (r.get("logger") or "")
                    + " " + json.dumps(r.get("meta") or {}, ensure_ascii=False)
                ).lower()
                if s not in hay:
                    continue
            out.append(r)
        return out[-limit:] if limit > 0 else out

    def clear(self) -> None:
        """Test-only — purge the ring + subscriber list."""
        with self._lock:
            self._ring.clear()
            self._subs = []


def _safe_put_nowait(q: asyncio.Queue, item: dict) -> None:
    """Put with backpressure: if the subscriber's queue is full,
    drop the event silently. A pause-tabbed WebUI client must not
    block publishers."""
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


def _coerce_to_set(v) -> "set[str] | None":
    if v is None:
        return None
    if isinstance(v, str):
        return {v}
    return set(v)


# Module-level singleton. The test suite resets via `BUS.clear()`.
BUS = LogBus()


class LogBusHandler(logging.Handler):
    """Bridges stdlib `logging` → `LogBus`. Attach to the root logger
    in `main.py` once at startup; every `log.info(...)` in the codebase
    then flows through the bus as a side effect.

    Crash safety: `emit` swallows every exception. A logging handler
    that raises crashes the producer — never worth it for a UI
    feature. The single private flag in `handle()` prevents
    `handle → publish → log → handle` recursion if anything inside
    the bus accidentally logs."""

    _LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "critical",
    }

    def __init__(self, bus: "LogBus | None" = None):
        super().__init__()
        self._bus = bus or BUS
        self._in_handle = threading.local()

    def handle(self, record: logging.LogRecord) -> bool:
        # Guard at the handle() boundary so the whole emit chain — including
        # any subclass behavior after super().emit() returns — is covered.
        # Without this, a subclass that logs after delegating up would
        # re-enter the handler (the flag would have already been reset).
        if getattr(self._in_handle, "active", False):
            return False
        self._in_handle.active = True
        try:
            return super().handle(record)
        finally:
            self._in_handle.active = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            try:
                msg = record.getMessage()
            except Exception:
                msg = record.msg or ""
            meta: dict = {}
            if record.exc_info:
                try:
                    import traceback
                    meta["traceback"] = "".join(
                        traceback.format_exception(*record.exc_info)
                    )
                except Exception:
                    pass
            level = self._LEVEL_MAP.get(record.levelno, "info")
            ev = LogEvent(
                ts=time.time(),
                level=level,
                source="python",
                logger=record.name or "",
                message=msg,
                meta=meta,
            )
            self._bus.publish(ev)
        except Exception:
            pass


# ─── Convenience publishers — keep call sites tiny ──────────────────


def publish_tool_event(
    *, name: str, args: dict, result_preview: str = "",
    is_error: bool = False, request_id: str = "",
) -> None:
    """Called from `unified_agent._on_tool_call`. Keeps the cross-
    cutting wiring to one line at the call site."""
    BUS.publish(LogEvent(
        ts=time.time(),
        level="error" if is_error else "info",
        source="tool",
        logger=name or "",
        message=f"{name}({', '.join((args or {}).keys())}) -> {result_preview}".strip(),
        meta={
            "args": args or {},
            "result_preview": result_preview,
            "is_error": is_error,
        },
        request_id=request_id,
    ))


def publish_job_event(
    *, job_id: str, new_status: str, prev_status: str = "",
    error: str = "",
) -> None:
    """Called from `jobs.py` whenever a Job transitions state."""
    level = "error" if new_status in ("failed", "interrupted") else "info"
    BUS.publish(LogEvent(
        ts=time.time(),
        level=level,
        source="job",
        logger=f"job/{job_id[:12]}",
        message=f"{prev_status or '?'} -> {new_status}"
                + (f": {error}" if error else ""),
        meta={
            "job_id": job_id,
            "from": prev_status,
            "to": new_status,
            "error": error,
        },
    ))


def publish_supervisor_event(
    *, job_id: str, decision: str, message: str = "",
) -> None:
    """Called from `job_supervisor.py` at decision points
    (done/escalate/retry/heartbeat)."""
    BUS.publish(LogEvent(
        ts=time.time(),
        level="info" if decision in ("done", "heartbeat") else "warning",
        source="supervisor",
        logger=f"supervisor/{job_id[:12]}",
        message=message or decision,
        meta={"job_id": job_id, "decision": decision},
    ))


def publish_agent_event(
    *, event: str, message: str, request_id: str = "",
) -> None:
    """Called from `agent.progress(...)` so the same thinking-trace
    events that feed the chat SSE also land in the Logs tab."""
    BUS.publish(LogEvent(
        ts=time.time(),
        level="info",
        source="agent",
        logger=event or "",
        message=message or "",
        request_id=request_id,
    ))
