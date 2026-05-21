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
