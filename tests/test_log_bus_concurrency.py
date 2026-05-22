"""Concurrent publishers + subscribers don't race or drop normal load.

The bus uses RLock around the ring; subscriber queues are bounded so
backpressure surfaces as silent drops (which is correct), not as a
publisher block. This test pins both invariants."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest


@pytest.fixture
def isolated_bus(tmp_path, monkeypatch):
    from backend import log_bus as _lb
    monkeypatch.setattr(_lb, "_logs_dir", lambda: tmp_path)
    _lb.BUS.clear()
    yield _lb.BUS
    _lb.BUS.clear()


def test_concurrent_publishers_no_loss(isolated_bus):
    from backend.log_bus import BUS, LogEvent
    N = 200
    threads = 8

    def _worker(tid):
        for i in range(N):
            BUS.publish(LogEvent(
                ts=time.time(), level="info", source="python",
                logger=f"t{tid}", message=f"{tid}:{i}",
            ))

    ts = [threading.Thread(target=_worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rows = BUS.tail(limit=0)  # 0 = unlimited
    assert len(rows) == threads * N


def test_publisher_does_not_block_when_subscriber_is_slow(isolated_bus):
    """Subscriber queues have maxsize=2; publishers must keep flowing
    even if the consumer never drains."""
    from backend.log_bus import BUS, LogEvent

    async def _run():
        q = BUS.subscribe(maxsize=2)
        # Don't drain q at all.
        t0 = time.monotonic()
        for i in range(50):
            BUS.publish(LogEvent(
                ts=time.time(), level="info", source="python",
                logger="t", message=f"m{i}",
            ))
        elapsed = time.monotonic() - t0
        BUS.unsubscribe(q)
        return elapsed

    elapsed = asyncio.run(_run())
    # 50 publishes through a stalled-queue bus must be near-instant
    # (no awaits, no blocking). 1 second is generous.
    assert elapsed < 1.0
