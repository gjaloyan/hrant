"""Autonomic scheduler — asyncio tick loop that respects the kill switch.

D-01 version: fires a single `on_tick` callable at regular intervals, with
exception isolation. Full L0 routing and multi-cadence ticks (fast/medium/
slow/nightly) come in D-02.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .kill_switch import KillSwitch

log = logging.getLogger(__name__)


class AutonomicScheduler:
    def __init__(
        self,
        kill_switch: KillSwitch,
        on_tick: Callable[[], None],
        tick_interval_seconds: float = 30.0,
    ) -> None:
        self._kill_switch = kill_switch
        self._on_tick = on_tick
        self._interval = tick_interval_seconds
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="autonomic-scheduler")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._interval + 1.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def interval(self) -> float:
        return self._interval

    def set_interval(self, seconds: float) -> None:
        """Live-update the tick interval. The current sleep finishes
        on the old value (asyncio.wait_for is already pending); the
        NEXT iteration picks up the new value. Caller validates the
        range — anything stranger than that is the caller's problem."""
        self._interval = float(seconds)

    async def _loop(self) -> None:
        log.info("Autonomic scheduler loop starting (interval=%ss)", self._interval)
        try:
            while not self._stopping.is_set():
                if self._kill_switch.is_enabled():
                    try:
                        self._on_tick()
                    except Exception as exc:
                        log.warning("Autonomic tick raised: %s", exc)
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            log.info("Autonomic scheduler loop stopped")
