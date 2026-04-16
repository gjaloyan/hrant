"""Glue between the FastAPI app lifespan and the autonomic scheduler.

Reads env vars for configurability:
  AUTONOMIC_ENABLED_PATH - path to the kill-switch file
  AUTONOMIC_TICK_SECONDS - base tick interval (float)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .scheduler import AutonomicScheduler

log = logging.getLogger(__name__)


def _noop_tick() -> None:
    """D-01 placeholder tick - real routing comes in D-02."""
    return None


def build_scheduler() -> AutonomicScheduler:
    enabled_path = Path(os.environ.get("AUTONOMIC_ENABLED_PATH", str(DEFAULT_ENABLED_PATH)))
    interval = float(os.environ.get("AUTONOMIC_TICK_SECONDS", "30"))
    return AutonomicScheduler(
        kill_switch=KillSwitch(enabled_path),
        on_tick=_noop_tick,
        tick_interval_seconds=interval,
    )


async def start_autonomic_scheduler(scheduler: AutonomicScheduler) -> None:
    try:
        await scheduler.start()
        log.info("Autonomic scheduler started")
    except Exception as exc:
        log.error("Autonomic scheduler failed to start: %s", exc)


async def stop_autonomic_scheduler(scheduler: AutonomicScheduler) -> None:
    try:
        await scheduler.stop()
        log.info("Autonomic scheduler stopped")
    except Exception as exc:
        log.warning("Autonomic scheduler stop raised: %s", exc)
