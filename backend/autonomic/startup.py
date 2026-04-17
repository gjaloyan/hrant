"""Glue between the FastAPI app lifespan and the autonomic scheduler.

Reads env vars for configurability:
  AUTONOMIC_ENABLED_PATH   - kill-switch file (default: knowledge/autonomic/ENABLED)
  AUTONOMIC_TICK_SECONDS   - base tick interval (default: 30.0)
  AUTONOMIC_KNOWLEDGE_ROOT - knowledge dir for state builder (default: knowledge)
  AUTONOMIC_ERROR_LOG_PATH - error_log.jsonl path (default: knowledge/error_log.jsonl)
  AUTONOMIC_LEVER_LOG_PATH - lever_log.jsonl path (default: knowledge/autonomic/lever_log.jsonl)
  AUTONOMIC_PENDING_PATH   - pending_approvals.jsonl (default: knowledge/autonomic/pending_approvals.jsonl)
  AUTONOMIC_TICK_LOG_PATH  - tick_log.jsonl path (default: knowledge/autonomic/tick_log.jsonl)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .events import EventBus
from .executor import LeverExecutor
from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .layer0 import Layer0Engine, default_rules
from .levers import LeverRegistry, clear_registry, register_default_immune_levers
from .safety import SafetyGate
from .scheduler import AutonomicScheduler
from .state import StateSnapshotBuilder
from .tick import make_real_tick

log = logging.getLogger(__name__)


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


def build_scheduler() -> AutonomicScheduler:
    enabled_path = _env_path("AUTONOMIC_ENABLED_PATH", str(DEFAULT_ENABLED_PATH))
    interval = float(os.environ.get("AUTONOMIC_TICK_SECONDS", "30"))
    knowledge_root = _env_path("AUTONOMIC_KNOWLEDGE_ROOT", "knowledge")
    error_log = _env_path("AUTONOMIC_ERROR_LOG_PATH", "knowledge/error_log.jsonl")
    lever_log = _env_path("AUTONOMIC_LEVER_LOG_PATH", "knowledge/autonomic/lever_log.jsonl")
    pending = _env_path("AUTONOMIC_PENDING_PATH", "knowledge/autonomic/pending_approvals.jsonl")
    tick_log = _env_path("AUTONOMIC_TICK_LOG_PATH", "knowledge/autonomic/tick_log.jsonl")

    clear_registry()
    register_default_immune_levers()
    registry = LeverRegistry.instance()

    gate = SafetyGate(pending_approvals_path=pending)
    bus = EventBus()
    executor = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=knowledge_root,
        error_log_path=error_log,
        pending_approvals_path=pending,
        lever_log_path=lever_log,
    )
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=registry,
        executor=executor,
        tick_log_path=tick_log,
        event_bus=bus,
    )

    return AutonomicScheduler(
        kill_switch=KillSwitch(enabled_path),
        on_tick=tick,
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
