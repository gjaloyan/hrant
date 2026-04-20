"""Glue between the FastAPI app lifespan and the autonomic scheduler.

Reads env vars for configurability (see AUTONOMIC_*_PATH constants below).
build_scheduler returns a SchedulerBundle that main.py stashes on app.state
so the api.py router can reach gate / executor / builder / log paths.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .events import EventBus
from .executor import LeverExecutor
from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .layer0 import Layer0Engine, default_rules
from .levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
from .safety import SafetyGate
from .scheduler import AutonomicScheduler
from .state import StateSnapshotBuilder
from .tick import make_real_tick

log = logging.getLogger(__name__)


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


@dataclass
class SchedulerBundle:
    scheduler: AutonomicScheduler
    gate: SafetyGate
    executor: LeverExecutor
    builder: StateSnapshotBuilder
    registry: LeverRegistry
    kill_switch: KillSwitch
    lever_log_path: Path
    tick_log_path: Path


def build_scheduler() -> SchedulerBundle:
    enabled_path = _env_path("AUTONOMIC_ENABLED_PATH", str(DEFAULT_ENABLED_PATH))
    interval = float(os.environ.get("AUTONOMIC_TICK_SECONDS", "30"))
    knowledge_root = _env_path("AUTONOMIC_KNOWLEDGE_ROOT", "knowledge")
    error_log = _env_path("AUTONOMIC_ERROR_LOG_PATH", "knowledge/error_log.jsonl")
    lever_log = _env_path("AUTONOMIC_LEVER_LOG_PATH", "knowledge/autonomic/lever_log.jsonl")
    pending = _env_path("AUTONOMIC_PENDING_PATH", "knowledge/autonomic/pending_approvals.jsonl")
    tick_log = _env_path("AUTONOMIC_TICK_LOG_PATH", "knowledge/autonomic/tick_log.jsonl")

    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
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
    kill_switch = KillSwitch(enabled_path)
    scheduler = AutonomicScheduler(
        kill_switch=kill_switch,
        on_tick=tick,
        tick_interval_seconds=interval,
    )
    return SchedulerBundle(
        scheduler=scheduler,
        gate=gate,
        executor=executor,
        builder=builder,
        registry=registry,
        kill_switch=kill_switch,
        lever_log_path=lever_log,
        tick_log_path=tick_log,
    )


async def start_autonomic_scheduler(bundle: SchedulerBundle) -> None:
    try:
        await bundle.scheduler.start()
        log.info("Autonomic scheduler started")
    except Exception as exc:
        log.error("Autonomic scheduler failed to start: %s", exc)


async def stop_autonomic_scheduler(bundle: SchedulerBundle) -> None:
    try:
        await bundle.scheduler.stop()
        log.info("Autonomic scheduler stopped")
    except Exception as exc:
        log.warning("Autonomic scheduler stop raised: %s", exc)
