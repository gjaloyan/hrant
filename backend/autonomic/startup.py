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


def _autonomic_default_paths() -> dict[str, Path]:
    """Audit P1 #3 fix: anchor every default path under the user
    data_dir, not the cwd-relative `knowledge/...` literal.

    Before this, autonomic logs (`tick_log.jsonl`, `lever_log.jsonl`,
    etc.) wrote into `<repo>/knowledge/autonomic/` because `Path("knowledge")`
    resolves relative to cwd (the engine repo directory under systemd).
    Meanwhile `backend/api/health.py` looks under `paths.knowledge_dir()`
    (which is `~/.hrant/data/knowledge` on prod). Result: prod logs
    grew to 10 MB+ in the engine repo while `/api/health` reported
    autonomic 'down' because the dir it checked was empty.

    Env overrides (AUTONOMIC_KNOWLEDGE_ROOT, AUTONOMIC_LEVER_LOG_PATH,
    etc.) still win — this only changes the default fallback when
    no override is set.
    """
    try:
        from ..paths import knowledge_dir
        kdir = knowledge_dir()
    except Exception:
        # paths module not initialised (e.g. very early import in
        # tests) — fall back to the legacy cwd-relative default.
        kdir = Path("knowledge")
    return {
        "knowledge_root": kdir,
        "error_log": kdir / "error_log.jsonl",
        "lever_log": kdir / "autonomic" / "lever_log.jsonl",
        "pending": kdir / "autonomic" / "pending_approvals.jsonl",
        "tick_log": kdir / "autonomic" / "tick_log.jsonl",
    }


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
    # Effective interval: knowledge/autonomic_settings.json > env > 30s.
    # Live updates after boot go through scheduler.set_interval() —
    # see backend.api.autonomic.put_autonomic_settings.
    from .settings import resolve_tick_interval
    interval = resolve_tick_interval()
    # Audit P1 #3 fix: defaults anchor under paths.knowledge_dir()
    # (~/.hrant/data/knowledge on prod), not cwd-relative "knowledge".
    # Env overrides still win.
    _defaults = _autonomic_default_paths()
    knowledge_root = _env_path("AUTONOMIC_KNOWLEDGE_ROOT", str(_defaults["knowledge_root"]))
    error_log = _env_path("AUTONOMIC_ERROR_LOG_PATH", str(_defaults["error_log"]))
    lever_log = _env_path("AUTONOMIC_LEVER_LOG_PATH", str(_defaults["lever_log"]))
    pending = _env_path("AUTONOMIC_PENDING_PATH", str(_defaults["pending"]))
    tick_log = _env_path("AUTONOMIC_TICK_LOG_PATH", str(_defaults["tick_log"]))

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


def unreachable_levers() -> list[str]:
    """Lever modules that no Layer0 rule can ever select.

    A lever with no rule is not "idle waiting for its condition" — it is
    unreachable code that reads as an autonomous capability. Measured
    2026-08-09 on prod against 29 lever modules and a 21-rule table: EIGHT
    could never fire, including `self_heal` and `service_repair`, which have
    knowledge modules on disk describing what they do. If the box breaks,
    nothing repairs it, while the documentation says otherwise.

    Logged at startup so this cannot silently rot again — a capability that
    quietly cannot run is the same class of lie as a budget cap that cannot
    cap.
    """
    import os
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rules = set(re.findall(
            r"FIRE_[A-Z_]+",
            open(os.path.join(here, "layer0.py"), encoding="utf-8").read()))
        mods = [f[:-3] for f in os.listdir(os.path.join(here, "levers"))
                if f.endswith(".py") and f != "__init__.py"]
    except Exception:
        return []
    return sorted(m for m in mods if f"FIRE_{m.upper()}" not in rules)


async def start_autonomic_scheduler(bundle: SchedulerBundle) -> None:
    try:
        await bundle.scheduler.start()
        log.info("Autonomic scheduler started")
        _orphans = unreachable_levers()
        if _orphans:
            log.warning(
                "%d lever module(s) have NO Layer0 rule and can never fire: "
                "%s — they read as capabilities the agent does not have",
                len(_orphans), ", ".join(_orphans),
            )
    except Exception as exc:
        log.error("Autonomic scheduler failed to start: %s", exc)


async def stop_autonomic_scheduler(bundle: SchedulerBundle) -> None:
    try:
        await bundle.scheduler.stop()
        log.info("Autonomic scheduler stopped")
    except Exception as exc:
        log.warning("Autonomic scheduler stop raised: %s", exc)
