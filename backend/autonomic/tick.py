"""Real scheduler tick — builds state, evaluates L0, runs lever, logs tick."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .events import EventBus
from .executor import LeverExecutor
from .layer0 import Layer0Engine
from .levers import LeverRegistry
from .state import StateSnapshotBuilder
from .types import TickDecision, utcnow

log = logging.getLogger(__name__)


def make_real_tick(
    builder: StateSnapshotBuilder,
    engine: Layer0Engine,
    registry: LeverRegistry,
    executor: LeverExecutor,
    tick_log_path: Path,
    event_bus: EventBus | None = None,
) -> Callable[[], None]:
    tick_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _tick() -> None:
        state = builder.build()
        decision = engine.evaluate(state)
        executed = False
        note = ""
        if decision.lever is not None:
            lever = registry.get(decision.lever)
            if lever is None:
                note = f"unknown_lever:{decision.lever}"
                log.warning(note)
            else:
                executor.execute(lever, decision.params, state)
                executed = True
        _append_tick_log(tick_log_path, decision, executed=executed, note=note)
        if event_bus is not None:
            try:
                event_bus.publish(
                    "tick.completed",
                    {
                        "source": decision.source.value,
                        "lever": decision.lever,
                        "reason": decision.reason,
                        "executed": executed,
                    },
                )
            except Exception as exc:
                log.warning("tick.completed publish failed: %s", exc)

    return _tick


def _append_tick_log(
    path: Path,
    decision: TickDecision,
    *,
    executed: bool,
    note: str,
) -> None:
    entry: dict[str, Any] = {
        "ts": utcnow().isoformat(),
        "source": decision.source.value,
        "lever": decision.lever,
        "params": decision.params,
        "reason": decision.reason,
        "rule_name": decision.rule_name,
        "executed": executed,
        "note": note,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("tick_log append failed: %s", exc)
