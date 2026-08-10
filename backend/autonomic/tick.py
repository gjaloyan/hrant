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
from .types import LeverStatus, TickDecision, TickDecisionSource, utcnow

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
        decision = _next_decision(state, engine)
        executed = False
        note = ""
        if decision.lever is not None:
            lever = registry.get(decision.lever)
            if lever is None:
                note = f"unknown_lever:{decision.lever}"
                log.warning(note)
            else:
                report = executor.execute(lever, decision.params, state)
                executed = True
                _record_immune_outcome(decision, report)
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


def _next_decision(state, engine: Layer0Engine) -> TickDecision:
    """A queued follow-up outranks the periodic table.

    Layer 0 is a schedule: things that should happen eventually. A follow-up
    is a reaction to something that already happened — an error matched, a
    repair planned — and making it wait behind the nightly log rotation is how
    a two-step repair takes until tomorrow. Draining one per tick keeps the
    "one lever per tick" invariant the executor and the safety gate assume.
    """
    from .followups import FOLLOWUPS
    try:
        fu = FOLLOWUPS.pop()
    except Exception as exc:
        log.warning("followup pop failed: %s", exc)
        fu = None
    if fu is not None:
        params = dict(fu.params)
        if fu.signature_id:
            params.setdefault("_signature_id", fu.signature_id)
        return TickDecision(
            source=TickDecisionSource.L0_IMMUNE,
            lever=fu.lever,
            params=params,
            reason=fu.reason or f"follow_up from {fu.origin or 'unknown'}",
            rule_name=f"followup:{fu.id}",
        )
    return engine.evaluate(state)


def _record_immune_outcome(decision: TickDecision, report) -> None:
    """Close the learning loop: did the repair this signature prescribed work?

    Only the REPAIR step counts. FIRE_SELF_HEAL merely names a lever, so
    scoring its success would record a perfect record for signatures whose
    fixes never work — the exact self-congratulatory metric that makes a
    health dashboard worse than none.
    """
    sig_id = (decision.params or {}).get("_signature_id")
    if not sig_id or decision.lever == "FIRE_SELF_HEAL":
        return
    success = report is not None and \
        getattr(report, "status", None) is LeverStatus.SUCCESS
    try:
        from .immune import FireLog, SignatureStore
        SignatureStore().record_outcome(sig_id, success)
        FireLog().note_outcome(sig_id, success)
    except Exception as exc:
        log.warning("immune outcome record failed for %s: %s", sig_id, exc)


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
