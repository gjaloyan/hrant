"""LeverExecutor — ties SafetyGate, Lever.run, lever_log, and event bus."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .events import EventBus
from .lever import Lever
from .safety import SafetyDecision, SafetyGate
from .types import LeverReport, LeverStatus, StateSnapshot, utcnow

log = logging.getLogger(__name__)

DEFAULT_LEVER_LOG_PATH = Path("knowledge/autonomic/lever_log.jsonl")


class LeverExecutor:
    def __init__(
        self,
        gate: SafetyGate,
        lever_log_path: Path | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._gate = gate
        self._log_path = lever_log_path or DEFAULT_LEVER_LOG_PATH
        self._bus = event_bus
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        lever: Lever,
        params: dict[str, Any],
        state: StateSnapshot,
    ) -> LeverReport | None:
        decision = self._gate.evaluate(lever, params)
        if decision is SafetyDecision.BLOCK:
            log.info("LeverExecutor: BLOCK %s", lever.name)
            return None
        if decision is SafetyDecision.QUEUE_FOR_APPROVAL:
            log.info("LeverExecutor: QUEUE %s", lever.name)
            return None

        if not lever.preconditions(state):
            now = utcnow()
            report = LeverReport(
                lever=lever.name,
                params=dict(params),
                started_at=now,
                finished_at=now,
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="preconditions_false",
            )
            self._persist(report)
            return report

        started = utcnow()
        try:
            report = lever.run(dict(params), {"state": state})
        except Exception as exc:
            report = LeverReport(
                lever=lever.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={},
                reason=f"exception:{exc}",
            )
        self._persist(report)
        return report

    def _persist(self, report: LeverReport) -> None:
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(report.to_jsonl())
        except OSError as exc:
            log.warning("Could not append lever_log: %s", exc)
        if self._bus is not None:
            try:
                self._bus.publish(
                    "lever.executed",
                    {
                        "lever": report.lever,
                        "status": report.status.value,
                        "reason": report.reason,
                    },
                )
            except Exception as exc:
                log.warning("Event bus publish failed: %s", exc)
