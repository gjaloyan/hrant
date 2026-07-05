"""Layer 0 reflex engine — rule-based pure-Python decisions per tick."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from .types import StateSnapshot, TickDecision, TickDecisionSource

log = logging.getLogger(__name__)


@dataclass
class LayerZeroRule:
    name: str
    predicate: Callable[[StateSnapshot], bool]
    lever: str
    params: dict = field(default_factory=dict)
    cooldown_seconds: float = 30.0


class Layer0Engine:
    def __init__(self, rules: list[LayerZeroRule]) -> None:
        self._rules = list(rules)
        self._last_fired: dict[str, float] = {}

    def evaluate(self, state: StateSnapshot) -> TickDecision:
        now = time.monotonic()
        last_cooldown_hit: TickDecision | None = None
        for rule in self._rules:
            try:
                matched = bool(rule.predicate(state))
            except Exception as exc:
                log.warning("Layer0 rule %r predicate raised: %s", rule.name, exc)
                continue
            if not matched:
                continue
            last = self._last_fired.get(rule.name)
            if last is not None and (now - last) < rule.cooldown_seconds:
                if last_cooldown_hit is None:
                    last_cooldown_hit = TickDecision(
                        source=TickDecisionSource.L0_REFLEX,
                        lever=None,
                        params={},
                        reason=f"cooldown:{rule.name}",
                        rule_name=rule.name,
                    )
                continue
            self._last_fired[rule.name] = now
            return TickDecision(
                source=TickDecisionSource.L0_REFLEX,
                lever=rule.lever,
                params=dict(rule.params),
                reason=f"rule_matched:{rule.name}",
                rule_name=rule.name,
            )
        if last_cooldown_hit is not None:
            return last_cooldown_hit
        return TickDecision(
            source=TickDecisionSource.L0_REFLEX,
            lever=None,
            params={},
            reason="idle_no_rules_matched",
        )


def default_rules() -> list[LayerZeroRule]:
    return [
        LayerZeroRule(
            name="disk_low",
            predicate=lambda s: s.disk_free_gb < 2.0,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "disk_low"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="memory_low",
            predicate=lambda s: s.memory_free_gb < 0.5,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "memory_low"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="cpu_high",
            predicate=lambda s: s.cpu_load_1m > 4.0,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "cpu_high"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="errors_present",
            predicate=lambda s: len(s.recent_errors) > 0,
            lever="FIRE_ERROR_TRIAGE",
            params={},
            cooldown_seconds=120.0,
        ),
        LayerZeroRule(
            # Starvation fix (2026-06-12): this rule lived LAST in
            # the list, and the engine is first-match-wins. After a
            # service restart every periodic rule's cooldown is
            # clear, so they all queue ahead by list order and a DUE
            # REMINDER waited 5+ minutes for a tick slot (caught
            # live: integrity + goal_propose + capability_scan each
            # took a tick while a due message sat pending). Now the
            # FIRST of the always-true periodic rules: reactive rules
            # (disk/memory/cpu/errors — the four above) still preempt
            # genuinely urgent conditions, but among housekeeping,
            # user-facing delivery wins every time.
            name="scheduled_messages_tick",
            predicate=lambda s: True,
            lever="FIRE_SCHEDULED_MESSAGES",
            params={},
            cooldown_seconds=60.0,
        ),
        LayerZeroRule(
            name="integrity_tick",
            predicate=lambda s: True,
            lever="FIRE_INTEGRITY_HEARTBEAT",
            params={},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="goal_propose_tick",
            predicate=lambda s: True,
            lever="FIRE_GOAL_PROPOSE",
            params={},
            cooldown_seconds=3600.0,
        ),
        # `consolidation_tick` was retired 2026-05-27 (audit T3.2).
        # The dedicated consolidation scheduler in
        # `backend/consolidation/scheduler.py` already runs daily +
        # idle-aware. The layer-0 rule fired ~86 times/day and was
        # ~99% no-ops ("no_unconsolidated_sessions"), polluting the
        # lever_log without doing additional work. The lever class
        # stays registered so it can still be invoked manually via
        # the autonomic API.
        LayerZeroRule(
            name="capability_scan_tick",
            predicate=lambda s: True,
            lever="FIRE_CAPABILITY_SCAN",
            params={},
            cooldown_seconds=21600.0,
        ),
        LayerZeroRule(
            name="self_study_tick",
            predicate=lambda s: True,
            lever="FIRE_SELF_STUDY",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="graph_maintenance_tick",
            predicate=lambda s: True,
            lever="FIRE_GRAPH_MAINTENANCE",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="proactive_learn_tick",
            predicate=lambda s: True,
            lever="FIRE_PROACTIVE_LEARN",
            params={},
            cooldown_seconds=3600.0,
        ),
        LayerZeroRule(
            # Drive user/learning goals to completion, one subtask per fire
            # (2026-06-25 audit: those goal types were created but never
            # executed by anything). Long cooldown bounds the spend.
            name="goal_drive_tick",
            predicate=lambda s: True,
            lever="FIRE_GOAL_DRIVE",
            params={},
            cooldown_seconds=7200.0,
        ),
        LayerZeroRule(
            name="note_curation_tick",
            predicate=lambda s: True,
            lever="FIRE_NOTE_CURATION",
            params={},
            cooldown_seconds=604800.0,
        ),
        LayerZeroRule(
            name="model_eval_tick",
            predicate=lambda s: True,
            lever="FIRE_MODEL_EVAL",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="session_archive_tick",
            predicate=lambda s: True,
            lever="FIRE_SESSION_ARCHIVE",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="cost_audit_tick",
            predicate=lambda s: True,
            lever="FIRE_COST_AUDIT",
            params={},
            cooldown_seconds=3600.0,
        ),
        LayerZeroRule(
            name="self_reflection_tick",
            predicate=lambda s: True,
            lever="FIRE_SELF_REFLECTION",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="finetune_qc_tick",
            predicate=lambda s: True,
            lever="FIRE_FINETUNE_QC",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="gap_detection_tick",
            predicate=lambda s: True,
            lever="FIRE_GAP_DETECTION",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="log_rotation_tick",
            predicate=lambda s: True,
            lever="FIRE_LOG_ROTATION",
            params={},
            cooldown_seconds=86400.0,  # daily
        ),
        LayerZeroRule(
            name="goal_executor_tick",
            predicate=lambda s: True,
            lever="FIRE_GOAL_EXECUTOR",
            params={},
            cooldown_seconds=86400.0,  # daily
        ),
        LayerZeroRule(
            name="fact_embedding_backfill_tick",
            predicate=lambda s: True,
            lever="FIRE_FACT_EMBEDDING_BACKFILL",
            params={},
            cooldown_seconds=86400.0,  # daily
        ),
        # scheduled_messages_tick moved to the head of the list
        # (right after the safety triad) 2026-06-12 — see the
        # starvation-fix comment there.
    ]
