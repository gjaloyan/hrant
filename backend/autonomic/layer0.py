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
                return TickDecision(
                    source=TickDecisionSource.L0_REFLEX,
                    lever=None,
                    params={},
                    reason=f"cooldown:{rule.name}",
                    rule_name=rule.name,
                )
            self._last_fired[rule.name] = now
            return TickDecision(
                source=TickDecisionSource.L0_REFLEX,
                lever=rule.lever,
                params=dict(rule.params),
                reason=f"rule_matched:{rule.name}",
                rule_name=rule.name,
            )
        return TickDecision(
            source=TickDecisionSource.L0_REFLEX,
            lever=None,
            params={},
            reason="idle_no_rules_matched",
        )
