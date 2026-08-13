"""Per-turn capability surface and execution-budget broker.

The model proposes tool calls; code decides which capabilities are visible and
whether one concrete call may execute.  Keeping both decisions in one object
prevents the schema filter, audit guard, and loop budget from drifting into
independent name lists inside the main agent loop.

Normal turns preserve the owner's intentionally generous execution policy.
Audit turns are explicitly diagnostic and receive a bounded profile so a small
model cannot turn a service inspection into hundreds of probes or an unbounded
context-refeeding loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from .tool_registry import ToolCallSemantics, ToolRegistry


_DEFAULT_AUDIT_ITERATIONS = 32
_DEFAULT_AUDIT_TOOL_CALLS = 32
_DEFAULT_AUDIT_INPUT_TOKENS = 60_000


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


@dataclass(frozen=True)
class ExecutionBudget:
    """Hard execution limits for one main tool loop.

    A value of zero disables a tool/input limit.  ``max_iterations`` is always
    positive because every provider needs at least one opportunity to answer.
    """

    profile: str
    max_iterations: int
    max_tool_calls: int = 0
    max_input_tokens: int = 0


@dataclass(frozen=True)
class CapabilityDecision:
    allowed: bool
    tool: str
    semantics: ToolCallSemantics
    reason: str = ""
    error_code: str = ""


def budget_from_config(
    *, audit_mode: bool, normal_max_iterations: int,
) -> ExecutionBudget:
    """Resolve a fresh budget at turn entry so live config applies instantly."""
    try:
        from .config import CONFIG
        router_cfg = CONFIG.router or {}
    except Exception:
        router_cfg = {}

    if audit_mode:
        return ExecutionBudget(
            profile="audit",
            max_iterations=_positive_int(
                router_cfg.get("audit_loop_max_iterations"),
                _DEFAULT_AUDIT_ITERATIONS,
            ),
            max_tool_calls=_positive_int(
                router_cfg.get("audit_loop_max_tool_calls"),
                _DEFAULT_AUDIT_TOOL_CALLS,
            ),
            max_input_tokens=_positive_int(
                router_cfg.get("audit_loop_input_budget"),
                _DEFAULT_AUDIT_INPUT_TOKENS,
            ),
        )

    return ExecutionBudget(
        profile="normal",
        max_iterations=_positive_int(normal_max_iterations, 500),
        # Disabled by default.  This is an opt-in emergency lever, not a
        # replacement for the owner's generous normal-loop policy.
        max_tool_calls=_non_negative_int(
            router_cfg.get("tool_loop_max_tool_calls"), 0,
        ),
        # Provider loops already enforce this existing setting.  Mirroring it
        # here makes the receipt truthful and blocks before a handler too.
        max_input_tokens=_non_negative_int(
            router_cfg.get("tool_loop_input_budget"), 0,
        ),
    )


class CapabilityBroker:
    """One authority for schema exposure and concrete call admission."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        audit_mode: bool = False,
        budget: Optional[ExecutionBudget] = None,
    ) -> None:
        self.registry = registry
        self.audit_mode = bool(audit_mode)
        self.budget = budget or budget_from_config(
            audit_mode=self.audit_mode,
            normal_max_iterations=500,
        )
        self._attempted = 0
        self._allowed = 0
        self._denied = 0
        self._input_tokens_observed = 0
        self._exhaustion_reason = ""

    def visible_names(self) -> set[str]:
        """Current base + loaded bundle + runtime capability surface."""
        from .tool_bundles import (
            BASE_TOOLS, expand_loaded, get_loaded_bundles,
        )

        allowed = set(BASE_TOOLS) | expand_loaded(get_loaded_bundles())
        # Skill and MCP tools are registered dynamically and cannot appear in
        # the static bundle catalog.  Preserve their normal-mode reachability.
        allowed |= {
            name for name, tool in self.registry.tools.items()
            if not str(getattr(tool, "origin", "")).startswith("builtin")
        }
        allowed &= set(self.registry.tools)
        if self.audit_mode:
            allowed = self.registry.audit_visible_names(allowed)
        return allowed

    def tool_schema(self) -> list[dict[str, Any]]:
        return self.registry.to_anthropic_list(
            filter_names=self.visible_names(),
        )

    def iteration_limit(self, requested: Optional[int] = None) -> int:
        """Clamp an optional cascade request to the turn's hard ceiling."""
        if requested is None:
            return self.budget.max_iterations
        requested_n = _positive_int(requested, self.budget.max_iterations)
        return min(requested_n, self.budget.max_iterations)

    def authorize(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
        *,
        input_tokens_used: int = 0,
    ) -> CapabilityDecision:
        """Decide before handler execution and account for the attempt."""
        self._attempted += 1
        self.observe_input_tokens(input_tokens_used)
        semantics = self.registry.resolve_call_semantics(name, arguments)

        if name not in self.registry.tools:
            return self._deny(
                name, semantics,
                reason="tool is not registered in this process",
                error_code="TOOL_NOT_FOUND",
            )

        if self.audit_mode and not semantics.audit_allowed:
            return self._deny(
                name, semantics,
                reason=(
                    "read-only audit policy refuses this concrete call "
                    f"(resolved effect: {semantics.effect.value})"
                ),
                error_code="CAPABILITY_DENIED",
            )

        if name not in self.visible_names():
            return self._deny(
                name, semantics,
                reason=(
                    "tool is registered but is not available in the current "
                    "base/bundle/runtime capability surface"
                ),
                error_code="CAPABILITY_NOT_AVAILABLE",
            )

        if (
            self.budget.max_input_tokens > 0
            and self._input_tokens_observed >= self.budget.max_input_tokens
        ):
            self._exhaustion_reason = "input_token_budget"
            return self._deny(
                name, semantics,
                reason=(
                    f"turn input-token budget reached "
                    f"({self._input_tokens_observed} >= "
                    f"{self.budget.max_input_tokens})"
                ),
                error_code="TURN_BUDGET_EXCEEDED",
            )

        if (
            self.budget.max_tool_calls > 0
            and self._allowed >= self.budget.max_tool_calls
        ):
            self._exhaustion_reason = "tool_call_budget"
            return self._deny(
                name, semantics,
                reason=(
                    f"turn tool-call budget reached "
                    f"({self._allowed} >= {self.budget.max_tool_calls})"
                ),
                error_code="TURN_BUDGET_EXCEEDED",
            )

        self._allowed += 1
        return CapabilityDecision(
            allowed=True,
            tool=name,
            semantics=semantics,
        )

    def _deny(
        self,
        name: str,
        semantics: ToolCallSemantics,
        *,
        reason: str,
        error_code: str,
    ) -> CapabilityDecision:
        self._denied += 1
        return CapabilityDecision(
            allowed=False,
            tool=name,
            semantics=semantics,
            reason=reason,
            error_code=error_code,
        )

    def observe_input_tokens(self, value: Any) -> None:
        """Update receipt telemetry even when no further tool is proposed."""
        self._input_tokens_observed = max(
            self._input_tokens_observed,
            _non_negative_int(value, 0),
        )

    def denial_result(self, decision: CapabilityDecision) -> tuple[str, bool]:
        """Stable machine-readable denial for the model and trace receipt."""
        if decision.error_code == "TOOL_NOT_FOUND":
            return f"[tool '{decision.tool}' not found in registry]", True
        payload = {
            "ok": False,
            "error": decision.error_code or "CAPABILITY_DENIED",
            "tool": decision.tool,
            "effect": decision.semantics.effect.value,
            "reason": decision.reason,
            "budget": self.snapshot(),
        }
        if decision.error_code == "TURN_BUDGET_EXCEEDED":
            payload["instruction"] = (
                "Stop calling tools and return the best evidence-based partial "
                "report now; name what remains unchecked."
            )
        return json.dumps(payload, ensure_ascii=False), True

    def snapshot(self) -> dict[str, Any]:
        return {
            "profile": self.budget.profile,
            "max_iterations": self.budget.max_iterations,
            "max_tool_calls": self.budget.max_tool_calls,
            "max_input_tokens": self.budget.max_input_tokens,
            "tool_calls_attempted": self._attempted,
            "tool_calls_allowed": self._allowed,
            "tool_calls_denied": self._denied,
            "input_tokens_observed": self._input_tokens_observed,
            "exhausted": bool(self._exhaustion_reason),
            "exhaustion_reason": self._exhaustion_reason,
        }

    def prompt_block(self) -> str:
        """Concise contract shown to the model for bounded profiles."""
        b = self.budget
        calls = str(b.max_tool_calls) if b.max_tool_calls else "unlimited"
        tokens = str(b.max_input_tokens) if b.max_input_tokens else "unlimited"
        return (
            "# EXECUTION BUDGET\n"
            f"Profile: {b.profile}. Maximum LLM iterations: "
            f"{b.max_iterations}; allowed tool calls: {calls}; accumulated "
            f"input-token threshold: {tokens}. If the broker returns "
            "TURN_BUDGET_EXCEEDED, stop probing and report the evidence "
            "already collected plus the remaining uncertainty."
        )


__all__ = [
    "CapabilityBroker", "CapabilityDecision", "ExecutionBudget",
    "budget_from_config",
]
