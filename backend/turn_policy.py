"""Per-turn execution policy.

The normal agent is allowed to act and to learn from a turn.  Audit turns are
different: they may inspect, reason, and verify, but they must not mutate the
world or feed the result back into conversational/learning memory.

The policy lives in a ContextVar so deeply nested tool handlers and delegated
execution paths see the same boundary without threading another boolean
through every function signature.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnPolicy:
    mode: str = "normal"
    read_only: bool = False
    persist_cognitive_state: bool = True
    enforce_action_progress: bool = True


NORMAL_POLICY = TurnPolicy()
AUDIT_POLICY = TurnPolicy(
    mode="audit",
    read_only=True,
    persist_cognitive_state=False,
    enforce_action_progress=False,
)

_current_policy: ContextVar[TurnPolicy] = ContextVar(
    "hrant_turn_policy", default=NORMAL_POLICY,
)


def begin_turn(*, audit_mode: bool = False):
    """Install a fresh policy for the current turn and return its token."""
    return _current_policy.set(AUDIT_POLICY if audit_mode else NORMAL_POLICY)


def reset_turn(token) -> None:
    try:
        _current_policy.reset(token)
    except (LookupError, ValueError):
        pass


def current_policy() -> TurnPolicy:
    return _current_policy.get()


def is_audit_mode() -> bool:
    return current_policy().mode == "audit"
