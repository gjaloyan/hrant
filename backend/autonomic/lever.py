"""Base class for all autonomic levers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    StateSnapshot,
)

_REQUIRED_CLASS_ATTRS = (
    "name",
    "category",
    "safety",
    "executor",
    "estimated_cost",
    "required_context",
)


class Lever(ABC):
    """Abstract base for autonomic levers.

    Subclasses MUST declare: name, category, safety, executor,
    estimated_cost, required_context. Missing any of these triggers
    TypeError at instantiation.
    """

    name: ClassVar[str]
    category: ClassVar[LeverCategory]
    safety: ClassVar[LeverSafety]
    executor: ClassVar[str]
    estimated_cost: ClassVar[Cost]
    required_context: ClassVar[list[str]]

    def __init__(self) -> None:
        for attr in _REQUIRED_CLASS_ATTRS:
            if not hasattr(type(self), attr) or getattr(type(self), attr, None) is getattr(Lever, attr, None):
                raise TypeError(
                    f"{type(self).__name__} missing required attribute {attr!r}"
                )

    @abstractmethod
    def preconditions(self, state: StateSnapshot) -> bool:
        """Return True if the lever is allowed to run in this state."""

    @abstractmethod
    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        """Execute the lever. MUST return a LeverReport."""

    def rollback(self, report: LeverReport) -> None:
        """Optional rollback. Default: noop."""
        return None
