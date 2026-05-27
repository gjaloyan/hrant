"""Base class for all autonomic levers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from .types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    StateSnapshot,
)


def resolve_knowledge_path(p: Path | str) -> Path:
    """Resolve a lever's relative `knowledge/...` path against the
    user's data directory.

    Audit 2026-05-27 #5 found ~23 levers using `Path("knowledge/...")`
    as their DEFAULT — those resolve against the SERVICE'S CWD
    (`/home/hrant/hrant/knowledge/...`), NOT the user's actual data
    dir (`~/.hrant/data/knowledge/...`). Result: 8 days of self_
    reflection_log + model_eval_log silently written to a phantom
    path while the audit tooling and the rest of the agent looked
    in the right place and saw an empty file.

    This helper is the surgical fix: legacy `DEFAULT_X = Path("knowledge/foo")`
    constants stay (so existing tests passing absolute tmp_path
    values keep working), but `run()` passes them through this
    resolver so the on-disk write lands where the audit grep'd.

    Behaviour:
      - Absolute path → returned untouched.
      - Path starting with `knowledge/` → re-rooted at
        `paths.knowledge_dir()`.
      - Other relatives → resolved against CWD (legacy behaviour).
    """
    p = Path(p)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "knowledge":
        from backend import paths as _paths
        return _paths.knowledge_dir().joinpath(*parts[1:])
    return p.resolve()

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
