"""Safety gate: classifies lever execution requests by their safety tier."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .lever import Lever
from .types import LeverSafety, utcnow


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    QUEUE_FOR_APPROVAL = "queue_for_approval"
    BLOCK = "block"


DEFAULT_PENDING_PATH = Path("knowledge/autonomic/pending_approvals.jsonl")


class SafetyGate:
    """Enforces green/yellow/red policy on lever execution.

    - GREEN: ALLOW (executor may run immediately).
    - YELLOW: QUEUE_FOR_APPROVAL (writes to pending_approvals.jsonl, NOT executed).
    - RED: BLOCK (refused entirely; autonomic cannot trigger RED levers).
    """

    def __init__(self, pending_approvals_path: Path | None = None) -> None:
        self._pending_path = pending_approvals_path or DEFAULT_PENDING_PATH
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._pending_path.exists():
            self._pending_path.touch()

    def evaluate(self, lever: Lever, params: dict[str, Any]) -> SafetyDecision:
        if lever.safety == LeverSafety.GREEN:
            return SafetyDecision.ALLOW
        if lever.safety == LeverSafety.YELLOW:
            self._queue(lever, params)
            return SafetyDecision.QUEUE_FOR_APPROVAL
        return SafetyDecision.BLOCK

    def _queue(self, lever: Lever, params: dict[str, Any]) -> None:
        entry = {
            "lever": lever.name,
            "params": params,
            "requested_at": utcnow().isoformat(),
            "status": "pending",
        }
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def list_pending(self) -> list[dict[str, Any]]:
        if not self._pending_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._pending_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
