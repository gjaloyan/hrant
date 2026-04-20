"""Safety gate: classifies lever execution requests by their safety tier."""
from __future__ import annotations

import json
import secrets
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

    def _queue(self, lever: Lever, params: dict[str, Any]) -> str:
        entry_id = secrets.token_hex(6)
        entry = {
            "id": entry_id,
            "lever": lever.name,
            "params": params,
            "requested_at": utcnow().isoformat(),
            "status": "pending",
        }
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry_id

    def list_pending(self) -> list[dict[str, Any]]:
        if not self._pending_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._pending_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def remove_pending(self, entry_id: str) -> bool:
        entries = self.list_pending()
        kept = [e for e in entries if e.get("id") != entry_id]
        if len(kept) == len(entries):
            return False
        tmp = self._pending_path.with_suffix(self._pending_path.suffix + ".tmp")
        tmp.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept),
            encoding="utf-8",
        )
        tmp.replace(self._pending_path)
        return True
