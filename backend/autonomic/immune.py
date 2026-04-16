"""Immune signature store — matches error entries to known fix recipes."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SIGNATURES_PATH = Path("knowledge/immune/signatures.jsonl")


@dataclass
class ImmuneSignature:
    id: str
    pattern: dict[str, Any]
    severity: str
    fix_lever: str
    fix_params: dict[str, Any]
    observed_count: int = 0
    success_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImmuneSignature":
        return cls(
            id=data["id"],
            pattern=dict(data["pattern"]),
            severity=data["severity"],
            fix_lever=data["fix_lever"],
            fix_params=dict(data.get("fix_params", {})),
            observed_count=int(data.get("observed_count", 0)),
            success_rate=data.get("success_rate"),
        )


class SignatureStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_SIGNATURES_PATH

    def load(self) -> list[ImmuneSignature]:
        if not self._path.exists():
            return []
        out: list[ImmuneSignature] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ImmuneSignature.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning("Skipping malformed signature line: %s", exc)
                continue
        return out

    def match(self, error_entry: dict[str, Any]) -> ImmuneSignature | None:
        msg = str(error_entry.get("message", ""))
        src = str(error_entry.get("source", ""))
        svc = error_entry.get("service")
        for sig in self.load():
            pat = sig.pattern
            if pat.get("source") != src:
                continue
            regex = pat.get("msg_regex", "")
            try:
                if not re.search(regex, msg):
                    continue
            except re.error as exc:
                log.warning("Bad regex in signature %s: %s", sig.id, exc)
                continue
            if "service" in pat and pat["service"] != svc:
                continue
            return sig
        return None

    def record_outcome(self, signature_id: str, success: bool) -> None:
        sigs = self.load()
        found = False
        for sig in sigs:
            if sig.id == signature_id:
                found = True
                prior_count = sig.observed_count
                prior_success_count = int(round((sig.success_rate or 0.0) * prior_count))
                new_count = prior_count + 1
                new_success_count = prior_success_count + (1 if success else 0)
                sig.observed_count = new_count
                sig.success_rate = new_success_count / new_count if new_count else None
                break
        if not found:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            for sig in sigs:
                f.write(json.dumps(sig.to_dict(), ensure_ascii=False) + "\n")
