"""Immune signature store — matches error entries to known fix recipes."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SIGNATURES_PATH = Path("knowledge/immune/signatures.jsonl")
DEFAULT_FIRES_PATH = Path("knowledge/immune/fires.json")

# Levers a signature is allowed to name as its fix. A signature is a rule that
# makes the machine act on its own, so the set of things it can trigger is a
# closed list, not whatever string happened to be written in a JSONL file.
ALLOWED_FIX_LEVERS = frozenset({
    "FIRE_SERVICE_REPAIR",
    "FIRE_TOOL_INSTALL",
    "FIRE_LOG_ROTATION",
    "FIRE_GRAPH_REBUILD",
    "FIRE_EMBEDDING_BACKFILL",
    "FIRE_FACT_EMBEDDING_BACKFILL",
    "FIRE_INTEGRITY_HEARTBEAT",
})

# How long a signature stays quiet after firing. A repair that needs to run
# more than once an hour is not a repair.
DEFAULT_COOLDOWN_SECONDS = 3600.0

# Consecutive failed fixes before a signature is quarantined. Three tries is
# generous; a fourth is a loop, and a loop that restarts services is worse
# than the fault it was chasing.
MAX_CONSECUTIVE_FAILURES = 3


def resolve_immune_path(p: Path) -> Path:
    """Re-root a relative `knowledge/...` default at the real data dir.

    Audit 2026-05-27 #5 all over again: DEFAULT_SIGNATURES_PATH is relative,
    so `SignatureStore()` with no argument resolved against the SERVICE's cwd
    (`~/hrant/knowledge/immune/...`) while everything else — the API, the
    audit tooling, the data dir — looked in `~/.hrant/data/knowledge/immune/`.
    Nothing noticed, because nothing had ever written a signature."""
    from .lever import resolve_knowledge_path
    return resolve_knowledge_path(p)


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
        # Callers that pass a path own it verbatim (tests, the API). The
        # default is re-rooted, so the no-argument constructor and the API
        # finally read the same file.
        self._path = path if path is not None \
            else resolve_immune_path(DEFAULT_SIGNATURES_PATH)

    @property
    def path(self) -> Path:
        return self._path

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

    def add(self, sig: ImmuneSignature) -> tuple[bool, str]:
        """Append a signature. Returns (ok, message).

        Until 2026-08-10 nothing in the codebase could write one: the store
        was read-only, signatures.jsonl did not exist on prod, and `match()`
        had no callers. A matcher with an empty, unwritable rulebook is an
        elaborate way to return None."""
        if sig.fix_lever not in ALLOWED_FIX_LEVERS:
            return False, (f"fix_lever {sig.fix_lever!r} is not one a "
                           f"signature may trigger; allowed: "
                           f"{sorted(ALLOWED_FIX_LEVERS)}")
        if not str(sig.pattern.get("source") or "").strip():
            return False, "pattern.source is required (e.g. 'tool', 'service')"
        regex = str(sig.pattern.get("msg_regex") or "")
        if not regex:
            return False, "pattern.msg_regex is required"
        try:
            re.compile(regex)
        except re.error as exc:
            return False, f"pattern.msg_regex does not compile: {exc}"
        if any(s.id == sig.id for s in self.load()):
            return False, f"signature id {sig.id!r} already exists"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sig.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            return False, f"could not write signature: {exc}"
        return True, f"signature {sig.id} added"

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


class FireLog:
    """Per-signature firing history: the storm guard.

    `SignatureStore.record_outcome` tracks a lifetime success rate, which is
    useful for judging a signature and useless for deciding whether to fire
    it right now. This answers the operational question instead: has this
    signature fired recently, and has it been failing?

    Kept separate from the signature file on purpose — signatures are a
    rulebook worth reading and editing by hand; this is runtime state.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None \
            else resolve_immune_path(DEFAULT_FIRES_PATH)

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        except OSError as exc:
            log.warning("immune: could not persist fire log: %s", exc)

    def may_fire(self, signature_id: str, *,
                 cooldown: float = DEFAULT_COOLDOWN_SECONDS,
                 now: float | None = None) -> tuple[bool, str]:
        """(allowed, why-not). Quarantine outranks cooldown."""
        rec = self._load().get(signature_id) or {}
        if int(rec.get("consecutive_failures", 0)) >= MAX_CONSECUTIVE_FAILURES:
            return False, "quarantined"
        last = float(rec.get("last_fired", 0.0) or 0.0)
        current = time.time() if now is None else now
        if last and (current - last) < cooldown:
            return False, "cooling_down"
        return True, ""

    def note_fired(self, signature_id: str, *, now: float | None = None) -> None:
        data = self._load()
        rec = data.setdefault(signature_id, {})
        rec["last_fired"] = time.time() if now is None else now
        rec["fire_count"] = int(rec.get("fire_count", 0)) + 1
        self._save(data)

    def note_outcome(self, signature_id: str, success: bool) -> None:
        data = self._load()
        rec = data.setdefault(signature_id, {})
        if success:
            rec["consecutive_failures"] = 0
        else:
            rec["consecutive_failures"] = \
                int(rec.get("consecutive_failures", 0)) + 1
        self._save(data)

    def quarantined(self) -> list[str]:
        return [sid for sid, rec in self._load().items()
                if int(rec.get("consecutive_failures", 0))
                >= MAX_CONSECUTIVE_FAILURES]

    def stats(self) -> dict[str, dict[str, Any]]:
        return self._load()
