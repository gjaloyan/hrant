"""Did the agent's own fix actually work?

The self-modification loop had a closed front half and an open back half.
Front half (2026-08-08): a patch runs its tests before it is kept, and a
failing test rolls the whole thing back — that answers "did I break
anything". Back half: nothing. A proposal names a problem, gets applied, and
the agent never learns whether the problem stopped.

Measured over 74 production turns on 2026-08-11: four
`propose_self_modification` calls, zero checks that any of them helped. Two
pytest runs. Zero `git log`. The agent proposes into silence.

That silence is most of the gap between "can read code and fix bugs" and
"cannot". A human — or the assistant doing this repair work — changes
something, watches, and learns from the result. Without the watching step
there is no learning at all, only guessing that happens to be recorded.

So: when a patch that targets a tool is applied, an entry is opened here.
If that same tool fails again afterwards, the entry is marked as not having
helped. The next time the agent is told "this tool keeps failing", it is also
told what it already tried and how that went — which is the difference
between diagnosing and re-diagnosing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("knowledge/self_mod_outcomes.json")

# Keep the file readable and bounded; older attempts stop being useful once a
# tool has been rewritten several times.
MAX_ENTRIES = 200

# A tool that fails once right after a patch may just be the same broken
# world; the patch deserves a moment. Failures within this window still count
# — the point is to catch "the fix did not take", not to be lenient — but the
# entry records how long it survived so the agent can tell "instantly" from
# "held for a day".
VERDICT_PENDING = "pending"
VERDICT_DID_NOT_HELP = "did_not_help"


@dataclass
class ModOutcome:
    proposal_id: str = ""
    title: str = ""
    tools: list[str] = field(default_factory=list)
    applied_at: float = field(default_factory=time.time)
    verdict: str = VERDICT_PENDING
    failures_after: int = 0
    first_failure_after: float = 0.0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModOutcome":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)

    def held_for_seconds(self) -> float:
        end = self.first_failure_after or time.time()
        return max(0.0, end - self.applied_at)


def tools_from_paths(paths: list[str]) -> list[str]:
    """`backend/tools/agent_browser.py` -> `agent_browser`.

    Only the tools directory: a patch to unified_agent.py touches everything
    and claiming it "targets" a specific tool would produce false verdicts.
    """
    out = []
    for p in paths or []:
        s = str(p).replace("\\", "/")
        if "/tools/" not in s or not s.endswith(".py"):
            continue
        name = s.rsplit("/", 1)[-1][:-3]
        if name and name != "__init__" and name not in out:
            out.append(name)
    return out


class OutcomeStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def _resolve(self) -> Path:
        if self._path is not None:
            return self._path
        from .config import CONFIG
        return Path(CONFIG.knowledge["base_dir"]) / "self_mod_outcomes.json"

    def _load(self) -> list[ModOutcome]:
        try:
            raw = json.loads(self._resolve().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        out = []
        for d in raw:
            try:
                out.append(ModOutcome.from_dict(d))
            except TypeError:
                continue
        return out

    def _save(self, items: list[ModOutcome]) -> None:
        p = self._resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps([i.to_dict() for i in items[-MAX_ENTRIES:]],
                           ensure_ascii=False, indent=1),
                encoding="utf-8")
        except OSError as e:
            log.warning("self_mod_outcomes: could not persist: %s", e)

    # ── the loop ────────────────────────────────────────────────────

    def record_applied(self, *, proposal_id: str, title: str,
                       paths: list[str]) -> "ModOutcome | None":
        """Open an entry when a tool-targeting patch lands."""
        tools = tools_from_paths(paths)
        if not tools:
            return None            # nothing tool-specific to watch
        items = self._load()
        entry = ModOutcome(proposal_id=proposal_id, title=title or "",
                           tools=tools)
        items.append(entry)
        self._save(items)
        return entry

    def note_tool_failure(self, tool: str, message: str = "") -> int:
        """A tool failed. Mark any patch that claimed to fix it.

        Returns how many entries were updated. Cheap and silent: this runs on
        the error path, where nothing may raise.
        """
        if not tool:
            return 0
        items = self._load()
        touched = 0
        now = time.time()
        for e in items:
            if tool not in e.tools:
                continue
            if now < e.applied_at:
                continue
            e.failures_after += 1
            if not e.first_failure_after:
                e.first_failure_after = now
            e.last_error = str(message or "")[:300]
            e.verdict = VERDICT_DID_NOT_HELP
            touched += 1
        if touched:
            self._save(items)
        return touched

    def history_for(self, tool: str, *, limit: int = 3) -> list[ModOutcome]:
        """Most recent attempts on this tool, newest first."""
        if not tool:
            return []
        hits = [e for e in self._load() if tool in e.tools]
        return sorted(hits, key=lambda e: e.applied_at, reverse=True)[:limit]

    def stats(self) -> dict[str, int]:
        items = self._load()
        return {
            "total": len(items),
            "pending": sum(1 for e in items if e.verdict == VERDICT_PENDING),
            "did_not_help": sum(1 for e in items
                                if e.verdict == VERDICT_DID_NOT_HELP),
        }


OUTCOMES = OutcomeStore()


def prior_attempts_note(tool: str) -> str:
    """One paragraph for the self-repair marker: what was already tried.

    Empty when nothing was. This is the feedback the agent has never had —
    without it, the fourth attempt at a tool looks exactly like the first.
    """
    try:
        history = OUTCOMES.history_for(tool)
    except Exception:
        return ""
    if not history:
        return ""
    lines = ["", "⚠️ YOU HAVE PATCHED THIS TOOL BEFORE:"]
    for e in history:
        when = time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(e.applied_at))
        if e.verdict == VERDICT_DID_NOT_HELP:
            held = e.held_for_seconds()
            unit = (f"{held / 3600:.1f}h" if held >= 3600
                    else f"{held / 60:.0f}m")
            lines.append(
                f"  • {when} — \"{e.title[:70]}\" — DID NOT FIX IT "
                f"(failed again after {unit}; {e.failures_after} failures "
                f"since)")
        else:
            lines.append(
                f"  • {when} — \"{e.title[:70]}\" — no failures since")
    lines.append(
        "  Do not re-apply a fix that is listed as DID NOT FIX IT. If your "
        "new diagnosis matches an old one, the diagnosis was wrong — look "
        "somewhere else, and prefer the environment over the source.")
    return "\n".join(lines)
