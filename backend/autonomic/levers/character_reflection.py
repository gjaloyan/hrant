"""FIRE_CHARACTER_REFLECTION — the agent looks at who it has been.

FIRE_SELF_STUDY keeps 202 files describing how the agent WORKS, refreshed
within minutes of any code change. Nothing kept the description of who it IS
up to date: soul.md was last written by a human on 2026-07-07 and had not
moved since, through a month of daily work.

This lever closes that asymmetry. It reads consolidated sessions — real
conversations, not synthetic probes — against the current soul, and asks one
question: is there something durable about being useful to THIS person that
the character does not yet say?

Two properties keep it honest:

  * It proposes; it never writes. Every revision goes to the owner with a
    diff and two buttons (soul_evolution.py).
  * It requires a PATTERN. A single interaction is a mood, not a character,
    so the prompt demands recurrence across sessions and the lever refuses to
    run on fewer than MIN_SESSIONS of material.

It also refuses to queue a second revision while one is pending. An agent
that mails its person three character changes a night is not reflecting.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..lever import Lever, resolve_knowledge_path
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

from backend.llm import router, TaskType

log = logging.getLogger(__name__)

DEFAULT_SESSIONS_PATH = Path("knowledge/sessions.json")

# Below this there is not enough lived material to tell a pattern from a mood.
MIN_SESSIONS = 4
MAX_SESSIONS = 12
MAX_TRANSCRIPT_CHARS = 12000

CHARACTER_SYSTEM = """You are reflecting on your own character.

You will be given your current character file (soul.md) and summaries of
recent real conversations with the person you work for. Decide whether those
conversations revealed something DURABLE about being useful to them that your
character file does not already say.

Return strictly JSON:
{
  "revise": true | false,
  "rationale": "one or two sentences: why this belongs in your character",
  "evidence": "the recurring pattern you observed, citing what happened",
  "old_excerpt": "text copied VERBATIM from the character file, or \\"\\" to append",
  "new_excerpt": "what it becomes"
}

Rules — read them before answering:
- Default to {"revise": false}. Most weeks change nothing about who you are.
- Propose only a pattern you saw across SEVERAL conversations. One
  interaction is a mood, not a character.
- A fact about the person (their timezone, their projects, their preferences)
  is NOT character — it belongs in their profile, not in your soul. Character
  is how you behave: what you refuse, what you volunteer, what you check
  before claiming done, how you say hard things.
- `old_excerpt` must be copied character-for-character from the file you were
  given and must appear there exactly once. If you cannot copy it exactly,
  use "" and write a new passage to append instead.
- Change one thing. Do not restyle the document.
- Never propose removing a commitment your person gave you. You may sharpen
  it; you may not delete it."""


class FIRE_CHARACTER_REFLECTION(Lever):
    name = "FIRE_CHARACTER_REFLECTION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN          # it proposes; the owner writes
    executor = "claude"
    estimated_cost = Cost(seconds=40.0, tokens_in=8000, tokens_out=700)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any],
            context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        from backend.soul_evolution import SOUL_EVOLUTION

        if SOUL_EVOLUTION.list(status="pending"):
            # The owner already has one waiting. Piling on is how a thoughtful
            # proposal becomes noise that gets dismissed unread.
            return self._report(params, started, LeverStatus.SKIPPED,
                                {}, "revision_already_pending")

        sessions_path = resolve_knowledge_path(
            params.get("sessions_path") or DEFAULT_SESSIONS_PATH)
        transcript, used = self._recent_material(sessions_path, params)
        if used < MIN_SESSIONS:
            return self._report(params, started, LeverStatus.SKIPPED,
                                {"sessions": used}, "insufficient_history")

        try:
            soul = self._soul_text()
        except OSError as exc:
            return self._report(params, started, LeverStatus.FAILURE,
                                {}, f"soul_unreadable:{exc}")

        try:
            data = router().call_json(
                TaskType.TASK_ANALYSIS,
                CHARACTER_SYSTEM,
                f"CURRENT CHARACTER FILE (soul.md):\n{soul}\n\n"
                f"RECENT CONVERSATIONS ({used} sessions):\n{transcript}",
                max_tokens=1200,
                temperature=0.3,
            )
        except Exception as exc:
            log.warning("character_reflection: cortex call failed: %s", exc)
            return self._report(params, started, LeverStatus.FAILURE,
                                {"sessions": used}, f"cortex_failed:{exc}")

        if not isinstance(data, dict) or not data.get("revise"):
            return self._report(params, started, LeverStatus.SUCCESS,
                                {"sessions": used, "proposed": 0},
                                "no_change_warranted")

        rev = SOUL_EVOLUTION.propose(
            target="soul",
            rationale=str(data.get("rationale") or ""),
            old_excerpt=str(data.get("old_excerpt") or ""),
            new_excerpt=str(data.get("new_excerpt") or ""),
            evidence=str(data.get("evidence") or ""),
        )
        if rev is None:
            # Almost always a paraphrased old_excerpt. Worth seeing in the
            # log: it is the difference between "nothing to say" and "said
            # something unusable".
            return self._report(params, started, LeverStatus.SUCCESS,
                                {"sessions": used, "proposed": 0},
                                "revision_rejected_by_validation")

        return self._report(params, started, LeverStatus.SUCCESS,
                            {"sessions": used, "proposed": 1,
                             "revision_id": rev.id},
                            f"proposed:{rev.id}")

    # ── helpers ─────────────────────────────────────────────────────

    def _soul_text(self) -> str:
        from backend.identity import IDENTITY
        return IDENTITY.soul_path.read_text(encoding="utf-8")

    def _recent_material(self, sessions_path: Path,
                         params: dict[str, Any]) -> tuple[str, int]:
        """Summaries of the most recent consolidated sessions.

        Consolidated only: a summary is what memory_consolidation already
        distilled from a real conversation, which keeps this lever off raw
        transcripts and off sessions that were never really had."""
        import json
        try:
            blob = json.loads(sessions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "", 0
        sessions = [s for s in (blob.get("sessions") or [])
                    if isinstance(s, dict) and str(s.get("summary") or "").strip()]
        limit = int(params.get("max_sessions", MAX_SESSIONS))
        chosen = sessions[-limit:]
        chunks = []
        for s in chosen:
            when = str(s.get("started") or s.get("id") or "")
            chunks.append(f"— {when}\n{str(s.get('summary')).strip()}")
        text = "\n\n".join(chunks)
        if len(text) > MAX_TRANSCRIPT_CHARS:
            text = text[-MAX_TRANSCRIPT_CHARS:]
        return text, len(chosen)

    def _report(self, params: dict[str, Any], started, status: LeverStatus,
                outcome: dict[str, Any], reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status,
            outcome=outcome,
            reason=reason,
        )
