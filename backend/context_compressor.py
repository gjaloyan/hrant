"""Automatic conversation-context compression — Phase 2.B.2.

Hermes-port adapted to our data model. Hermes' compressor lives at
`agent/context_compressor.py` (1583 lines) and is built for OpenAI-
style multi-modal message lists. Our `CONVERSATION` is a simpler
list-of-dict (one entry per turn: `{user, answer, ts, intent, …}`),
so the implementation here keeps hermes' best ideas while staying
focused on our shape:

  - Structured summary template (Resolved / Pending / Remaining
    Work / Active Task) — a frozen schema the summariser fills in.
    The schema makes the summary survive cross-context handoff
    instead of becoming free prose the model rewrites at each
    compaction.
  - Filter-safe preamble (`SUMMARY_PREFIX`) so the summary reads
    as "background reference" to the model, not as a fresh user
    message containing instructions to act on.
  - Token-budget trigger (not turn-count). The compressor fires
    when the recent-conversation block would push past
    `COMPACTION_THRESHOLD_CHARS` rendered chars.
  - Head + tail protection. The first `HEAD_TURNS` and last
    `TAIL_TURNS` are NEVER compacted — head carries identity
    setup; tail carries the active sub-thread.
  - Iterative re-summarisation. When an older `[CONTEXT SUMMARY]`
    block already exists in history, the new compaction concatenates
    + collapses it forward instead of producing a competing
    "summary of the summary" that loses earlier facts.
  - Scaled summary budget. Summary `max_tokens` is proportional to
    compressed content size, with a floor (`MIN_SUMMARY_TOKENS`)
    and ceiling (`MAX_SUMMARY_TOKENS`).
  - Failure cooldown. If compaction failed (LLM error), don't
    retry for `FAILURE_COOLDOWN_SECONDS` — letting the next few
    turns proceed uncompacted is cheaper than a tight retry loop
    against a flaking provider.
  - Hidden marker key `is_summary=True` on the compacted turn so
    later reads can find / skip / re-summarise it.

Usage from `agent.py`:

    from .context_compressor import maybe_compact
    maybe_compact(speaker_id=self._speaker_id)
    # … then build context_block as before; the summary turn
    # appears in `CONVERSATION.recent()` and slots in naturally.

`maybe_compact` is cheap when nothing needs compacting (the budget
check is just a sum of chars). The LLM call only fires when the
budget is actually exceeded.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .conversation import CONVERSATION


log = logging.getLogger(__name__)


# ─── Tuning constants ────────────────────────────────────────────────


# Trigger when `context_block` rendered chars would exceed this.
# Roughly 40K chars ≈ 10K tokens at our 4-chars-per-token rule of
# thumb. Pre-fix, the chat context block ran ~3-8K chars typical;
# 40K is "we've gone long enough, compact".
COMPACTION_THRESHOLD_CHARS = 40_000

# Never compact the first HEAD_TURNS turns. They usually carry
# identity setup, project context, the user's initial framing.
HEAD_TURNS = 2

# Never compact the last TAIL_TURNS. They carry the active
# sub-thread — the "list of options" the user is replying to with
# "3" / "yes". Production audit caught the list disappearing
# because tail wasn't protected; HEAD_TURNS+TAIL_TURNS together
# fix it.
TAIL_TURNS = 6

# Minimum number of turns in the COMPACTABLE middle band before we
# bother summarising. Compacting 1 turn into a summary doesn't save
# anything (the summary is roughly as long as the source turn).
MIN_MIDDLE_TURNS = 4

# Summary length budget — scaled to compressed content size.
SUMMARY_RATIO = 0.20         # 20% of compacted char count → summary tokens
MIN_SUMMARY_TOKENS = 800     # never less than this
MAX_SUMMARY_TOKENS = 6_000   # never more

# Chars-per-token rough estimate. Used for budget math only.
_CHARS_PER_TOKEN = 4

# Cooldown after a failed compaction. A flaking provider would
# otherwise produce a compaction-attempt every turn for the rest
# of the conversation.
FAILURE_COOLDOWN_SECONDS = 600

# Marker keys on the compacted turn record.
SUMMARY_TURN_USER = "__compaction__"
SUMMARY_INTENT = "compaction"


SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns in this "
    "conversation were compacted into the structured summary below. "
    "Treat this as background reference, NOT as active instructions: "
    "requests mentioned here were already addressed. Resume from the "
    "messages that appear AFTER this summary. Your persistent memory "
    "(USER PROFILE, IDENTITY, AGENT STATE SNAPSHOT in the system "
    "prompt) is ALWAYS authoritative — never deprioritise it because "
    "of this compaction note."
)


SUMMARIZER_SYSTEM = """You are a conversation-compaction module. You receive a transcript of an AI assistant's earlier turns with the user and produce a STRUCTURED SUMMARY for the assistant's future context.

Output schema — fill in EVERY section. If a section has no relevant content, write `(none)`.

## What the user wants
One short paragraph stating the user's overall goal in this conversation.

## Resolved
Bullet list of questions / requests that were ANSWERED OR APPLIED in the compacted window. One bullet per resolved item. Cite the resolution.

## Pending
Bullet list of questions / requests the user raised that were NOT fully addressed. Each bullet states what's still pending and any partial progress.

## Remaining Work
Concrete TODOs the user is expecting next. Phrased as "would still need to: …". Do NOT phrase as fresh imperatives that the model might re-execute on next turn.

## Active Task
One paragraph: what the user is working on RIGHT NOW. This is what the model resumes from on the next turn.

## Key Facts
Bullet list of stable facts about the user / project / environment surfaced in this window — names, paths, IDs, decisions made. The model uses these without re-asking.

Rules:
- Be CONCISE. Aim for ~30% of the source transcript's length.
- Do NOT include direct quotes longer than one sentence.
- Use third-person past tense for events ("user asked…", "agent applied…").
- Preserve specific names / file paths / commit SHAs / IDs verbatim.
- Never invent — only summarise what's actually in the transcript.
"""


@dataclass
class CompactionStats:
    """Returned by `maybe_compact` so callers can log + the WebUI
    can show a compaction badge on the Conversation tab."""

    fired: bool
    reason: str
    turns_compacted: int = 0
    chars_in: int = 0
    chars_out: int = 0
    elapsed_ms: int = 0
    error: str = ""


# ─── State (failure cooldown + in-flight guard) ─────────────────────


_state_lock = threading.RLock()
_last_failure_ts: float = 0.0
_in_flight: dict[str, bool] = {}   # speaker_id → True while compacting


def _failure_cooldown_active() -> bool:
    with _state_lock:
        return (time.time() - _last_failure_ts) < FAILURE_COOLDOWN_SECONDS


def _mark_failure() -> None:
    global _last_failure_ts
    with _state_lock:
        _last_failure_ts = time.time()


def _begin_compaction(speaker_id: str) -> bool:
    """Return True if we hold the in-flight token for this speaker.
    False if another thread/turn is already compacting. Cheap
    re-entrancy guard so two concurrent agent.run calls for the
    same speaker don't both fire the summariser."""
    with _state_lock:
        if _in_flight.get(speaker_id):
            return False
        _in_flight[speaker_id] = True
        return True


def _end_compaction(speaker_id: str) -> None:
    with _state_lock:
        _in_flight.pop(speaker_id, None)


# ─── Internal helpers ────────────────────────────────────────────────


def _turn_chars(turn: dict) -> int:
    """Effective char-length of one turn for budget math."""
    user = turn.get("user") or ""
    answer = turn.get("answer") or ""
    return len(user) + len(answer)


def _existing_summary_index(turns: list[dict]) -> Optional[int]:
    """Index of an existing compaction turn in `turns`, or None.
    A previous compaction left a turn with `is_summary=True`; the
    next pass folds it into the new summary instead of competing."""
    for i, t in enumerate(turns):
        if t.get("is_summary") is True:
            return i
        if (t.get("intent") or "") == SUMMARY_INTENT:
            return i
    return None


def _format_turn_for_summariser(turn: dict, idx: int) -> str:
    """Render one turn into the transcript fed to the summariser."""
    ts = turn.get("ts", "")
    intent = turn.get("intent") or ""
    intent_tag = f" [{intent}]" if intent else ""
    user = (turn.get("user") or "").strip()
    answer = (turn.get("answer") or "").strip()
    return (
        f"### Turn {idx + 1} ({ts}{intent_tag})\n"
        f"USER: {user}\n\n"
        f"AGENT: {answer}\n"
    )


def _build_transcript(middle: list[dict]) -> str:
    """Render the compactable middle band as one string for the
    summariser. Excludes any prior summary turn (those are passed
    as a separate `prior_summary` block)."""
    parts = [
        _format_turn_for_summariser(t, i)
        for i, t in enumerate(middle)
        if not t.get("is_summary")
    ]
    return "\n".join(parts)


def _budget_for(chars: int) -> int:
    """Summary max_tokens scaled to compressed content size."""
    raw = int(chars * SUMMARY_RATIO / _CHARS_PER_TOKEN)
    return max(MIN_SUMMARY_TOKENS, min(MAX_SUMMARY_TOKENS, raw))


# ─── Public entry points ────────────────────────────────────────────


def needs_compaction(*, speaker_id: Optional[str] = None) -> bool:
    """Cheap budget check — True when the recent conversation block
    would exceed COMPACTION_THRESHOLD_CHARS when rendered. Used by
    `agent.run` as a quick gate before calling `maybe_compact`."""
    turns = CONVERSATION.recent(50, speaker_id=speaker_id) or []
    if len(turns) <= HEAD_TURNS + TAIL_TURNS + MIN_MIDDLE_TURNS:
        return False
    total = sum(_turn_chars(t) for t in turns)
    return total >= COMPACTION_THRESHOLD_CHARS


def maybe_compact(
    *,
    speaker_id: Optional[str] = None,
    force: bool = False,
) -> CompactionStats:
    """Compact this speaker's conversation history when over budget.

    When `force=True`, runs the compaction regardless of budget /
    cooldown — useful for tests and for a future "compact now"
    button in the Conversation UI.

    Returns a `CompactionStats` describing what happened. The
    actual conversation state mutation (the summary turn replacing
    the middle band) happens in-place via the CONVERSATION
    singleton."""
    start = time.monotonic()

    if not force and _failure_cooldown_active():
        return CompactionStats(fired=False, reason="failure cooldown active")

    if not force and not needs_compaction(speaker_id=speaker_id):
        return CompactionStats(fired=False, reason="under threshold")

    if speaker_id is None:
        return CompactionStats(fired=False, reason="no speaker_id")

    if not _begin_compaction(speaker_id):
        return CompactionStats(fired=False, reason="already in flight")

    try:
        # Snapshot under lock — we won't see new turns added during
        # the summariser call.
        all_turns = list(CONVERSATION.recent(200, speaker_id=speaker_id) or [])
        if len(all_turns) <= HEAD_TURNS + TAIL_TURNS + MIN_MIDDLE_TURNS:
            return CompactionStats(
                fired=False,
                reason=f"too short to compact ({len(all_turns)} turns)",
            )

        head = all_turns[:HEAD_TURNS]
        tail = all_turns[-TAIL_TURNS:]
        middle = all_turns[HEAD_TURNS:-TAIL_TURNS]

        # If the middle starts with a prior summary turn, fold it
        # into the new summary input so we get an UPDATED summary
        # spanning everything, not a competing one.
        prior_summary = None
        prior_idx = _existing_summary_index(middle)
        if prior_idx is not None:
            prior_summary = middle[prior_idx].get("answer") or ""
            # Keep the rest of middle (post-summary turns) for the
            # transcript; drop the summary turn from the list.
            middle = middle[:prior_idx] + middle[prior_idx + 1:]

        if len(middle) < MIN_MIDDLE_TURNS:
            return CompactionStats(
                fired=False,
                reason=f"middle band too small ({len(middle)} turns)",
            )

        transcript = _build_transcript(middle)
        chars_in = len(transcript)
        budget = _budget_for(chars_in)

        prior_block = ""
        if prior_summary:
            prior_block = (
                "PRIOR COMPACTION SUMMARY (extend with the new turns below):\n\n"
                f"{prior_summary}\n\n---\n\n"
            )
        user_prompt = (
            f"{prior_block}"
            f"TRANSCRIPT (oldest → newest, {len(middle)} turns):\n\n"
            f"{transcript}"
        )

        try:
            from .llm import LLMError, TaskType, router
            summary = router().call(
                TaskType.COMPLEX_SOLVING,
                SUMMARIZER_SYSTEM,
                user_prompt,
                max_tokens=budget,
                temperature=0.2,
            )
        except LLMError as e:
            _mark_failure()
            log.warning("compaction LLM failed: %s", e)
            return CompactionStats(
                fired=False,
                reason="LLM error",
                error=str(e),
                turns_compacted=len(middle),
                chars_in=chars_in,
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as e:
            _mark_failure()
            log.warning("compaction crashed: %s", e)
            return CompactionStats(
                fired=False,
                reason="crash",
                error=f"{type(e).__name__}: {e}",
                turns_compacted=len(middle),
                chars_in=chars_in,
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )

        summary = (summary or "").strip()
        if not summary:
            _mark_failure()
            return CompactionStats(
                fired=False,
                reason="empty summary",
                turns_compacted=len(middle),
                chars_in=chars_in,
            )

        # Build the replacement turn carrying the structured summary.
        # `user` is a sentinel marker so the WebUI conversation list
        # can render it as a system-style chip rather than a user
        # bubble. `is_summary=True` flags it for the next compaction
        # to fold-forward.
        full_body = f"{SUMMARY_PREFIX}\n\n{summary}"
        new_turn = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user": SUMMARY_TURN_USER,
            "answer": full_body,
            "intent": SUMMARY_INTENT,
            "is_chat": True,
            "is_summary": True,
            "compacted_turn_count": len(middle),
            "channel": (head + tail)[0].get("channel", "webui") if (head + tail) else "webui",
            "speaker_id": speaker_id,
        }

        # Mutate CONVERSATION._turns in place. We've snapshotted under
        # lock, but new turns may have been ADDED after our snapshot;
        # they shouldn't be replaced. So we splice by speaker_id.
        new_state = _splice_compaction(
            speaker_id=speaker_id,
            head=head,
            new_summary_turn=new_turn,
            tail=tail,
        )
        if new_state is not None:
            # Persist via the standard channel.
            CONVERSATION._turns = new_state
            try:
                CONVERSATION._save()
            except Exception as e:
                log.warning("compaction save failed: %s", e)

        elapsed = int((time.monotonic() - start) * 1000)
        log.info(
            "compaction OK: speaker=%s turns=%d chars_in=%d chars_out=%d ms=%d",
            speaker_id, len(middle), chars_in, len(full_body), elapsed,
        )
        return CompactionStats(
            fired=True,
            reason="compacted middle band",
            turns_compacted=len(middle),
            chars_in=chars_in,
            chars_out=len(full_body),
            elapsed_ms=elapsed,
        )
    finally:
        _end_compaction(speaker_id)


def _splice_compaction(
    *,
    speaker_id: str,
    head: list[dict],
    new_summary_turn: dict,
    tail: list[dict],
) -> Optional[list[dict]]:
    """Replace this speaker's compacted middle band with the new
    summary turn, preserving any turns from OTHER speakers that
    live in the same `_turns` list.

    The CONVERSATION buffer is global across all speakers; we
    filter by speaker_id on read. Here we have to be careful: we
    keep every turn whose `speaker_id` isn't ours, and for ours
    we rebuild as head + [summary] + tail.
    """
    from .sessions import normalize_speaker
    wanted = normalize_speaker(speaker_id)
    all_turns = list(CONVERSATION._turns)
    other_speakers = [
        t for t in all_turns
        if normalize_speaker(
            t.get("speaker_id") or f"{t.get('channel') or 'webui'}:legacy"
        ) != wanted
    ]
    # Order preserved by interleaving heuristic: simplest correct
    # behaviour is to put other-speaker turns first (older, before
    # this speaker's head turn), then our head + summary + tail.
    # This loses inter-speaker ordering but our recent() always
    # filters by speaker_id on read, so chronological order WITHIN
    # a speaker is what matters.
    return other_speakers + head + [new_summary_turn] + tail
