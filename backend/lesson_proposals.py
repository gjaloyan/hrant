"""Turn a behavioural lesson into a change the owner can approve.

The meta-learner spots things like "claims a file path exists without
checking" and filed them as goals. Measured on prod 2026-09-01: 88 of 97
active goals had NO subtasks at all -- no plan, nothing executable -- so
nothing could act on them and the stale sweep retired them after 14 days.
Over three months that produced 447 distinct suggestions, 5 approved.

The mistake was the shape, not the volume. A behavioural lesson is not a
project; it is a RULE, and the agent's rules live in `prompt_modules.py`.
So instead of a goal nobody can execute, this writes the actual one-line
edit that adds the rule to the `m11_lessons` module, and files it as a
self-modification proposal -- which already has a Telegram approval flow,
a real diff to read, and an apply path that compiles and runs the tests.

One queue instead of two, and the thing the owner approves is the change
itself rather than a wish that still needs someone to implement it.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

MODULE_PATH = "backend/prompt_modules.py"

# The insertion point. Kept as a literal comment inside the module body so
# the diff is an exact, idempotent replacement rather than a line number
# that drifts the moment anything above it changes.
ANCHOR = (
    "<!-- LESSONS ANCHOR — new lessons are inserted directly above this line -->"
)

# Below this, two lessons are treated as the same lesson. The meta-learner
# rephrases the same complaint endlessly -- at 0.72 similarity the 509
# archived goals still counted as 447 "distinct" ones, which is how a queue
# fills with paraphrases of one idea.
SIMILARITY = 0.72

# Every approved lesson costs tokens on EVERY turn, and the default prompt
# has a hard 17k budget. Without a ceiling this path would reproduce the
# problem it was built to fix — an unbounded queue — just in the prompt
# instead of in goals.json. At the cap the agent stops proposing and says
# so, which is the owner's cue to prune what no longer earns its place.
MAX_LESSONS = 12


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9Ѐ-ӿ԰-֏ ]+", " ",
                  (text or "").lower()).strip()


def _too_similar(a: str, b: str) -> bool:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= SIMILARITY


def existing_lessons(body: str) -> list[str]:
    """The lesson bullets already in the module body."""
    return [ln.strip()[2:].strip()
            for ln in (body or "").splitlines()
            if ln.strip().startswith("- ")]


def already_known(lesson: str, body: str, pending: list) -> Optional[str]:
    """Why this lesson should NOT be proposed, or None if it should.

    Checks the module itself AND the pending queue: proposing a rule that
    is already waiting for approval is the duplication that made the old
    queue unreadable.
    """
    current = existing_lessons(body)
    if len(current) >= MAX_LESSONS:
        return (f"the lessons module is full ({len(current)}/{MAX_LESSONS}); "
                "prune one before adding another")
    for known in current:
        if _too_similar(lesson, known):
            return f"already a rule: {known[:60]}"
    for p in pending or []:
        if getattr(p, "status", "") != "pending":
            continue
        title = getattr(p, "title", "") or getattr(p, "description", "")
        if title.startswith("Lesson:") and _too_similar(lesson, title[7:]):
            return f"already proposed: {title[:60]}"
    return None


def build_edit(lesson: str, evidence: str = "") -> dict:
    """The literal old_code / new_code pair that inserts one lesson.

    `old_code` is the anchor alone, so the edit applies cleanly no matter
    how many lessons are already there and no matter what changed above.
    """
    line = f"- {lesson.strip().rstrip('.')}."
    if evidence:
        line += f"  <!-- {evidence.strip()[:120]} -->"
    return {
        "module": MODULE_PATH,
        "old_code": ANCHOR,
        "new_code": f"{line}\n\n{ANCHOR}",
    }


def propose_lesson(lesson: str, *, evidence: str = "",
                   requester: str = "meta_learner") -> Optional[object]:
    """File one behavioural lesson as a reviewable proposal.

    Returns the Proposal, or None when it was a duplicate, empty, or the
    store refused it. Never raises: this runs inside the meta-learner's
    analysis pass, and a failure here must not lose the analysis.
    """
    lesson = (lesson or "").strip()
    if len(lesson) < 12:
        return None
    try:
        # The prompt is English-only, and the meta-learner writes in the
        # language of the conversation it learned from — the goals on prod
        # carried fixes like "Требовать проверять каждое дело по официальному
        # источнику". Left alone, this path would have written Russian rules
        # into the system prompt. Meaning-preserving, not word-for-word.
        from .meaning_translate import needs_translation, to_english
        if needs_translation(lesson):
            lesson = (to_english(lesson) or lesson).strip()
            if needs_translation(lesson):
                log.info("lesson not proposed (translation failed): %s",
                         lesson[:60])
                return None

        from .prompt_modules import MODULES
        from .self_modifier import SELF_MODIFIER

        body = MODULES["m11_lessons"].body
        skip = already_known(lesson, body, list(SELF_MODIFIER._proposals))
        if skip:
            log.info("lesson not proposed (%s): %s", skip, lesson[:60])
            return None

        from .self_modifier import Proposal, _fire_proposal_created

        edit = build_edit(lesson, evidence)
        proposal = Proposal(
            module=MODULE_PATH,
            title=f"Lesson: {lesson[:70]}",
            description=(
                "Add a standing rule to the agent's LESSONS LEARNED prompt "
                "module." + chr(10) + chr(10) + lesson
            ),
            old_code=edit["old_code"],
            new_code=edit["new_code"],
            impact="One line added to the system prompt. No code paths change.",
            risk="low",
            reasoning=evidence or "Recurring failure seen by the meta-learner.",
            # The prompt has no unit test of its own; importing the module is
            # the honest check that the edit did not break the file, and the
            # apply engine compiles it either way.
            test_commands=["python -c 'import backend.prompt_modules'"],
            success_criteria="The rule appears in the LESSONS LEARNED module.",
            rollback_plan="Remove the added bullet from m11_lessons.",
        )
        with SELF_MODIFIER._LOCK:
            SELF_MODIFIER._proposals.append(proposal)
        SELF_MODIFIER._save()
        # The Telegram DM with its approve/reject buttons hangs off this
        # subscription. A proposal nobody is told about is the exact failure
        # this whole path exists to end.
        _fire_proposal_created(proposal)
        return proposal

    except Exception as exc:
        log.warning("propose_lesson failed for %r: %s", lesson[:50], exc)
        return None
