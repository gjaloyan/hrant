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
from pathlib import Path
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

# Every approved lesson costs tokens on EVERY turn, so this module needs a
# ceiling or it reproduces the unbounded queue it was built to replace,
# just in the prompt instead of in goals.json.
#
# The first version capped the COUNT at twelve, which was the wrong unit.
# Five approved lessons averaged 290 characters each and pushed the
# default prompt from 16862 to 18434 against a 17000 budget — a count cap
# cannot see that coming. Characters are what get billed, so characters
# are what is capped.
MAX_MODULE_CHARS = 1800

# One rule, one sentence. The model writes paragraphs when left alone —
# the longest of the first five ran to 331 characters for something
# sayable in eighty — and a rule nobody finishes reading is not a rule.
MAX_LESSON_CHARS = 200

# Kept as a backstop so a pile of very short lessons cannot become a wall.
MAX_LESSONS = 12


# The file the lessons live in, as it exists RIGHT NOW. An applied
# self-mod patches this file; the running process keeps whatever it
# imported at start-up. Measuring the ceiling against the in-memory copy
# therefore under-counts by exactly the lessons the agent has applied
# since the last restart -- which is how prod reached 1951 characters
# against an 1800 cap on 2026-09-03.
MODULE_SOURCE = Path(__file__).with_name("prompt_modules.py")

_BODY_OPEN = '_M11_BODY = """'


def module_body_on_disk() -> Optional[str]:
    """The lessons module body as the file currently has it.

    Returns None when the file cannot be read or the marker is missing;
    callers fall back to the imported body. Reading the source as text
    rather than importing it keeps this free of side effects -- the point
    is to look at a file the running process has deliberately not
    reloaded.
    """
    try:
        text = MODULE_SOURCE.read_text(encoding="utf-8")
    except Exception as exc:
        log.info("lessons module unreadable on disk (%s); using the "
                 "imported copy", exc)
        return None
    start = text.find(_BODY_OPEN)
    if start < 0:
        return None
    start += len(_BODY_OPEN)
    end = text.find('"""', start)
    if end < 0:
        return None
    return text[start:end]


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
    if len(lesson) > MAX_LESSON_CHARS:
        return (f"too long ({len(lesson)} chars, limit {MAX_LESSON_CHARS}); "
                "state the rule in one sentence")
    current = existing_lessons(body)
    if len(current) >= MAX_LESSONS:
        return (f"the lessons module is full ({len(current)}/{MAX_LESSONS}); "
                "prune one before adding another")
    # The real ceiling: what this module costs on every turn.
    #
    # Counted against the module PLUS everything already proposed and not
    # yet decided. The first version checked the module alone, so nine
    # proposals generated in one pass all measured themselves against an
    # empty module, all passed, and all applied — leaving 2732 characters
    # in a 1800-character module and a prompt 900 over budget. A ceiling
    # checked against a state that no longer exists by the time the change
    # lands is not a ceiling.
    claimed = sum(
        len(lesson_in(getattr(p, "new_code", "")) or "") + 8
        for p in (pending or [])
        if getattr(p, "status", "") == "pending"
        and (getattr(p, "title", "") or "").startswith("Lesson:")
    )
    if len(body) + claimed + len(lesson) + 8 > MAX_MODULE_CHARS:
        return (f"no room left ({len(body)} in the module, {claimed} already "
                f"proposed, limit {MAX_MODULE_CHARS}); decide on the pending "
                "ones or prune a rule first")
    for known in current:
        if _too_similar(lesson, known):
            return f"already a rule: {known[:60]}"
    # The meaning check ran against the pending queue only, so a lesson
    # restating an APPROVED rule in other words went straight through.
    # Prod 2026-09-02 carried nine rules where five ideas existed, and two
    # of the copies pushed back toward asking where the original had been
    # corrected to act. Character similarity cannot see that; the cheap
    # comparison above runs first so this only reaches genuinely new text.
    for known in current:
        if _means_the_same(lesson, known):
            return f"already a rule (same meaning): {known[:50]}"
    for p in pending or []:
        if getattr(p, "status", "") != "pending":
            continue
        title = getattr(p, "title", "") or ""
        if not title.startswith("Lesson:"):
            continue
        # The title is clipped to 70 characters; comparing against THAT
        # made every long lesson look unlike its own duplicate. The full
        # text is in new_code, which is what actually lands in the module.
        other = lesson_in(getattr(p, "new_code", "")) or title[7:]
        if _too_similar(lesson, other):
            return f"already proposed: {other[:60]}"
        # Character similarity cannot see a paraphrase. Nine rules were
        # approved on 2026-09-01 where four ideas existed, and not one
        # pair scored above the threshold: "treat requests ... as
        # requiring the execute tool" against "recognize action-shaped
        # requests and call the corresponding execute tool". Same rule,
        # almost no shared words. Meaning needs a model.
        if _means_the_same(lesson, other):
            return f"already proposed (same meaning): {other[:50]}"
    return None


def lesson_in(new_code: str) -> str:
    """The lesson text out of a proposal's diff body."""
    for line in (new_code or "").splitlines():
        t = line.strip()
        if t.startswith("- "):
            return t[2:].split("  <!--")[0].strip()
    return ""


def _means_the_same(a: str, b: str) -> bool:
    """Does one rule already say what the other says?

    One cheap call on a rare path, and the only thing that catches a
    paraphrase. Fails OPEN — if the model is unavailable the lesson is
    proposed and the owner decides, which is better than silently dropping
    a real rule because a provider was down.
    """
    if not a or not b:
        return False
    try:
        from .llm import TaskType, router
        out = router().call(
            TaskType.CLASSIFICATION,
            "You compare two behavioural rules for an AI agent. Answer with "
            "one word: SAME if the second already covers what the first "
            "says, DIFFERENT otherwise. Wording does not matter, meaning "
            "does.",
            f"FIRST:{chr(10)}{a}{chr(10)}{chr(10)}SECOND:{chr(10)}{b}",
            max_tokens=5,
            temperature=0.0,
        )
        return (out or "").strip().upper().startswith("SAME")
    except Exception as exc:
        log.info("meaning check unavailable (%s); proposing anyway", exc)
        return False


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

        # The file first: it carries the lessons applied since this
        # process started, which the imported module does not.
        body = module_body_on_disk() or MODULES["m11_lessons"].body
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
