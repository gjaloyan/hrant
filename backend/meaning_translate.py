"""Render a non-English message in English, by meaning rather than by word.

The owner's rule is that prompts are English. His voice notes are Armenian
and Russian, so something has to bridge them, and the obvious bridge is
the wrong one: Whisper's own `translate` task emits a literal gloss AND
discards the original, leaving nothing to check when it is wrong.

That mattering is not hypothetical. On 2026-08-31 a Russian note came back
as "добавить армянский язык в распознавание ДИЧИ" — game birds instead of
speech, one letter apart. A literal translation would have carried the
error through with a straight face; the original is what lets anyone
notice.

So: transcribe natively, translate for MEANING with the model that is
already reasoning about the turn, and keep both. The English becomes the
task; the original travels with it, because the agent answers the owner in
his own language and needs to know what he actually said.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_TRANSLATE_SYSTEM = (
    "You render a spoken message in English for another assistant to act "
    "on. This is not a dictionary exercise.\n\n"
    "Carry the INTENT. What is the person asking for, and what would they "
    "consider a correct response? Say that, the way a fluent bilingual "
    "colleague would relay it — not the way a dictionary would.\n\n"
    "Rules:\n"
    "- Keep names, numbers, dates, times, file paths, URLs, model names "
    "and identifiers EXACTLY as spoken. Never localise or 'correct' them.\n"
    "- Speech recognition makes mistakes. When a word is clearly a "
    "mis-hearing of a word that fits the sentence, use the sensible one "
    "and mark it: 'speech recognition (heard: X)'.\n"
    "- Preserve the register: a terse instruction stays terse, an "
    "irritated one stays irritated. Tone is information about what the "
    "person needs.\n"
    "- Do not answer, explain, summarise, or add anything. Output only "
    "the English rendering.\n"
    "- If the text is already English, return it unchanged."
)

# Scripts that mean "not English". Latin text is passed through untouched
# rather than round-tripped through a model that could only damage it.
_NON_LATIN = re.compile(r"[Ѐ-ӿ԰-֏؀-ۿ一-鿿]")


def needs_translation(text: str) -> bool:
    """True when the text contains a script the owner's rule excludes."""
    return bool(_NON_LATIN.search(text or ""))


def to_english(text: str, *, max_tokens: int = 700) -> str:
    """The English rendering, or the original text on any failure.

    Never raises and never returns empty: a translation that fails must
    leave the message intact rather than replace it with nothing. The
    caller keeps the original regardless, so the worst case here is that
    the agent reads the message in its own language — which is exactly
    what it did before this existed.
    """
    src = (text or "").strip()
    if not src or not needs_translation(src):
        return src
    try:
        from .llm import router, TaskType
        out = router().call(
            TaskType.CLASSIFICATION,
            system=_TRANSLATE_SYSTEM,
            user=src,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        log.warning("meaning translation failed, keeping original: %s", e)
        return src
    return (out or "").strip() or src


def render_for_prompt(original: str) -> str:
    """Both readings, labelled, for the turn to act on.

    The English leads because it is what the rule asks for and what the
    agent reasons in. The original follows because the agent replies in
    the owner's language, and because a translation nobody can check is a
    single point of failure over a channel already known to mis-hear.
    """
    src = (original or "").strip()
    if not src:
        return ""
    english = to_english(src)
    if english == src:
        return src
    return (
        f"{english}\n\n"
        f"[spoken by the user, verbatim — reply in THIS language: {src}]"
    )
