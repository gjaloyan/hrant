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
    "colleague would relay it - not the way a dictionary would.\n\n"
    "Rules:\n"
    "- Keep names, numbers, dates, times, file paths, URLs, model names "
    "and identifiers EXACTLY as spoken. Never localise or correct them.\n"
    "- Speech recognition makes mistakes. When a word is clearly a "
    "mis-hearing of a word that fits the sentence, use the sensible one "
    "and mark it: speech recognition (heard: X).\n"
    "- Preserve the register: a terse instruction stays terse, an "
    "irritated one stays irritated. Tone is information about what the "
    "person needs.\n"
    "- Do not answer, explain, summarise, or add anything. Output only "
    "the English rendering.\n"
    "- If the text is already English, return it unchanged.\n\n"
    "WHEN IT IS NOT SPEECH AT ALL. Recognition sometimes fails so badly "
    "that the text is not a sentence in any language: misspelt words that "
    "resolve to nothing, a name mangled past recognition, grammar that "
    "does not hold together. Do NOT invent a plausible request out of it. "
    "Reply with exactly:\n"
    "  GARBLED: <what little, if anything, seems recoverable>\n"
    "That is a real answer and a useful one. Guessing turns noise into a "
    "fluent sentence nobody said, which is worse than admitting the audio "
    "was not understood."
)

# Typed text needs a different brief: there is no recogniser to blame, so
# "GARBLED" would only invite the model to refuse a hurried message.
_TRANSLATE_WRITTEN_SYSTEM = (
    "You render a typed message in English for another assistant to act "
    "on. This is not a dictionary exercise.\n\n"
    "Carry the INTENT. What is the person asking for, and what would they "
    "consider a correct response? Say that, the way a fluent bilingual "
    "colleague would relay it - not the way a dictionary would.\n\n"
    "Rules:\n"
    "- People type fast: read through typos, missing letters and broken "
    "grammar to what was meant.\n"
    "- Keep names, numbers, dates, times, file paths, URLs, model names, "
    "identifiers and legal or technical terms EXACTLY as written. Never "
    "localise or correct them. Where a term matters, add the original in "
    "brackets after the English.\n"
    "- Preserve the register: a terse instruction stays terse, an "
    "irritated one stays irritated.\n"
    "- Do not answer, explain, summarise, or add anything. Output only "
    "the English rendering.\n"
    "- If the text is already English, return it unchanged."
)

# The marker the translator returns instead of a confabulated sentence.
_GARBLED = "GARBLED:"

# Scripts that mean "not English". Latin text is passed through untouched
# rather than round-tripped through a model that could only damage it.
_NON_LATIN = re.compile(r"[Ѐ-ӿ԰-֏؀-ۿ一-鿿]")

# Armenian letters (the main block and the ligatures). Typed Armenian is the
# one written input the owner asked to translate (2026-10-11): the models read
# it less well than Russian, and the fallback model least of all.
_ARMENIAN = re.compile(r"[Ա-֏ﬓ-ﬗ]")


def needs_translation(text: str) -> bool:
    """True when the text contains a script the owner's rule excludes."""
    return bool(_NON_LATIN.search(text or ""))


def is_armenian(text: str) -> bool:
    """A word or more of Armenian, not a stray letter in another language."""
    return len(_ARMENIAN.findall(text or "")) >= 3


def _switch(key: str) -> bool:
    try:
        from .config import CONFIG
        return bool((CONFIG._data.get("conversation") or {}).get(key, True))
    except Exception:
        return True


def echo_voice_enabled() -> bool:
    """`conversation.echo_voice_transcript` (default on)."""
    return _switch("echo_voice_transcript")


def translate_armenian_text_enabled() -> bool:
    """`conversation.translate_armenian_text` (default on)."""
    return _switch("translate_armenian_text")


_ECHO_LIMIT = 3500


def voice_echo(transcript: str) -> str:
    """What the sender sees before the answer: the words as heard.

    Taken from Hermes, which echoes every transcript back. A mis-heard
    word is then visible at once, instead of surfacing three messages
    later as an answer to a question nobody asked. No label, so it reads
    the same in every language.
    """
    t = " ".join((transcript or "").split())
    if len(t) > _ECHO_LIMIT:
        t = t[:_ECHO_LIMIT].rstrip() + " …"
    return f"🎙 \"{t}\"" if t else ""


async def voice_turn_text(transcript: str, reply) -> str:
    """A voice note's text for the turn. First shows the sender the words as
    heard through `reply` (an async send, e.g. Telegram's `reply_text`), then
    returns the English rendering with the original kept. Never raises."""
    import asyncio
    raw = (transcript or "").strip()
    if not raw:
        return raw
    if echo_voice_enabled() and voice_echo(raw):
        try:
            await reply(voice_echo(raw))
        except Exception as e:
            log.debug("voice echo failed: %s", e)
    try:
        return await asyncio.to_thread(render_for_prompt, raw)
    except Exception as e:
        log.debug("meaning translation skipped: %s", e)
        return raw


async def typed_turn_text(text: str) -> str:
    """A typed message's text for the turn: Armenian gets an English rendering
    with the original kept; anything else passes through. Never raises."""
    import asyncio
    src = (text or "").strip()
    if not src or not translate_armenian_text_enabled() or not is_armenian(src):
        return text
    try:
        return await asyncio.to_thread(render_for_prompt, src, spoken=False)
    except Exception as e:
        log.debug("Armenian text translation skipped: %s", e)
        return text


def to_english(text: str, *, max_tokens: int = 700, spoken: bool = True) -> str:
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
            system=_TRANSLATE_SYSTEM if spoken else _TRANSLATE_WRITTEN_SYSTEM,
            user=src,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        log.warning("meaning translation failed, keeping original: %s", e)
        return src
    return (out or "").strip() or src


def render_for_prompt(original: str, *, spoken: bool = True) -> str:
    """Both readings, labelled, for the turn to act on.

    The English leads because it is what the rule asks for and what the
    agent reasons in. The original follows because the agent replies in
    the owner's language, and because a translation nobody can check is a
    single point of failure over a channel already known to mis-hear.

    When the text does not parse as speech, the turn is told so instead
    of being handed a tidy sentence. Measured 2026-09-01: the recogniser
    produced noise, this layer rendered it as a fluent English request,
    and the agent answered it. Four notes in a row went that way and it
    never once wondered why the owner had started talking nonsense.
    """
    src = (original or "").strip()
    if not src:
        return ""
    english = to_english(src, spoken=spoken)
    if english == src:
        return src
    if spoken and english.strip().startswith(_GARBLED):
        salvage = english.split(":", 1)[-1].strip()
        return _unintelligible_block(src, salvage)
    how = "spoken" if spoken else "written"
    return (
        f"{english}\n\n"
        f"[{how} by the user, verbatim, in the language they used: {src}]"
    )


def _unintelligible_block(src: str, salvage: str) -> str:
    """What the turn sees when the audio did not survive recognition.

    Written as an instruction, not a warning. A warning is something a
    model reads and then proceeds anyway, which is exactly what happened
    when the only signal was that the text looked odd.
    """
    hint = f" The only part that may be real: {salvage}" if salvage else ""
    return (
        "SPEECH RECOGNITION FAILED ON THIS MESSAGE. What came through is "
        "not a sentence in any language, so there is no request in it to "
        "act on." + hint + "\n\n"
        "Do NOT guess what was meant, and do NOT answer as though you "
        "understood. Tell the user plainly, in his own language, that you "
        "could not make out the audio, quote back what you heard so he can "
        "see it, and ask him to repeat or type it.\n\n"
        f"[what recognition produced, verbatim: {src}]"
    )
