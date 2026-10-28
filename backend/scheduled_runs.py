"""Turns nobody typed: scheduled tasks and reminders running in the background.

What went wrong on prod, 2026-10-24..27, with one owner request — "check
PEPE every thirty minutes, write to me only on a signal":

* every run's final answer was mailed to him, "no signal, I sent nothing"
  included, every half hour — he asked for silence and the agent agreed,
  but delivery was not the agent's to decide;
* each run scheduled the next one itself, and every new row brought its
  own "task set, tap Cancel" card;
* the runs executed inside his chat thread, saw the whole conversation,
  and many of them answered a fortnight-old question about his car.

The shape that works is the one Hermes uses for its cron jobs, and this
module is that shape: a run is isolated from the chat; its final answer IS
the message, delivered as written; "[SILENT]" means there is nothing to
say and nothing is sent; "[DONE]" ends the series; a run never schedules,
moves or cancels anything (the schedule repeats it); and it is shown its
previous report so it does not say the same thing twice. Whatever is
actually sent is written into the person's chat, so "what is this?" has
an answer.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging

log = logging.getLogger(__name__)

SILENT = "[SILENT]"
DONE = "[DONE]"

_BACKGROUND: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hrant_background_run", default=False)


def is_background() -> bool:
    """True inside a scheduled task or reminder run."""
    return bool(_BACKGROUND.get())


@contextlib.contextmanager
def background_scope(active: bool = True):
    token = _BACKGROUND.set(bool(active))
    try:
        yield
    finally:
        _BACKGROUND.reset(token)


REFUSAL = (
    "refused: this is a scheduled run in the background. The schedule repeats "
    "it by itself, so do not create, move or cancel schedules from here, and "
    "nobody is waiting to answer a question. Do the task and write the result "
    "as your final answer; end with [DONE] on its own line if the task is "
    "finished for good, or answer [SILENT] if there is nothing to report."
)


def run_prompt(instruction: str, *, previous: str = "", what: str = "task") -> str:
    """The message a background run is woken with."""
    head = (
        f"[SCHEDULED {what.upper()} RUN — you are running in the background, "
        "not in a conversation. Nobody is waiting and you cannot ask "
        "questions.\n"
        "DELIVERY: your final answer is sent to the user exactly as you write "
        "it, as a message from you. Write that message itself — the result, "
        "the reminder, the report — never a description of what you did or "
        "checked. Do not send messages with tools.\n"
        f"SILENT: if, by the task's own terms, there is nothing the user needs "
        f"to hear this time, answer exactly {SILENT} and nothing else; nothing "
        f"is sent. {SILENT} is a literal token: never translate it and never "
        "combine it with other text.\n"
        f"DONE: if the task's goal has been reached for good and it should not "
        f"run again, put {DONE} on the last line by itself; the schedule "
        "stops.\n"
        "RECURSION: this run belongs to an existing schedule that repeats on "
        "its own. Never create, move or cancel schedules from here.]"
    )
    prev = (previous or "").strip()
    if prev:
        head += (
            "\n\n## Your previous report\n"
            "What this task last sent the user. Do not repeat what it already "
            "said; report only what is new.\n\n" + prev[:3000]
        )
    return head + "\n\n## The task\n" + (instruction or "").strip()


# Spellings of "nothing to send" a model writes when it drops the brackets.
_SILENCE_LINES = {"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"}


def _is_silence_line(line: str) -> bool:
    return line.strip().strip(".!*_`'\"").strip().upper() in _SILENCE_LINES


def parse_outcome(answer: str) -> tuple[str, bool, bool]:
    """(text to deliver, silent, done) from a run's final answer.

    Silent when the answer is empty, opens with [SILENT], or has a silence
    marker as its own first or last line — Hermes's rule for its cron lane.
    2026-10-27 15:15: a run on the fallback model wrote its whole check and
    put [SILENT] under it; only a leading token counted then, so the owner
    got the check with "[SILENT]" at the bottom. A marker inside a sentence
    is still content and is delivered.
    """
    text = (answer or "").strip()
    done = False
    lines = text.splitlines()
    kept = [ln for ln in lines if ln.strip() != DONE]
    if len(kept) != len(lines):
        done = True
        text = "\n".join(kept).strip()
    if DONE in text and text.rstrip().endswith(DONE):
        done = True
        text = text.rstrip()[: -len(DONE)].strip()
    rows = [ln for ln in text.splitlines() if ln.strip()]
    silent = (not rows or text.upper().startswith(SILENT)
              or _is_silence_line(rows[0]) or _is_silence_line(rows[-1]))
    return ("" if silent else text), silent, done


def record_in_chat(speaker_id: str, text: str, label: str) -> None:
    """Write a message the agent sent on its own into the person's chat, so
    the conversation that follows can see it. Never raises."""
    body = (text or "").strip()
    if not speaker_id or not body:
        return
    try:
        from .threads import THREADS, latest_key_for
        key = latest_key_for(speaker_id) or speaker_id
        THREADS.append(key, user=f"[{label}]", answer=body, speaker_id=speaker_id)
        try:
            from .conversation import CONVERSATION
            CONVERSATION.add_turn(
                f"[{label}]", body, intent="scheduled", is_chat=False,
                confidence=70, topics_used=[], channel="scheduled",
                speaker_id=speaker_id, session_key=key,
            )
        except Exception as e:
            log.debug("record_in_chat: conversation buffer not updated: %s", e)
    except Exception as e:
        log.debug("record_in_chat failed: %s", e)
