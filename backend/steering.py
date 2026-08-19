"""Messages the user sends while a turn is already running.

Before this, every inbound message created its own job. Writing "no, not
that one — the second case" while the agent was mid-task did not correct
the task; it opened a second one, and the first kept going with the
instruction the user had already withdrawn. The owner reported it as:
he wrote to the agent while a task was executing, and it opened a new
task instead of taking the correction into the running one.

So an inbound message that arrives while its speaker already has a turn
in flight is parked here instead, and the running turn picks it up at its
next tool result — the same interception point that already carries the
budget, no-progress and proof markers.

Two decisions worth stating, because both had a tempting wrong answer.

**Everything is absorbed; nothing is classified.** A mid-turn message
might be a correction, a new task, or a passing remark, and there is no
reliable way to tell from the text — every attempt is keyword routing
wearing a hat. The model is already reading the conversation and is far
better placed to judge, so it gets the message verbatim and decides.
It may act on it, defer it, or say it will handle it after.

**Nothing is silently dropped.** A turn can end before it reads its
queue: it may be on its last iteration, or already writing the answer.
Undelivered messages are handed back by `close_turn` / `pop_orphans` so
the caller can dispatch them as a normal turn — the user is never left
waiting on a message that was quietly swallowed.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

# One turn should not be able to absorb an unbounded backlog: a user
# hammering the keyboard would otherwise grow the tool result without
# limit. Beyond this the extra messages stay queued for the next turn.
MAX_PENDING_PER_JOB = 8
# Orphans are messages a finished turn never read. Bounded per speaker so a
# channel that stops collecting cannot grow memory without limit.
MAX_ORPHANS_PER_SPEAKER = 16


@dataclass
class SteeringMessage:
    text: str
    speaker_id: str = ""
    channel: str = ""
    # Set when the running turn has shown this to the model. Delivered
    # messages must not be re-dispatched as their own turn afterwards.
    delivered: bool = False
    attachments: list = field(default_factory=list)


_lock = threading.Lock()
_queues: "dict[str, list[SteeringMessage]]" = {}
# job_id -> speaker_id, so an arriving message can find the turn that is
# already serving that speaker without the caller knowing job ids.
_owners: "dict[str, str]" = {}
# speaker_id -> messages a finished turn never read.
_orphans: "dict[str, list[SteeringMessage]]" = {}


def register_turn(job_id: str, speaker_id: str) -> None:
    """Announce that `job_id` is now the live turn for `speaker_id`."""
    if not job_id:
        return
    with _lock:
        _owners[job_id] = speaker_id or ""
        _queues.setdefault(job_id, [])


def active_job_for(speaker_id: str) -> Optional[str]:
    """The live turn serving this speaker, if any.

    Speaker-scoped rather than global: two people talking to the agent at
    once must not steer each other's work.
    """
    if not speaker_id:
        return None
    with _lock:
        for job_id, owner in _owners.items():
            if owner == speaker_id:
                return job_id
    return None


def enqueue(job_id: str, text: str, *, speaker_id: str = "",
            channel: str = "", attachments: Optional[list] = None) -> bool:
    """Park a message for a running turn. False if it could not be taken
    (unknown turn, empty text, or the queue is full) — the caller must
    then fall back to starting a normal turn rather than dropping it."""
    text = (text or "").strip()
    if not text or not job_id:
        return False
    with _lock:
        q = _queues.get(job_id)
        if q is None:
            return False
        if len([m for m in q if not m.delivered]) >= MAX_PENDING_PER_JOB:
            return False
        q.append(SteeringMessage(text=text, speaker_id=speaker_id,
                                 channel=channel,
                                 attachments=list(attachments or [])))
    return True


def take(job_id: str) -> "list[SteeringMessage]":
    """Hand the running turn everything queued for it, marking it
    delivered. Empty list when there is nothing — the common case, so it
    stays cheap enough to call on every tool result."""
    if not job_id:
        return []
    with _lock:
        q = _queues.get(job_id)
        if not q:
            return []
        fresh = [m for m in q if not m.delivered]
        for m in fresh:
            m.delivered = True
    return fresh


def has_pending(job_id: str) -> bool:
    if not job_id:
        return False
    with _lock:
        return any(not m.delivered for m in _queues.get(job_id, []))


def close_turn(job_id: str) -> "list[SteeringMessage]":
    """End the turn's registration; return and park whatever it never read.

    Called from the same `finally` that releases the turn's browser, so it
    runs even when the turn ends by raising. A turn can finish before
    reading its queue — it may be on its last iteration, or already
    composing the answer — and those messages must not evaporate. They are
    moved to a per-speaker orphan list for `pop_orphans`, and also
    returned, so a caller holding the job id can act immediately.
    """
    if not job_id:
        return []
    with _lock:
        q = _queues.pop(job_id, []) or []
        speaker = _owners.pop(job_id, "")
        left = [m for m in q if not m.delivered]
        if left:
            bucket = _orphans.setdefault(speaker, [])
            bucket.extend(left)
            # Bounded: a channel that never collects must not grow memory
            # without limit. The newest are the ones still worth answering.
            if len(bucket) > MAX_ORPHANS_PER_SPEAKER:
                del bucket[:-MAX_ORPHANS_PER_SPEAKER]
    return left


def pop_orphans(speaker_id: str) -> "list[SteeringMessage]":
    """Messages that arrived during a turn which never read them.

    The caller owes these a turn of their own: from the user's side they
    were sent and never answered, which is indistinguishable from the bug
    this module exists to fix.
    """
    with _lock:
        return _orphans.pop(speaker_id or "", [])


def render_marker(messages: "list[SteeringMessage]") -> str:
    """The block the running turn sees appended to its tool result.

    Phrased so the model treats it as the user speaking NOW, mid-task,
    and re-reads its own plan against it. It deliberately does not say
    "this is a correction" — that would be the classification this module
    refuses to make. The model decides what the message is.
    """
    if not messages:
        return ""
    body = "\n".join(f'  "{m.text}"' for m in messages)
    plural = "messages" if len(messages) > 1 else "a message"
    return (
        f"\n\n💬 **THE USER SENT {plural.upper()} WHILE YOU WERE WORKING**\n"
        f"{body}\n\n"
        "This arrived just now, during this task — it is not a new "
        "conversation and there is no second turn coming to handle it. "
        "Read it against what you are currently doing and decide what it "
        "means:\n"
        "  - a correction or a change of target -> adjust NOW, and do not "
        "finish the version the user has just moved away from;\n"
        "  - extra information or a constraint -> fold it into the work in "
        "flight;\n"
        "  - a separate request -> finish what you are doing, then handle "
        "it in the same turn, or say plainly in your answer that it is "
        "still outstanding.\n"
        "Acknowledge it in your final answer either way. Silently "
        "continuing as if it had not arrived is the failure this exists "
        "to prevent."
    )
