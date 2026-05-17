"""Sticky-request detector: spot when the user has repeated the same
request multiple turns in a row without seeing a real change.

The pattern from the production Telegram audit:

  User: Измени голос на мужской                  → Agent: Понял (no tool)
  User: Но ты все еще отвечаешь женским голосом  → Agent: Понял (no tool)
  User: Почему ты не изменяешь голос?            → Agent: Понял (no tool)
  User: Я тебе говорю, измени голос на мужской   → Agent: Понял (no tool)

Each turn was classified as `preference` → ack → save fact → no
change. The user got more frustrated and repeated the directive
4 times before the agent did anything tool-shaped.

The detector inspects the per-speaker conversation memory:
  - last K user messages mention the SAME system attribute
    (voice / language / model / config / setting)
  - corresponding agent answers were SHORT acks (<150 chars) AND
    carried no recorded tool calls

When the pattern matches, the agent's system prompt for the
next turn picks up a `# STICKY REQUEST DETECTED` block telling
the LLM to escalate to tool use instead of producing another
acknowledgement. Same destination as the `_looks_like_system_directive`
short-circuit, different entry signal (we caught a behavioural
failure that the regex missed).
"""
from __future__ import annotations

import re
from typing import Any

from .conversation import CONVERSATION


# Default detection window. K turns back, looking for a SAME-topic
# repeat in M of them. Numbers chosen from the production case:
# the user repeated "change voice" 4 times in 5 turns before the
# agent noticed. We fire at 2 repeats so the SECOND repetition
# triggers escalation — not after the user has already repeated 4×.
_DEFAULT_WINDOW = 5
_DEFAULT_MIN_REPEATS = 2

# Acknowledgement-shaped agent answers. If the agent said one of
# these and didn't run a tool, that's the "ack without action"
# failure mode the detector is looking for.
_ACK_PATTERNS = (
    "понял", "поняла", "будет", "буду",
    "okay", "got it", "ok,", "okay,", "understood",
    "noted", "запомн", "запомнил", "запомню",
    "хорошо", "конечно", "sure", "alright",
)


# System attribute keywords reused from agent._SYSTEM_ATTRIBUTE_RE
# but as a flat set for cheap repeated-attribute matching.
_ATTR_PATTERNS = {
    "voice":     re.compile(r"\b(?:voice|tts|голос|озвучк[ауи]+)\b", re.IGNORECASE),
    "language":  re.compile(r"\b(?:language|язык)\b", re.IGNORECASE),
    "model":     re.compile(r"\b(?:model|модель)\b", re.IGNORECASE),
    "provider":  re.compile(r"\b(?:provider|провайдер)\b", re.IGNORECASE),
    "channel":   re.compile(r"\b(?:channel|канал)\b", re.IGNORECASE),
    "tone":      re.compile(r"\b(?:tone|tон|стиль)\b", re.IGNORECASE),
    "speed":     re.compile(r"\b(?:speed|скорост)\b", re.IGNORECASE),
    "backend":   re.compile(r"\b(?:backend|бэкенд)\b", re.IGNORECASE),
    "config":    re.compile(r"\b(?:config|конфиг|настройк[уаи]+|setting)\b", re.IGNORECASE),
}


def _attributes_mentioned(text: str) -> set[str]:
    """Return the SET of system attributes referenced in `text`.
    Empty set → no system attribute → can't be a sticky request
    about a setting."""
    if not text:
        return set()
    out: set[str] = set()
    for name, rx in _ATTR_PATTERNS.items():
        if rx.search(text):
            out.add(name)
    return out


def _looks_like_ack_only(answer: str) -> bool:
    """True for short acknowledgement-shaped answers that almost
    certainly didn't apply a change. Heuristic: under ~150 chars
    AND starts with / contains an ack phrase. Long answers — even
    if they LOOK like acks — are usually doing something more
    substantial (citing files, explaining, asking back)."""
    if not answer:
        return True  # missing answer is even worse
    body = answer.strip()
    if len(body) > 150:
        return False
    low = body.lower()
    return any(p in low for p in _ACK_PATTERNS)


def detect_sticky_request(
    *,
    current_user_message: str,
    speaker_id: str | None,
    session_key: str | None = None,
    window: int = _DEFAULT_WINDOW,
    min_repeats: int = _DEFAULT_MIN_REPEATS,
) -> dict[str, Any]:
    """Inspect the last `window` turns of this thread's conversation
    history and decide whether the CURRENT user message is a sticky
    repeat of an earlier same-topic request that the agent failed to
    apply. Filtering is per-thread (`session_key`) when supplied —
    same person in two chats can't trigger stickiness across them —
    otherwise per-speaker (legacy).

    Returns:
      {
        'sticky': bool,
        'attribute': str | "",  # the system attribute being repeated
        'repeats': int,         # how many same-attribute messages found
        'reason': str           # short explanation for prompts / logs
      }

    Best-effort: any failure swallowed → returns `{sticky: False, ...}`.
    """
    empty = {
        "sticky": False,
        "attribute": "",
        "repeats": 0,
        "reason": "",
    }
    current_attrs = _attributes_mentioned(current_user_message)
    if not current_attrs:
        return empty
    try:
        # `window+1` so we always have at least one PRIOR turn even
        # when the CURRENT one was already added to memory before
        # we ran the check. Some callers (handle_message) add the
        # turn AFTER agent.run, but we don't want to be sensitive
        # to call-order quirks.
        if session_key:
            recent = CONVERSATION.recent(window + 1, session_key=session_key) or []
        else:
            recent = CONVERSATION.recent(window + 1, speaker_id=speaker_id) or []
    except Exception:
        return empty
    # Walk backwards (newest first) looking for prior user messages
    # that mention any of the SAME attributes as the current one,
    # paired with short ack-only agent answers.
    repeats = 0
    matched_attr = ""
    for turn in reversed(recent):
        user_msg = turn.get("user") or ""
        # Strip the channel prefix `[channel:username]` if present.
        clean_user = re.sub(r"^\[[^\]]+\]\s*", "", user_msg)
        if clean_user.strip() == current_user_message.strip():
            # Don't count the CURRENT turn itself if it leaked in.
            continue
        prior_attrs = _attributes_mentioned(clean_user)
        overlap = prior_attrs & current_attrs
        if not overlap:
            continue
        # Only count when the agent's reply LOOKED like an ack
        # without action. That's the failure mode we care about —
        # repeated successful answers shouldn't trigger escalation.
        agent_reply = turn.get("answer") or ""
        if not _looks_like_ack_only(agent_reply):
            continue
        repeats += 1
        matched_attr = next(iter(overlap)) if not matched_attr else matched_attr
    if repeats < min_repeats:
        return empty
    return {
        "sticky": True,
        "attribute": matched_attr,
        "repeats": repeats,
        "reason": (
            f"The user has asked about {matched_attr} in {repeats} prior "
            f"turn(s) and your previous answers were short acknowledgements "
            f"that didn't apply a change. ESCALATE: actually apply the "
            f"change this turn via `set_setting(key, value)` (preferred) "
            f"or `terminal_exec` / `run_python`, then report the diff."
        ),
    }


def render_sticky_block(info: dict[str, Any]) -> str:
    """Format `detect_sticky_request` output as a prompt block.
    Empty when `sticky=False`."""
    if not info.get("sticky"):
        return ""
    return (
        "# STICKY REQUEST DETECTED\n"
        f"{info['reason']}\n"
    )
