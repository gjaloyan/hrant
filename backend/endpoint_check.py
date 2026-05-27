"""Endpoint-aware post-hoc verifier signal.

The legacy verifier returns a confidence based on whether *claims*
in the answer are *verifiable* against grounding (tool outputs +
notes). It does NOT check whether the answer DELIVERED against the
shape of the user's request.

The 2026-05-26 terminal-bench turn ("can you run terminal-bench")
exercised this gap: 17 inspect calls, zero `start_background_job`,
no deliverable — but every CLAIM the agent made ("I checked
harbor", "docker is available") was verifiable, so the verifier
returned 75.

This module adds an orthogonal check: for action-verb requests,
require at least one execute-class tool call OR a MEDIA: file
delivery in the answer. If neither, cap the confidence at 30 and
mark the turn endpoint-missed.

The keyword list here is for ANALYTICS (confidence calibration),
not ROUTING. The agent's runtime behavior is unaffected — only the
confidence label changes.
"""
from __future__ import annotations


# Tools whose presence in the trace counts as "the agent actually
# did something". `terminal_exec` is deliberately excluded — it's
# ambiguous (inspection vs execution by command, which we'd need
# to parse). `run_python` is included because it's almost always
# used for file-writes / computation in our codebase.
_EXECUTE_TOOLS: frozenset[str] = frozenset({
    "set_setting",
    "save_user_fact",
    "run_python",
    "start_background_job",
    "define_task_endpoint",
    "complete_supervisor",
    "schedule_message",
    "grant_telegram_access",
    "revoke_telegram_access",
    "approve_pairing",
    "propose_skill",
    "propose_self_modification",
    "delegate",
    "ask_user",
    "sandbox_exec",
    "agent_browser",
})


# Substrings that signal the user wants the agent to DO something.
# Multilingual (English / Russian / Armenian). We match as
# case-insensitive substrings, so e.g. "run" matches "run terminal-
# bench" but not "running" — close enough; false negatives are
# safer than false positives (a false-positive caps confidence on
# a chat turn).
_ACTION_VERB_TOKENS: tuple[str, ...] = (
    # English (with trailing space to avoid matching "running" / "runt")
    "run ", "launch ", "start ", "execute ", "send ", "deploy ",
    "install ", "build ", "make ", "create ", "fix ", "apply ",
    "delete ", "remove ", "set ", "change ", "update ",
    "schedule ", "post ", "submit ", "push ",
    "save ", "write ",
    # Russian — root forms; suffix variations covered by substring.
    "запусти", "запустите", "запускай",
    "сделай", "сделайте",
    "отправ",  # отправь / отправьте / отправляй
    "поставь", "ставь",
    "измени", "поменяй",
    "ускорь", "замедл",
    "удали", "удалите",
    "создай", "создайте",
    "напиши", "напишите",
    "примени", "применить",
    "запиши", "сохрани",
    # Armenian — small set
    "գրիր", "ուղարկ",
)


def _detect_action_verbs(text: str) -> bool:
    """True if `text` looks like a directive — user wants the agent
    to DO something, not just answer."""
    if not text:
        return False
    low = text.lower()
    return any(tok in low for tok in _ACTION_VERB_TOKENS)


def endpoint_met(
    *, task: str, answer: str, tool_names: list[str],
) -> bool:
    """Did the agent's trace deliver against the user's request?

    Non-action requests automatically pass (no expectation of
    state change). Action requests require either:
      - at least one execute-class tool call, OR
      - a `MEDIA:` line in the answer (file delivery convention).
    """
    if not _detect_action_verbs(task):
        return True
    for name in (tool_names or []):
        if name in _EXECUTE_TOOLS:
            return True
    if answer and "MEDIA:" in answer:
        return True
    return False


# Cap applied when endpoint is missed. 30 is below the "low_confidence"
# threshold (~50) used by the daily report — a flag the
# meta_learner / human reviewer will see as needing attention.
_MISSED_ENDPOINT_CAP: int = 30


def cap_confidence_for_endpoint(
    *, task: str, answer: str, tool_names: list[str], confidence: int,
) -> int:
    """If the endpoint isn't met, clip confidence at
    _MISSED_ENDPOINT_CAP. Never raises confidence — already-low
    scores pass through unchanged."""
    if endpoint_met(task=task, answer=answer, tool_names=tool_names):
        return confidence
    return min(confidence, _MISSED_ENDPOINT_CAP)
