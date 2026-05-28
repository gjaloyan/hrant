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

This module adds an orthogonal check: for action requests, require
at least one execute-class tool call OR a MEDIA: file delivery in
the answer. If neither, an LLM judge decides whether the request was
satisfied (language-agnostic, no keyword lists). If not satisfied,
confidence is capped at 30 and the turn is marked endpoint-missed.
"""
from __future__ import annotations


# Tools whose presence in the trace UNAMBIGUOUSLY signals "the
# agent took a state-changing action against the user's request".
# Deliberately narrow: `terminal_exec` and `run_python` are
# EXCLUDED because they can be inspection (`which X`, `2+2`) OR
# execution (`pip install X`, `Path.write_text(...)`). Counting
# them would falsely pass the endpoint check for an inspection-
# only trace — which is exactly what the 2026-05-27 smoke test of
# `run terminal-bench` revealed: 36 terminal_exec + 24 run_python
# without one `start_background_job`, agent claimed "ran 3 tasks",
# verifier accepted because run_python was in this set.
_EXECUTE_TOOLS: frozenset[str] = frozenset({
    "set_setting",
    "save_user_fact",
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

_ENDPOINT_JUDGE_SYSTEM = """You judge whether an assistant's answer DELIVERED what the user's request required.

You are given the user's request and the assistant's answer. No action/execute tool was called this turn.

Rules:
- If the request was purely informational (a question, explanation, opinion, small talk) it is satisfied by a relevant answer -> endpoint_met = true.
- If the request demanded an ACTION, a state change, or a concrete result (run/execute/send/create something, change a setting, produce a file) it is satisfied ONLY if the answer actually delivers that result. A bare "done" / "I did it" with no evidence -> endpoint_met = false.
- An HONEST "I cannot do X (reason), but here is a concrete plan / alternative / proposal" counts as satisfied -> endpoint_met = true.
- Judge in the user's own language; the request may be in any language.

Return strictly JSON: {"endpoint_met": true|false, "reason": "short"}"""


def _llm_endpoint_met(task: str, answer: str) -> bool:
    """LLM judgment replacing the old keyword action-verb detection.
    Fails OPEN (returns True) on any LLM/infra error so a verifier-side
    failure never spuriously caps a good turn."""
    from .llm import router, TaskType
    try:
        data = router().call_json(
            TaskType.CLASSIFICATION,
            _ENDPOINT_JUDGE_SYSTEM,
            f"USER REQUEST:\n{task}\n\nASSISTANT ANSWER:\n{answer}",
            max_tokens=120, temperature=0.0,
        )
        return bool(data.get("endpoint_met", True))
    except Exception:
        return True


def endpoint_met(*, task: str, answer: str, tool_names: list[str]) -> bool:
    """Did the answer deliver against the request? Cheap deterministic
    checks first (an execute-class tool ran, or a file was delivered via
    MEDIA:), otherwise defer to an LLM judgment (language-agnostic,
    no keyword lists)."""
    for name in (tool_names or []):
        if name in _EXECUTE_TOOLS:
            return True
    if answer and "MEDIA:" in answer:
        return True
    return _llm_endpoint_met(task, answer)


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
