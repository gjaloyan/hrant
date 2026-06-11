"""Post-verifier critic-revise pass — best-of-2 on answer quality.

AGI-roadmap item (2026-06-11). The legacy deep_agent tier had a
self-critic retry loop (pipeline/critic.py mixin); the unified loop
dropped it. Since then the verifier's findings — contradictions,
unsupported claims — were LOGGED (confidence dip, daily report) but
nothing ever tried to FIX the answer before the user saw it.

This module closes that loop with one bounded revision:

  1. `should_critique(vr, ...)` — fire only on CONTENT problems:
     real contradictions (delivery markers like `endpoint_not_met`
     are excluded — those are process failures handled by the
     self-correction / lessons pipeline) or a low content score
     with unsupported claims.
  2. `revise(...)` — one LLM call that sees the critique block and
     a READ-ONLY tool subset (read_file / locate_symbol /
     search_knowledge / fetch_url / web_search). It can re-check
     evidence but cannot take new actions — its job is to fix
     CLAIMS, not to redo the task. Even hallucinated tool names
     are refused at the execute_tool guard.
  3. `revise_and_pick(...)` — re-verifies the revision and keeps
     whichever of {original, revised} scores higher on CONTENT
     confidence. The endpoint/delivery state is carried over
     unchanged (a read-only revision can't deliver an action).

Cost: fires only when the verifier found problems — on a healthy
turn this module costs nothing.
"""
from __future__ import annotations

import logging
from typing import Optional

from .models import VerificationResult

log = logging.getLogger(__name__)


# Tools the revision call may use. Strictly read-only — a revision
# that "fixes" the answer by launching new work would invalidate the
# verify-and-compare contract (and N revisions would mean N side
# effects).
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "locate_symbol",
    "search_knowledge",
    "fetch_url",
    "web_search",
})

_CONTENT_CONFIDENCE_BAR = 60
_MAX_REVISION_ITERATIONS = 4
_MAX_REVISION_TOKENS = 2000

# Delivery-class contradiction markers appended by the endpoint cap
# and the psm empty-diff check. They describe PROCESS failures the
# revision can't fix (read-only), so they neither trigger the critic
# nor appear in its critique block.
_DELIVERY_MARKERS = ("endpoint_not_met:", "empty_propose_self_modification:")


def content_contradictions(vr: VerificationResult) -> list[str]:
    """vr.contradictions minus the delivery-class markers."""
    out: list[str] = []
    for c in vr.contradictions or []:
        text = str(c or "")
        if any(text.startswith(m) for m in _DELIVERY_MARKERS):
            continue
        if text.strip():
            out.append(text)
    return out


def _content_score(vr: VerificationResult) -> int:
    """Pre-clip claim score when a delivery clip happened, else the
    plain confidence."""
    if vr.content_confidence is not None:
        return int(vr.content_confidence)
    return int(vr.confidence)


def should_critique(
    vr: VerificationResult,
    *,
    answer: str,
    is_chat: bool = False,
    supervisor_mode: bool = False,
    pending_question: bool = False,
) -> tuple[bool, str]:
    """(fire, reason). Content problems only — delivery failures are
    the self-correction / lessons pipeline's job."""
    if supervisor_mode:
        return False, "supervisor-turn"
    if is_chat:
        return False, "chat-turn"
    if pending_question:
        return False, "ask_user-pending"
    if not (answer or "").strip():
        return False, "empty-answer"
    contras = content_contradictions(vr)
    if contras:
        return True, f"{len(contras)} contradiction(s)"
    if (
        _content_score(vr) < _CONTENT_CONFIDENCE_BAR
        and (vr.unverified_claims or [])
    ):
        return True, (
            f"content-confidence-{_content_score(vr)} with "
            f"{len(vr.unverified_claims)} unverified claim(s)"
        )
    return False, "no-content-problems"


def build_critique(vr: VerificationResult, prev_answer: str) -> str:
    """Standalone critique block (adapted from the legacy
    SelfCriticMixin._build_critique). Lists only CONTENT problems."""
    parts = [
        "# CRITIQUE OF YOUR PREVIOUS ANSWER",
        f"Your previous answer scored {_content_score(vr)}% confidence.",
        "The verifier found the following problems:",
        "",
    ]
    contras = content_contradictions(vr)
    if vr.unverified_claims:
        parts.append("## Unverified claims (no evidence found):")
        for c in vr.unverified_claims[:8]:
            parts.append(f"- {c}")
    if contras:
        parts.append("")
        parts.append("## Contradictions (conflict with sources):")
        for c in contras[:8]:
            parts.append(f"- {c}")
    parts.append("")
    parts.append(f"## Your previous answer (to revise):\n{prev_answer[:2000]}")
    parts.append(
        "\nRewrite the answer fixing ONLY these issues. Re-check "
        "evidence with the read-only tools if needed. Remove or "
        "soften claims you cannot support; keep everything that was "
        "correct. Do NOT start new work — no new actions are "
        "available in this pass. Keep the user's language and the "
        "original answer's format."
    )
    return "\n".join(parts)


def revise(
    *,
    task: str,
    answer: str,
    vr: VerificationResult,
    system_prompt: str,
    on_tool_call=None,
) -> Optional[str]:
    """One revision call with the read-only tool subset. Returns the
    revised answer text or None on failure / empty output."""
    from .llm import LLMError, TaskType, router
    from .tool_registry import get_registry

    registry = get_registry()
    tools = registry.to_anthropic_list(filter_names=set(READ_ONLY_TOOLS))

    def _execute(name, args):
        # Belt-and-suspenders: even if the model hallucinates a tool
        # name outside the filtered schema, refuse to run it.
        if name not in READ_ONLY_TOOLS:
            return (
                f"[tool '{name}' is not available in the revision pass "
                f"— read-only tools only: {sorted(READ_ONLY_TOOLS)}]",
                True,
            )
        return registry.execute(name, args)

    critique = build_critique(vr, answer)
    try:
        out = router().call_with_tools(
            TaskType.SELF_CRITIC,
            system_prompt,
            f"{task}\n\n{critique}",
            tools=tools,
            execute_tool=_execute,
            max_tokens=_MAX_REVISION_TOKENS,
            temperature=0.2,
            max_iterations=_MAX_REVISION_ITERATIONS,
            on_tool_call=on_tool_call,
        )
    except LLMError as e:
        log.warning("answer_critic revision failed: %s", e)
        return None
    out = (out or "").strip()
    return out or None


def revise_and_pick(
    *,
    task: str,
    answer: str,
    vr: VerificationResult,
    system_prompt: str,
    tool_context: str = "",
    on_tool_call=None,
) -> tuple[Optional[str], VerificationResult]:
    """Revise, re-verify, keep the better of {original, revised} by
    CONTENT score. Returns (revised_answer, revised_vr) when the
    revision wins, (None, original_vr) otherwise. Delivery state
    (endpoint_met + its confidence clip) carries over unchanged —
    a read-only revision cannot deliver an action."""
    revised = revise(
        task=task, answer=answer, vr=vr,
        system_prompt=system_prompt, on_tool_call=on_tool_call,
    )
    if not revised or revised.strip() == (answer or "").strip():
        return None, vr

    try:
        from .verifier import verify
        new_vr = verify(
            question=task,
            answer=revised,
            notes_text="",
            used_topics=[],
            tool_context=tool_context,
        )
    except Exception as e:
        log.warning("answer_critic re-verify failed: %s", e)
        return None, vr

    # Carry the delivery judgment over; re-apply the clip when the
    # endpoint was missed (the revision can't have fixed delivery).
    new_vr.endpoint_met = vr.endpoint_met
    if vr.endpoint_met is False:
        new_vr.content_confidence = new_vr.confidence
        new_vr.confidence = min(new_vr.confidence, 30)
        try:
            new_vr.contradictions.append(
                "endpoint_not_met: action-verb request without "
                "execute-class tool call or MEDIA: delivery"
            )
        except Exception:
            pass

    if _content_score(new_vr) > _content_score(vr):
        return revised, new_vr
    return None, vr
