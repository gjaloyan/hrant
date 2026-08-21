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

# Defaults. The live values come from CONFIG.verification, which the Settings
# UI writes and runtime_config validates (2026-08-09 dead-code audit): before
# this, `critic_threshold`, `critic_max_retries` and `critic_retry_token_budget`
# were offered as sliders, accepted, range-checked and persisted — and read by
# nobody. The owner could tune when the critic fires and change nothing.
_CONTENT_CONFIDENCE_BAR = 60
_MAX_REVISION_ITERATIONS = 4
_MAX_REVISION_TOKENS = 2000


def _cfg(key: str, default):
    """Read a verification knob, falling back to the module default.

    Read per call, not cached at import: the Settings UI writes these live and
    a slider that needs a restart to take effect is only half-connected.
    """
    try:
        from .config import CONFIG
        val = (CONFIG.verification or {}).get(key)
        return default if val is None else type(default)(val)
    except Exception:
        return default


def content_confidence_bar() -> int:
    return _cfg("critic_threshold", _CONTENT_CONFIDENCE_BAR)


def rewriting_enabled() -> bool:
    """May the critic REPLACE the agent's answer?

    Read live, so the owner can flip it from Settings without a restart —
    the point of the flag is that HE decides, and a switch needing a
    redeploy is not a decision he can make.
    """
    return bool(_cfg("critic_rewrites_answer", False))


# `critic_max_retries` and `critic_retry_token_budget` are deliberately NOT
# wired here. They name a RETRY LOOP — the legacy pipeline/critic.py mixin —
# and the unified critic makes exactly ONE revision pass. Mapping them onto
# `max_iterations` (a tool-loop depth) and `max_tokens` (a per-call cap) would
# look like connecting the sliders while silently changing what they mean:
# critic_retry_token_budget=60000 would have become a 60k per-call max_tokens,
# 30x the 2000 that is actually right for a revision. They are removed from
# the Settings whitelist instead — a knob for a mechanism that no longer
# exists should not be offered.

# Delivery-class contradiction markers appended by the endpoint cap
# and the psm empty-diff check. They describe PROCESS failures the
# revision can't fix (read-only), so they neither trigger the critic
# nor appear in its critique block.
_DELIVERY_MARKERS = ("endpoint_not_met:", "empty_propose_self_modification:")

# Phrases that mark a revision RETRACTING the previous answer's claims
# rather than fixing them. Matched case-insensitively on the revised text.
# Both languages: the user is Russian-speaking, the agent answers in kind.
_RETRACTION_MARKERS: tuple[str, ...] = (
    "cannot confirm", "can not confirm", "cannot honestly confirm",
    "unable to confirm", "was not verified", "not verified",
    "i did not actually", "no proof that",
    "не могу подтвердить", "не могу честно подтвердить",
    "не подтверждено", "нет доказательств", "не был проверен",
    "не была проверена", "не проверено",
)


def looks_like_retraction(text: str) -> bool:
    """True when a revision walks the previous answer's claims back."""
    low = (text or "").lower()
    return any(m in low for m in _RETRACTION_MARKERS)


def blind_retraction(revised: str, *, verify_calls: int) -> bool:
    """A revision that RETRACTS work while having verified NOTHING.

    Root cause of a real prod failure (2026-07-21): the agent correctly
    edited a PDF, the verifier flagged the "file was created" claim as
    unverified, and the critic — told to soften unsupportable claims —
    replied "I cannot honestly confirm the PDF was changed". The file was
    fine; the user simply never got it. Retracting real work without
    spending one read_file is worse than the original imperfect answer,
    so the caller keeps the original when this fires."""
    return bool(revised) and verify_calls == 0 and looks_like_retraction(revised)


# A revision must EARN its acceptance. The old rule was "any higher score
# wins", and the score measures groundedness — how many claims are backed —
# so deleting a claim raises it mechanically. Under that rule the optimal
# revision is the empty one: an answer that asserts nothing scores near
# 100. The critic was structurally paid to strip.
#
# Measured on the owner's turns, 2026-08-21: a revision scoring 42 against
# 37 replaced an answer about the differences between two legal codes with
# a list of things it would not claim. He asked what changed and received
# an inventory of doubts, and the machine recorded it as an improvement.
#
# A large gain is allowed to shrink the answer — that is a real correction.
# A small gain must keep the substance, or it is the score being gamed.
SUBSTANTIAL_GAIN = 15
MIN_RETENTION = 0.75


def revision_wins(
    *, old_score: int, new_score: int, old_text: str, new_text: str,
) -> tuple[bool, str]:
    """Should the revision replace the original? Returns (yes, why).

    Deliberately ignores WHY the text shrank. A revision that grounds the
    same content scores better without shrinking; one that shrank a lot
    for a couple of points removed something the user wanted, whatever it
    tells itself about honesty.
    """
    gain = int(new_score) - int(old_score)
    if gain <= 0:
        return False, f"no gain ({old_score} -> {new_score})"
    old_len = len((old_text or "").strip())
    new_len = len((new_text or "").strip())
    retention = (new_len / old_len) if old_len else 1.0
    if gain >= SUBSTANTIAL_GAIN:
        return True, f"substantial gain (+{gain})"
    if retention >= MIN_RETENTION:
        return True, f"+{gain} and kept {retention:.0%} of the answer"
    return False, (
        f"+{gain} bought by cutting {1 - retention:.0%} of the answer — "
        f"the score rises when claims are deleted, so a small gain that "
        f"costs this much content is not an improvement"
    )


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
    the self-correction / lessons pipeline's job.

    Returns False outright unless `verification.critic_rewrites_answer`
    is on. Verification itself still runs and still reports; what the
    flag controls is whether its findings may REPLACE what the agent
    wrote. Off by default at the owner's instruction — he judges the
    answer, and he cannot argue with a rewrite he never sees.
    """
    if not rewriting_enabled():
        return False, "critic-rewrite-disabled"
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
        _content_score(vr) < content_confidence_bar()
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
        "\nRewrite the answer fixing ONLY these issues.\n\n"
        "CHECK BEFORE YOU RETRACT. The read-only tools ARE available "
        "to you right now — use them. If a flagged claim is about an "
        "artifact YOU produced this turn (a file you wrote, a server "
        "you started, a record you saved), VERIFY it: read_file the "
        "path, extract the text, list the directory. A claim you can "
        "cheaply check must be checked, never softened blind. "
        "Retracting work you actually did is itself a falsehood — and "
        "it costs the user the result.\n\n"
        "Only soften a claim when you tried to verify it and could "
        "not, and then say what you checked and what was missing. "
        "Keep everything that was correct, including any MEDIA: line "
        "delivering a file to the user. Do NOT start NEW work (no "
        "new edits, no new builds) — verification of existing work is "
        "not new work. Keep the user's language and the original "
        "answer's format."
    )
    return "\n".join(parts)


def revise(
    *,
    task: str,
    answer: str,
    vr: VerificationResult,
    system_prompt: str,
    on_tool_call=None,
    tool_calls_out: list | None = None,
) -> Optional[str]:
    """One revision call with the read-only tool subset. Returns the
    revised answer text or None on failure / empty output.

    `tool_calls_out`, when given, receives the name of every tool the
    revision actually ran — the caller uses it to tell a checked
    correction apart from a blind retraction (see `blind_retraction`)."""
    from .llm import LLMError, TaskType, router
    from .tool_registry import get_registry

    registry = get_registry()
    tools = registry.to_anthropic_list(filter_names=set(READ_ONLY_TOOLS))

    def _execute(name, args):
        if tool_calls_out is not None:
            tool_calls_out.append(name)
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
    revision_tools: list[str] = []
    revised = revise(
        task=task, answer=answer, vr=vr,
        system_prompt=system_prompt, on_tool_call=on_tool_call,
        tool_calls_out=revision_tools,
    )
    if not revised or revised.strip() == (answer or "").strip():
        return None, vr

    # Refuse a retraction that verified nothing: keeping an imperfect but
    # truthful answer beats shipping "I cannot confirm" over work that was
    # actually done (2026-07-21 PDF incident — see `blind_retraction`).
    if blind_retraction(revised, verify_calls=len(revision_tools)):
        log.warning(
            "answer_critic: rejecting blind retraction (0 verification "
            "tool calls); keeping the original answer",
        )
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

    won, why = revision_wins(
        old_score=_content_score(vr), new_score=_content_score(new_vr),
        old_text=answer or "", new_text=revised,
    )
    if won:
        log.info("answer_critic: revision accepted — %s", why)
        return revised, new_vr
    log.warning("answer_critic: revision rejected — %s", why)
    return None, vr
