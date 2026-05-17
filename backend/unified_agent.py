"""Unified agent loop — Claude-Code / Hermes style.

The single biggest architectural difference between hrant and the
agents the user said «understand right» (Hermes, Claude Code): we
have intent CLASSIFICATION (a separate LLM call that routes the
turn into `chat` / `preference` / `task` pipelines, each with
different tool access and prompts), they have a single tool-use
loop where the LLM sees ALL tools every turn and decides for
itself what to do.

The audit caught the failure mode: when our `_classify_intent`
mislabels a directive ("Increase voice speed by 30%") as
`preference`, the agent lands in `_save_preference` which has NO
tool access. It saves a fact, says "Понял", does nothing. Patching
regexes (which I'd been doing) is whack-a-mole — there's always
another phrasing.

This module is the unified path. Wired in `agent.Agent.run` behind
the `HRANT_UNIFIED_AGENT=1` env flag so it can run side-by-side
with the legacy pipeline for A/B verification. When stable it
becomes default and the legacy classifier / preference / pipeline-
tier code gets removed (Phase C).

Shape of the unified turn:

  1.  Pre-flight (cheap, no LLM):
      - context_compressor.maybe_compact     — long-history hygiene
      - sticky_requests.detect               — behavioural backstop
      - self_state.current_self_state        — what the agent IS
      - HYBRID.search(task, limit=3)         — auto-recall (cheap)
      - GOALS.tick_interaction               — autonomic
  2.  Build system prompt:
      - identity preamble (soul + name + user profile)
      - STATE SNAPSHOT (mutable settings, tools, configs)
      - speaker permissions (owner / trusted / guest)
      - sticky-request hint (when fired)
      - related notes from auto-recall
      - recent conversation
      - RULES block (anti-hallucination + use-tools-when-asked)
  3.  ONE tool-loop call:
      - router.call_with_tools(task_type, system, user, tools,
        execute_tool, max_iterations=8)
      - LLM decides everything: chat, tool use, set_setting,
        save_user_fact, delegate, refuse, ask clarifying Q
  4.  Post-hoc:
      - memory_extractor.extract_and_store   — feeds KG
      - verifier (opt-in via marker OR auto for named-entity-heavy
        answers — same rule as legacy task_mode)
      - CONVERSATION.add_turn                — history persistence
      - EVALUATOR.log                         — analytics

All the hrant-unique subsystems (verifier, KG, memory extractor,
goals, autonomic, consolidation, self-mod, subagents, sticky,
compressor) survive — they're either pre-flight or post-hoc steps
that don't need the pipeline-tier branch.

What's GONE:
  - `_classify_intent`                     — LLM decides
  - `_handle_preference` / `_save_preference` — `save_user_fact` tool
  - `_pick_pipeline_mode`                  — single tier
  - `_chat_reply` / `_solve`               — replaced by call_with_tools
  - All directive / preference / recall regex routers

The replacement system prompt makes the swap real: a strong RULES
block tells the LLM "take action, don't acknowledge; use
set_setting for config, not terminal_exec hacks; never claim
tools are unavailable".
"""
from __future__ import annotations

import logging
import os
import time as _time
from typing import Optional

from .models import (
    AgentAnswer,
    LLMCallDetail,
    ThinkingStep,
    TokenUsage,
    ToolCallDetail,
    VerificationResult,
)


log = logging.getLogger(__name__)


# ─── Feature flag ──────────────────────────────────────────────────


def unified_enabled() -> bool:
    """Unified path is now the DEFAULT (Phase C, after A/B verified
    on the server). Opt-OUT via `HRANT_LEGACY_PIPELINE=1` for the
    rare emergency rollback during the post-flip window. Once Stage
    3 lands and the legacy code is deleted, the opt-out becomes a
    no-op too — but keeping the gate here for one more deploy lets
    us flip back via env-only without a code change."""
    v = (os.getenv("HRANT_LEGACY_PIPELINE", "") or "").strip().lower()
    return v not in ("1", "true", "yes", "on")


# ─── RULES block (the single biggest difference from legacy) ──────


_UNIFIED_RULES = """# UNIFIED AGENT RULES

You are a single-loop agent. Every turn you receive: the user's
message + your identity / state snapshot / permissions / recent
conversation / related notes / available tools. You decide
everything yourself. There is no upstream classifier routing you
into a tool-less branch.

## Apply, don't acknowledge
When the user requests a change ("change X", "set Y", "increase Z",
"измени X", "ускорь Y", or any equivalent phrasing in any language),
APPLY the change THIS TURN via a tool call. Then report a
one-sentence confirmation of WHAT changed and WHERE.

DO NOT say "Понял, буду X" / "Got it, will do X" / "Sure, I'll X"
as a final answer without having called a tool that applies X.
That is the single most common failure mode the production audit
caught — repeated 4 times in one conversation. An acknowledgement
without a corresponding tool call is a LIE; never produce one.

## Pick the right tool

  - System / config change (voice, model, language, rate, …) →
    `set_setting(key, value)`. The MUTABLE SETTINGS block lists
    every key + its current value + valid choices. One tool call,
    not 4-6 of hand-editing JSON via terminal_exec.

  - Stable user-profile fact (language preference, style, "my
    name is X", interaction rule) → `save_user_fact(category,
    fact)`. NOT for one-off task state; only for traits the
    agent should remember across sessions.

  - Knowledge lookup → `search_knowledge(query)` first; if empty
    or stale, fall back to `web_search` / `read_file`.

  - Read source code → `locate_symbol` FIRST, then `read_file`
    with start_line / end_line. Don't dump 60KB files when you
    need 30 lines.

  - Owner-only shell inspection (status, logs, file content) →
    `terminal_exec`. NOT a substitute for `set_setting` when a
    setting exists.

  - Multi-step research / code-review → `delegate(role, task)`
    to a specialised subagent (researcher / coder / reviewer).

  - Self-mod (structural code changes the user requested) →
    `propose_self_modification(description, files, rationale)`.

## Refusals must be honest

NEVER say "tools are disabled" / "инструменты отключены" / "I
can't apply" when tools are listed above. The tools listed ARE
available this turn — refusal is only valid when:
  1. The setting / file / API genuinely doesn't exist, AND
  2. You've tried at least one tool to verify, AND
  3. You explain WHAT'S missing and offer a concrete next step.

If a tool call failed, try a DIFFERENT tool — don't surrender.

## Chat vs task

Not every turn needs a tool. Casual chat ("hi", "thanks", "how
are you"), recall ("what is my name?", "what voice are you
using?"), and small acknowledgements answer directly from
context. Use the STATE SNAPSHOT for recall answers — don't guess.

But: any message that looks like a directive, a request to
change state, a question about external facts you don't already
know, or a multi-step problem — USE TOOLS.

## Repeated request → escalate

When the conversation history shows the user has raised the
same topic 2+ times and your prior replies were short
acknowledgements (the STICKY REQUEST DETECTED block flags this
when it fires), you have ALREADY failed once or twice. Don't
fail again: pick the most aggressive tool that could apply
the change (set_setting / terminal_exec / propose_self_modification)
and use it this turn."""


# ─── Auto-recall ───────────────────────────────────────────────────


def _auto_recall_block(task: str, *, limit: int = 3) -> str:
    """Cheap pre-flight: hybrid-search for notes related to the
    user's message and inject as a context block. Saves the LLM
    a `search_knowledge` tool call when the relevant notes are
    obvious from the message text.

    Skipped on very short messages (< 20 chars) — short messages
    usually don't have enough signal for useful retrieval."""
    if not task or len(task.strip()) < 20:
        return ""
    try:
        from .hybrid_searcher import HYBRID
        # Re-use the post-Phase-2-audit min_raw_score of 0.55 so
        # we don't pull noise (Mercury → Scary Movie).
        hits = HYBRID.search(task, limit=limit)
    except Exception:
        return ""
    if not hits:
        return ""
    lines = ["# AUTO-RECALL (related notes — read via read_file if relevant)"]
    for h in hits:
        e = h.entry
        lines.append(
            f"- {e.topic} (cat: {e.category}, score: {h.score:.2f}, "
            f"source: {h.source}) → {e.path}"
        )
    return "\n".join(lines)


# ─── Main entry point ──────────────────────────────────────────────


def run_unified(
    *,
    agent,
    task: str,
    project: Optional[str],
    attachments: Optional[list[str]],
    channel: str,
    speaker_id: str,
) -> AgentAnswer:
    """Execute one unified-loop turn. Called from `Agent.run` when
    the env flag is set. `agent` is the calling Agent instance —
    we pull `progress`, `_record_llm_call`, `_trace`, etc. from it
    so the dev panel / SSE stream still receives all the same
    events the legacy pipeline emitted."""
    # Late imports to avoid cycles.
    from . import roles as _roles
    from . import sticky_requests as _sticky
    from . import context_compressor as _cc
    from . import self_state as _ss
    from .conversation import CONVERSATION
    from .evaluator import EVALUATOR, EvalEntry
    from .goals import GOALS
    from .identity import IDENTITY
    from .knowledge_manager import KM  # noqa: F401 — used by side-effects
    from .llm import LLMError, TaskType, TOKENS, router
    from .memory_extractor import MEMORY
    from .tool_registry import get_registry

    # Pre-flight 1: long-history compaction (no-op when under budget).
    try:
        _cc.maybe_compact(speaker_id=speaker_id)
    except Exception as e:
        log.debug("unified: compaction failed (non-fatal): %s", e)

    # Pre-flight 2: sticky-request detection (renders into prompt).
    sticky_info = _sticky.detect_sticky_request(
        current_user_message=task, speaker_id=speaker_id,
    )
    sticky_block = _sticky.render_sticky_block(sticky_info)

    # Pre-flight 3: progress event.
    agent.progress("unified", "single-loop turn starting")

    # System prompt assembly.
    perms = _roles.permissions_block(speaker_id)
    try:
        state = _ss.current_self_state(speaker_id=speaker_id)
        snapshot = _ss.render_snapshot(state)
    except Exception:
        snapshot = ""

    recall = _auto_recall_block(task)
    convo = CONVERSATION.context_block(n=6, speaker_id=speaker_id)

    system_parts = [
        IDENTITY.preamble(speaker_id=speaker_id),
    ]
    if snapshot:
        system_parts.append(f"---\n\n{snapshot}")
    if sticky_block:
        system_parts.append(f"---\n\n{sticky_block}")
    if recall:
        system_parts.append(f"---\n\n{recall}")
    if convo:
        system_parts.append(f"---\n\n{convo}")
    system_parts.append(f"---\n\n{_UNIFIED_RULES}")
    system_parts.append(f"---\n\n{perms}")
    system_prompt = "\n\n".join(system_parts)

    # Tool surface = whole registry. The LLM picks.
    registry = get_registry()
    tools_schema = registry.to_anthropic_list()

    # Tool execution + progress wiring. Reuse the agent's existing
    # `_on_tool_call` shape so the dev panel + SSE event stream
    # gets the same events the legacy pipeline emits.
    tool_outputs: list[str] = []
    _DEFAULT_CAP = 1500
    _tool_cap = {
        "read_file": 20000,
        "view_file": 20000,
        "read_note": 8000,
        "list_files": 4000,
        "glob": 4000,
        "grep": 4000,
        "search": 4000,
        "search_knowledge": 4000,
    }

    def _on_tool_call(name: str, args: dict, result: str, is_error: bool) -> None:
        preview = (result or "").strip().splitlines()[0][:80] if result else ""
        tag = "tool_error" if is_error else "tool"
        _trace_result_cap = 4000
        full_result = result or ""
        full_len = len(full_result)
        preview_body = full_result[:_trace_result_cap]
        detail = ToolCallDetail(
            name=name,
            args=args or {},
            result=preview_body,
            result_truncated=full_len > _trace_result_cap,
            result_full_len=full_len,
            is_error=bool(is_error),
        )
        agent.progress(
            tag,
            f"{name}({', '.join((args or {}).keys())}) -> {preview}",
            tool_call=detail,
        )
        if result:
            cap = _tool_cap.get(name, _DEFAULT_CAP) if not is_error else 2000
            snippet = result[:cap]
            if len(result) > cap:
                snippet += f"\n…[+{len(result) - cap} more chars truncated]"
            tag_str = f"[{name} ERROR]" if is_error else f"[{name}]"
            tool_outputs.append(f"{tag_str} {snippet}")

    def _execute_with_progress(name: str, args: dict):
        preview = ", ".join(str(k) for k in (args or {}).keys())
        agent.progress(
            "tool_starting",
            f"{name}({preview})",
            tool_call=ToolCallDetail(
                name=name,
                args=args or {},
                result="",
                result_truncated=False,
                result_full_len=0,
                is_error=False,
            ),
        )
        return registry.execute(name, args)

    # The big call. max_iterations: legacy solve used 6; unified
    # turns can do more work in one loop (research + apply +
    # report) so we widen a bit. Falls under failover chain.
    t0 = _time.monotonic()
    usage_before = TOKENS.request_usage()
    try:
        answer = router().call_with_tools(
            TaskType.COMPLEX_SOLVING,
            system_prompt,
            task,
            tools=tools_schema,
            execute_tool=_execute_with_progress,
            max_tokens=4000,
            temperature=0.3,
            max_iterations=10,
            on_tool_call=_on_tool_call,
            attachments=attachments or None,
        )
    except LLMError as e:
        # Bubble up — outer run() / SSE handler classifies and
        # surfaces to the user.
        raise

    # Record the (super) LLM call for the dev panel. We treat the
    # whole tool loop as one labelled call — per-iteration detail
    # is already visible via the `tool` / `tool_error` trace
    # entries.
    agent._record_llm_call(
        label="_unified",
        task_type=TaskType.COMPLEX_SOLVING,
        system=system_prompt,
        user=task,
        response=str(answer),
        duration_ms=int((_time.monotonic() - t0) * 1000),
        usage_before=usage_before,
    )

    # Post-hoc: memory extraction.
    try:
        MEMORY.extract_and_store(
            task, answer, intent="task",
            confidence=100, contradictions=0,
            speaker_id=speaker_id,
        )
    except Exception as e:
        log.debug("unified: memory extract failed: %s", e)

    # Post-hoc: optional verifier. Same threshold as legacy
    # task_mode — only fires when there's grounding material to
    # verify against (notes + tool outputs).
    vr = VerificationResult(confidence=85)
    if tool_outputs:
        try:
            from .verifier import verify
            vr = verify(
                question=task,
                answer=answer,
                notes_text="",  # auto-recall is already in system prompt
                used_topics=[],
                tool_context="\n\n".join(tool_outputs),
            )
        except Exception as e:
            log.debug("unified: verifier failed: %s", e)

    # Goals + conversation persistence + evaluator log.
    try:
        GOALS.tick_interaction()
    except Exception:
        pass

    CONVERSATION.add_turn(
        task, answer or "",
        intent="task", is_chat=False,
        confidence=vr.confidence,
        topics_used=[],
        channel=channel,
        speaker_id=speaker_id,
    )

    try:
        EVALUATOR.log(EvalEntry(
            question=task,
            intent="unified",
            confidence=vr.confidence,
            topics_used=[],
            contradictions=len(vr.contradictions),
            unverified=len(vr.unverified_claims),
            verified=len(vr.verified_claims),
        ))
    except Exception:
        pass

    return AgentAnswer(
        answer=answer or "",
        verification=vr,
        learned_topics=[],
        used_topics=[],
        project=project,
        is_chat=False,
        mode="unified",
        token_usage=agent._get_token_usage(),
        thinking_trace=agent._trace,
        llm_calls=agent._llm_calls,
    )
