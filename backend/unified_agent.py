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

import json
import logging
import os
import re
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
from .system_prompt_sections import assemble as _assemble_prompt


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


# Audit T5 (May 2026) split the original ~14KB rules monolith into a
# core always-on block + four scenario blocks loaded on signal.
# Phase 1 (May 2026) further extracted the core into named sections
# living in `backend/system_prompt_sections.py` — SECTIONS dict +
# DEFAULT_ORDER + `assemble()`. `_UNIFIED_RULES_CORE` is now the
# result of `_assemble_prompt()` at import time; the same name is
# preserved so existing tests that grep for sentences still work.
# A profile override may replace any section at runtime — see
# `pipeline_profile.active_overrides()` (Task 2+).

def _classify_model_size(model_name: str | None) -> str:
    """Map an active model ID to one of `small` / `medium` / `large`
    for the prompt-module `requires_model_size` predicate.

    Frontier-class names (Claude 4.x, GPT-4.x/5, Gemini 2.x, Mistral
    Large, Command R+) → `large`. Mid-tier (Haiku, GPT-4-mini, Llama
    70B, Mistral medium) → `medium`. Local small (Llama 7-13B,
    Gemma, Phi, Qwen <14B) → `small`. Unknown → `large` (safe
    default — small-model module is opt-in, not opt-out)."""
    if not model_name:
        return "large"
    n = model_name.lower()
    # Explicit small-model identifiers used in local setups.
    small_markers = (
        "7b", "8b", "13b", "phi-", "gemma-2b", "gemma-7b",
        "llama-3.2-1b", "llama-3.2-3b", "qwen2.5-7b", "qwen2.5-3b",
        "tinyllama", "stablelm",
    )
    if any(m in n for m in small_markers):
        return "small"
    medium_markers = ("haiku", "mini", "70b", "mistral-medium")
    if any(m in n for m in medium_markers):
        return "medium"
    return "large"


def _unified_rules_core(ctx=None) -> str:
    """Live system-prompt body for a given TurnContext.

    Reads the active pipeline profile's `prompt_overrides` (5s
    in-process cache, invalidated on switch). The overrides dict
    may carry a `modules` key (v2 path, used by `build_prompt`) and
    historically a `sections` key (v1 path, used by `assemble`). v2
    is authoritative now; `sections` is silently ignored.

    `ctx=None` returns the default-context prompt, which is what
    the import-time `_UNIFIED_RULES_CORE` constant snapshots."""
    from .prompt_modules import build_prompt, TurnContext
    if ctx is None:
        ctx = TurnContext()
    try:
        from .pipeline_profile import active_overrides
        overrides = active_overrides().get("prompt_overrides")
    except Exception:
        overrides = None
    return build_prompt(ctx, overrides)


# Back-compat: tests that grep `_UNIFIED_RULES_CORE` still work
# because the module-level attribute reflects current defaults at
# import time. Live callers go through the function.
_UNIFIED_RULES_CORE = _unified_rules_core()


# ─── Scenario blocks — loaded on demand by `_build_rules_for_turn` ─

# Loaded when the user's message looks like a runtime-bug report
# (keywords: error, exception, "не работает", "buttons don't work",
# etc.). Avoids paying for ~1KB on every chat / settings turn.
_RULES_JOURNAL_FIRST = """## Diagnose runtime bugs from the journal FIRST

When the bug shows up as a runtime artefact — HTTP error code in
a log, Python exception in a traceback, "buttons don't work", "no
voice playback", "service crashed" — your FIRST tool call should
be `terminal_exec("journalctl --user -u hrant -n 200 --no-pager
| grep -iE 'error|exception|traceback|<symptom>'")` (or `journalctl
--user -u hrant --since '15 min ago' --no-pager` for time-bounded).
Read the actual HTTP responses / exception messages BEFORE
diving into the code. Reading code first when the journal already
shows you the layer that's failing is hours wasted.

Concrete example from the May 19 incident:
  - symptom: "Telegram buttons don't work"
  - good first call: `journalctl --user -u hrant -n 200 | grep -i
    callback` → would have shown `answerCallbackQuery → 400` and
    pointed straight at PTB's update-dispatch layer.
  - bad first call: `locate_symbol("handle_callback_query")` +
    read_file backend/channels.py — chases the dispatch logic when
    the bug is in the queue-up-front Application config.

If the journal call is empty / has no error pattern, THEN dive
into code. The order matters: observable failure mode → journal →
code, not the other way round."""


# Loaded when an attachment is in play OR when this turn is likely
# to produce a file the user should receive. Both signals point at
# the MEDIA: convention being relevant.
_RULES_MEDIA_CONVENTION = """## Sending files back (MEDIA: convention)

You DO have a way to attach files to your reply — it's NOT a tool,
it's a CONVENTION the Telegram bridge parses. To deliver a file
(processed video, generated image, PDF, audio, anything) to the
user, include a line on its own in your answer in the exact form:

  MEDIA:/absolute/path/to/file.ext

The bridge (`backend/channels.py::_strip_and_send_media`) detects
each such line, picks the right Telegram send method by extension
(reply_video for .mp4/.mov/.webm, reply_photo for .jpg/.png,
reply_audio for .mp3/.ogg, reply_document for everything else),
sends the file as a real attachment, AND strips the line from the
textual reply so the user just sees clean text + the file bubble
underneath.

Constraints:
  - Path MUST be absolute and live under `~/.hrant/data/` or
    `/tmp/` (safety allowlist; other paths are silently refused
    and the line stays inline so you see your mistake).
  - One MEDIA: line per file. Multiple lines mean multiple
    attachments.
  - Don't say "I can't send files" — you can. You just write a
    MEDIA: line.

Example final answer after processing a video:

  Готово — логотип убран. Длительность сохранена, аудио без re-encode.

  MEDIA:/home/hrant/.hrant/data/workspace/outbox/clip_no_logo.mp4"""


# Loaded when attachments are in play. Tells the agent which inspection
# tool matches which file shape.
_RULES_FILE_TYPES = """## File types — which path handles which

When a file attachment is in play (sha256 reference from the user
or a file you saved to `outbox/`), pick the right inspection path.
DON'T invent a tool — these are the actual capabilities:

  - **Images** (jpg/png/webp/gif): use `analyze_image(sha, question)`
    — multimodal LLM answers about the visible content. For
    coordinate-style answers ask for `x=… y=… w=… h=…` explicitly.
  - **Voice / audio**: transcript is already attached on the
    Attachment record (`meta.transcript`) — TRANSCRIBER auto-runs
    on every incoming voice AND audio file. Read it directly from
    the attachment metadata; do NOT re-transcribe.
  - **Video**: call `backend.tools.video_processor.preprocess_video(sha)`
    via run_python — returns frame_shas (image sha256s) + audio
    transcript. Each frame sha is then usable with `analyze_image`.
  - **PDF / DOCX**: use `read_file(path)` — pypdf / python-docx are
    available.
  - **Text-like files** (.txt .md .json .yaml .toml .csv .html .xml
    .css .sh .c .cpp .rs .go .java + most code suffixes): use
    `read_file(path)`. Supports `start_line` / `end_line` for slices.
  - **Spreadsheets** (.xlsx .ods) and **archives** (.zip .tar.gz):
    no dedicated tool — use `run_python` with pandas / openpyxl /
    zipfile / tarfile.
  - **Unknown binary**: use `run_python` to probe (head bytes, file
    magic) before deciding how to process.

To DELIVER any file back to the user, write a MEDIA: line — see
the "Sending files back" block above."""


# Loaded only when the sticky-request detector fires (STICKY REQUEST
# DETECTED block in the prompt). Otherwise this rule is dead weight.
_RULES_REPEATED_REQUEST = """## Repeated request → escalate

When the conversation history shows the user has raised the
same topic 2+ times and your prior replies were short
acknowledgements (the STICKY REQUEST DETECTED block flags this
when it fires), you have ALREADY failed once or twice. Don't
fail again: pick the most aggressive tool that could apply
the change (set_setting / terminal_exec / propose_self_modification)
and use it this turn."""


# Injected when the runtime detector at `_recent_refusal_pattern`
# sees the user's previous assistant message was a meta-cognitive
# refusal ("честно: не могу подтвердить", "I cannot confirm that…").
# This is the structural escalation of the `re_prompt_resilience`
# section that's already in the core prompt — same intent, harder
# tone, fires only when the pattern actually shows. The trigger is
# narrow on purpose: only when the user is now re-prompting
# (current turn message is short ≤ 80 chars OR repeats a recent
# theme) AND the prior agent answer matches refusal phrasing.
_RULES_REPEAT_REFUSAL = """## REPEAT-REFUSAL ALERT — execute now, do not investigate

Your previous turn ended with a meta-cognitive refusal ("честно:
я не могу подтвердить, что X сделано" / "I cannot confirm that
X was completed"). The user re-prompted. This is the exact loop
the `re_prompt_resilience` rule exists to break.

DO NOT do another round of `read_file` / `locate_symbol` /
`terminal_exec` environment inspection — the environment has not
changed and re-inspecting it will produce the same refusal again.

This turn: the FIRST tool call must be the actual work the user
asked for (run the command, write the file, start the bench).
Inspection is allowed only AFTER the execution attempt and only
to verify the result.

If a hard blocker truly exists, your ONLY legitimate exit is
`ask_user(question=..., options=[...])` with 2-4 concrete options
identifying the blocker AND how the user can unblock it. A
free-text "I cannot confirm" answer this turn is forbidden — the
post-turn rewriter will overwrite it."""


# Full surface — kept as the concat of core + every scenario block.
# Tests that grep `_UNIFIED_RULES` for any sentence still pass. At
# runtime, `run_unified` uses `_build_rules_for_turn(...)` to assemble
# the trimmed per-turn version.
_UNIFIED_RULES = "\n\n".join([
    _UNIFIED_RULES_CORE,
    _RULES_JOURNAL_FIRST,
    _RULES_MEDIA_CONVENTION,
    _RULES_FILE_TYPES,
    _RULES_REPEATED_REQUEST,
    _RULES_REPEAT_REFUSAL,
])


# Audit follow-up — "this is an LLM agent, we don't need keywords in
# code". The previous version of this section had two keyword-based
# classifiers:
#
#   - `_BUG_REPORT_KEYWORDS_RE` / `_looks_like_bug_report(task)`:
#     ~25-keyword regex (Russian + English) to decide whether to
#     inject the journal-first scenario block.
#   - `_ACTION_VERBS` / `_is_trivial_chat(task, ...)`: ~60-verb list
#     to decide whether the turn is "casual chat" and gets a
#     minimal prompt vs the full preamble.
#
# Both REMOVED. Reasoning: when an LLM is in the loop, the CORRECT
# place for fuzzy classification (chat vs task, bug vs not) is in
# the LLM's reasoning, guided by prompt instructions — not a hand-
# curated keyword list that misses every dialect, slang, and new
# verb that isn't pre-listed. Brittle by construction; the LLM is
# already paid for, let it think.
#
# What replaced them:
#   - Bug-report → journal-first rule is now ALWAYS included in
#     `_RULES_JOURNAL_FIRST` (loaded by default; 1.2 KB cost on every
#     turn is cheaper than a misclassification). LLM reads the rule
#     and applies it when its judgement says "this is a runtime bug".
#   - Chat-vs-task → the "Chat vs task" rule already in CORE tells
#     the LLM "not every turn needs a tool — casual chat / recall /
#     small acks answer from context, but directives / state changes
#     / multi-step problems USE TOOLS". LLM decides; full preamble
#     always.
#
# Kept structural signals (NOT keyword-based) in `_build_rules_for_turn`:
#   - `has_attachments` — direct attribute of the request (file/sha
#     refs present). No NLP needed.
#   - `sticky_fired` — comes from the dedicated sticky-request
#     detector that watches the conversation buffer.


# Audit follow-up — `_try_chat_path`: LLM-based fast lane for
# casual chats. Replaces the T8 keyword classifier that was
# removed earlier (Russian + English action-verb lists were
# brittle). Instead of deciding "chat vs task" via Python regex,
# we let the LLM decide ITSELF — it gets a tiny prompt with no
# tools and either answers directly OR emits `ESCALATE: <reason>`
# and we route through the full agent.
#
# Cost:
#   - chat-shaped turn that LLM handles directly: ~1-2 k tokens
#     (vs ~15-17 k on the full unified path).
#   - false-positive escalate: ~1.5 k extra tokens for the chat
#     attempt + full path runs afterwards.
# Net: huge win on real chats, small loss on misclassified tasks.
# The decider is the LLM, not a hand-curated list.

_CHAT_FAST_PATH_RULES = """# AGENT — CHAT FAST PATH

You are running in a single-call chat lane. You have NO tools this
turn. The user's message is short and looks casual.

Goal: if you can answer directly (greeting, acknowledgement,
recall like "what voice / model / settings am I on", brief
opinion), do so. Use the STATE SNAPSHOT above for any recall —
don't guess settings.

If you actually NEED tools (file operations, settings changes,
external lookups, running code, multi-step problems, file
delivery) — do NOT try to answer. Respond with EXACTLY one line:

  ESCALATE: <one-sentence reason>

The system will then route the same message through the full
agent with all tools available. Don't apologise, don't explain
how the routing works, don't promise "I'll do it next turn" —
just `ESCALATE: ...` and stop.

Match the user's language. Be brief.
"""


def _try_chat_path(
    *,
    task: str,
    agent,
    speaker_id: str,
    snapshot: str,
    convo: str,
) -> str | None:
    """Try a single-call chat lane. Returns the answer if the LLM
    chose to answer directly, or `None` if it emitted ESCALATE: /
    looks like it needs tools (caller falls back to the full
    unified path).

    The decision is the LLM's — no keyword list, no regex over the
    user message. We just provide a minimal prompt + zero tools
    and trust the model to pick the cheaper option.

    Fail-safe: any exception (router down, parse failure, weird
    output) returns None so the full path runs unchanged. Fast
    path failure must not break a turn.
    """
    try:
        from .llm import router as _router, TaskType as _TT
        from .identity import IDENTITY
        sys_parts = [IDENTITY.preamble(speaker_id=speaker_id)]
        if snapshot:
            sys_parts.append(f"---\n\n{snapshot}")
        if convo:
            sys_parts.append(f"---\n\n{convo}")
        sys_parts.append(f"---\n\n{_CHAT_FAST_PATH_RULES}")
        sys_prompt = "\n\n".join(sys_parts)

        # Audit P0 fix: the previous version called
        # `router().call_with_tools(..., tools=[])`. Router's
        # `_supports_tools()` returns False for `tools=[]` (line 2710
        # of llm.py: `if not tools: return False`), so `call_with_tools`
        # rejected the call and the try/except swallowed the exception
        # — fast-chat NEVER fired in production. A "2+2" turn that
        # should have cost ~1-2 k tokens was hitting the full unified
        # path (12 k+ tokens) silently.
        #
        # The correct primitive for a no-tool single-shot LLM call
        # is `router.call(task_type, system, user, ...)` — same
        # provider chain + failover, no tool-loop infrastructure.
        answer = _router().call(
            _TT.QUICK_ANSWER,
            sys_prompt,
            task,
            max_tokens=600,
            temperature=0.4,
        )
    except Exception as e:
        log.debug("chat fast path failed (exception): %s", e)
        return None

    if not answer or not isinstance(answer, str):
        return None
    head = answer.lstrip()
    if head.upper().startswith("ESCALATE:"):
        try:
            agent.progress("chat_fast_path", f"escalating: {head[9:120].strip()}")
        except Exception:
            pass
        return None
    if "<tool_call" in head[:300]:
        # LLM emitted a tool-call XML dump — definitely wanted tools.
        try:
            agent.progress("chat_fast_path", "escalating: tool-call XML in output")
        except Exception:
            pass
        return None
    try:
        agent.progress("chat_fast_path", "answered directly")
    except Exception:
        pass
    return answer


_REFUSAL_PHRASES = (
    "не могу подтвердить",
    "честно: я не могу",
    "честно: по предоставленному",
    "i cannot confirm",
    "i can't confirm",
    "i cannot verify",
    "i can't verify",
    "не могу подтвер",
    "по предоставленному investigation",
)


def _recent_refusal_pattern(
    *, session_key: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> bool:
    """Detect the meta-cognitive refusal loop caught in the audit.

    Returns True iff the most recent prior assistant message in this
    session starts with (or contains in its first 200 chars) one of
    the known refusal phrases. The current turn's user message is
    NOT checked — the rule fires regardless of how the user
    re-prompts, because re-prompting a refusal is the failure mode.
    """
    try:
        from .conversation import CONVERSATION as _CONV
        recent = _CONV.recent(
            n=1, session_key=session_key, speaker_id=speaker_id,
        )
    except Exception:
        return False
    if not recent:
        return False
    last = recent[-1]
    ans = ((last.get("answer") or "")[:300]).lower()
    if not ans.strip():
        return False
    return any(phrase in ans for phrase in _REFUSAL_PHRASES)


def _build_rules_for_turn(
    *,
    ctx=None,
    has_attachments: bool = False,
    sticky_fired: bool = False,
    repeat_refusal: bool = False,
) -> str:
    """T5: compose the rules string for this specific turn. Core block
    is always on (~13 KB); structural-signal scenario blocks load
    when their attribute is set.

    No keyword-based classification. The LLM gets the same preamble
    every turn (modulo structural signals) and decides for itself
    when each rule applies.

    Signals:
      - `has_attachments`: file/sha refs in the turn → load MEDIA +
        file-types blocks (the LLM needs the inspection cheatsheet
        and the delivery convention).
      - `sticky_fired`: STICKY REQUEST DETECTED → load
        repeated-request escalation block.
      - `repeat_refusal`: previous assistant message matched the
        meta-cognitive refusal phrasing — load the REPEAT-REFUSAL
        ALERT block to escalate the resilience rule from "always-on
        prompt" into a forceful per-turn directive.

    Journal-first (for bug reports) is always present — the rule is
    in the prompt every turn, applied by the LLM when its judgement
    triggers it. We trust the LLM more than a Russian-plus-English
    bug-keyword regex.
    """
    parts = [_unified_rules_core(ctx), _RULES_JOURNAL_FIRST]
    if has_attachments:
        parts.append(_RULES_FILE_TYPES)
        # MEDIA convention is most useful alongside an inbound
        # attachment (likely to produce a delivered file) but we add
        # it on attachment presence as a proxy.
        parts.append(_RULES_MEDIA_CONVENTION)
    if sticky_fired:
        parts.append(_RULES_REPEATED_REQUEST)
    if repeat_refusal:
        parts.append(_RULES_REPEAT_REFUSAL)
    return "\n\n".join(parts)


# ─── Auto-recall ───────────────────────────────────────────────────


_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


def _rewrite_xml_tool_call_dump(answer: str, agent) -> str:
    """If the final LLM answer is an XML-style tool-call dump (the
    failure mode that fires when max_iterations is hit and the model
    can't execute another real tool), rewrite it as a plain-language
    status report based on the agent's actual trace.

    A "dump" is detected when the first significant chunk of the
    answer (first 200 non-whitespace chars) opens with `<tool_call
    name=...>`. Plain answers that merely mention the words "tool
    call" in prose are untouched.
    """
    if not answer or not isinstance(answer, str):
        return answer or ""
    stripped = answer.lstrip()
    if not stripped.startswith("<tool_call"):
        return answer

    m = _XML_TOOL_CALL_RE.search(stripped[:300])
    intended_tool = m.group(1) if m else "(unknown tool)"

    # Summarise what DID run, from the trace.
    tool_names_run: list[str] = []
    for step in (getattr(agent, "_trace", None) or []):
        ev = getattr(step, "event", "") or ""
        if ev not in ("tool", "tool_error"):
            continue
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if name:
            tool_names_run.append(name)
    n_ran = len(tool_names_run)
    last_few = ", ".join(tool_names_run[-4:]) if tool_names_run else "none"

    return (
        f"⚠️ I hit the iteration ceiling without finishing.\n\n"
        f"Tools I actually ran ({n_ran} of {n_ran} budgeted): {last_few}.\n"
        f"The next call I wanted to make was `{intended_tool}` but the "
        f"loop budget ran out before I could execute it.\n\n"
        f"Common ways to unblock me:\n"
        f"  • tell me the missing detail (e.g. logo bounding box "
        f"`x=… y=… w=… h=…` if this was a video task);\n"
        f"  • narrow the request (one objective per turn);\n"
        f"  • or just say \"continue\" and I'll resume with a fresh "
        f"iteration budget.\n\n"
        f"_(Never emit `&lt;tool_call …&gt;` XML as the final answer "
        f"again — that's a runtime artefact, not a tool I support.)_"
    )


# Keyword-based refusal detection (`_REFUSAL_OPENER_RE`,
# `_POLICY_REFUSAL_KEYWORDS_RE`, `_is_policy_refusal`) was removed
# on 2026-05-21 along with the rewriter it powered
# (`_rewrite_refusal_without_attempts`). The user explicitly asked
# to drop all keyword logic from the pipeline; refusing or rewriting
# the LLM's answer based on substring matches was the worst
# offender. The TSP rule "do not refuse before 2 distinct tools"
# still lives in the system prompt — the LLM enforces it itself.
# `_is_russian_dominant` below stays (script counting, not
# keyword-based) — used by other helpers for language picking.


def _is_russian_dominant(text: str) -> bool:
    """Naive script detection — count cyrillic vs latin letters. Used
    by the refusal rewriter to pick a response language that matches
    what the agent (and presumably the user) was already speaking.

    M1: ties resolve to Russian, not English. Hrant's owner speaks
    Russian; an English rewrite on a mixed/balanced answer feels
    worse than the inverse. Equal-count text is far more likely to
    be a Russian answer that quoted a few English tool names than
    the other way round.
    """
    if not text:
        return False
    cyr = 0
    lat = 0
    for c in text:
        lc = c.lower()
        if "а" <= lc <= "я" or lc == "ё":
            cyr += 1
        elif "a" <= lc <= "z":
            lat += 1
    if cyr == 0 and lat == 0:
        # Pure symbols / digits / emoji — default to Russian for this
        # bot (the owner speaks Russian; English rewrite on a symbol
        # answer would be a sudden language switch).
        return True
    return cyr >= lat


def _count_distinct_tools_called(agent) -> tuple[int, set[str]]:
    """Walk the agent trace; return (count, names) of distinct tools
    that were called this turn (whether they succeeded or errored)."""
    names: set[str] = set()
    for step in (getattr(agent, "_trace", None) or []):
        ev = getattr(step, "event", "") or ""
        if ev not in ("tool", "tool_error"):
            continue
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        n = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if n:
            names.add(n)
    return len(names), names


# `REFUSAL_ATTEMPT_BAR` was used by the deleted refusal rewriter
# (the keyword-based one). The "2 distinct tools before refusing"
# rule still lives in the system prompt for the LLM to honour.


# Post-turn skill_creator reflection — settings.
#
# Background: H3 added the `skill_creator` builtin skill that the
# agent is supposed to load on its own after a non-trivial workflow.
# A May 2026 audit of 110 prod turns showed the agent NEVER fires it:
# 41 non-trivial turns (≥3 distinct tools), 0 `load_skill('skill_creator')`
# calls, 0 `propose_skill` calls. A rule paragraph in _UNIFIED_RULES
# isn't enough — the LLM just ignores it.
#
# Fix: a structural out-of-band LLM call after the user-visible answer.
# It runs the 3-gate checklist on the just-finished turn and either
# calls propose_skill or returns "no skill needed". Cost ~$0.01 per
# qualifying turn; latency hidden from user (runs after answer ships).

SKILL_REFLECTION_TOOL_BAR = 3
SKILL_REFLECTION_MAX_ITERATIONS = 4
_REFLECTION_TOOL_ALLOWLIST = frozenset({
    "list_skills", "load_skill", "propose_skill",
})


# T3 (no-progress detector): consecutive identical tool-result hashes
# this many in a row → append the 🔄 NO PROGRESS marker. 3 is a
# sweet spot: 2 catches an intentional re-read (false fire), 4+
# misses the 20-iteration probe loops the May 2026 audit caught.
_NOPROGRESS_WINDOW = 3


# I2: cap on the number of auto-fired install proposals per turn.
# Without this, a skill with N missing required_tools would generate
# N separate Telegram DMs to the owner in a single burst. Five is the
# pragmatic ceiling: enough for any one skill's reasonable dep set,
# small enough not to be a DM-flood. Remaining missing tools surface
# as a "deferred" line so the LLM and the user still see them.
AUTO_PROPOSE_CAP = 5


# Audit T1+T4: token budget thresholds (SIGNAL, not enforcement).
#
# The May 2026 cost audit found the agent burns ~$40/month largely on
# re-feeding context to the model (input:output ratio 40:1, median
# 110k tokens/turn). Hard refuse-caps would regress "agent is smart" —
# the user explicitly wants signal-not-enforcement so the LLM sees
# the cost and chooses to wrap up, the same way TSP gates work.
#
# Mechanism: after each tool call, the cumulative token usage for the
# turn gets formatted into a tiny marker and APPENDED to the tool
# result that goes back to the LLM. The LLM sees it as part of the
# tool feedback ("...your output. 🟡 BUDGET: 12000/10000 used —
# wrap up.") and reacts. No raise, no enforcement — just visibility.
#
# Numbers are aggressive on purpose (audit recommended "less is
# better"). Env overrides — HRANT_TOKEN_SOFT_PER_TURN and
# HRANT_TOKEN_HARD_PER_TURN — let an operator widen for benchmark
# runs without a redeploy. Setting either to 0 disables the marker.
import os as _os_for_token_budget

TOKEN_SOFT_PER_TURN = int(
    _os_for_token_budget.environ.get("HRANT_TOKEN_SOFT_PER_TURN", "0")
)
TOKEN_HARD_PER_TURN = int(
    _os_for_token_budget.environ.get("HRANT_TOKEN_HARD_PER_TURN", "0")
)
# Defaults flipped 2026-05-21 — user explicit decision: "no limits,
# agent need to have a free work opportunity". Budget markers used
# to default to 10k/30k and inject 🟡/🔴 warnings into tool results,
# nudging the model to wrap up. The mechanism stays in case an
# operator wants to opt back in via env vars, but it's OFF by
# default. The previous 30k threshold was advisory anyway — audit
# observed a turn going to 343k input despite the marker firing.


_TRUNCATION_HINTS = {
    "read_file": (
        "Call again with `start_line=N` / `end_line=M` to read the "
        "specific range you actually need, or use `grep` to find "
        "the line(s) before reading."
    ),
    "view_file": (
        "Call again with `start_line=N` / `end_line=M` to read just "
        "the relevant range."
    ),
    "terminal_exec": (
        "Pipe through `head -100` / `tail -100` / `grep -n PATTERN` "
        "next time, or write to a file and read with start_line/"
        "end_line."
    ),
    "web_search": (
        "Narrow the query (add the year, a specific error code, the "
        "library name) so fewer results match."
    ),
    "fetch_url": (
        "Use `read_file` on a saved excerpt, or fetch a deeper URL "
        "that's already filtered (e.g. the API endpoint, not the "
        "human-readable index page)."
    ),
}


# T7: structured digest headers per tool result.
#
# The original audit recommended a "context ledger" replacing prior
# tool results with summary facts. The anthropic SDK's tool-use loop
# doesn't allow mid-loop history rewriting, so we do the next-best
# thing: every tool result gets a 1-3 line digest header PREPENDED
# to the body before truncation. Across N iterations the LLM can
# scroll the conversation and see at-a-glance what it has already
# learned ("[read_file backend/channels.py] 47L 4234c — handle_callback_query,
# CallbackQueryHandler") even after the body is truncated by T2 caps.
#
# Each digester is fast (no LLM call) — pure pattern extraction.

_SYMBOL_PATTERN_RE = re.compile(
    r"^\s*(def\s+\w+|class\s+\w+|async\s+def\s+\w+|#{1,6}\s+\S[^\n]*)",
    re.MULTILINE,
)


def _digest_read_file(args: dict, result: str) -> str:
    """Header for read_file / view_file results. Calls out path,
    size, and top 5 symbol-like lines (def/class/headings)."""
    path = (args or {}).get("path") or (args or {}).get("file") or "?"
    n_lines = result.count("\n") + (1 if result and not result.endswith("\n") else 0)
    chars = len(result)
    symbols = [m.group(1).strip() for m in _SYMBOL_PATTERN_RE.finditer(result)][:5]
    sym_str = "; ".join(s[:60] for s in symbols) if symbols else "no symbols"
    return f"[read_file {path}] {n_lines}L {chars}c — {sym_str}"


def _digest_terminal_exec(args: dict, result: str) -> str:
    """Header for terminal_exec. Shows command head + outcome shape."""
    cmd = ((args or {}).get("command") or (args or {}).get("cmd") or "").strip()
    cmd_head = cmd[:80]
    n_lines = result.count("\n")
    low = result.lower()
    if "traceback" in low or "error:" in low[:200] or "fatal:" in low[:200]:
        outcome = "ERROR"
    elif "permission denied" in low or "not found" in low[:200]:
        outcome = "REFUSED"
    elif n_lines == 0 and len(result) < 40:
        outcome = "empty"
    else:
        outcome = "ok"
    return f"[terminal_exec `{cmd_head}`] {outcome}, {n_lines}L {len(result)}c"


def _digest_web_search(args: dict, result: str) -> str:
    """Header for web_search. Counts hits + extracts first 2 result
    titles if visible."""
    query = (args or {}).get("query") or "?"
    n_lines = result.count("\n")
    # Heuristic: count "- " or " 1. " style hit markers.
    hits = result.count("\n- ") + result.count("\n1. ") + result.count("\n2. ")
    return f"[web_search '{query[:60]}'] {max(1, hits)} hits, {n_lines}L"


def _digest_fetch_url(args: dict, result: str) -> str:
    """Header for fetch_url. Shows URL + page size + likely title."""
    url = (args or {}).get("url") or "?"
    chars = len(result)
    # Try to grab a <title> if present.
    m_title = re.search(r"<title[^>]*>([^<]{1,120})</title>", result, re.IGNORECASE)
    title = (m_title.group(1).strip() if m_title else "")[:80]
    title_part = f" — '{title}'" if title else ""
    return f"[fetch_url {url[:80]}] {chars}c{title_part}"


def _digest_locate_symbol(args: dict, result: str) -> str:
    """Header for locate_symbol. Reports symbol + path + hit count."""
    sym = (args or {}).get("symbol") or "?"
    path = (args or {}).get("path") or "?"
    # locate_symbol returns JSON list — count entries.
    hits = result.count('"start_line"')
    return f"[locate_symbol {sym}@{path}] {hits} hit(s)"


def _digest_search_knowledge(args: dict, result: str) -> str:
    """Header for search_knowledge. Counts hits."""
    query = (args or {}).get("query") or "?"
    hits = result.count('"topic"') or result.count("\n- ")
    return f"[search_knowledge '{query[:60]}'] {max(0, hits)} hit(s)"


_DIGESTERS = {
    "read_file": _digest_read_file,
    "view_file": _digest_read_file,
    "terminal_exec": _digest_terminal_exec,
    "web_search": _digest_web_search,
    "fetch_url": _digest_fetch_url,
    "locate_symbol": _digest_locate_symbol,
    "search_knowledge": _digest_search_knowledge,
}


def _digest_tool_result(name: str, args: dict, result: str) -> str:
    """Return a one-line digest header for known tools, or empty
    string for tools that don't need one (set_setting, propose_*,
    etc. — those have short results already)."""
    if not result:
        return ""
    digester = _DIGESTERS.get(name)
    if digester is None:
        return ""
    try:
        return digester(args or {}, result or "")
    except Exception:
        return ""


def _truncation_hint(name: str) -> str:
    """T2: tell the LLM HOW to ask for less. Without a hint the LLM
    re-calls the same tool hoping for less data; with a hint it uses
    the parameterised version (range, grep, narrower query)."""
    return _TRUNCATION_HINTS.get(
        name,
        "Call again with a narrower scope or use a more specific tool.",
    )


def _format_token_budget_marker(used: int) -> str:
    """Return the marker text to append to a tool result, based on
    cumulative tokens used in the current turn.

    Returns:
        "" when below soft, no need to disturb the LLM with bookkeeping.
        "🟡 BUDGET ..." when soft <= used < hard. Nudges the LLM to
            consider wrapping up the current investigation.
        "🔴 BUDGET ..." when used >= hard. Tells the LLM to stop
            probing and issue a partial report.

    Both thresholds are env-overridable. Setting either to 0 disables
    the marker (useful for benchmark / explicit long-run modes).
    """
    if not used:
        return ""
    if TOKEN_HARD_PER_TURN > 0 and used >= TOKEN_HARD_PER_TURN:
        return (
            f"\n\n🔴 **TOKEN BUDGET EXCEEDED** "
            f"({used:,} tokens consumed this turn; hard limit "
            f"{TOKEN_HARD_PER_TURN:,}). Stop probing, write a partial "
            f"report with what you have, and end the turn. Do not call "
            f"more tools unless absolutely required to deliver. The "
            f"user pays per token — this turn is now expensive."
        )
    if TOKEN_SOFT_PER_TURN > 0 and used >= TOKEN_SOFT_PER_TURN:
        return (
            f"\n\n🟡 **TOKEN BUDGET WARNING** "
            f"({used:,} tokens used; soft limit "
            f"{TOKEN_SOFT_PER_TURN:,}). Consider wrapping up: one or "
            f"two more focused tool calls and a delivery, not "
            f"open-ended exploration."
        )
    return ""


# `_rewrite_refusal_without_attempts` was deleted 2026-05-21 along
# with its keyword regex tripwires. The TSP attempt-bar rule still
# lives in the system prompt — the LLM enforces "do not refuse
# before 2 distinct tools" itself. The post-processing rewriter
# was the most brittle keyword-driven piece of the pipeline and
# the user explicitly asked it gone.


# ─── Post-turn skill_creator reflection (Option C from H3 audit) ───


def _should_reflect_for_skill(agent, answer: str) -> tuple[bool, str]:
    """Decide whether to run the post-turn skill-creator reflection.

    Returns (should_run, reason) — the reason string is for telemetry
    so the owner can audit why most turns skip reflection (most do).

    Gates:
      1. Answer is a non-empty string.
      2. Answer doesn't open with a refusal pattern (failed turns
         don't yield reusable workflows).
      3. Answer isn't a rewriter output (iteration-ceiling rewrite,
         refusal-rewrite) — those are recovery messages, not solutions.
      4. The trace shows at least `SKILL_REFLECTION_TOOL_BAR` distinct
         tool calls — a real composed workflow, not a one-tool answer.
      5. The turn didn't already call propose_skill or
         load_skill('skill_creator') — no double-firing.
      6. The skill_creator skill itself is loaded (couldn't run the
         reflection without its body anyway).
    """
    if not answer or not isinstance(answer, str):
        return False, "empty-or-nonstring-answer"
    head = answer[:300]
    # The keyword-based "refusal opener" gate used to live here.
    # Removed 2026-05-21 — skill reflection skips on low tool-count
    # turns anyway (see SKILL_REFLECTION_TOOL_BAR below), so a
    # failed turn naturally won't reflect. The iteration-ceiling
    # rewriter marker (different mechanism) still short-circuits.
    rewriter_marker_starts = (
        "⚠️ I hit the iteration ceiling",
    )
    for marker in rewriter_marker_starts:
        if head.startswith(marker):
            return False, "rewriter-output"
    n_distinct, names = _count_distinct_tools_called(agent)
    if n_distinct < SKILL_REFLECTION_TOOL_BAR:
        return False, f"only-{n_distinct}-distinct-tools"
    for step in (getattr(agent, "_trace", None) or []):
        ev = getattr(step, "event", "") or ""
        if ev not in ("tool", "tool_error"):
            continue
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        nm = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if nm == "propose_skill":
            return False, "already-proposed"
        if nm == "load_skill":
            args = (
                getattr(tc, "args", None)
                or (tc.get("args") if isinstance(tc, dict) else None)
                or {}
            )
            if isinstance(args, dict) and args.get("name") == "skill_creator":
                return False, "skill_creator-already-loaded"
    from .skills import SKILLS as _SKILLS
    if _SKILLS.get("skill_creator") is None:
        return False, "skill_creator-skill-not-loaded"
    return True, "ok"


def _post_turn_skill_reflection(
    agent,
    task: str,
    answer: str,
    speaker_id: str,
) -> None:
    """Out-of-band reflection: after a non-trivial turn finishes,
    spawn a small LLM call to decide whether to propose_skill.

    Fires only when `_should_reflect_for_skill` says yes. Reuses
    the global tool registry's execute path so a `propose_skill`
    call here actually creates the user-tier draft + fires the
    on_skill_proposed callback (owner gets Telegram DM as usual).

    Tool surface is filtered to: list_skills, load_skill, propose_skill.
    The reflection cannot accidentally trigger another nested turn of
    work — no read_file, terminal_exec, run_python, etc.

    Failures are swallowed — reflection is best-effort and must NEVER
    affect the user-visible turn outcome.

    Merge-existing-skill behaviour: when the catalog (passed in the
    system prompt) shows a near-match, the reflection LLM can call
    `load_skill(name)` to read its body, then call
    `propose_skill(name=<same>, ...)` to overwrite the user-tier
    skill with a merged version (preserving useful old parts, adding
    new patterns from this turn). The `upsert_user_skill` path writes
    by name, so propose_skill with an existing name replaces it
    (and the new version is still DISABLED until owner approves).
    """
    should, reason = _should_reflect_for_skill(agent, answer)
    if not should:
        try:
            agent.progress("skill_reflection", f"skipped: {reason}")
        except Exception:
            pass
        return

    n_distinct, names = _count_distinct_tools_called(agent)
    try:
        agent.progress(
            "skill_reflection",
            f"starting (distinct_tools={n_distinct})",
        )
    except Exception:
        pass

    try:
        from .skills import SKILLS as _SKILLS
        from .llm import router, TaskType
        from .tool_registry import get_registry

        sk = _SKILLS.get("skill_creator")
        if sk is None:  # defensive — gate should have caught it
            return

        registry = get_registry()
        full_schema = registry.to_anthropic_list()
        filtered = [
            t for t in full_schema
            if t.get("name") in _REFLECTION_TOOL_ALLOWLIST
        ]
        if not filtered:
            log.warning("skill_reflection: no allowlisted tools in registry")
            return

        # Catalog block — names + descriptions + tags + source. Not
        # full bodies (those are heavy; reflection can `load_skill`
        # the one it actually wants to merge).
        cat_lines = ["# CURRENT SKILL CATALOG (for dedup / merge check)"]
        for s in _SKILLS.list():
            if not s.enabled:
                continue
            trig = ", ".join(s.triggers) if s.triggers else "—"
            tags = ", ".join(s.tags) if s.tags else "—"
            cat_lines.append(
                f"- **{s.name}** ({s.source}): {s.description} "
                f"_(triggers: {trig}; tags: {tags})_"
            )
        catalog_block = "\n".join(cat_lines)

        reflection_system = (
            "You are running the post-turn skill_creator reflection. "
            "A non-trivial workflow just completed. Decide whether it "
            "should become a reusable skill — or whether an EXISTING "
            "skill should be IMPROVED with what this turn taught.\n\n"
            "Constraints:\n"
            "  - Make at most one `propose_skill(...)` call total.\n"
            "  - If a near-match appears in the catalog below, call "
            "`load_skill(name)` first to read its body, then call "
            "`propose_skill(name=<same>, ...)` to OVERWRITE the user-"
            "tier skill with a merged version (preserve useful old "
            "parts, add new patterns from this turn). Skills upsert "
            "by name — same name = replace.\n"
            "  - If no skill is warranted (Gate 1/2/3 failure), do "
            "NOT call any tool; reply with one plain-text line "
            "starting `no skill needed — <reason>`.\n"
            "  - Do not chain unrelated work; this is reflection, "
            "not execution.\n\n"
            + (sk.body or "")
            + "\n\n---\n\n"
            + catalog_block
        )

        tool_summary = ", ".join(sorted(names))
        reflection_user = (
            f"=== USER ASKED ===\n{(task or '')[:1500]}\n\n"
            f"=== DISTINCT TOOLS USED THIS TURN ===\n"
            f"{tool_summary}  (count: {n_distinct})\n\n"
            f"=== AGENT'S FINAL ANSWER (first 2500 chars) ===\n"
            f"{(answer or '')[:2500]}\n\n"
            f"Walk skill_creator's 3 gates. Then either call "
            f"`propose_skill(...)` once (with merge if a near-match "
            f"exists in the catalog) or reply 'no skill needed' as "
            f"plain text. Do not chain multiple tools."
        )

        def _exec(name, args):
            return registry.execute(name, args)

        def _on_tool(name, args, result, is_error):
            try:
                preview = (result or "").strip().splitlines()[0][:80] if result else ""
                tag = "skill_reflection_tool_error" if is_error else "skill_reflection_tool"
                agent.progress(
                    tag, f"{name}({', '.join((args or {}).keys())}) -> {preview}",
                )
            except Exception:
                pass

        out = router().call_with_tools(
            TaskType.SKILL_REFLECTION,
            reflection_system,
            reflection_user,
            tools=filtered,
            execute_tool=_exec,
            max_tokens=2000,
            temperature=0.2,
            max_iterations=SKILL_REFLECTION_MAX_ITERATIONS,
            on_tool_call=_on_tool,
        )
        try:
            preview = (out or "").strip()[:300]
            agent.progress("skill_reflection", f"done: {preview}")
        except Exception:
            pass
    except Exception as e:
        # Best-effort — never break the user-visible turn.
        log.warning("post-turn skill reflection failed: %s", e)
        try:
            agent.progress("skill_reflection", f"error: {type(e).__name__}: {e}")
        except Exception:
            pass


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
    session_key: str | None = None,
    job_id: str | None = None,
    supervisor_mode: bool = False,
    supervisor_job_id: str | None = None,
) -> AgentAnswer:
    """Execute one unified-loop turn. Called from `Agent.run` when
    the env flag is set. `agent` is the calling Agent instance —
    we pull `progress`, `_record_llm_call`, `_trace`, etc. from it
    so the dev panel / SSE stream still receives all the same
    events the legacy pipeline emitted.

    `session_key` isolates conversation threads — Wife in a group
    chat and Wife in a DM share `speaker_id` (identity / roles) but
    get distinct session_keys (thread). When unset, falls back to
    speaker_id (one thread per speaker — the WebUI's behaviour).

    Supervisor-mode is the audit T6 follow-up for background-job
    completion. When `supervisor_mode=True` and `supervisor_job_id`
    points at a job that just finished, the turn:
      • injects a SUPERVISOR_MODE block at the top of the system
        prompt with explicit retry/deliver/escalate rules;
      • binds an `_active_supervisor_job_id` context var so the
        `complete_supervisor` tool knows which job to seal;
      • skips post-turn hooks (memory extract, skill_reflection,
        conversation log, sessions row) — supervisor turns are
        internal plumbing, not user conversation, and we don't
        want them to pollute long-term memory or trigger skill
        proposals based on synthetic system messages."""
    # Phase 2 (2026-05-23): reset the per-turn tool-bundle state at
    # the very start of every turn. The ContextVar's default is an
    # empty frozenset, but a previous turn in the same process may
    # have set it; without this reset the next turn would start
    # with stale bundles loaded.
    from .tool_bundles import set_loaded_bundles as _po_set_loaded_bundles
    _po_set_loaded_bundles(set())
    skey = (session_key or "").strip() or speaker_id
    # Late imports to avoid cycles.
    from . import roles as _roles
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
    # Per-thread — same person in two chats grows long history
    # independently.
    try:
        _cc.maybe_compact(speaker_id=speaker_id, session_key=skey)
    except Exception as e:
        log.debug("unified: compaction failed (non-fatal): %s", e)

    # Sticky-request detection was dropped 2026-05-21 (keyword-based
    # regexes per system attribute — voice / language / model / etc.)
    # and the `backend.sticky_requests` module was purged 2026-05-23
    # (audit Important #8). TSP rule "Apply, don't acknowledge" +
    # reasoning_routing high effort cover the same ground.

    # Pre-flight 3: progress event.
    agent.progress("unified", "single-loop turn starting")

    # Pre-flight 4: skills. Two things happen here:
    #   1. Each skill's `handler.py` (if any) registers its tools
    #      into the global ToolRegistry — so `registry.to_anthropic_list`
    #      below sees skill-provided tools as well as builtins.
    #   2. `match(task)` walks every enabled skill's triggers; on a
    #      hit the FULL `system_block()` (description + when_to_use
    #      + body) is appended to the prompt so the model gets the
    #      step-by-step instructions in this same turn.
    # This was lost during the Phase C migration from agent.py → run_unified;
    # restoring it brings skill-driven workflows back on the production path.
    from .skills import SKILLS as _SKILLS
    try:
        _SKILLS.ensure_loaded()
    except Exception as e:
        log.debug("SKILLS.ensure_loaded failed (non-fatal): %s", e)
    try:
        matched_skills = _SKILLS.match(task) or []
    except Exception:
        matched_skills = []
    if matched_skills:
        agent.progress(
            "skill",
            "active skills: " + ", ".join(s.name for s in matched_skills),
        )

    # H4: semantic fallback. Only fires when trigger/tag match got
    # nothing — explicit match always wins over fuzzy match. We surface
    # the top candidates as a tiny hint, NOT by inlining their bodies,
    # so the model still has to call `load_skill(name)` to actually
    # pull the instructions. That preserves the "index first → load
    # on demand" budget the user designed for.
    semantic_suggestions: list = []
    if not matched_skills:
        try:
            semantic_suggestions = _SKILLS.semantic_match(task, limit=2) or []
        except Exception:
            semantic_suggestions = []
        if semantic_suggestions:
            agent.progress(
                "skill",
                "semantic suggestions: " + ", ".join(s.name for s in semantic_suggestions),
            )

    # Skill missing-tools reporting. Pre-2026-05-21 this block
    # auto-fired `installer.propose()` (Telegram approval DM) for
    # each missing dep — that ceremony was retired. We now just
    # COLLECT the missing tools and surface them in the system
    # block; the LLM decides whether to `pip install` / `apt install`
    # via terminal_exec when it actually needs them. This avoids
    # installing things proactively that the skill might never
    # exercise during the turn.
    auto_proposed: list[dict] = []
    auto_propose_deferred: int = 0
    if matched_skills:
        seen_pairs: set[tuple[str, str]] = set()
        for sk in matched_skills:
            try:
                missing = _SKILLS.missing_tools_with_manager_for(sk)
            except Exception:
                missing = []
            for m in missing:
                pkg_name = m.get("name") or ""
                mgr = m.get("manager") or "pip"
                key = (mgr, pkg_name)
                if not pkg_name or key in seen_pairs:
                    continue
                seen_pairs.add(key)
                auto_proposed.append({
                    "name": pkg_name, "manager": mgr,
                    "skill": sk.name, "code": None,
                    "status": "missing",
                })
        if auto_proposed:
            agent.progress(
                "install",
                "missing tools (install via terminal_exec when needed): "
                + ", ".join(
                    f"{p['name']}({p['manager']})"
                    for p in auto_proposed
                ),
            )

    # System prompt assembly.
    perms = _roles.permissions_block(speaker_id)
    try:
        state = _ss.current_self_state(speaker_id=speaker_id)
        snapshot = _ss.render_snapshot(state)
    except Exception:
        snapshot = ""

    recall = _auto_recall_block(task)
    # Per-thread conversation context — Wife's DM thread and Wife's
    # group-chat thread don't leak into each other's prompts even
    # though both have the same speaker_id.
    convo = CONVERSATION.context_block(n=6, session_key=skey)

    # Audit follow-up — LLM-based fast chat path. Cheap turns
    # (greetings, recall, acknowledgements) shouldn't pay the full
    # ~15 KB preamble + tool-loop overhead. Structural gate: try the
    # fast lane only when:
    #   - no attachments (file delivery needs the full preamble),
    #   - no matched skill (skill match = real task by definition),
    #   - message is reasonably short (≤500 chars; long is task-shaped).
    # The decision IS the LLM's: it gets a tiny prompt with no tools
    # and either answers directly or emits `ESCALATE: <reason>` to
    # fall through to the full path. No keyword regex.
    if (
        not attachments
        and not matched_skills
        and len(task or "") <= 500
    ):
        chat_answer = _try_chat_path(
            task=task,
            agent=agent,
            speaker_id=speaker_id,
            snapshot=snapshot,
            convo=convo,
        )
        if chat_answer is not None:
            # Fast-path turn — still write a minimal artifact under
            # workspace/turns/<id>.json so every turn has an audit
            # trace and /api/turns/<id> resolves. Skipping the
            # artifact would silently lose ~30-40% of turns (the
            # chat-shaped ones) from the dev panel + audits.
            turn_id = ""
            try:
                from .workspace import get_workspace
                from datetime import datetime as _dt_fp
                import uuid as _uuid
                turn_id = (
                    f"{_dt_fp.now().strftime('%Y%m%d_%H%M%S')}_"
                    f"{_uuid.uuid4().hex[:8]}"
                )
                tu_now = TOKENS.request_usage()
                artifact = {
                    "turn_id": turn_id,
                    "ts": _dt_fp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": task,
                    "answer": chat_answer,
                    "intent": "chat",
                    "is_chat": True,
                    "confidence": 85,
                    "topics": [],
                    "channel": channel,
                    "speaker_id": speaker_id,
                    "session_key": skey,
                    "mode": "fast_chat",
                    "token_usage": tu_now,
                    "n_tool_calls": 0,
                    "n_llm_calls": tu_now.get("llm_calls", 1) if isinstance(tu_now, dict) else 1,
                    "thinking_trace": [],
                    "llm_calls": [],
                    "verification": {
                        "confidence": 85,
                        "verified_claims": [],
                        "unverified_claims": [],
                        "contradictions": [],
                        "notes_used": [],
                    },
                }
                if job_id:
                    artifact["job_id"] = job_id
                get_workspace().save_turn(turn_id, artifact)
                try:
                    setattr(agent, "_last_turn_id", turn_id)
                except Exception:
                    pass
            except Exception as e:
                log.debug("unified fast-path: save_turn failed: %s", e)

            # The inner `call_with_tools` already pushed token usage
            # through TokenTracker.record so per-day counters and
            # request_usage() are up-to-date for the parent turn.
            from .models import VerificationResult as _VR
            return AgentAnswer(
                answer=chat_answer,
                verification=_VR(confidence=85),
                is_chat=True,
                mode="fast_chat",
                turn_id=turn_id,
            )

    # Audit follow-up: the T8 keyword-based "trivial chat" classifier
    # was removed (see comment above `_build_rules_for_turn`). Full
    # preamble + catalog go to every turn; the LLM uses the "Chat vs
    # task" rule in core to decide brevity.
    try:
        catalog = _SKILLS.catalog_block()
    except Exception:
        catalog = ""

    system_parts = [
        IDENTITY.preamble(speaker_id=speaker_id),
    ]
    if snapshot:
        system_parts.append(f"---\n\n{snapshot}")
    if recall:
        system_parts.append(f"---\n\n{recall}")
    if convo:
        system_parts.append(f"---\n\n{convo}")
    if catalog:
        system_parts.append(f"---\n\n{catalog}")
    if semantic_suggestions:
        hint_lines = [
            "# SEMANTIC SUGGESTIONS",
            "(No trigger/tag fired, but these skills look topically close. "
            "If one fits, call `load_skill(name)` to pull its body before "
            "starting the work.)",
        ]
        for sk in semantic_suggestions:
            hint_lines.append(f"- **{sk.name}** — {sk.description}")
        system_parts.append("---\n\n" + "\n".join(hint_lines))
    if auto_proposed:
        # Tell the LLM which deps the matched skills need but aren't
        # installed. The agent decides whether to actually install
        # them via `terminal_exec` (pip/apt/npm/cargo) before
        # invoking the skill, or to pivot to an approach that
        # doesn't need them. The owner-approval ceremony is gone —
        # `terminal_exec` runs install commands directly.
        ap_lines = [
            "# MISSING TOOLS (skills need these — install via terminal_exec when needed)",
            "(Skills matched this turn declared dependencies that aren't on this host. "
            "Install them with `terminal_exec` BEFORE invoking the skill if you decide "
            "to use it: `pip install <name>` / `apt install <name>` / `npm install <name>` "
            "/ `cargo install <name>`. The package will be importable in the NEXT turn — "
            "the current turn's Python interpreter has its site-packages already loaded.)",
        ]
        for p in auto_proposed:
            ap_lines.append(
                f"- `{p['name']}` via `{p['manager']}` (skill `{p['skill']}`)"
            )
        system_parts.append("---\n\n" + "\n".join(ap_lines))
    for sk in matched_skills:
        try:
            missing = _SKILLS.missing_tools_for(sk)
        except Exception:
            missing = []
        try:
            system_parts.append(f"---\n\n{sk.system_block(missing_tools=missing)}")
        except Exception as e:
            log.debug("skill %s system_block failed: %s", sk.name, e)
    # Audit follow-up: keyword-based trivial-chat / bug-report
    # classifiers removed. Rules are composed from structural signals
    # only (attachments, sticky-request flag). The LLM uses the
    # "Chat vs task" + "Diagnose runtime bugs from journal" rules
    # always present in the preamble to decide its own behaviour.
    _repeat_refusal = _recent_refusal_pattern(
        session_key=skey, speaker_id=speaker_id,
    )
    # Derive the TurnContext for v2 prompt-module loader. The fields
    # decide which conditional modules fire (M2 / M4 / M7 / M9).
    from .prompt_modules import TurnContext as _PromptCtx
    from .tool_bundles import get_loaded_bundles as _get_loaded_bundles
    _channel_norm = channel if channel in ("webui", "telegram", "voice", "cli", "api") else "webui"
    try:
        from .llm import router as _router_for_size
        _active_llm = getattr(_router_for_size(), "_active_llm", None)
        _active_model = getattr(_active_llm, "model", None) if _active_llm else None
    except Exception:
        _active_model = None
    _turn_ctx = _PromptCtx(
        turn_type="supervisor" if supervisor_mode else "task",
        channel=_channel_norm,  # type: ignore[arg-type]
        loaded_bundles=frozenset(_get_loaded_bundles()),
        model_size=_classify_model_size(_active_model),  # type: ignore[arg-type]
    )
    _rules_for_turn = _build_rules_for_turn(
        ctx=_turn_ctx,
        has_attachments=bool(attachments),
        sticky_fired=False,
        repeat_refusal=_repeat_refusal,
    )
    system_parts.append(f"---\n\n{_rules_for_turn}")
    system_parts.append(f"---\n\n{perms}")
    # Supervisor-mode preamble. The user-visible chat rules above
    # ("answer in Russian / no preamble / ...") still apply but the
    # supervisor block OVERRIDES the "ask before acting" / "DM the
    # user about progress" defaults. The supervisor turn is internal
    # plumbing — its only outputs are tool calls (retry / mark
    # terminal) and at most ONE user-visible DM at the end of the
    # chain.
    if supervisor_mode:
        sup_block = (
            "# SUPERVISOR MODE\n\n"
            "You are running as a background-job supervisor, not a "
            "chat reply. The synthetic `BACKGROUND_JOB_COMPLETED:` "
            "message in this turn was generated by the system on "
            "job completion — there is no human waiting on the "
            "other end of THIS turn.\n\n"
            "Your job: decide RETRY / DONE / ESCALATE and execute. "
            "Do NOT chat. Do NOT introduce yourself. Do NOT ask "
            "questions. Do NOT send progress messages. The user "
            "only sees the FINAL DM you compose via "
            "`complete_supervisor`.\n\n"
            "Tools you should use here:\n"
            "  • `read_file` / `terminal_exec` — diagnose the failure "
            "(read logs, check filesystem state).\n"
            "  • `start_background_job` — when you decide RETRY, "
            "spawn a CHILD job with `parent_job_id` set to the "
            "completed job's id and `retry_count` incremented. The "
            "supervisor will be re-invoked on the child's "
            "completion. After spawning the retry, finish this "
            "turn with a short status line (no DM).\n"
            "  • `complete_supervisor(decision, final_message)` — "
            "when DONE or ESCALATE. `decision` is 'done' or "
            "'escalate'. `final_message` is the structured DM the "
            "user will receive (one message; mention problems hit, "
            "fixes applied, and the final result OR what blocks "
            "you).\n\n"
            "Rules:\n"
            "  - Apply trivial fixes (path typo, `python`→`python3`, "
            "missing flag) WITHOUT asking. Just patch the command "
            "and re-spawn.\n"
            "  - If the previous decisions in `supervisor_history` "
            "tried the same fix and it failed twice, ESCALATE — "
            "don't loop on the same broken approach.\n"
            "  - When the user's original request said 'publishable "
            "result on N items' and the job only ran 1 item, that "
            "is NOT done — RETRY with the corrected scope.\n"
            "  - The user accepts up to 10 silent retry attempts. "
            "Past that, ESCALATE with diagnostic.\n"
            "  - Final DMs are Russian (default user language) and "
            "follow this shape: short status, what problems hit, "
            "what fixed them, final result or blocker.\n\n"
            "TASK ENDPOINT GATE (Phase 3):\n"
            "If a `=== TASK ENDPOINT EVALUATION ===` block appears "
            "in this turn's input, it is the AUTHORITATIVE checklist "
            "for what counts as DONE. Each critical criterion marked "
            "❌ (unmet) BLOCKS `complete_supervisor(decision='done')` "
            "at the code level — the tool will refuse and tell you "
            "which criteria are blocking. Your options when criteria "
            "are unmet:\n"
            "  (a) RETRY — patch the command/inputs to satisfy the "
            "criterion and call start_background_job again with "
            "endpoint_id inherited from this job.\n"
            "  (b) ESCALATE — write a final_message naming the "
            "criteria you couldn't meet + what you tried + what the "
            "user needs to clarify.\n"
            "  (c) OVERRIDE (rare, escape hatch) — call "
            "complete_supervisor with `criteria_overrides={\"<id>\": "
            "\"<concrete evidence the criterion IS met despite "
            "check_cmd>\"}` if you've verified from logs/output "
            "that the check_cmd has a bug or path issue. Each "
            "override needs a real explanation; bare 'looks fine' "
            "is not acceptable."
        )
        system_parts.append(f"---\n\n{sup_block}")
    system_prompt = "\n\n".join(system_parts)

    # Tool surface = base set + currently-loaded bundles. The LLM
    # picks; mid-turn it can call `load_tool_bundle(name)` to unlock
    # a niche bundle, which mutates the per-turn ContextVar; the next
    # iteration's `tools_provider` callback (see below) sees the new
    # set. Phase 2 (2026-05-23).
    # MUST be after SKILLS.ensure_loaded() above so skill-provided
    # tools (handler.py register) make it into the schema.
    registry = get_registry()

    def _current_tool_schema_for_turn() -> list[dict]:
        """Re-derive the tool schema from the current loaded-bundle
        state. Called by `call_with_tools` before each LLM iteration
        so a mid-turn `load_tool_bundle` is reflected in the next
        request."""
        from .tool_bundles import (
            BASE_TOOLS, expand_loaded, get_loaded_bundles,
        )
        loaded = get_loaded_bundles()
        allowed = set(BASE_TOOLS) | expand_loaded(loaded)
        return registry.to_anthropic_list(filter_names=allowed)

    # Initial schema is the cold-start (base-only) view — passed for
    # back-compat with any provider path that ignores tools_provider.
    tools_schema = _current_tool_schema_for_turn()

    # Tool execution + progress wiring. Reuse the agent's existing
    # `_on_tool_call` shape so the dev panel + SSE event stream
    # gets the same events the legacy pipeline emits.
    tool_outputs: list[str] = []
    # T2 (audit cost-reduction): tighten per-tool token caps. Old
    # caps (e.g. read_file at 20000) fed huge file bodies back to
    # the LLM on every iteration — the May 2026 cost audit traced
    # the 40:1 input:output ratio largely to this. New caps cut the
    # default by 4-5x; the LLM is now expected to use start_line /
    # end_line on read_file (it already supports them) or `grep`
    # for targeted lookups. Errors get a smaller cap because they
    # rarely need full traceback context.
    _DEFAULT_CAP = 1500
    _tool_cap = {
        "read_file": 4000,
        "view_file": 4000,
        "read_note": 4000,
        "list_files": 2000,
        "glob": 2000,
        "grep": 4000,
        "search": 4000,
        "search_knowledge": 4000,
    }
    # T3 (no-progress detector): tracks the last `_NOPROGRESS_WINDOW`
    # tool-result hashes. If the LAST `_NOPROGRESS_WINDOW` tool calls
    # produced identical hashes, the LLM is in a probing loop —
    # rereading the same file, re-running the same grep, etc. We
    # append a 🔄 marker telling it to stop and write the partial
    # report. Same shape as the token budget marker. Window constant
    # is module-level for tests + audit visibility.
    _recent_tool_hashes: list[str] = []

    # AskUserQuestion follow-up: when the agent calls the `ask_user`
    # tool, the handler persists a PendingQuestion and returns a
    # sentinel JSON with `awaiting_input=True`. We detect that in
    # `_execute_with_progress`, stash the question_id here, and after
    # the tool loop ends we attach the structured question payload to
    # the AgentAnswer so the SSE / Telegram / WebUI rendering layer
    # can surface it as a clean card. The mutable dict pattern is
    # used because Python <3.10 / lint targets don't let us reassign
    # `nonlocal` from a nested def cleanly.
    _pending_question_id = {"v": ""}

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
        # Side-publish to LogBus so the WebUI Logs tab sees this
        # tool call alongside Python logging + job state + agent
        # progress. Best-effort: never block or crash the tool loop
        # on a logging concern.
        try:
            from .log_bus import publish_tool_event as _pub_tool
            _pub_tool(
                name=name,
                args=args or {},
                result_preview=preview,
                is_error=is_error,
                request_id=getattr(agent, "_last_turn_id", "") or "",
            )
        except Exception:
            pass

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
        raw_result, is_error = registry.execute(name, args)
        # AskUserQuestion sentinel detection. The `ask_user` tool
        # returns a JSON payload that includes `awaiting_input=True`
        # + a `question_id`. When we see it, stash the id and
        # rewrite the tool result with a strong instruction so the
        # LLM ends the turn cleanly instead of trying to compose
        # more text on top of "I asked the user X".
        if (
            name == "ask_user"
            and raw_result
            and not is_error
            and not _pending_question_id["v"]
        ):
            try:
                _parsed = json.loads(raw_result)
            except Exception:
                _parsed = None
            if (
                isinstance(_parsed, dict)
                and _parsed.get("ok")
                and _parsed.get("awaiting_input")
                and _parsed.get("question_id")
            ):
                _pending_question_id["v"] = str(_parsed["question_id"])
                # Replace the tool result with a directive — short and
                # unambiguous. The outer turn will overwrite `answer`
                # anyway, but if the LLM still tries to compose a
                # final paragraph here we don't want it to confuse
                # itself with the full tool JSON.
                raw_result = (
                    "QUESTION_SHOWN_TO_USER. The user is now seeing "
                    "a structured question card with the options "
                    "you provided. END THE TURN immediately — reply "
                    "with an empty string or a one-line "
                    "acknowledgement. Do NOT call any more tools."
                )
        # Audit follow-up: the previous order (markers appended →
        # truncation at end) clipped the markers exactly when they
        # were most needed (long tool outputs). New order:
        #   1. compute T7 digest header (from FULL raw_result)
        #   2. T2 truncation cuts the BODY only
        #   3. assemble: digest + truncated_body + budget/no-progress markers
        # That keeps markers visible regardless of body size.

        # T7: digest header from the full result.
        digest = ""
        if raw_result and not is_error:
            try:
                digest = _digest_tool_result(name, args, raw_result)
            except Exception:
                digest = ""

        # T2: truncate the BODY only — markers go outside this.
        body = raw_result or ""
        if body and not is_error:
            cap = _tool_cap.get(name, _DEFAULT_CAP)
            if len(body) > cap:
                hint = _truncation_hint(name)
                body = (
                    body[:cap]
                    + f"\n\n…[+{len(raw_result) - cap:,} more chars truncated. "
                    + hint + "]"
                )

        # T1+T4: token-budget marker. Audit follow-up: count
        # `input_tokens` not `total_tokens` — output is ~2.5% of input,
        # the bloat we're trying to surface is input replay across
        # iterations. A 30 k input cap on a turn that emitted 5 k of
        # output is the warning the LLM needs to see; total-token-
        # based markers fire too late.
        marker_budget = ""
        try:
            usage = TOKENS.request_usage()
            input_so_far = usage.get("input_tokens", 0)
            marker_budget = _format_token_budget_marker(input_so_far)
        except Exception:
            marker_budget = ""

        # T3: no-progress detector — hash the FULL raw_result head
        # (before truncation) so identical-with-noise outputs still
        # match. Marker appended OUTSIDE truncation so it survives.
        marker_nopr = ""
        try:
            import hashlib as _hl
            head = (raw_result or "")[:500]
            h = _hl.sha1(f"{name}:{head}".encode("utf-8", "replace")).hexdigest()[:16]
            _recent_tool_hashes.append(h)
            if len(_recent_tool_hashes) > _NOPROGRESS_WINDOW:
                _recent_tool_hashes.pop(0)
            if (
                len(_recent_tool_hashes) == _NOPROGRESS_WINDOW
                and len(set(_recent_tool_hashes)) == 1
            ):
                marker_nopr = (
                    f"\n\n🔄 **NO PROGRESS DETECTED** — the last "
                    f"{_NOPROGRESS_WINDOW} tool calls produced identical "
                    f"output (same hash {h}). You're in a probing loop. "
                    f"Stop re-reading / re-running the same probe. Either "
                    f"switch to a different tool / strategy, or write the "
                    f"partial report with what you have and end the turn."
                )
                _recent_tool_hashes.clear()
        except Exception:
            marker_nopr = ""

        # Assemble: digest (top, never clipped) + truncated body +
        # markers (bottom, never clipped). The body is the only part
        # that may shrink.
        out_parts = []
        if digest:
            out_parts.append(digest)
        out_parts.append(body)
        if marker_budget:
            out_parts.append(marker_budget.lstrip())
        if marker_nopr:
            out_parts.append(marker_nopr.lstrip())
        final = "\n\n".join(p for p in out_parts if p)
        return final, is_error

    # The big call. max_iterations: legacy solve used 6; unified
    # turns can do more work in one loop (research + apply +
    # report) so we widen a bit. Falls under failover chain.
    t0 = _time.monotonic()
    usage_before = TOKENS.request_usage()
    # Snapshot the request-call log length so we can emit a
    # per-iteration LLMCallDetail for each tool-loop API call after
    # the loop completes. Pre-fix the whole loop showed up as a
    # single `_unified` aggregate in the turn artifact — fine for
    # the cumulative cost number but useless for "which iteration
    # burned the tokens". Audit follow-up 2026-05-21.
    _calls_before_loop = TOKENS.request_calls_count()
    try:
        answer = router().call_with_tools(
            TaskType.COMPLEX_SOLVING,
            system_prompt,
            task,
            tools=tools_schema,
            tools_provider=_current_tool_schema_for_turn,
            execute_tool=_execute_with_progress,
            max_tokens=4000,
            temperature=0.3,
            # 10 was too tight for multi-step media work: the
            # post-mortem on the logo-removal task showed all 10
            # iterations spent on metadata probes + frame extraction
            # before the agent could reach the actual delogo render.
            # 20 still bounds runaway loops but leaves room for the
            # natural shape of complex workflows (load skill → find
            # file → probe → sample frames → analyze_image x N →
            # render → verify → deliver).
            max_iterations=20,
            on_tool_call=_on_tool_call,
            attachments=attachments or None,
        )
    except LLMError as e:
        # Bubble up — outer run() / SSE handler classifies and
        # surfaces to the user.
        raise

    # Post-parser: when the model exhausts max_iterations and still
    # wants to call another tool, some providers (Codex / gpt-5.5 in
    # particular) emit the next intended call as XML-like text in
    # the final synthesis turn:
    #   <tool_call name="terminal_exec">
    #     <arg name="cmd">...</arg>
    #   </tool_call>
    # That XML never executed — it's just text the user sees as a
    # broken-looking message. Rewrite it into a plain-language
    # status report so the human knows what was tried and what
    # input is missing.
    answer = _rewrite_xml_tool_call_dump(answer, agent)
    # 2026-05-21: refusal-rewriter dropped. The previous version
    # ran a ~25-keyword regex over the answer head ("не могу",
    # "I can't", "tools are not available") and a second regex to
    # exclude legitimate privacy refusals. With the trust-LLM
    # direction the user explicitly asked for ("remove keyword
    # logic fully from agent pipeline"), the answer stands as-is —
    # if the LLM gives up too early, that's a prompt issue to fix
    # via the TSP rules + reasoning_routing, not a regex
    # post-processor.

    # H3 enforcement: post-turn skill_creator reflection. Out-of-band
    # LLM call that walks skill_creator's 3 gates against this turn's
    # trace, then optionally fires propose_skill (with merge-existing
    # support when a near-match is in the catalog). Gated tightly so
    # most turns skip — only fires when ≥3 distinct tools ran AND
    # the answer isn't a refusal/rewriter output AND no propose_skill
    # was already called this turn. See `_should_reflect_for_skill`.
    # Supervisor turns SKIP skill_reflection — they're internal
    # plumbing, not workflows worth distilling into skills.
    if not supervisor_mode:
        try:
            _post_turn_skill_reflection(agent, task, answer, speaker_id)
        except Exception as _e:
            log.warning("skill reflection top-level swallow: %s", _e)

    # Per-iteration telemetry: walk the new CallRecord entries
    # produced inside the tool loop and emit a compact LLMCallDetail
    # for each. We skip `system_redacted` / `user_redacted` (would
    # balloon turn artifacts — same system prompt × N iterations);
    # the value of these entries is the per-iter token breakdown,
    # not re-dumping the prompt. The aggregate `_unified` call below
    # stays for back-compat with the existing UI.
    try:
        iter_records = TOKENS.request_calls_since(_calls_before_loop)
        for rec in iter_records:
            tt = (rec.get("task_type") or "").strip()
            if not tt:
                continue
            agent._llm_calls.append(LLMCallDetail(
                label=f"_unified:{tt}",
                task_type=tt,
                model=(rec.get("model") or ""),
                system_redacted="",
                user_redacted="",
                response_preview="",
                duration_ms=int(rec.get("duration_ms") or 0),
                input_tokens=int(rec.get("input_tokens") or 0),
                output_tokens=int(rec.get("output_tokens") or 0),
            ))
    except Exception as e:
        log.debug("unified: per-iter telemetry skipped: %s", e)

    # Record the (super) LLM call for the dev panel. We treat the
    # whole tool loop as one labelled call — per-iteration detail
    # is the records emitted above; this aggregate preserves the
    # cumulative cost number that pre-2026-05-21 UIs read.
    agent._record_llm_call(
        label="_unified",
        task_type=TaskType.COMPLEX_SOLVING,
        system=system_prompt,
        user=task,
        response=str(answer),
        duration_ms=int((_time.monotonic() - t0) * 1000),
        usage_before=usage_before,
    )

    # AskUserQuestion attachment — resolved EARLY so the question
    # text lands in conversation history, the saved turn artifact,
    # and the session row (not just the in-memory AgentAnswer).
    # Pre-fix the overwrite happened AFTER `CONVERSATION.add_turn`,
    # so the conversation log saved `answer=""` and the resume turn
    # (fired when the user clicks an option) read the prior turn as
    # an empty assistant message — no context for what was asked,
    # so the LLM either re-fired `ask_user` in a loop or produced
    # garbage. "agent stops working after I answer" (2026-05-21).
    question_payload: Optional[dict] = None
    if _pending_question_id["v"]:
        try:
            from .tools import ask_user as _aq
            q = _aq.STORE.get(_pending_question_id["v"])
            if q is not None:
                question_payload = q.to_dict()
                preview_q = q.question or "(no text)"
                # Compose a conversation-readable representation that
                # also carries the option labels. The resume turn
                # sees this in history and understands what menu it
                # offered — picks the right branch on the user's
                # "My choice: …" follow-up without re-firing ask_user.
                opt_lines = []
                for opt in q.options:
                    lbl = (opt.get("label") or "").strip()
                    if lbl:
                        opt_lines.append(f"  • {lbl}")
                if opt_lines:
                    answer = (
                        f"❓ {preview_q}\n\n"
                        f"Options offered:\n" + "\n".join(opt_lines)
                    )
                else:
                    answer = f"❓ {preview_q}"
        except Exception as e:
            log.warning("ask_user payload attach failed: %s", e)

    # Post-hoc: memory extraction. Supervisor turns SKIP — the
    # synthetic "BACKGROUND_JOB_COMPLETED: ..." text would otherwise
    # land in the user's long-term memory as if they had said it.
    # ask_user turns also SKIP — the "❓ …" placeholder isn't a fact
    # worth storing, and routing it through the extractor wastes
    # tokens for zero learning signal.
    if not supervisor_mode and not question_payload:
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

    # Supervisor turns DON'T enter the user-facing conversation log.
    # They're internal plumbing — including them would make the
    # conversation buffer look like "user asked status, agent asked
    # status, user asked status…" because the synthetic
    # BACKGROUND_JOB_COMPLETED messages would surface as turns.
    if not supervisor_mode:
        CONVERSATION.add_turn(
            task, answer or "",
            intent="task", is_chat=False,
            confidence=vr.confidence,
            topics_used=[],
            channel=channel,
            speaker_id=speaker_id,
            session_key=skey,
        )

    # Audit fix #2: persist a full Session row from EVERY agent.run
    # path, not just WebUI's /api/chat. Pre-fix, Telegram and CLI
    # turns updated CONVERSATION but not SESSIONS, so they were
    # invisible in the WebUI Sessions panel even though their
    # conversation history was retrievable. Now run_unified is the
    # single source of truth for both — every caller (WebUI, TG, CLI)
    # gets a Session entry on the new thread keyed by session_key.
    try:
        from .sessions import SESSIONS
        from datetime import datetime as _dt
        n_tools = sum(
            1 for s in (agent._trace or [])
            if getattr(s, "tool_call", None)
            and getattr(s, "event", "") in ("tool", "tool_error")
        )
        tu = agent._get_token_usage()
        turn_record = {
            "ts": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": task,
            "answer": answer or "",
            "intent": "task",
            "is_chat": False,
            "confidence": vr.confidence,
            "topics": [],
            "channel": channel,
            "speaker_id": speaker_id,
            "session_key": skey,
            "token_usage": tu.model_dump() if tu else None,
            "n_tool_calls": n_tools,
            # Audit follow-up — `len(agent._llm_calls)` lied: the
            # unified turn records ONE super-call but the underlying
            # `call_with_tools` may have done 5-15 actual LLM
            # requests. Take the truth from TokenTracker which is
            # incremented on every API call. Fall back to the old
            # value only if usage wasn't reported (legacy path).
            "n_llm_calls": (
                (tu.llm_calls if tu and getattr(tu, "llm_calls", 0) else None)
                or len(agent._llm_calls or [])
            ),
        }
        if job_id:
            turn_record["job_id"] = job_id

        # Persist the FULL turn artifact under workspace/turns/<id>.json
        # so /api/turns/<id> can lazy-load the thinking_trace / tool
        # outputs / verification details for any past turn — including
        # Telegram and CLI turns. Pre-fix, only the WebUI path even
        # mentioned `save_turn` was *defined* (workspace.py) but never
        # called from anywhere; every turn carried `turn_id: None` and
        # the WebUI's lazy-load endpoint always 404'd. Wire up the
        # write here so all channels get it for free.
        try:
            from .workspace import get_workspace
            import uuid as _uuid
            turn_id = (
                f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"
            )
            artifact = dict(turn_record)
            artifact["turn_id"] = turn_id
            artifact["thinking_trace"] = [
                (s.model_dump() if hasattr(s, "model_dump") else dict(s))
                for s in (agent._trace or [])
            ]
            artifact["llm_calls"] = [
                (c.model_dump() if hasattr(c, "model_dump") else dict(c))
                for c in (agent._llm_calls or [])
            ]
            artifact["verification"] = (
                vr.model_dump() if hasattr(vr, "model_dump") else dict(vr)
            )
            get_workspace().save_turn(turn_id, artifact)
            turn_record["turn_id"] = turn_id
            # Stamp on the in-memory AgentAnswer so callers (WebUI
            # SSE response) can deep-link without re-deriving.
            try:
                setattr(agent, "_last_turn_id", turn_id)
            except Exception:
                pass
        except Exception as e:
            log.debug("unified: save_turn failed (non-fatal): %s", e)

        SESSIONS.add_turn(
            turn_record,
            speaker_id=speaker_id,
            session_key=skey,
        )
    except Exception as e:
        log.debug("unified: sessions add_turn failed (non-fatal): %s", e)

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

    # NOTE: `question_payload` and the `answer = "❓ …"` overwrite
    # for the AskUserQuestion path are resolved EARLIER (right after
    # `_record_llm_call`) so the question text propagates into
    # conversation history + saved turn artifact + session row —
    # not just the in-memory AgentAnswer returned below.

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
        turn_id=getattr(agent, "_last_turn_id", "") or "",
        question=question_payload,
    )
