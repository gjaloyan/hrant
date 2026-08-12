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

This module IS the path. It was originally wired behind an
`HRANT_UNIFIED_AGENT=1` env flag for side-by-side A/B against the
legacy pipeline; that migration finished, the flag is read nowhere,
and this text was still describing the transition as if it were
ongoing (corrected 2026-08-09). Treat any remaining reference to a
"legacy pipeline tier" as history, not as a live alternative.

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
    ToolCallDetail,
    VerificationResult,
)
# V2 (2026-05-27): the legacy system_prompt_sections module was
# retired. `_unified_rules_core()` now calls `prompt_modules.build_prompt`
# directly; the SECTIONS dict + assemble() shim are gone.


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


# RULES surface (2026-05-27 v2 architecture):
#   - The core block is assembled per-turn by
#     `prompt_modules.build_prompt(ctx, overrides)` from M1-M9
#     modules. `_unified_rules_core(ctx)` delegates to that.
#   - Scenario blocks below (journal-first, media-convention,
#     file-types, repeated-request, repeat-refusal) are still
#     appended on demand by `_build_rules_for_turn` because they
#     trigger on per-turn structural signals (attachments, sticky
#     pattern, refusal detector), not module predicates.
#   - `_UNIFIED_RULES_CORE` is the default-context build at import
#     time; legacy tests that grep it for phrases still work.
#   - Per-profile overrides go in `prompt_overrides["modules"]`
#     (see `pipeline_profile.active_overrides()`).

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


# Always-on: the agent MUST run the real test suite before declaring
# done. Closes the "I made it compile, looks good" failure mode that
# accounts for 6 of the 11 terminal-bench failures on the 2026-06-04
# baseline. Universal — many real-world tasks have test suites.
_RULES_VERIFY_TESTS = """## Before declaring a task done — run the real tests

If the workspace contains a test suite (any of: `/tests/` directory,
`test_*.py` files at the project root, a `Makefile` `test` target,
`pytest.ini`, or `pyproject.toml` with `[tool.pytest.ini_options]`),
you MUST execute it and observe a passing run BEFORE composing your
final answer.

"It compiles", "my sample input works", or "I checked the obvious
case" are NOT verification — only the real test suite's pass signal
counts. If tests fail, fix the cause and re-run. Do not synthesize
the final answer while any test is red.

If you cannot find a test suite, look for verifier hints in the
task instruction (paths under `/tests/`, "the verifier expects X",
"run pytest /tests"). Run them. If you genuinely cannot find any
verification mechanism, say so explicitly in your final answer —
don't pretend a check happened."""


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

  Done — logo removed. Duration preserved, audio kept without re-encode.

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

If you actually NEED tools (saving or remembering a fact/preference
— `запомни` / `сохрани` / remember / save persist to memory and
REQUIRE a tool, file operations, settings changes, external lookups,
running code, multi-step problems, file delivery) — do NOT try to
answer. Respond with EXACTLY one line:

  ESCALATE: <one-sentence reason>

The system will then route the same message through the full
agent with all tools available. Don't apologise, don't explain
how the routing works, don't promise "I'll do it next turn", and
NEVER claim you saved or remembered something here — you have no
tools to do it, so a "Запомнил/Saved" reply would be a lie. Just
`ESCALATE: ...` and stop.

Match the user's language. Be brief.
"""


import re as _re_tcd

# `tool_name(` at the very start of the answer. We confirm the name
# is a REAL registered tool before escalating, so legit prose that
# happens to contain `print(x)` or `f(a, b)` is never misread.
_BARE_CALL_RE = _re_tcd.compile(r"^\s*([a-z_][a-z0-9_]{2,})\s*\(")
# Wrappers some models put around a tool call that we must peel off
# before the bare-call check: a ```tool_code```/<tool_code> block and
# a `print(...)` / `tool.run(...)`-style code shell.
_TOOL_CODE_WRAPPER_RE = _re_tcd.compile(
    r"^\s*(?:```+\s*(?:tool_code|python|py)?\s*|<tool_code>\s*|print\s*\()",
    _re_tcd.IGNORECASE,
)


def _looks_like_tool_call_dump(head: str) -> bool:
    """True when the answer head IS a tool call dumped as text — the
    bare `web_search(query=...)` form OR a `<tool_code>`/```tool_code```
    code block OR a `print(web_search(...))` shell (Gemini/code-style).
    Conservative: only fires when, after peeling wrappers, the head
    starts with a REAL registered tool name, so ordinary prose with
    `print(x)` or `f(a, b)` is never misread."""
    if not head:
        return False
    s = head.lstrip()
    # Peel up to two layers of wrapper (e.g. ```tool_code\nprint( ).
    for _ in range(2):
        m = _TOOL_CODE_WRAPPER_RE.match(s)
        if not m:
            break
        s = s[m.end():].lstrip()
    m = _BARE_CALL_RE.match(s)
    if not m:
        return False
    name = m.group(1)
    try:
        from .tool_registry import get_registry
        return name in get_registry().tools
    except Exception:
        return False


# A no-tool chat-lane answer that ASSERTS it saved/remembered something
# is a fabricated action — the lane has no tools to persist anything.
# Anchored at the answer head so a mid-sentence "I could save that" or
# "если хочешь, я могу сохранить" never trips it. Boolean only: on a
# match the caller escalates to the full path where save_user_fact runs.
_SAVE_CLAIM_RE = _re_tcd.compile(
    r"^\s*(?:"
    r"запомнил|запомнила|запомню|запомним|"
    r"сохранил|сохранила|сохраню|"
    r"записал|записала|запишу|"
    r"занёс|занес|"
    r"буду помнить|"
    r"saved|noted|remembered|"
    r"i['’`]?ve saved|i saved|"
    r"i['’`]?ve noted|i noted|"
    r"i['’`]?ve remembered|i remembered|"
    r"i['’`]?ll remember|i will remember|"
    r"added to memory"
    r")\b",
    _re_tcd.IGNORECASE,
)


def _claims_save_without_tool(head: str | None) -> bool:
    """True when a no-tool chat-lane answer asserts it saved/remembered
    something (e.g. "Запомнил: ...", "Saved your preference."). The lane
    has zero tools, so the claim is an apply-don't-acknowledge lie —
    escalate so the full path actually persists it. Caught 2026-06-16:
    "запомни ... 10-19" got a "Запомнил: ..." reply with nothing saved."""
    if not head:
        return False
    return _SAVE_CLAIM_RE.match(head) is not None


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
    # ESCALATE marker can appear (a) at the very start of the answer
    # (the disciplined shape the prompt asks for) OR (b) on any line
    # AFTER preamble prose. The defensive scan was added 2026-06-09
    # after Task 4 of the agent improvement loop caught a case where
    # the LLM wrote "I need to delegate ... \n\nESCALATE: <reason>" —
    # the prose-then-ESCALATE shape made the prior `startswith` check
    # miss the marker, and the full preamble (including the literal
    # ESCALATE line) leaked into the user-facing answer.
    escalate_reason: str | None = None
    if head.upper().startswith("ESCALATE:"):
        escalate_reason = head[9:120].strip()
    else:
        for line in answer.splitlines():
            stripped = line.lstrip()
            if stripped.upper().startswith("ESCALATE:"):
                escalate_reason = stripped[9:120].strip()
                break
    if escalate_reason is not None:
        try:
            agent.progress("chat_fast_path", f"escalating: {escalate_reason}")
        except Exception:
            pass
        return None
    if "<tool_call" in head[:300] or "<tool_code" in head[:300]:
        # LLM emitted a tool-call XML / code-block dump — wanted tools.
        try:
            agent.progress("chat_fast_path", "escalating: tool-call block in output")
        except Exception:
            pass
        return None
    if _looks_like_tool_call_dump(head):
        # LLM emitted a bare function-call dump as its answer, e.g.
        # `web_search(query="...")` — the parenthesised form the XML
        # guard above missed. Caught in Gor's real history 2026-06-13:
        # a chat turn answered with the literal `web_search(...)` text
        # instead of running it, so the user got a non-answer and had
        # to poke "Hrant?". Escalate to the full tool loop.
        try:
            agent.progress(
                "chat_fast_path", "escalating: bare tool-call in output",
            )
        except Exception:
            pass
        return None
    if _claims_save_without_tool(head):
        # The lane has no tools, so an answer asserting it saved /
        # remembered something is fabricated. Escalate so the full path
        # actually calls save_user_fact instead of lying "Запомнил".
        try:
            agent.progress(
                "chat_fast_path", "escalating: claimed save with no tool",
            )
        except Exception:
            pass
        return None
    # The general case (2026-08-08, from the owner's real Telegram log). The
    # regex above only catches SAVE-shaped claims. On 2026-08-08 the fast lane
    # answered "Ок, Гор — делаем быстрый MVP сейчас." — a commitment to do
    # work, with zero tools — and returned before any gate ran, so nothing
    # started. Forty-five minutes later "status?" reported "nothing is
    # running". The lane must not be able to promise an action it has no
    # tools to take.
    #
    # `unbacked_action_claim` is the existing language-agnostic judge for
    # exactly this, so no new keyword list. It costs one cached
    # classification call on a lane that exists to skip a ~15 KB preamble and
    # a whole tool loop — cheap next to silently losing an afternoon.
    try:
        from .endpoint_check import unbacked_action_claim
        _claim = unbacked_action_claim(task, head or "", [])
    except Exception:
        _claim = ""
    if _claim:
        try:
            agent.progress(
                "chat_fast_path",
                f"escalating: promised an action with no tool — {_claim[:60]}",
            )
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
    parts = [_unified_rules_core(ctx), _RULES_JOURNAL_FIRST, _RULES_VERIFY_TESTS]
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


_FINDINGS_TOTAL_CAP = 4000
_FINDINGS_PER_CALL_CAP = 320
_FINDINGS_MAX_CALLS = 14


_DEFAULT_LOOP_ITERATIONS = 500
# A correction round is a second chance inside an existing turn, not a fresh
# turn, so it gets a smaller budget than the main loop — but a real one.
# Hermes gives its subagents 50 against a parent's 500; the same ratio here.
_DEFAULT_CORRECTION_ITERATIONS = 50


def _configured_correction_iterations() -> int:
    """Budget for one self-correction round. See `_configured_loop_iterations`."""
    try:
        from .config import CONFIG
        n = int(CONFIG.router.get("correction_max_iterations",
                                  _DEFAULT_CORRECTION_ITERATIONS))
    except Exception:
        return _DEFAULT_CORRECTION_ITERATIONS
    return n if n >= 1 else _DEFAULT_CORRECTION_ITERATIONS


def _configured_loop_iterations() -> int:
    """How many tool rounds one turn may take. See config.py's long note.

    Read at call time, not import time, so `set_setting` takes effect without
    a restart. Defensive: a corrupt or absent setting must never make a turn
    unable to run — it falls back to the documented default rather than
    raising, and a value below 1 would silently produce a turn that cannot
    call a single tool.
    """
    try:
        from .config import CONFIG
        n = int(CONFIG.router.get("tool_loop_max_iterations",
                                  _DEFAULT_LOOP_ITERATIONS))
    except Exception:
        return _DEFAULT_LOOP_ITERATIONS
    return n if n >= 1 else _DEFAULT_LOOP_ITERATIONS


def _turn_findings(agent, previous_answer: str = "") -> str:
    """What this turn has ALREADY established, for the correction round.

    Measured 2026-08-10 on the owner's DataLex task. The corrective re-prompt
    passes only `task + corrective` to `call_with_tools`, which starts a fresh
    message list — so the agent entered round two knowing nothing it had just
    spent fifty tool calls learning. The trace shows exactly that: it had
    found `#search_form` and enumerated its controls, then opened the site's
    home page and began again. Twice, once per round.

    That is not the model being forgetful. We deleted its notes and asked why
    it started over. The corrective even says "keep going with it THIS TURN",
    which was impossible to obey.

    Newest calls first: the tail is where the turn actually got to, and it is
    what survives the budget when a turn made fifty calls.
    """
    steps = list(getattr(agent, "_trace", None) or [])
    lines: list[str] = []
    used = 0
    for _step in reversed(steps):
        if len(lines) >= _FINDINGS_MAX_CALLS or used >= _FINDINGS_TOTAL_CAP:
            break
        tc = getattr(_step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None)
        if not name:
            continue
        args = getattr(tc, "args", None)
        if args is None and isinstance(tc, dict):
            args = tc.get("args")
        result = getattr(tc, "result", None)
        if result is None and isinstance(tc, dict):
            result = tc.get("result")
        is_error = bool(getattr(tc, "is_error", False) or (
            tc.get("is_error") if isinstance(tc, dict) else False))
        arg_s = ", ".join(f"{k}={str(v)[:90]}" for k, v in (args or {}).items())
        res_s = str(result or "").strip().replace("\n", " ")[:_FINDINGS_PER_CALL_CAP]
        mark = "ERROR" if is_error else "ok"
        entry = f"  [{mark}] {name}({arg_s}) -> {res_s}"
        lines.append(entry)
        used += len(entry)
    if not lines and not (previous_answer or "").strip():
        return ""
    out = ["[WHAT YOU ALREADY DID THIS TURN — do not start over]"]
    if (previous_answer or "").strip():
        out.append("Your previous answer was:\n"
                   + previous_answer.strip()[:800])
    if lines:
        # Collected newest-first so the budget keeps the TAIL — where the turn
        # actually got to — then printed in the order they happened, which is
        # how they read.
        out.append(f"Your last {len(lines)} tool call(s), in order:")
        out.extend(reversed(lines))
        out.append("Continue from here. Re-running what already succeeded "
                   "wastes the turn; the results above are still valid.")
    return "\n".join(out)


def _turn_tool_names(agent) -> list[str]:
    """Names of tools actually called this turn, read from the trace."""
    out: list[str] = []
    for _step in (getattr(agent, "_trace", None) or []):
        tc = getattr(_step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None)
        if name is None and isinstance(tc, dict):
            name = tc.get("name")
        if isinstance(name, str):
            out.append(name)
    return out


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


_BG_TRAILING_AMP = re.compile(r"[^&]\s+&\s*$")
_BG_NOHUP = re.compile(r"(^|;|\|\||&&|;)\s*nohup\s+")
_BG_SETSID = re.compile(r"(^|;|\|\||&&|;)\s*setsid\s+")
_BG_DISOWN = re.compile(r"(^|\s)disown(\b|$)")

# Substrings that indicate a wait/poll for a backgrounded job.
_WAIT_HINTS: tuple[str, ...] = (
    "wait $",      # `wait $PID` or `wait $!`
    "wait %",      # job-spec wait
    "while ps",    # busy-wait pattern
    "until [",     # busy-wait pattern
    "tail -f",     # following the job's log to completion
)


def _command_looks_backgrounded(cmd: str) -> bool:
    """Heuristic match for the four supported backgrounding shapes
    (trailing `&`, `nohup`, `setsid`, `disown`). Conservative on
    purpose — better miss a legitimate fire-and-forget than
    false-positive on `make build && make test`."""
    if not cmd:
        return False
    if _BG_TRAILING_AMP.search(cmd):
        return True
    if _BG_NOHUP.search(cmd):
        return True
    if _BG_SETSID.search(cmd):
        return True
    if _BG_DISOWN.search(cmd):
        return True
    return False


def _command_looks_like_wait(cmd: str) -> bool:
    """Did this command wait/poll for a background job to finish?"""
    if not cmd:
        return False
    low = cmd.lower()
    return any(hint in low for hint in _WAIT_HINTS)


_ARTIFACT_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".csv", ".png", ".jpg", ".jpeg", ".mp4", ".mp3",
    ".zip", ".pptx", ".txt", ".md", ".json", ".webm", ".ogg", ".wav", ".svg",
)
# Paths the Telegram bridge is allowed to attach (see _RULES_MEDIA_CONVENTION).
# `/tmp/` was REMOVED on 2026-08-06: it is where the agent keeps its own
# working material, so every scratch .txt matched here and this gate then
# told the agent to attach it. That is how a per-engine measurement dump
# shipped as the answer to "calibrate the search engines". A real
# deliverable belongs in outbox; scratch does not become a deliverable by
# living in a directory the bridge happens to allow.
_DELIVERABLE_DIRS = ("outbox",)


def _detect_undelivered_artifact(trace, answer: str) -> str:
    """Return the artifact path this turn produced but never delivered, or "".

    Prod incident 2026-07-21: the agent correctly rewrote an invoice PDF into
    workspace/outbox/ and answered without a MEDIA: line, so the owner never
    received the file and read the turn as a failure. Producing a file the
    user asked for and not attaching it is an undelivered result, not a
    finished task. Deterministic — no LLM call."""
    if "MEDIA:" in (answer or ""):
        return ""
    if not trace:
        return ""
    for step in reversed(list(trace)):
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        blob = " ".join(str(x) for x in (
            (getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else {}) or {}),
            (getattr(tc, "result", "") or (tc.get("result") if isinstance(tc, dict) else "") or ""),
        ))
        for m in re.finditer(r"[/\w.\-]+\.\w{2,5}", blob):
            path = m.group(0)
            low = path.lower()
            if not low.endswith(_ARTIFACT_EXTENSIONS):
                continue
            if not any(d in path for d in _DELIVERABLE_DIRS):
                continue
            if path.startswith("/") and path in (answer or ""):
                # Mentioned in the answer but not as a MEDIA: line — still
                # undelivered, and worth correcting.
                return path
            if path.startswith("/"):
                return path
    return ""


def _detect_background_not_awaited(trace) -> bool:
    """True iff at least one terminal_exec command in `trace` was
    backgrounded AND no LATER command in the same trace waited/polled.
    Returns False on empty/None trace.

    Deterministic — no LLM call."""
    if not trace:
        return False
    bg_index = None
    for i, step in enumerate(trace):
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if name != "terminal_exec":
            continue
        args = getattr(tc, "args", None) or (
            tc.get("args") if isinstance(tc, dict) else {}
        )
        cmd = (args or {}).get("command") or ""
        if bg_index is None and _command_looks_backgrounded(cmd):
            bg_index = i
            continue
        if bg_index is not None and _command_looks_like_wait(cmd):
            return False
    return bg_index is not None


def _detect_truncated_then_refusal(trace, answer: str) -> bool:
    """True iff the LAST terminal_exec call in `trace` returned a
    truncated result AND `answer` matches one of the existing
    refusal phrases. This narrowly catches the failure mode where
    Hrant's needed evidence was clipped past the 1500-char cap and
    the agent then refused to commit.

    Deterministic — no LLM call. Reuses _REFUSAL_PHRASES so future
    additions to the refusal list are picked up automatically.
    """
    if not trace or not answer:
        return False
    head = answer[:300].lower()
    if not any(phrase in head for phrase in _REFUSAL_PHRASES):
        return False
    # Walk trace backwards to find the last terminal_exec call.
    for step in reversed(list(trace)):
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if name != "terminal_exec":
            continue
        was_truncated = getattr(tc, "result_truncated", False)
        if was_truncated is None and isinstance(tc, dict):
            was_truncated = tc.get("result_truncated", False)
        return bool(was_truncated)
    return False


# Tokens that mean "the agent learned a test suite exists":
_TESTS_DISCOVERY_TERMINAL_PREFIXES: tuple[str, ...] = (
    "ls /tests",
    "find /tests",
    "cat /tests/",
    "head /tests/",
    "tail /tests/",
)

# Tokens that mean "the agent ran the actual test suite":
_TESTS_RUN_TOKENS: tuple[str, ...] = (
    "pytest",         # also covers `python -m pytest`
    "unittest",       # covers `python -m unittest`
    "make test",
)


def _detect_tests_exist_not_run(trace) -> bool:
    """True iff (a) at least one tool call in `trace` indicates the
    agent discovered the existence of a test suite AND (b) no tool
    call in the same trace actually executed it.

    Deterministic — no LLM call."""
    if not trace:
        return False
    discovered = False
    ran = False
    for step in trace:
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        args = getattr(tc, "args", None) or (
            tc.get("args") if isinstance(tc, dict) else {}
        ) or {}
        if name == "terminal_exec":
            cmd = (args.get("command") or "").strip()
            cmd_low = cmd.lower()
            # Discovery: command starts with a discovery prefix, OR a
            # discovery prefix appears after ; or && (chained commands).
            if any(
                cmd.startswith(p)
                or ("; " + p) in cmd
                or (";" + p) in cmd
                or ("&& " + p) in cmd
                or ("&&" + p) in cmd
                for p in _TESTS_DISCOVERY_TERMINAL_PREFIXES
            ):
                discovered = True
            # Run: any of the run-tokens substring-matches the command.
            if any(token in cmd_low for token in _TESTS_RUN_TOKENS):
                ran = True
        elif name == "read_file":
            path = (args.get("path") or "")
            if path.startswith("/tests/") or path == "/tests":
                discovered = True
    return discovered and not ran


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
# Audit 2026-06-10 I4: don't run skill_reflection on low-confidence
# turns — the verifier already flagged the answer as questionable
# and we don't want to distill a flawed workflow into the catalog.
# 50 lines up with the "low_confidence" threshold the daily report
# uses (cap_confidence_for_endpoint clips to 30 on endpoint-miss,
# so a missed endpoint AND a low base verifier score will both gate
# correctly here).
SKILL_REFLECTION_CONFIDENCE_FLOOR = 50
SKILL_REFLECTION_MAX_ITERATIONS = 4
_REFLECTION_TOOL_ALLOWLIST = frozenset({
    "list_skills", "load_skill", "propose_skill",
})


# T3 (no-progress detector): consecutive identical tool-result hashes
# this many in a row → append the 🔄 NO PROGRESS marker. 3 is a
# sweet spot: 2 catches an intentional re-read (false fire), 4+
# misses the 20-iteration probe loops the May 2026 audit caught.
_NOPROGRESS_WINDOW = 3


# AUTO_PROPOSE_CAP was removed 2026-08-09. It capped auto-fired INSTALL
# proposals per turn — a DM-flood guard for a mechanism deleted on 2026-05-21
# when the install gate went away and the agent started installing via
# terminal_exec directly. It had no readers left, but read as a live safety
# limit: six lines explaining a flood it could not prevent.


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


def _should_reflect_for_skill(
    agent, answer: str, *,
    verifier_confidence: int | None = None,
    endpoint_was_met: bool | None = None,
) -> tuple[bool, str]:
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
      7. (audit 2026-06-10 I4) verifier confidence isn't below the
         floor — low-confidence turns don't yield trustworthy skills.
      8. (audit 2026-06-10 I4) endpoint_met != False — pure-research
         turns (read-only inspects, no MEDIA: delivery) don't have a
         reusable workflow shape, only a question-answering shape.
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
    # Gate 7: low-confidence answers don't yield reusable skills. If
    # the verifier marked the turn as questionable (e.g. unbacked
    # claim, contradictions, hallucinated source), spending another
    # LLM trip to canonize the workflow is wasted — the skill would
    # carry the same flaw forward.
    if (
        verifier_confidence is not None
        and verifier_confidence < SKILL_REFLECTION_CONFIDENCE_FLOOR
    ):
        return False, f"low-confidence-{verifier_confidence}"
    # Gate 8: pure-research turn (no state change, no delivery). The
    # endpoint judge already concluded the answer didn't deliver an
    # action shape — that's a Q-and-A turn, not a workflow. Skill
    # reflection would just propose "search_knowledge then summarize"
    # which is what the agent does by default.
    if endpoint_was_met is False:
        return False, "endpoint-not-met"
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
    *,
    verifier_confidence: int | None = None,
    endpoint_was_met: bool | None = None,
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
    should, reason = _should_reflect_for_skill(
        agent, answer,
        verifier_confidence=verifier_confidence,
        endpoint_was_met=endpoint_was_met,
    )
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

    Audit 2026-06-11: also surfaces top consolidated long-term FACTS
    (memory_facts.jsonl via the fact vector store). Pre-fix, the
    1400+ facts the nightly consolidation distilled were reachable
    ONLY through an explicit `search_facts` tool call the LLM rarely
    thought to make — write-side worked, read-side didn't.

    Skipped on very short messages (< 20 chars) — short messages
    usually don't have enough signal for useful retrieval."""
    if not task or len(task.strip()) < 20:
        return ""
    note_hits = []
    try:
        from .hybrid_searcher import HYBRID
        # 2026-08-08 audit: the comment here used to claim a min_raw_score of
        # 0.55 was applied. `HybridSearcher.search()` has no such parameter —
        # only `find_best()` does — so the guard was documented and never
        # ran. Worse, `search()` min-max normalises, so its top hit is ALWAYS
        # 1.00 no matter how weak the match, and that fabricated 1.00 was
        # printed into the prompt as if it were evidence. Use the call that
        # actually enforces the floor.
        # find_best returns a single entry, so it is used here as the GATE it
        # is: when nothing clears the raw floor it returns None and we emit no
        # notes at all. The list itself still comes from search().
        if HYBRID.find_best(task, min_raw_score=0.55) is None:
            note_hits = []
        else:
            note_hits = HYBRID.search(task, limit=limit)
    except Exception:
        note_hits = []
    fact_hits: list[dict] = []
    try:
        from .fact_search import search_facts
        # No limit=2-always. search_facts now applies a measured score floor
        # and drops synthetic (audit/benchmark) authors, so this returns
        # nothing at all when nothing relevant exists — which is the honest
        # answer, and what the old top-2 could never say.
        fact_hits = search_facts(task, limit=2)
    except Exception:
        fact_hits = []
    if not note_hits and not fact_hits:
        return ""
    lines = ["# AUTO-RECALL (related notes — read via read_file if relevant)"]
    for h in note_hits:
        e = h.entry
        # No score here on purpose. `search()` min-max normalises, so the top
        # hit is ALWAYS 1.00 and the last is ALWAYS 0.00 regardless of how
        # weak the matches are — printing that as "score" fed the model a
        # fabricated confidence. The list is ordered; the order is the signal.
        lines.append(
            f"- {e.topic} (cat: {e.category}, source: {h.source}) → {e.path}"
        )
    if fact_hits:
        lines.append("Long-term facts (consolidated memory):")
        for f in fact_hits:
            summary = str(f.get("summary") or "").strip()
            if summary:
                # Carry the score, as the note lines above already do. Without
                # it the model receives an unqualified assertion it has no way
                # to discount — and these were being injected into EVERY turn.
                lines.append(f"- {summary} (score: {float(f.get('score') or 0):.2f})")
    return "\n".join(lines)


# ─── Self-correction gating ────────────────────────────────────────


_OPEN_STATUS_MARKER = "[TURN GATE] NOT DONE"


def _append_open_status(answer: str, tag: str) -> str:
    """Append the honest status line when the turn is STILL open after the
    correction rounds.

    Written by code on purpose. Composing an accurate account of its own
    incompleteness is the exact thing the agent fails at — three separate
    2026-08-05/06 turns ended with a confident success report over unfinished
    work. Asking it once more to write that sentence is asking the same
    question that already got the wrong answer twice.
    """
    text = (answer or "").rstrip()
    if _OPEN_STATUS_MARKER in text:
        return text
    detail = ""
    try:
        from . import turn_contract as _tc
        detail = _tc.render_user_block()
    except Exception:
        detail = ""
    return (
        f"{text}\n\n---\n{_OPEN_STATUS_MARKER} ({tag}) — after two correction "
        "rounds this turn still did not satisfy the request. Treat the report "
        "above as unverified: check what was actually changed before relying "
        "on it." + (f"\n\n{detail}" if detail else "")
    )


def _decide_self_correction(
    *,
    task: str,
    answer: str,
    turn_tools: list[str],
    trace=None,
    speaker_id: str = "",
) -> tuple[str, str]:
    """Decide whether the just-finished turn needs a corrective re-prompt.

    Returns `(tag, corrective_text)`. Empty `corrective_text` ("") means
    no correction needed. The `tag` is a short label for telemetry
    (logged via `agent.progress`).

    Two failure modes are covered, both decided by an LLM judge:

      (A) ZERO-tool turn that CLAIMS an action — the agent just talked,
          said "I saved it" / "done" with no save_user_fact / no
          set_setting. `unbacked_action_claim` catches it.

      (B) TOOLFUL turn that did NOT deliver — the agent ran a long
          investigation (read_file, terminal_exec, locate_symbol, …)
          and then gave up with a meta-cognitive "I can't confirm" /
          "no verifiable result", without taking the action the user
          asked for (e.g. `start_background_job` for a bench task).
          Detected via the same `endpoint_met` judge the post-hoc
          verifier already uses to cap confidence — but here we
          re-prompt the LLM to either DO the thing or escalate
          honestly with concrete alternatives.

    The toolful branch only fires when no execute-class tool was
    called. A turn that already called `start_background_job`,
    `set_setting`, `complete_supervisor`, etc. is considered to have
    taken action; we don't second-guess it.
    """
    if not (answer or "").strip():
        return "", ""
    # NOTE ON ORDER (2026-08-06): the undelivered-artifact block used to run
    # HERE, first. It matched on a file existing, which is far weaker than the
    # structural gates below, yet it preempted every one of them — including
    # the free plan-incomplete backstop. An unfinished turn that happened to
    # leave a file behind got "attach the file" instead of "you have 3 steps
    # left". It now runs last, after everything that can prove the turn is
    # structurally unfinished.
    # Block 3 — background-not-awaited. Fire FIRST because it's the
    # most specific pattern (the agent literally spawned a process
    # and walked away).
    if _detect_background_not_awaited(trace):
        corrective = (
            "You spawned a background process (nohup/&/setsid/disown) "
            "and never waited for it to finish. Wait for it now: use "
            "`wait $!` (if you have the PID), or poll with `while ps -p "
            "$PID >/dev/null 2>&1; do sleep 5; done`, or `tail -f` the "
            "job's log until you see its completion marker. Then verify "
            "the expected artifact actually exists on disk before "
            "composing your final answer."
        )
        return "background-not-awaited", corrective
    # Block 2 — truncated-then-refusal. Specific recovery path: tell
    # the agent to narrow the output via tail/head/grep so the actual
    # evidence fits the 1500-char tool-result cap.
    if _detect_truncated_then_refusal(trace, answer):
        corrective = (
            "Your last terminal_exec output was truncated at the "
            "1500-char cap and the part you needed to act on didn't "
            "fit. Re-run the command with the output narrowed: pipe "
            "through `tail -200`, `head -200`, or `grep -n PATTERN` so "
            "only the relevant slice comes back. Read the actual "
            "evidence before composing the final answer — do not "
            "refuse based on truncated output."
        )
        return "truncated-then-refusal", corrective
    # Block 1b — tests-exist-not-run, bench-mode only. The universal
    # prompt rule from Task 1 already told the agent to run tests;
    # this branch is the structural backstop when the agent ignored
    # it. Bench-harness only: we don't want to force /tests checks on
    # the WebUI owner asking "what's in /tests/".
    if speaker_id == "webui:bench-harness" and _detect_tests_exist_not_run(trace):
        corrective = (
            "You discovered a test suite under /tests/ but never ran "
            "it. Run the actual tests NOW (`pytest /tests/ -v`, or "
            "`python -m pytest /tests/`, or `make test`, whichever "
            "matches the project setup) and observe a passing run "
            "BEFORE composing your final answer. If the tests fail, "
            "fix the cause and re-run — do not synthesize the final "
            "answer while any test is red."
        )
        return "tests-exist-not-run", corrective
    # Plan-incomplete backstop (2026-06-11). Deterministic and free —
    # checked BEFORE the LLM-judge branches. When the agent declared
    # a checklist via set_plan this turn and is now synthesizing with
    # steps still pending, the work is structurally unfinished no
    # matter how confident the prose sounds. One corrective re-prompt
    # listing exactly what remains; the agent finishes the steps,
    # marks them skipped with a reason, or honestly escalates.
    try:
        from .tools.plan_scratchpad import unfinished_steps
        _pending = unfinished_steps()
    except Exception:
        _pending = []
    if _pending:
        listed = "\n".join(f"  {i}. {text}" for i, text in _pending[:8])
        corrective = (
            f"Your plan for this turn still has {len(_pending)} "
            f"unfinished step(s):\n{listed}\n\n"
            f"Do NOT finalize yet. For each remaining step, either "
            f"(a) execute it now and mark it via "
            f"`update_plan(step, 'done')`, or (b) mark it "
            f"`update_plan(step, 'skipped', note='why')` if it is "
            f"genuinely not needed, or (c) if you are blocked, say so "
            f"explicitly and offer options via `ask_user`. Then write "
            f"the final answer reflecting the TRUE completion state."
        )
        return f"plan-incomplete — {len(_pending)} pending", corrective
    # Turn contract — deterministic and free, so it runs with the other
    # structural gates rather than behind an LLM judgment. The turn changed
    # state and never demonstrated the change took effect. This is the one
    # gate that reads the WORLD (a shell check's exit status) rather than the
    # agent's account of it; everything above still grades output.
    try:
        from . import turn_contract as _tc
        if _tc.is_open():
            return "contract-open", _tc.corrective_text()
    except Exception:
        pass
    # Block 0 — undelivered artifact (2026-07-21 PDF incident): a file the
    # user asked for exists on disk and the answer has no MEDIA: line, so the
    # user receives nothing.
    #
    # The corrective below deliberately does NOT hand over the finished
    # `MEDIA:<path>` line. The 2026-08-06 version did, and because
    # `endpoint_met` then treated that substring as proof of delivery, an
    # unfinished turn was being told the exact string that would silence
    # every remaining check. A gate must never name the artifact that
    # satisfies it. The question it asks now is the one that matters —
    # whether the file is the user's or the agent's own working material.
    _artifact = _detect_undelivered_artifact(trace, answer)
    if _artifact:
        corrective = (
            f"This turn produced `{_artifact}` and your answer does not "
            "deliver it. Did the USER ask for this file?\n"
            "  (a) Yes — read it back, state in one sentence what you "
            f"confirmed is in it, and put `MEDIA:{_artifact}` on its own "
            "line so the bridge attaches it.\n"
            "  (b) No — it is your own working material (measurements, "
            "scratch, an intermediate dump). Then it is NOT a deliverable "
            "and you must not attach it. Instead, name the part of the "
            "request that is still not done, and either do it now or say "
            "plainly that it is undone.\n"
            "Handing over your own scratch output as the result is a "
            "failed task, not a delivery."
        )
        return "undelivered-artifact", corrective
    from .endpoint_check import (
        _DELIVERY_TOOLS as _ENDPOINT_EXECUTE_TOOLS,
        endpoint_met,
        unbacked_action_claim,
    )
    if not turn_tools:
        claim = unbacked_action_claim(task, answer, [])
        if not claim:
            return "", ""
        corrective = (
            f'Your previous answer claimed: "{claim}". But you called no '
            f"tool this turn, so nothing actually performed it. Either call "
            f"the correct tool NOW to actually do it, then confirm in one "
            f"sentence; or rewrite your final answer to state honestly that "
            f"you did not do it. Never claim an action you did not perform."
        )
        return f"unbacked claim — {claim[:60]}", corrective
    if any(t in _ENDPOINT_EXECUTE_TOOLS for t in turn_tools):
        return "", ""
    if endpoint_met(task=task, answer=answer, tool_names=turn_tools):
        return "", ""
    shown = ", ".join(turn_tools[:6])
    if len(turn_tools) > 6:
        shown += f", … (+{len(turn_tools) - 6} more)"
    # Don't assert "all read-only" on a turn that ran twenty shell commands —
    # `terminal_exec`/`run_python` are deliberately outside _EXECUTE_TOOLS
    # because they can be either, and a corrective whose first sentence is
    # visibly false teaches the model to discount the rest of it.
    _mutation_capable = ("terminal_exec", "run_python", "save_to_workspace",
                         "save_knowledge", "pdf_edit")
    _ran_shell = any(t in _mutation_capable for t in turn_tools)
    # 2026-08-10: instruments (agent_browser, sandbox_exec, delegate,
    # start_background_job) left _DELIVERY_TOOLS, so they now reach this
    # corrective. Calling 32 of them "read-only greps" would be visibly false
    # — the exact thing the note above warns teaches the model to discount
    # everything that follows.
    from .endpoint_check import _INSTRUMENT_TOOLS
    _instruments = [t for t in turn_tools if t in _INSTRUMENT_TOOLS]
    if _instruments:
        _opening = (
            f"you drove {_instruments[0]} {len(_instruments)} time(s) and it "
            f"produced no result the user can use, and no concrete blocker"
        )
    elif _ran_shell:
        _opening = (
            "none of them is a tool that records a completed action, and "
            "nothing in your answer shows the change actually taking effect"
        )
    else:
        _opening = "ALL of them were read-only (inspection, reads, greps)"
    corrective = (
        f"Your previous turn called {len(turn_tools)} tool(s) "
        f"({shown}) but {_opening}. You delivered no state-changing action "
        f"AND did not honestly state a concrete blocker. This is "
        f"the long-investigation giveup failure mode.\n\n"
        f"Pick ONE of two paths NOW:\n"
        f"  (a0) If you were already using the right instrument and simply "
        f"stopped short — keep going with it THIS TURN until you have the "
        f"actual result. Do not switch tools to look busy, and do not end "
        f"the turn by describing the next step: take it.\n"
        f"  (a) Take the actual action — `start_background_job` "
        f"for any long-running task (benchmarks, builds, "
        f"transcodes; the supervisor will iterate fixes/retries "
        f"up to 10 times on completion automatically, so you do "
        f"NOT need to babysit it in this turn), `set_setting` "
        f"for config, `save_user_fact`/`save_to_workspace` for "
        f"persistence, `terminal_exec` for short executions, "
        f"`schedule_message` for delivery, etc. Spawn the work "
        f"and end the turn with a short status line.\n"
        f"  (b) Honestly escalate — rewrite the final answer to "
        f"name the SPECIFIC blocker (missing credential, "
        f"ambiguous spec, dependency you can't install) AND "
        f"propose 1-3 concrete alternatives the user can pick "
        f"from via `ask_user`.\n\n"
        f"Do not end this turn with another investigation summary."
    )
    # The tag is what the USER sees in the [TURN GATE] NOT DONE line, so it
    # must not describe 33 browser calls as "read-only tools" — the same
    # visibly-false wording the corrective itself was just fixed for.
    _what = (f"{len(_instruments)} {_instruments[0]} calls" if _instruments
             else f"{len(turn_tools)} read-only tools")
    return (f"toolful no-deliver — {_what}", corrective)


# ─── Build-without-frame nudge (structural, no keyword routing) ─────
# The soul/skill telling the agent to frame a big build is a soft nudge the
# model under-applies (it jumps to building, especially when a detail like a
# port is given). This is a STRUCTURAL backstop that reads the agent's OWN
# behaviour — not the user's words: once it has taken several build-write
# actions WITHOUT calling frame_problem, inject a one-time, timely nudge so it
# can stop and frame mid-flow. Soft (the agent ignores it on a non-system
# build), so no false-positive cost; behavioural, so no keyword classification.
_BUILD_WRITE_TOOLS = frozenset({"save_to_workspace", "terminal_exec", "run_python"})
_BUILD_FRAME_THRESHOLD = 4   # soft FRAME-CHECK marker fires here
_BUILD_BLOCK_THRESHOLD = 4   # hard block: refuse build-writes past this w/o a frame


_VERIFY_TOOLS = frozenset({"verify_web"})


def _note_verify_tools(state: dict, tool_name: str) -> None:
    """Mark the turn as having verified its work (a verification-class tool
    ran). Even an unreachable-result is a look at reality — the gate's goal is
    'look before you claim', not 'the look must succeed'."""
    if tool_name in _VERIFY_TOOLS:
        state["verified"] = True


def _should_block_done(state: dict, tool_name: str, args: dict) -> bool:
    """Hard gate: in a turn that made build-writes, refuse to mark a tracker
    step `done` until a verification ran. The shop audits showed the agent
    claiming 'done' three times while a human had to headless-check the page.
    Behavioral (agent's own tool stream, no keywords); review-only turns (no
    build-writes) mark steps freely."""
    return (
        tool_name == "update_step"
        and str((args or {}).get("status", "")).lower() == "done"
        and state.get("writes", 0) > 0
        and not state.get("verified")
    )


def _should_block_build(state: dict, tool_name: str) -> bool:
    """Hard gate decision: refuse a build-write tool once the agent has built
    several times this turn WITHOUT calling frame_problem. Soft nudges (soul,
    skill, the FRAME-CHECK marker) don't stop the build-eager model, so this
    blocks until it frames. Behavioural — keyed on the agent's own tool stream,
    not the user's words. A frame is one cheap call; an unframed system build is
    the decorative-demo trap. Trivial tasks (<= threshold build-writes) pass."""
    return (
        tool_name in _BUILD_WRITE_TOOLS
        and not state.get("framed")
        and state.get("writes", 0) >= _BUILD_BLOCK_THRESHOLD
    )


# How many failures of the SAME tool in one turn before the agent is told to
# stop working around it and repair it. Three: one failure is noise, two can
# be a transient, three in a row is a defect.
_SELF_REPAIR_AFTER = 3


def _self_repair_marker(tool: str, n: int, last_error: str) -> str:
    """Tell the agent that a repeatedly-failing tool is ITS OWN BUG.

    Not a suggestion to try harder — a redirect. The agent has
    propose_self_modification always-on (2026-08-09) and the handler source
    is readable with read_file; what was missing was any signal connecting
    "this tool keeps failing" to "so fix the tool".

    REWRITTEN 2026-08-11, because the first version sent it to the wrong
    place and it obeyed. When agent_browser started failing with

        Auto-launch failed: Chrome exited early without writing
        DevToolsActivePort ... FATAL:sandbox/linux/suid/client/setuid_sandbox

    this marker fired correctly, and the agent did exactly what it said: it
    grepped the repo, ran locate_symbol, read backend/tools/agent_browser.py
    twice, checked its imports and npm packages. All of that was faithful and
    all of it was useless — the source was fine. Leaked Chrome sessions had
    exhausted memory, so a NEW launch died. Nothing in the handler could show
    that. It then spent 150k tokens working around the tool.

    The diagnosis needed two cheap steps the old text never mentioned: read
    what KIND of error it is, and measure the machine (`free -m`,
    `pgrep -c chrome`). Plus a third the agent has never once used in 74
    turns: `git log` — the browser broke because of a change made to it the
    previous day, and the commit message said so.

    So the order now goes error -> environment -> what changed -> source.
    Reading the source is step FOUR, not step one.
    """
    head = (last_error or "").strip().splitlines()
    first = head[0][:200] if head else ""
    # What was already tried on this tool, and whether it worked. Without
    # this the fourth attempt looks exactly like the first.
    try:
        from .self_mod_outcomes import prior_attempts_note
        prior = prior_attempts_note(tool)
    except Exception:
        prior = ""
    return (
        f"🔧 **THIS IS YOUR BUG** — `{tool}` has now failed {n} times this "
        f"turn with the same class of error:\n"
        f"    {first}\n\n"
        "Retrying it will fail again. Diagnose it IN THIS ORDER — the first "
        "two steps are cheap and are where most tool failures actually "
        "live:\n"
        "  1. READ THE ERROR ABOVE and decide what KIND of failure it is. "
        "A message about a process dying, a port, a socket, memory, disk, a "
        "permission, a certificate or a missing binary is the ENVIRONMENT, "
        "not the source code. Reading the handler will not show it to you.\n"
        "  2. If it looks environmental, MEASURE the machine before "
        "theorising: `terminal_exec` with `free -m`, `df -h`, "
        "`pgrep -c <process>`, `systemctl --user status <unit>`, "
        "`journalctl --user -u <unit> -n 30`. A tool that worked an hour ago "
        "and fails now is almost never a code change — something on the box "
        "moved.\n"
        "  3. CHECK WHAT CHANGED: `cd ~/hrant && git log --oneline -15` and "
        "`git log -p -3 -- <the tool's file>`. You are modified regularly, "
        "including by yourself. A tool that broke today was very likely "
        "changed yesterday, and the commit message usually names the "
        f"reason.\n"
        f"  4. ONLY THEN read the source: `read_file` the handler (grep the "
        f"repo for `{tool}`) — for a wrong path, a stale package name, a "
        "missing fallback.\n"
        "  5. Fix it with `propose_self_modification` (always available, no "
        "bundle needed) — or, if the fault is in the environment rather than "
        "the code, repair the environment directly with `terminal_exec` and "
        "say what you did.\n"
        "  6. If you genuinely cannot repair it, say so plainly and name the "
        "blocker WITH the measurement that proves it — do NOT keep calling "
        "the tool.\n"
        "Working around a broken tool leaves it broken for the next turn, "
        "and for the owner.\n"
        + prior +
        "\n--- original tool result follows ---\n\n"
    )


def _build_frame_marker(state: dict, tool_name: str, is_error: bool) -> str:
    """Mutate `state` and return a one-time FRAME-CHECK marker (or "").

    `state` keys: writes (int), framed (bool), fired (bool)."""
    if tool_name in ("frame_problem", "create_tracker"):
        state["framed"] = True
        return ""
    if not is_error and tool_name in _BUILD_WRITE_TOOLS:
        state["writes"] = state.get("writes", 0) + 1
    if (state.get("writes", 0) >= _BUILD_FRAME_THRESHOLD
            and not state.get("framed")
            and not state.get("fired")):
        state["fired"] = True
        return (
            "🧭 **FRAME-CHECK** — you've taken several build actions without "
            "calling `frame_problem`. IF you are building an app / shop / site / "
            "system: STOP now and `frame_problem` the FULL component map "
            "(subsystems — accounts, payments, admin, inventory, search, "
            "security, a real DB, …), confirm scope with `ask_user`, and be "
            "honest that an MVP is a slice, not the whole. If this is NOT a "
            "system build, ignore this."
        )
    return ""


# ─── Experience loop, write side: auto case notes ───────────────────
# The "own knowledge first" pillar stood on an almost-empty KB because
# save_knowledge was voluntary and the model rarely called it. Framed builds
# are the highest-value experience — the frame (components/scope) + outcome is
# a complete case — so persist it STRUCTURALLY at turn end, assembled
# deterministically (no extra LLM call). Next similar task recalls it via
# search_knowledge / the auto-recall block.

# Re-audit 2026-07-06: the verifier scores long framed build turns 30-50
# (honest partial-status answers sit at 50), so the original >=60 gate never
# passed in practice and ZERO cases were written. Real signal: delivery
# (endpoint_met) not False + moderate confidence.
_CASE_MIN_CONFIDENCE = 40


def _auto_case_note(*, task: str, answer_head: str, tools_used: list,
                    confidence: int, frame: dict | None,
                    endpoint_met: bool | None = None) -> bool:
    """Save a compact case note for a successful framed turn. Returns True
    when a note was written. Never raises — this is best-effort plumbing."""
    try:
        if not frame or confidence < _CASE_MIN_CONFIDENCE:
            return False
        if endpoint_met is False:
            return False
        title = str(frame.get("title") or "").strip()
        if not title:
            return False
        comps = frame.get("components") or []
        comp_lines = "\n".join(
            f"- {c.get('name')}{' (mvp)' if c.get('mvp') else ' (deferred)'}"
            for c in comps if isinstance(c, dict) and c.get("name")
        )
        tool_counts: dict[str, int] = {}
        for t in tools_used or []:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        tools_line = ", ".join(
            f"{k}×{v}" if v > 1 else k for k, v in tool_counts.items()
        )
        body = (
            f"Task: {task.strip()[:300]}\n\n"
            f"Component map ({len(comps)}):\n{comp_lines}\n\n"
            f"Scope: {frame.get('proposed_scope', '')}\n\n"
            f"How: {tools_line}\n\n"
            f"Outcome (confidence {confidence}): {answer_head.strip()[:400]}"
        )
        kw = [str(frame.get("domain") or "").strip()] + [
            str(c.get("name")) for c in comps[:6]
            if isinstance(c, dict) and c.get("name")
        ]
        from .knowledge_manager import KM
        KM.save_note(
            topic=f"Case: {title}"[:120],
            body=body,
            category="projects",
            keywords=[k for k in kw if k],
            source="auto-case (experience loop)",
            confidence="partial",
        )
        return True
    except Exception as e:
        log.debug("auto case note failed (non-fatal): %s", e)
        return False


def _load_turn_frame(turn_started_at: float) -> dict | None:
    """Newest frame artifact written during this turn, or None."""
    try:
        from .paths import workspace_dir
        d = workspace_dir() / "frames"
        best, best_m = None, 0.0
        for p in d.glob("*.json"):
            m = p.stat().st_mtime
            if m >= turn_started_at - 1 and m > best_m:
                best, best_m = p, m
        if best is None:
            return None
        import json as _json
        return _json.loads(best.read_text(encoding="utf-8"))
    except Exception:
        return None


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
    # Every turn starts with the empty default bundle set. The
    # former scenario-specific auto-load (supervisor → {"bench"})
    # was retired 2026-05-27 when start_background_job /
    # define_task_endpoint / complete_supervisor moved to
    # BASE_TOOLS — supervisor turns no longer need a bundle dance.
    _po_set_loaded_bundles(set())
    # Reset the per-turn duplicate-call cache so each turn starts
    # with a clean slate (the cache is what makes the second
    # `terminal_exec("same cmd")` short-circuit with a DUPLICATE
    # CALL warning instead of re-running the handler).
    from .tool_registry import reset_per_turn_call_cache as _ptc_reset
    _ptc_reset()
    # The browser tool hands the CLI's own guide to the FIRST browser call of
    # a turn. Reset here so each turn gets it once — measured 2026-08-10, the
    # agent never fetched that guide on its own across four turns even though
    # the tool description points at it, and it contains the fact it kept
    # tripping over (refs go stale after any click or re-render).
    try:
        from .tools.agent_browser import (
            reap_orphan_sessions as _abg_reap,
            reset_guide_for_turn as _abg_reset,
        )
        _abg_reset()
        # Backstop for a turn killed before its own cleanup ran. Skips any
        # session whose job is still running, so a concurrent turn keeps its
        # page.
        _abg_reap()
    except Exception:
        pass
    # Open a per-turn endpoint-judgment cache. _decide_self_correction,
    # the verifier-cap branch, and cap_confidence_for_endpoint all
    # evaluate endpoint_met on identical (task, answer, tool_names) —
    # ~3 LLM CLASSIFICATION trips for one turn without this. The
    # cache makes the 2nd and 3rd calls instant. Audit 2026-06-10 (I3).
    # We deliberately don't reset() at function exit: the next turn's
    # begin_turn_cache() rebinds a fresh dict via ContextVar.set(), so
    # the prior dict becomes garbage. Tracking the token through
    # 1300+ lines of branches would just be footgun-bait.
    from .endpoint_check import begin_turn_cache as _ec_begin
    _ec_begin()
    # Turn contract — same ContextVar lifecycle and the same reasoning about
    # not tracking a reset token. A turn that changes state owes one proof
    # that the change took effect; see backend/turn_contract.py.
    try:
        from .turn_contract import begin_turn as _tc_begin
        _tc_begin()
    except Exception:
        pass
    # Reset the per-turn router fallback reason so the honest model notice
    # only reflects THIS turn's fallback (if any).
    try:
        from .llm import reset_turn_fallback_reason as _rfr
        _rfr()
    except Exception:
        pass
    # Reset the per-turn plan scratchpad (2026-06-11) — each turn
    # starts plan-less; set_plan/update_plan mutate it mid-turn and
    # _decide_self_correction refuses synthesis with pending steps.
    from .tools.plan_scratchpad import reset_plan as _plan_reset
    _plan_reset()
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
    # Wake-up context (audit 2026-06-11): yesterday's consolidated
    # narrative + open threads + failure lessons. Cached per calendar
    # day inside the consolidation module, so this is a dict lookup
    # on all but the first turn of the day. Closes the read-side gap
    # where digests were written nightly but never re-entered
    # cognition — the agent woke up amnesiac.
    try:
        from .consolidation.recall import yesterday_block as _yb
        yesterday = _yb()
    except Exception:
        yesterday = ""
    # Per-thread conversation context — Wife's DM thread and Wife's
    # group-chat thread don't leak into each other's prompts even
    # though both have the same speaker_id.
    # Ten, not six (2026-08-12). Six was enough for chat and far too few
    # for work: two `ask_user` round-trips cost four slots, and every
    # background job that finishes costs another, so the exchange the
    # owner actually cared about was routinely evicted before he
    # replied to it. `recent()` now also keeps human turns ahead of
    # machine ones inside the window.
    convo = CONVERSATION.context_block(n=10, session_key=skey)

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
    # ...and never for a turn that is RESUMING a paused task with the owner's
    # answer to an ask_user question (2026-08-08, from the day's real Telegram
    # log). "My choice: Быстрый MVP сейчас" is 43 chars with no attachment, so
    # the fast lane took it, answered "Ок, делаем" with ZERO tools, and the
    # work never started. Forty-five minutes later "status?" was answered with
    # "waiting for you to choose" — a choice already made. The reply to a
    # question the agent itself asked is the continuation of a task.
    try:
        from .tools.ask_user import (
            clear_question_resume as _clear_resume,
            is_question_resume as _is_resume,
        )
        _resuming = _is_resume()
        # Consume it. Left set, the flag would persist for the life of the
        # context and push every later turn off the fast lane — which the
        # tests for this fix caught before it shipped.
        _clear_resume()
    except Exception:
        _resuming = False
    if (
        not attachments
        and not matched_skills
        and not _resuming
        and len(task or "") <= 500
    ):
        chat_answer = _try_chat_path(
            task=task,
            agent=agent,
            speaker_id=speaker_id,
            # Fast path gets the wake-up block folded into the snapshot —
            # "what did you do yesterday?" is a chat-shaped question and
            # must not require the full tool loop to answer.
            snapshot=(
                f"{snapshot}\n\n{yesterday}" if yesterday else snapshot
            ),
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
                    "level": "L0_CHAT",
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

            # Persist the fast-path turn into CONVERSATION so the
            # next turn's `context_block(n=6, session_key=skey)` can
            # see it. Without this, fast-path turns are invisible to
            # future-turn back-references — caught 2026-06-09 D4 of
            # the deep agent-improvement loop, where the second turn
            # of a same-session pair could not recall the first
            # because turn 1 had landed on the fast path and was
            # never written to history.
            try:
                CONVERSATION.add_turn(
                    task, chat_answer,
                    intent="chat", is_chat=True,
                    confidence=85,
                    topics_used=[],
                    channel=channel,
                    speaker_id=speaker_id,
                    session_key=skey,
                )
            except Exception as e:
                log.debug("unified fast-path: CONVERSATION.add_turn failed: %s", e)

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

    # Trajectory memory (AGI roadmap #2, 2026-06-11): retrieve up to
    # two similar PAST SOLVED turns and show their tool chains +
    # outcomes. Computed here — after the fast-path gate — so chat
    # turns never pay the embed. Best-effort: embedder down = "".
    try:
        from .trajectory_memory import past_experience_block
        experience = past_experience_block(task)
    except Exception:
        experience = ""

    # NOW block (2026-06-12): the agent defaulted to UTC for every
    # user-facing time ("напомни в понедельник" became Monday 10:00
    # UTC = 14:00 for the user). Inject the user's local time + zone
    # so 'tomorrow' / 'on Monday' / reminder times resolve in THEIR
    # zone; UTC stays available for tool args that require it.
    now_block = ""
    try:
        from datetime import datetime as _now_dt, timezone as _now_tz
        from zoneinfo import ZoneInfo as _ZI
        from .settings import user_timezone as _user_tz
        _tzname = _user_tz()
        _utc_now = _now_dt.now(_now_tz.utc)
        _local_now = _utc_now.astimezone(_ZI(_tzname))
        _off = _local_now.strftime("%z")
        now_block = (
            "# NOW\n"
            f"User local time: {_local_now:%A %Y-%m-%d %H:%M} "
            f"({_tzname}, UTC{_off[:3]}:{_off[3:]})\n"
            f"UTC time: {_utc_now:%Y-%m-%d %H:%M}\n"
            "All user-facing times (reminders, 'tomorrow', 'on "
            "Monday', schedules you report back) are in the USER'S "
            "LOCAL timezone. Convert to UTC only for tool arguments "
            "that require UTC (e.g. schedule_message due_at)."
        )
    except Exception as e:
        log.debug("now-block assembly failed: %s", e)

    system_parts = [
        IDENTITY.preamble(speaker_id=speaker_id),
    ]
    if now_block:
        system_parts.append(f"---\n\n{now_block}")
    if snapshot:
        system_parts.append(f"---\n\n{snapshot}")
    if yesterday:
        system_parts.append(f"---\n\n{yesterday}")
    if recall:
        system_parts.append(f"---\n\n{recall}")
    if experience:
        system_parts.append(f"---\n\n{experience}")
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

    # Self-surface unresolved provider failures (audit 2026-05-28).
    # When the active LLM provider returned 402 / 401 / 5xx during
    # an earlier turn (especially silent supervisor turns), this
    # block tells the agent that something failed AND that it should
    # explain to the user + propose a fix (config change, model
    # swap, or self-modification). Skipped for supervisor turns —
    # those are non-user-facing.
    if not supervisor_mode:
        try:
            from .provider_error_log import recent_unresolved
            unresolved = recent_unresolved(within_hours=24)
        except Exception:
            unresolved = []
        if unresolved:
            issue_lines = [
                "# UNRESOLVED AGENT-SIDE FAILURES",
                "",
                "The following provider failures happened on recent turns "
                "and have NOT been explained to the user. If the user's "
                "current message looks related (e.g. asking why something "
                "didn't work, or checking on a background job that should "
                "have completed), include the diagnosis at the top of your "
                "answer + suggest a concrete fix (top up credits / switch "
                "model / add fallback provider / propose a code change via "
                "`propose_self_modification`). After explaining, call "
                "`acknowledge_provider_issue(error_id, resolution)` so the "
                "same issue is not re-surfaced next turn.",
                "",
            ]
            for r in unresolved[-5:]:
                ctx = r.get("context") or {}
                issue_lines.append(
                    f"- id={r.get('id')!r} | provider={r.get('provider')} "
                    f"model={r.get('model')} | HTTP {r.get('status_code')} "
                    f"({(ctx.get('category') or 'unknown')}): "
                    f"{(r.get('message') or '')[:120]}"
                )
            system_parts.append("---\n\n" + "\n".join(issue_lines))
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
            "  - SCOPE-PRESERVING RETRIES (audit 2026-05-28): a RETRY "
            "fixes HOW (flag names, paths, env, syntax). A RETRY "
            "MUST NOT change WHAT (the user's chosen `--agent`, "
            "dataset, model, task list, budget). If the user asked "
            "for `--agent codex` and codex setup is broken, that is "
            "an ESCALATE signal — NOT a 'switch to --agent oracle' "
            "retry. Oracle replays gold answers; benchmarking it "
            "silently delivers a different thing than the user "
            "asked for. The same applies to dataset substitution, "
            "task-list reduction, or scope-narrowing to claim "
            "success.\n"
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
        # Anything registered at RUNTIME is reachable too (2026-08-09
        # dead-code audit). BASE_TOOLS and the bundles are static frozensets
        # written by hand, so a tool a skill registers can never appear in
        # them — the filter silently dropped it from every request. Measured:
        # `calc` (origin "skill:calc") is registered on every boot and was in
        # neither set, while run_python's own description tells the model
        # "for pure arithmetic ALWAYS prefer `calc`". The model was being
        # instructed to call a tool it could not see, and the reachability
        # test could not catch it because that test reloads a registry
        # holding builtins only.
        try:
            allowed |= {
                name for name, tool in registry.tools.items()
                if not str(getattr(tool, "origin", "")).startswith("builtin")
            }
        except Exception:
            pass
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

    # Action-drift detector (audit follow-up 2026-06-03). Token-smart
    # mid-turn guard: count consecutive read-only tools without a
    # single execute-class action. When the count crosses the
    # threshold, inject a marker telling the LLM "stop investigating,
    # take action or write the blocker". Complements the end-of-turn
    # self-correction (which only fires AFTER the final answer is
    # composed): drift catches the bloat mid-flow so the agent
    # doesn't burn 40 read_files before snapping out of it.
    _drift_state = {"consecutive_readonly": 0, "marker_fired": 0}
    # Build-without-frame nudge state (see _build_frame_marker).
    _build_frame_state = {"writes": 0, "framed": False, "fired": False}
    # Repeated failures of the SAME tool, this turn (2026-08-10). The owner's
    # point, and he is right: a self-modifying agent that hits its own broken
    # tool should FIX it, not work around it. On 2026-08-10 agent_browser
    # failed on every call for ~140k tokens; the agent framed the problem,
    # probed PATH, tried to install a package that does not exist, waived
    # honestly and asked the owner — and never once considered that the
    # defect was in its own handler, which it has always-on tools to repair.
    # Nothing told it that a tool failing the same way repeatedly is a BUG IT
    # OWNS rather than an environment it must route around.
    _tool_failures: dict = {}
    # Wall-clock turn start — used to pick THIS turn's frame artifact for the
    # auto case note (file mtimes are wall-clock; t0 above is monotonic).
    _run_started_at = _time.time()

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
        # Record failures where the immune matcher can see them (2026-08-10).
        # Tool errors were visible ONLY in the turn's progress stream, which
        # nothing outlives the turn to read — so a tool failing the same way
        # every time left no trace anywhere. The owner's 2026-08-10
        # conversation shows agent_browser and npm failing repeatedly with
        # `404 Not Found` and `command not found`, retried across dozens of
        # calls, and afterwards there was nothing to learn from.
        if is_error:
            try:
                from .meta_learner import META_LEARNER
                META_LEARNER.log_tool_error(
                    tool=name, message=full_result[:600], args=args or {},
                    turn_id=getattr(agent, "_request_id", "") or "",
                )
            except Exception:
                pass
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
        # HARD build-without-frame gate (structural backstop — see
        # _should_block_build). Returns a blocked error WITHOUT executing, so
        # the model must call frame_problem before more building this turn.
        if _should_block_build(_build_frame_state, name):
            blocked = json.dumps({
                "ok": False,
                "error": (
                    f"BLOCKED: you've run {_build_frame_state.get('writes', 0)} "
                    "build actions without framing this work. Before ANY more "
                    "building this turn you MUST call `frame_problem` with the "
                    "FULL component map (real subsystems — not 8 surface items) "
                    "and confirm scope via `ask_user`. Then continue building "
                    "the confirmed scope. A frame is one cheap call."
                ),
            }, ensure_ascii=False)
            return blocked, True
        # HARD verify-before-done gate: built this turn => look at the result
        # (verify_web) before marking a tracker step done.
        if _should_block_done(_build_frame_state, name, args):
            blocked = json.dumps({
                "ok": False,
                "error": (
                    "BLOCKED: you made build changes this turn but haven't "
                    "verified the result. Call `verify_web(url, expect_texts=…)` "
                    "on what you built (does the page actually render? are the "
                    "products/data really there?) — THEN mark the step done. "
                    "Claiming done without looking is the failure mode this "
                    "gate exists to stop."
                ),
            }, ensure_ascii=False)
            return blocked, True
        _note_verify_tools(_build_frame_state, name)
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

        # Action-drift marker. The end-of-turn self-correction (in
        # `_decide_self_correction`) only inspects the FINAL answer,
        # so a turn that drifts through 40 read-only probes still
        # pays for 40 LLM round-trips before the corrective re-prompt
        # fires. Mid-turn marker injects a nudge inline so the LLM
        # can short-circuit the drift on the NEXT iteration. Token-
        # smart: this is text added to a tool result, not a fresh LLM
        # call.
        marker_drift = ""
        try:
            from .endpoint_check import _EXECUTE_TOOLS as _EXEC_T
            if not is_error and name in _EXEC_T:
                _drift_state["consecutive_readonly"] = 0
            elif not is_error:
                _drift_state["consecutive_readonly"] += 1
            cur = _drift_state["consecutive_readonly"]
            # Fire at 6 (first warning) and 12 (stronger). Beyond
            # that the end-of-turn self-correction will catch it
            # anyway; flooding markers wastes input tokens.
            if cur == 6 and _drift_state["marker_fired"] == 0:
                _drift_state["marker_fired"] = 1
                marker_drift = (
                    "\n\n⚠️ **ACTION DRIFT** — 6 consecutive "
                    "read-only tool calls without a single "
                    "execute-class action (set_setting, save_user_fact, "
                    "start_background_job, schedule_message, "
                    "complete_supervisor, ask_user, …). Investigation "
                    "without action is the long-giveup failure mode. "
                    "Decide NOW: take the action the user asked for, "
                    "or call `ask_user(...)` with a concrete blocker. "
                    "Do not run another read / grep / locate / journalctl."
                )
            elif cur == 12 and _drift_state["marker_fired"] < 2:
                _drift_state["marker_fired"] = 2
                marker_drift = (
                    "\n\n🛑 **STILL DRIFTING (12 read-only calls)** — "
                    "the prior ACTION DRIFT warning was ignored. "
                    "Stop investigating. Either spawn the action "
                    "with `start_background_job` / `set_setting` / "
                    "the appropriate execute-class tool NOW, or "
                    "exit via `ask_user` with 2-3 concrete options "
                    "naming the blocker. No more probes."
                )
        except Exception:
            marker_drift = ""

        # Build-without-frame nudge (structural backstop for the soft
        # soul/skill framing rule the build-eager model under-applies).
        try:
            marker_frame = _build_frame_marker(_build_frame_state, name, is_error)
        except Exception:
            marker_frame = ""

        # Turn contract — the obligation is raised at the MOMENT state
        # changes, not at the end of the turn when the agent has already
        # composed its success story. Same trigger as the build-frame
        # counter: the agent's own tool stream, no keywords.
        marker_self_repair = ""
        try:
            if is_error:
                _n = _tool_failures.get(name, 0) + 1
                _tool_failures[name] = _n
                if _n == _SELF_REPAIR_AFTER:
                    marker_self_repair = _self_repair_marker(name, _n, raw_result)
        except Exception:
            marker_self_repair = ""

        marker_contract = ""
        try:
            if not is_error and name in _BUILD_WRITE_TOOLS:
                from .turn_contract import note_mutation as _tc_note
                marker_contract = _tc_note()
        except Exception:
            marker_contract = ""

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
        if marker_drift:
            out_parts.append(marker_drift.lstrip())
        if marker_frame:
            out_parts.append(marker_frame.lstrip())
        if marker_contract:
            out_parts.append(marker_contract.lstrip())
        if marker_self_repair:
            out_parts.append(marker_self_repair.lstrip())
        final = "\n\n".join(p for p in out_parts if p)
        return final, is_error

    # The big call. Iteration budget comes from
    # `router.tool_loop_max_iterations` (200) — see the long note at its
    # definition in config.py. It was hardcoded 20, which every real task hit,
    # and a turn that runs out of iterations cannot call anything: the only
    # act left is prose, so it writes "the correct next step would be…"
    # instead of taking that step. Falls under failover chain.
    t0 = _time.monotonic()
    usage_before = TOKENS.request_usage()
    # Snapshot the request-call log length so we can emit a
    # per-iteration LLMCallDetail for each tool-loop API call after
    # the loop completes. Pre-fix the whole loop showed up as a
    # single `_unified` aggregate in the turn artifact — fine for
    # the cumulative cost number but useless for "which iteration
    # burned the tokens". Audit follow-up 2026-05-21.
    _calls_before_loop = TOKENS.request_calls_count()

    def _run_main_loop(model_override=None, iterations=None, user_task=None) -> str:
        return router().call_with_tools(
            TaskType.COMPLEX_SOLVING,
            system_prompt,
            user_task if user_task is not None else task,
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
            max_iterations=(
                iterations if iterations is not None
                else _configured_loop_iterations()
            ),
            on_tool_call=_on_tool_call,
            attachments=attachments or None,
            model_override=model_override,
        )

    # Model cascade (AGI roadmap #1, 2026-06-12). When enabled, the
    # turn first runs the FULL tool loop on the configured small tier;
    # the answer is judged by the verifier on the STRONG model (small
    # judges produce false negatives — 2026-06-11 battery); on gate
    # failure the loop re-runs on the active model. The per-turn
    # duplicate-call cache turns the small attempt's tool results
    # into a warm cache for the escalation re-run. Supervisor turns
    # skip the cascade — their decisions seal retry chains and must
    # stay on the strong model.
    _cascade_prevr = None  # strong-verifier result reused post-hoc
    _cascade_cfg = None
    if not supervisor_mode:
        try:
            from .cascade import gate_passes as _c_gate, route as _c_route
            _cascade_cfg = _c_route()
        except Exception:
            _cascade_cfg = None
    try:
        _small_ok = False
        if _cascade_cfg is not None:
            _c_pid, _c_model, _c_threshold, _c_small_iters = _cascade_cfg
            agent.progress(
                "cascade",
                f"small-tier attempt: {_c_pid}/{_c_model} "
                f"(max {_c_small_iters} iterations)",
            )
            try:
                answer = _run_main_loop(
                    model_override=(_c_pid, _c_model),
                    iterations=_c_small_iters,
                )
                _small_ok = True
            except LLMError as _ce:
                agent.progress(
                    "cascade",
                    f"small tier failed ({str(_ce)[:80]}) — escalating",
                )
        if _small_ok:
            try:
                from .verifier import verify as _gate_verify
                _g_vr = _gate_verify(
                    question=task,
                    answer=answer or "",
                    notes_text="",
                    used_topics=[],
                    tool_context="\n\n".join(tool_outputs),
                )
            except Exception as _ge:
                log.warning("cascade gate verify failed: %s", _ge)
                _g_vr = None
            _ok, _why = _c_gate(_g_vr, confidence_gate=_c_threshold)
            if _ok:
                _cascade_prevr = _g_vr
                agent.progress("cascade", f"small tier accepted ({_why})")
            else:
                agent.progress(
                    "cascade", f"escalating to active model ({_why})",
                )
                # The small attempt's plan checklist would trip the
                # plan-incomplete corrective against the strong
                # re-run; each attempt declares its own plan.
                try:
                    from .tools.plan_scratchpad import reset_plan as _c_pr
                    _c_pr()
                except Exception:
                    pass
                # Double-execution guard (2026-06-12 incident): tell
                # the strong rerun which EXECUTE-class actions the
                # small attempt already performed, so it verifies
                # their effects instead of repeating them. Identical
                # re-calls are deduped by the per-turn cache anyway;
                # this note covers same-action-different-args.
                _esc_task = task
                try:
                    from .endpoint_check import (
                        _EXECUTE_TOOLS as _C_EXEC,
                    )
                    _done = [
                        n for n in _turn_tool_names(agent) if n in _C_EXEC
                    ]
                    if _done:
                        seen: list[str] = []
                        for n in _done:
                            if n not in seen:
                                seen.append(n)
                        _esc_task = (
                            f"{task}\n\n[CASCADE HANDOFF] A previous "
                            f"attempt this turn already executed: "
                            f"{', '.join(seen)}. Check their results "
                            f"in the duplicate-call cache before "
                            f"acting — do NOT repeat completed "
                            f"actions; verify and build on them."
                        )
                except Exception:
                    pass
                answer = _run_main_loop(user_task=_esc_task)
        if not _small_ok:
            # No cascade configured, or the small tier raised —
            # either way the active model runs the loop.
            answer = _run_main_loop()
    except LLMError:
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

    # Self-correction: catch two failure modes via ONE corrective re-prompt.
    # See `_decide_self_correction` for the gating logic; both decisions
    # are LLM judgments (language-agnostic, no keyword lists). Supervisor
    # turns skip both — they're internal plumbing.
    _contract_status_tag = ""
    if not supervisor_mode and (answer or "").strip():
        # Two rounds, and the RE-ANSWER IS RE-GATED (2026-08-06). The single
        # -shot version accepted whatever came back unexamined, so a corrective
        # could be answered with a fresh false completion and shipped. When the
        # second round still leaves the turn open, the honest status line is
        # appended by CODE — the model never gets to write it, because writing
        # it is precisely what it does badly.
        _last_tag = ""
        for _round in range(2):
            _turn_tools = _turn_tool_names(agent)
            _correction_tag, _corrective = _decide_self_correction(
                task=task,
                answer=answer,
                turn_tools=_turn_tools,
                trace=getattr(agent, "_trace", None),
                speaker_id=speaker_id,
            )
            if not _corrective:
                _last_tag = ""
                break
            _last_tag = _correction_tag
            agent.progress("self_correct", f"{_correction_tag} — re-prompting")
            try:
                _findings = _turn_findings(agent, previous_answer=answer)
                answer = router().call_with_tools(
                    TaskType.COMPLEX_SOLVING,
                    system_prompt,
                    f"{task}\n\n"
                    + (f"{_findings}\n\n" if _findings else "")
                    + f"[SELF-CORRECTION REQUIRED]\n{_corrective}",
                    tools=_current_tool_schema_for_turn(),
                    tools_provider=_current_tool_schema_for_turn,
                    execute_tool=_execute_with_progress,
                    max_tokens=2000,
                    # Same wall as the main loop had, in the one place it
                    # hurts most: the corrective this round carries says
                    # "keep going with it THIS TURN until you have the actual
                    # result". With six iterations that is an order the agent
                    # cannot obey — it re-reads the page and is cut off again,
                    # which is how two correction rounds produced two more
                    # "the correct next step would be" paragraphs.
                    max_iterations=_configured_correction_iterations(),
                    on_tool_call=_on_tool_call,
                )
                answer = _rewrite_xml_tool_call_dump(answer, agent)
            except LLMError:
                break  # keep the current answer if the corrective call fails
        # The status line is NOT appended here. It is appended after the
        # answer_critic pass, near the return — see `_contract_status_tag`.
        # 2026-08-07, found by running this against the live agent: appending
        # before the critic feeds the code-written block back into the model,
        # which then edits it. An observed turn answered
        #   "Proof note corrected: I should not have said 'NOT PROVED — no
        #    proof registered'; the verifier shows proof was registered"
        # — the agent arguing with the gate's own text. Whatever the model can
        # rewrite, it will.
        _contract_status_tag = _last_tag

    # 2026-05-21: refusal-rewriter dropped. The previous version
    # ran a ~25-keyword regex over the answer head ("не могу",
    # "I can't", "tools are not available") and a second regex to
    # exclude legitimate privacy refusals. With the trust-LLM
    # direction the user explicitly asked for ("remove keyword
    # logic fully from agent pipeline"), the answer stands as-is —
    # if the LLM gives up too early, that's a prompt issue to fix
    # via the TSP rules + reasoning_routing, not a regex
    # post-processor.

    # NOTE: skill_reflection is now fired AFTER verifier + endpoint cap
    # so we can pass the confidence + endpoint-met flags into the gate
    # (audit 2026-06-10 I4) — see the block below labeled "skill
    # reflection (deferred)".

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
    # Escalation level (L0/L1/L2). Pure-action turns (only state-mutation
    # tools) are L1 and skip the claim verifier — endpoint_met already
    # confirmed delivery deterministically and there are no claims to ground.
    # `_level` is also stamped on the artifact below. See backend/escalation.py.
    from .escalation import decide_level, should_verify, tool_names_from_trace
    _level = decide_level(
        was_fast_chat=False,
        tool_names=tool_names_from_trace(agent._trace or []),
    )
    vr = VerificationResult(confidence=85)
    if _cascade_prevr is not None:
        # The cascade gate already verified THIS answer on the strong
        # model moments ago — reuse it instead of paying a second
        # verifier call on an accepted small-tier turn.
        vr = _cascade_prevr
    elif should_verify(tool_outputs, agent._trace or []):
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

    # Endpoint-aware cap (2026-05-27 audit T2.1). The legacy verifier
    # measures CLAIM-verifiability, not REQUEST-delivery. For action-
    # verb requests ("run X", "запусти Y", "send Z") require that
    # the trace contains at least one execute-class tool call OR
    # the answer delivers a file via MEDIA:. Otherwise clip the
    # confidence at 30 so the meta_learner sees the turn as a low-
    # confidence failure even when every claim was verifiable.
    _was_met: bool | None = None  # consumed by skill_reflection gate below
    try:
        from .endpoint_check import (
            cap_confidence_for_endpoint, endpoint_met,
        )
        _trace_tool_names: list[str] = []
        for _step in (agent._trace or []):
            tc = getattr(_step, "tool_call", None)
            if tc is None:
                continue
            # `tool_call` is a `ToolCallDetail` Pydantic model on
            # ThinkingStep — not a dict. Earlier code did
            # `isinstance(tc, dict)` and silently produced an empty
            # trace list, defeating the endpoint check. Pull `name`
            # via attribute access; fall back to dict semantics for
            # robustness across older trace formats.
            name = getattr(tc, "name", None)
            if name is None and isinstance(tc, dict):
                name = tc.get("name")
            if isinstance(name, str):
                _trace_tool_names.append(name)
        _was_met = endpoint_met(
            task=task, answer=answer or "", tool_names=_trace_tool_names,
        )
        # Grader calibration (2026-06-11): record the delivery
        # judgment separately so the learning loop can distinguish
        # "didn't deliver the action" from "bad content".
        vr.endpoint_met = _was_met
        if not _was_met:
            _capped = cap_confidence_for_endpoint(
                task=task, answer=answer or "",
                tool_names=_trace_tool_names, confidence=vr.confidence,
            )
            if _capped != vr.confidence:
                # Preserve the pre-clip content score for the
                # meta-learner / daily report before clipping.
                vr.content_confidence = vr.confidence
                # Surface the reason so the WebUI / daily report
                # explains the dip.
                try:
                    vr.contradictions.append(
                        "endpoint_not_met: action-verb request without "
                        "execute-class tool call or MEDIA: delivery"
                    )
                except Exception:
                    pass
                vr.confidence = _capped
    except Exception as exc:
        log.debug("unified: endpoint check failed: %s", exc)

    # Critic-revise pass (AGI roadmap, 2026-06-11). When the verifier
    # found CONTENT problems (real contradictions / unsupported claims
    # — delivery markers excluded), one bounded revision call with a
    # read-only tool subset tries to fix the claims, the revision is
    # re-verified, and the better of {original, revised} ships. Placed
    # BEFORE skill_reflection so the reflection gates on the improved
    # confidence. ask_user turns skip — the answer is about to be
    # replaced by the question payload anyway.
    if not supervisor_mode and not _pending_question_id["v"]:
        try:
            from .answer_critic import revise_and_pick, should_critique
            _crit_fire, _crit_why = should_critique(vr, answer=answer or "")
            if _crit_fire:
                agent.progress("self_critic", f"revising: {_crit_why}")
                _revised, _new_vr = revise_and_pick(
                    task=task,
                    answer=answer or "",
                    vr=vr,
                    system_prompt=system_prompt,
                    tool_context="\n\n".join(tool_outputs),
                    on_tool_call=_on_tool_call,
                )
                if _revised is not None:
                    answer = _revised
                    vr = _new_vr
                    agent.progress(
                        "self_critic",
                        f"revision kept (confidence {vr.confidence})",
                    )
                else:
                    agent.progress(
                        "self_critic", "revision rejected — original kept",
                    )
        except Exception as _e:
            log.warning("self-critic pass failed (non-fatal): %s", _e)

    # H3 enforcement: post-turn skill_creator reflection (deferred from
    # earlier in the function — audit 2026-06-10 I4). Out-of-band LLM
    # call that walks skill_creator's 3 gates against this turn's
    # trace, then optionally fires propose_skill (with merge-existing
    # support when a near-match is in the catalog). Gated tightly so
    # most turns skip — only fires when ≥3 distinct tools ran AND
    # the answer isn't a refusal/rewriter output AND no propose_skill
    # was already called this turn AND verifier confidence ≥ 50 AND
    # endpoint was met. See `_should_reflect_for_skill`. Supervisor
    # turns SKIP — internal plumbing, not workflows worth distilling.
    if not supervisor_mode:
        try:
            _post_turn_skill_reflection(
                agent, task, answer, speaker_id,
                verifier_confidence=vr.confidence,
                endpoint_was_met=_was_met,
            )
        except Exception as _e:
            log.warning("skill reflection top-level swallow: %s", _e)

    # propose_self_modification empty-diff check (audit 2026-05-28).
    # When the agent claims to have submitted a proposal but the
    # underlying record has no actual diff (old_code AND new_code
    # both empty), that's misleading — the agent reported success,
    # but the owner could approve a shell that won't apply anything.
    # Cap confidence and flag the contradiction so the WebUI / daily
    # report explains the dip.
    try:
        psm_in_trace = False
        for _step in (agent._trace or []):
            tc = getattr(_step, "tool_call", None)
            if tc is None:
                continue
            name = getattr(tc, "name", None)
            if name == "propose_self_modification":
                psm_in_trace = True
                break
        if psm_in_trace:
            from . import self_modifier as _smod
            empty_props: list[str] = []
            # Check the most-recently-created proposals — there's no
            # explicit linkage from the tool result back to the
            # proposal id without parsing the result JSON, so we
            # scan the latest 3 records (matches typical per-turn cap).
            for p in (_smod.SELF_MODIFIER._proposals or [])[-3:]:
                if not (p.old_code or "").strip() and \
                        not (p.new_code or "").strip():
                    empty_props.append(p.id)
            if empty_props:
                try:
                    vr.contradictions.append(
                        "empty_propose_self_modification: proposal(s) "
                        f"{empty_props} have no diff (old_code AND "
                        "new_code blank). The agent reported "
                        "'proposal submitted' but the record is a "
                        "description-only shell — approve will be a "
                        "no-op until a real diff is generated."
                    )
                except Exception:
                    pass
                # An empty-shell proposal is a DELIVERY failure (the
                # claimed action has no substance) — classify it with
                # the endpoint misses, preserve the content score.
                if vr.content_confidence is None:
                    vr.content_confidence = vr.confidence
                vr.endpoint_met = False
                vr.confidence = min(vr.confidence, 30)
    except Exception as exc:
        log.debug("unified: psm empty-diff check failed: %s", exc)

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
            "level": _level.name,
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
            # Trajectory memory: index this turn so future similar
            # tasks can retrieve the trajectory. Supervisor turns are
            # synthetic plumbing, not reusable experience. The module
            # applies its own quality gates (confidence, tool count).
            if not supervisor_mode:
                try:
                    from .trajectory_memory import index_turn
                    index_turn(turn_id, artifact)
                except Exception as e:
                    log.debug("trajectory index failed (non-fatal): %s", e)
                # Experience loop write side: a successful FRAMED turn
                # becomes a case note in the KB (structural, no LLM call).
                if _build_frame_state.get("framed"):
                    _auto_case_note(
                        task=task,
                        answer_head=answer or "",
                        tools_used=_turn_tool_names(agent),
                        confidence=int(vr.confidence or 0),
                        frame=_load_turn_frame(_run_started_at),
                        endpoint_met=vr.endpoint_met,
                    )
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
            content_confidence=vr.content_confidence,
            endpoint_met=vr.endpoint_met,
        ))
    except Exception:
        pass

    # Fine-tune auto-collection (AGI roadmap, 2026-06-12). High-trust
    # delivered turns become distillation data for the small local
    # model. The collector applies its own gates (confidence >= 85,
    # endpoint met, no contradictions, grounded in notes or tool
    # evidence) — most turns skip. Mirror image of the meta-learner
    # wire below: that one learns from failures, this one harvests
    # successes. Best-effort; never blocks the reply.
    if not supervisor_mode:
        try:
            from .finetune import collect_from_turn
            collect_from_turn(
                task=task,
                answer=answer or "",
                vr=vr,
                tool_names=_turn_tool_names(agent),
                is_chat=False,
                supervisor_mode=supervisor_mode,
                project=project,
            )
        except Exception as e:
            log.debug("finetune collection failed (non-fatal): %s", e)

        # Correction capture (AGI roadmap C, 2026-06-13). Reading the
        # real conversation history showed the gold signal — turns
        # where the human caught the agent wrong and it then fixed
        # itself — was lost: corrections score low confidence so the
        # collector above rejects them. This runs AFTER CONVERSATION.
        # add_turn (above) so recent(2) yields [prior, current]; an
        # LLM judge confirms the correction (no keyword matching).
        try:
            from .finetune import maybe_capture_correction
            maybe_capture_correction(
                is_chat=False,
                supervisor_mode=supervisor_mode,
                speaker_id=speaker_id,
                session_key=skey,
                project=project,
            )
        except Exception as e:
            log.debug("correction capture failed (non-fatal): %s", e)

    # Audit T3 follow-up (2026-05-27). The MetaLearner.analyze_failure
    # path was DEFINED but NEVER CALLED from anywhere — leaving the
    # whole learning loop dead since the unified-loop cutover. Wire
    # it here: on low-confidence turns, log the failure so
    # MetaLearner.stats() / extract_patterns() / FIRE_SELF_REFLECTION /
    # FIRE_GOAL_PROPOSE actually have material to work on. Best-effort
    # — never let analysis crash the user-facing reply.
    if vr.confidence < 60 and not supervisor_mode:
        try:
            MEMORY  # noqa: B018 — just ensures the module import succeeded
            from .meta_learner import META_LEARNER as _ML
            _ML.analyze_failure(
                question=task,
                answer=answer or "",
                verification=vr,
                intent="unified",
            )
        except Exception as exc:
            log.debug("unified: META_LEARNER.analyze_failure failed: %s", exc)

    # NOTE: `question_payload` and the `answer = "❓ …"` overwrite
    # for the AskUserQuestion path are resolved EARLIER (right after
    # `_record_llm_call`) so the question text propagates into
    # conversation history + saved turn artifact + session row —
    # not just the in-memory AgentAnswer returned below.

    # Honest model reporting: the model that ACTUALLY served this turn, and a
    # notice when the router silently fell back off the selected one.
    from .model_report import primary_model_used, fallback_note
    from .llm import turn_fallback_reason
    _model_used = primary_model_used(agent._llm_calls)
    try:
        from .providers import ACTIVE_MODEL as _AM
        _intended = (_AM.get() or {}).get("model", "")
    except Exception:
        _intended = ""
    _model_note = fallback_note(_intended, _model_used, turn_fallback_reason())
    # Turn-contract status, appended LAST — after the critic, after every
    # revision pass, immediately before the answer leaves. Nothing downstream
    # can edit or drop it, which is the point: the model must not get a chance
    # to soften the record of what it failed to prove.
    if not supervisor_mode:
        if _contract_status_tag:
            answer = _append_open_status(answer, _contract_status_tag)
        else:
            try:
                from . import turn_contract as _tc
                _block = _tc.render_user_block()
            except Exception:
                _block = ""
            if _block:
                answer = f"{(answer or '').rstrip()}\n\n{_block}"
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
        model_used=_model_used,
        model_note=_model_note,
    )
