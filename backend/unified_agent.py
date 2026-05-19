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

## Task Solver Process — execution first, explanation last

Treat every user task as an **execution request**, not a discussion
request. Do NOT lead with "I can't do this" / "tools are not
available" / "you need X library" — that's a limitation, and
limitations are reported AFTER attempting, never as the first reply.

**Bad opening:** "I cannot remove the logo because I don't have
video editing tools."

**Good opening:** "Я обработаю видео — извлеку кадр, найду рамку
логотипа, прогоню delogo через ffmpeg, проверю результат." (then
execute, then report.)

Walk these phases in order. Skip a phase only when its inputs are
clearly already in hand.

1. **Identify the final result + commit to a plan.** What does the
   user want as output — edited file, extracted text, generated PDF,
   fixed code, completed analysis? Before your first tool call this
   turn, commit to a ONE-SENTENCE plan stating WHAT you'll produce
   and HOW. Example: "Я обработаю видео — `preprocess_video` для
   кадра, найду рамку логотипа через `analyze_image`, прогоню ffmpeg
   `delogo` через `run_python`, верну файл через `MEDIA:`." The plan
   sentence is your own scaffolding — keep it short, make it visible,
   then execute. Hermes does this naturally; Hrant tends to skip it
   and drift mid-execution — make the plan explicit.

2. **Check available inputs.** Did the user attach a file? Is the
   sha256 visible in the message context? Did an earlier turn save
   anything to `outbox/`? Don't say "no file received" before
   walking AttachmentStore (sha refs in the current turn) and the
   recent conversation context. If a file is genuinely missing and
   the answer depends on it, then ask — not before.

3. **Try existing skills first.** Walk AVAILABLE SKILLS (the
   catalog block already in this prompt). Match by name /
   description / triggers / tags / `when_to_use`. If a skill fits,
   either let its auto-injected `## SKILL: <name>` block guide you
   or call `load_skill(name)` to pull a non-matched one. SEMANTIC
   SUGGESTIONS (if present) are second-pass candidates — load them
   if the trigger-matched ones don't fit.

4. **Universal fallback for unknowns.** If no skill applies and the
   task is non-trivial, call `load_skill("universal_resolver")` and
   walk its 7-phase workflow (understand → inventory → identify
   gaps → research → choose tools safely → test on a copy → solve
   and deliver). Don't reinvent.

5. **Execute, don't lecture.** When the path is clear, ACT. For a
   video-logo task: inspect the video → extract a frame → identify
   the logo region → run ffmpeg `delogo` / `boxblur` → re-probe the
   output → return the file via `MEDIA:/path`. Don't reply with
   "you could use ffmpeg" — DO use ffmpeg.

6. **Tools missing? Auto-install via the gate.** If a skill matches
   but requires a tool the host doesn't have, the AUTO-PROPOSED
   INSTALLS block in this prompt has already fired `propose_install`
   for it — tell the user to tap Approve in Telegram. If no AUTO-
   PROPOSED block is shown but you discover a gap during execution,
   call `propose_install(packages, manager, reason)` yourself. apt
   binaries are now supported (sudo -n apt-get install -y); pip /
   pipx / apt are all valid managers.

7. **Ask only when truly blocked.** Acceptable reasons to ask:
   (a) required file genuinely missing, (b) user's goal is ambiguous
   with multiple defensible interpretations, (c) action is
   destructive (delete / overwrite uncommitted / push to main),
   (d) needs credentials / external account access. Anything else —
   make the reasonable call and proceed.

8. **If you must report failure, show your work.** Format:
   - what you tried (tool names + arguments);
   - what failed (exit code / error message / verifier output);
   - what would unblock the next attempt (specific user input,
     install approval, scope narrowing).

   Bad: "I can't do this."
   Good: "I tried to open the .dwg with `read_file` (no DWG
   reader registered) and ran `search_package('libdwg')` — no
   official PyPI distro. Workaround: export to DXF on your side
   or approve `propose_install(['ezdxf'], 'pip')` and I'll convert."

### TSP operating limits (enforced — not guidelines)

- **Attempt bar — minimum 2 distinct tools before any refusal.**
  Before any "I can't do this" / "у меня нет тула" / "this isn't
  supported" / "среди инструментов нет" phrasing, your trace MUST
  show at least 2 *distinct* tool names (not two `read_file` calls
  on the same path) and at least one must be a real execution
  attempt, not just a state probe. A refusal-opener with <2 distinct
  tools is **automatically rewritten** by the bridge into a "what I
  tried / what's still needed" status — the user sees the rewrite,
  not your draft. That backstop exists so you recover gracefully;
  the safer path is to clear the attempt bar yourself.

- **Iteration budget — 30/50/20 split.** You have ~20 tool-call
  iterations per turn. Spend them:
  - first ~30% (≈ iterations 1–6): identify result + inspect inputs
    + skill match. If the path is clear after this, commit to it.
  - next ~50% (≈ iterations 7–16): execute the chosen path.
  - last ~20% (≈ iterations 17–20): verify the output + deliver
    via `MEDIA:`. Or, if execution stalled, write the structured
    failure report from phase 8.
  Past 70% of budget without execution progress → STOP probing and
  write the status report. The iteration-ceiling section below kicks
  in structurally if you ignore this.

- **Inspection cheatsheet — use the typed path, don't reinvent.**
  Phase 2 inspection has explicit per-file-type tools — see the
  "File types — which path handles which" section below. Quick
  index: video → `preprocess_video(sha)` (gives frame_shas +
  transcript); image → `analyze_image(sha, question)`; PDF / DOCX /
  text-like → `read_file(path)`; spreadsheet / archive → `run_python`
  with pandas / openpyxl / zipfile; unknown binary → `run_python`
  for head-bytes + file-magic probe before deciding the path.

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

  - Grant / revoke Telegram access (owner says "add my wife to
    trusted users", "give @lusine access", "remove X") →
    `grant_telegram_access(user_id, role, label)` or
    `revoke_telegram_access(user_id)`. ONE call, atomic across
    roles.json AND channels.json — never poke those files via
    terminal_exec yourself. Approve a pending pairing request
    (an unknown user wrote and we DMed you a code) →
    `approve_pairing(code_or_user_id, label)`. List who's waiting
    → `list_pending_pairings`. "Who has access?" / "who are my
    trusted users?" / "кто может писать боту" → `list_telegram_access`
    (NOT terminal_exec on roles.json).

  - Owner-only shell inspection (status, logs, file content) →
    `terminal_exec`. NOT a substitute for `set_setting` when a
    setting exists.

  - Multi-step research / code-review → `delegate(role, task)`
    to a specialised subagent (researcher / coder / reviewer).

  - Self-mod (structural code changes the user requested) →
    `propose_self_modification(description, files, rationale)`.
    **For small bug-fixes / one-line patches / config flag changes —
    DO NOT** wrap them in `propose_self_modification`. Just write the
    file directly via `run_python` (`open(path, "w").write(...)` or
    `pathlib.Path(path).write_text(...)`) or `terminal_exec` with
    `sed -i` / `cat > file` / heredoc. The PSM tool is for big
    architectural changes (multi-file refactors, new modules,
    cross-cutting redesigns) — using it for "add one flag to
    ApplicationBuilder" is friction theatre, not safety. The
    May 19 button-bug incident wasted 2 hours of user time because
    the agent kept refusing to write a one-line fix, citing PSM
    ceremony it didn't actually need.

## Diagnose runtime bugs from the journal FIRST

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
code, not the other way round.

## Skills come BEFORE ad-hoc tool loops

If the AVAILABLE SKILLS block lists a skill whose `description` /
`when_to_use` matches the task, FOLLOW THAT SKILL. Trigger-matched
skills already have their full instructions injected into this
prompt as a `## SKILL: <name>` block — read that block carefully,
then act on it.

If no trigger fired but the catalog hints a relevant skill, call
`load_skill(name)` to pull its body BEFORE attempting the work.
Skills capture pitfalls (ffmpeg flags, Telegram cache paths, edge
cases) that ad-hoc tool loops have to rediscover painfully.

### Universal fallback — unknown file / unknown task

If the catalogue is silent on the user's request — unknown file
format, unfamiliar software, conversion / extraction you've never
done, anything that makes you tempted to reply "I can't do this" —
**do NOT refuse**. Instead, call `load_skill("universal_resolver")`
and follow its 7-phase workflow: understand → inventory → identify
gaps → research → choose tools safely → test on a copy → solve
and deliver, then `propose_skill(...)` to capture what worked.

The resolver explicitly forbids two failure modes that have hit
us in production: (1) burning the iteration budget on probing
without a plan, and (2) running `pip install` / `apt install` via
terminal_exec. Both are guarded structurally:
  - Iteration ceiling section above kicks in before XML-dump.
  - Package installs go through `propose_install(packages, manager,
    reason)` — owner approves via Telegram inline buttons, and only
    then does the install actually run. terminal_exec REFUSES any
    `pip install` / `pipx inject` / `apt install` / `npm install` /
    `cargo install` etc. with a hint pointing at `propose_install`.

When you finish a non-trivial task that involved a sequence of
tool calls and produced a working result, do a post-task review:
call `load_skill("skill_creator")` and follow its 3-gate checklist
(non-trivial + verified-good + recurring shape). If all three
gates pass, call `propose_skill(...)`; otherwise reply "no skill
needed" and end. This keeps the catalogue clean — one-shot tasks
and recall don't pollute it, but real composed workflows get
captured for future reuse.

## Refusals must be honest

NEVER say "tools are disabled" / "инструменты отключены" / "I
can't apply" when tools are listed above. The tools listed ARE
available this turn — refusal is only valid when:
  1. The setting / file / API genuinely doesn't exist, AND
  2. You've tried at least one tool to verify, AND
  3. You explain WHAT'S missing and offer a concrete next step.

If a tool call failed, try a DIFFERENT tool — don't surrender.

## Iteration ceiling

You have a fixed budget of tool-call iterations per turn. When you
sense you're approaching it without a working result, STOP probing
and write a plain-language status report: what was tried, what's
missing, what concrete input from the user would unblock the next
attempt. NEVER output `<tool_call name="...">` XML in the final
answer — that's a runtime artefact, not a tool we support, and the
user sees it as broken output. If you need to make another tool
call, make it as a native tool-use; if you can't, just describe
what you would have done in plain text.

## Sending files back (MEDIA: convention)

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

  MEDIA:/home/hrant/.hrant/data/workspace/outbox/clip_no_logo.mp4

## File types — which path handles which

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

To DELIVER any file back to the user, write a MEDIA: line — that
section above covers it. The right Telegram bubble (video / photo /
audio / document) is picked from the extension; everything else
falls back to a document attachment.

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


# TSP-2: detect refusal-without-attempts.
#
# Patterns harvested from real production turns (workspace/turns/),
# e.g. "Gor, я не могу выполнить этот запрос", "среди доступных мне
# сейчас инструментов нет Telegram send_voice", "Я не могу здесь
# реально 'услышать' отправленное аудио". Each of those answers opened
# with a refusal phrase AND made <2 distinct tool calls. The TSP rule
# says: do not refuse before walking inspect → skill-match → universal
# fallback → at least one real execution attempt. This regex is the
# tripwire that fires the rewriter when the rule is violated.
_REFUSAL_OPENER_RE = re.compile(
    r"(?:"
    # Russian opens
    r"я\s+не\s+могу|"
    r"не\s+могу\s+(?:выполнить|сделать|обработать|помочь|это|такое)|"
    r"у\s+меня\s+нет\s+(?:доступа|инструмент|возможност|тул)|"
    r"среди\s+(?:доступных\s+(?:мне\s+)?)?(?:сейчас\s+)?инструмент\w*\s+нет|"
    r"инструмент\w*\s+(?:не\s+)?(?:отключ|недоступ|отсутству)|"
    r"невозможно\s+(?:выполнить|сделать|обработать)|"
    r"я\s+не\s+поддержива|"
    r"это\s+не\s+поддержива|"
    r"к\s+сожалени[юе],?\s+(?:я\s+)?не\s+мог|"
    # English opens
    r"\bi\s+(?:can(?:not|'?t)|am\s+(?:not\s+able|unable))\b|"
    r"\bi'?m\s+(?:not\s+able|unable)\b|"
    r"\bi\s+(?:don'?t|do\s+not)\s+have\s+(?:access|the\s+tool|a\s+tool)|"
    r"tools?\s+(?:are\s+)?(?:not\s+available|disabled|off|unavailable)|"
    r"this\s+(?:isn'?t|is\s+not)\s+supported|"
    r"unable\s+to\s+(?:do|perform|execute|complete)|"
    r"sorry,?\s+(?:i\s+)?(?:can(?:not|'?t)|don'?t\s+have)"
    r")",
    re.IGNORECASE | re.UNICODE,
)


# C1: policy / privacy / recall refusals are LEGITIMATE — they're not
# capability gaps and the rewriter must not touch them. Without this
# exclude the rewriter would mangle "I can't share private phone
# numbers" or "не могу показать user.md гостю" into "you didn't try
# the TSP" messages, which would be a lie. If any of these keywords
# appears in the same window where the refusal opener fires, leave
# the answer alone.
_POLICY_REFUSAL_KEYWORDS_RE = re.compile(
    r"(?:"
    # Russian — privacy / policy / recall / guest gating
    r"приват|конфиденциальн|"
    r"личн\w+\s+данн|"
    r"персональн\w+\s+данн|"
    r"чужи\w*\s+данн|"
    r"чужо\w*\s+(?:телефон|почт|email|номер|адрес)|"
    r"посторонн|"
    r"гостев|гостю|гостям|гостях|"
    r"закрыт(?:ый|ая|ое|ые)\s+(?:доступ|канал|чат|инфо)|"
    r"раскрыт(?:ь|ие)\s+(?:личн|персональн|конфиденц)|"
    r"в\s+моей\s+(?:памяти|базе|notes)\s+нет|"
    r"у\s+меня\s+нет\s+(?:верифицированн|проверенн|подтвержденн)|"
    # English — same axis
    r"\bprivate\s+(?:phone|number|email|address|data|info|details)|"
    r"\bpersonal\s+(?:data|info|details|information)|"
    r"\bsomeone(?:'s|s)?\s+(?:phone|number|email|address)|"
    r"\bthird(?:[\s-]+)party(?:'s)?\s+(?:phone|number|email|data|info)|"
    r"\bsensitive\s+(?:data|info|information)|"
    r"\bconfidential|"
    r"\bguest\s+(?:user|account|access)|"
    r"\bi\s+(?:don'?t|do\s+not)\s+have\s+(?:verified|reliable|trustworthy)|"
    r"\bno\s+verified\s+(?:public\s+)?(?:info|information)|"
    # Safety / abuse refusals
    r"\b(?:abuse|harassment|stalking|doxx?ing)\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _is_policy_refusal(text: str) -> bool:
    """Return True if the refusal-opener window also contains a
    policy / privacy / recall keyword that marks the refusal as
    legitimate (not a capability gap). Used by the rewriter to
    exclude refusals it should leave alone."""
    if not text:
        return False
    return bool(_POLICY_REFUSAL_KEYWORDS_RE.search(text))


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


# The minimum "I actually tried" bar before a refusal is allowed.
# Two distinct tool names — not two `read_file` calls on the same
# missing file, but two genuinely different probes / attempts.
REFUSAL_ATTEMPT_BAR = 2


# I2: cap on the number of auto-fired install proposals per turn.
# Without this, a skill with N missing required_tools would generate
# N separate Telegram DMs to the owner in a single burst. Five is the
# pragmatic ceiling: enough for any one skill's reasonable dep set,
# small enough not to be a DM-flood. Remaining missing tools surface
# as a "deferred" line so the LLM and the user still see them.
AUTO_PROPOSE_CAP = 5


def _rewrite_refusal_without_attempts(answer: str, agent) -> str:
    """If the final answer opens with a refusal pattern AND the trace
    shows fewer than `REFUSAL_ATTEMPT_BAR` distinct tool calls,
    rewrite into a 'what I tried / what's still needed' status with
    a TSP-anchored reset.

    Conservative on purpose: only fires when the agent literally
    refused without trying. A refusal AFTER 2+ distinct tools
    counts as a TSP-compliant honest report and is left untouched.

    Skips:
      - empty / non-string answers,
      - answers that don't open with a known refusal pattern,
      - answers where ≥ `REFUSAL_ATTEMPT_BAR` distinct tools ran.
    """
    if not isinstance(answer, str):
        return ""
    if not answer:
        return ""

    # Detect on the first 300 chars — refusals lead with the pattern;
    # checking the whole answer risks false positives in proper
    # status reports that quote a refusal example.
    head = answer[:300]
    if not _REFUSAL_OPENER_RE.search(head):
        return answer

    # C1: policy / privacy / recall refusals are NOT capability gaps
    # and must not be rewritten. Examples from real prod turns that
    # this guard protects:
    #   - "не могу показать `user.md` гостевому пользователю" (privacy)
    #   - "I can't help find or provide a private person's phone" (privacy)
    #   - "I don't have verified information about X" (recall)
    # We widen the window to 500 chars here because policy markers
    # often follow the opener (e.g. "Я не могу выполнить — это работа
    # с персональными данными").
    if _is_policy_refusal(answer[:500]):
        return answer

    n_distinct, names = _count_distinct_tools_called(agent)
    if n_distinct >= REFUSAL_ATTEMPT_BAR:
        return answer

    tools_str = ", ".join(sorted(names)) or "(none)"
    excerpt = head.strip().replace("\n", " ")[:160]
    if _is_russian_dominant(answer):
        return (
            f"⚠️ Я открыл ответ отказом, не пройдя Task Solver Process. "
            f"Перезапускаю.\n\n"
            f"Что я реально попробовал в этом турне: **{tools_str}** "
            f"(порог по TSP — минимум {REFUSAL_ATTEMPT_BAR} разных тулов "
            f"до фразы «не могу»).\n\n"
            f"По TSP до отказа я должен был:\n"
            f"  1. Определить конкретный output, который ты ждёшь.\n"
            f"  2. Проверить attachments / sha-ссылки / workspace.\n"
            f"  3. Пройти AVAILABLE SKILLS + SEMANTIC SUGGESTIONS, "
            f"`load_skill(name)` для подходящего.\n"
            f"  4. Если ничего не подошло — `load_skill(\"universal_resolver\")` "
            f"и его 7 фаз.\n"
            f"  5. И только тогда отчитаться форматом «что попробовал / "
            f"что упало / что нужно дальше».\n\n"
            f"Чтобы продолжить, выбери одно:\n"
            f"  • повтори запрос — я попробую execute сразу;\n"
            f"  • или скажи «посоветуй» / «объясни» — отвечу теорией "
            f"без выполнения;\n"
            f"  • или укажи что именно блокирует (отсутствует файл, "
            f"нужен пароль, выбор между вариантами).\n\n"
            f"_(Мой исходный отказ: «{excerpt}…»)_"
        )
    return (
        f"⚠️ I opened with a refusal without walking the Task Solver "
        f"Process. Resetting.\n\n"
        f"Tools I actually called this turn: **{tools_str}** "
        f"(TSP bar — at least {REFUSAL_ATTEMPT_BAR} distinct tools "
        f"before any \"I can't\").\n\n"
        f"Per TSP, before refusing I should have:\n"
        f"  1. Identified the concrete output you want.\n"
        f"  2. Inspected attachments / sha refs / workspace state.\n"
        f"  3. Walked AVAILABLE SKILLS + SEMANTIC SUGGESTIONS and "
        f"`load_skill(name)` for the best fit.\n"
        f"  4. Fallen back to `load_skill(\"universal_resolver\")` if "
        f"nothing matched.\n"
        f"  5. Only then reported \"tried X, failed at Y, need Z\".\n\n"
        f"To unblock me:\n"
        f"  • repeat the request — I'll execute on this next turn;\n"
        f"  • or say \"explain\" / \"advise\" — I'll answer with "
        f"theory and skip execution;\n"
        f"  • or name the specific blocker (missing file, password, "
        f"choice between outputs).\n\n"
        f"_(Original refusal: \"{excerpt}…\")_"
    )


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
) -> AgentAnswer:
    """Execute one unified-loop turn. Called from `Agent.run` when
    the env flag is set. `agent` is the calling Agent instance —
    we pull `progress`, `_record_llm_call`, `_trace`, etc. from it
    so the dev panel / SSE stream still receives all the same
    events the legacy pipeline emitted.

    `session_key` isolates conversation threads — Wife in a group
    chat and Wife in a DM share `speaker_id` (identity / roles) but
    get distinct session_keys (thread). When unset, falls back to
    speaker_id (one thread per speaker — the WebUI's behaviour)."""
    skey = (session_key or "").strip() or speaker_id
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
    # Per-thread — same person in two chats grows long history
    # independently.
    try:
        _cc.maybe_compact(speaker_id=speaker_id, session_key=skey)
    except Exception as e:
        log.debug("unified: compaction failed (non-fatal): %s", e)

    # Pre-flight 2: sticky-request detection (renders into prompt).
    # Sticky detection reads RECENT turns to catch repeat asks — must
    # be per-thread, not per-speaker, or a separate chat resurrects
    # stale "you didn't do X" signals from a completely different
    # conversation.
    sticky_info = _sticky.detect_sticky_request(
        current_user_message=task, speaker_id=speaker_id, session_key=skey,
    )
    sticky_block = _sticky.render_sticky_block(sticky_info)

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

    # H1-rev: auto-propose installs for missing required_tools.
    #
    # When a matched skill declares dependencies that aren't on this
    # host, fire `installer.propose` automatically — the owner gets a
    # Telegram DM with [Show][Approve][Reject] for each missing tool.
    # The LLM doesn't have to remember to call propose_install itself,
    # and once the owner taps Approve the install runs and the NEXT
    # turn will see the tool available.
    #
    # Dedup: within a turn, only propose each (name, manager) once.
    # Across turns, `installer.has_pending` skips if there's already a
    # pending request waiting for the owner — no spam DMs.
    #
    # I2: cap at AUTO_PROPOSE_CAP per turn. Without this, a skill with
    # 10 missing tools would fire 10 separate Telegram DMs in a single
    # message burst. After the cap, remaining tools surface as a
    # "deferred" line in the system block so the LLM still knows
    # they're needed.
    #
    # I3: the `requester` field always carries an "(auto-skill-match)"
    # suffix so journal readers can tell auto-fired requests from
    # explicit `propose_install` calls. The owner-only approval gate
    # is still in installer.py — this is just journal hygiene so a
    # guest who triggers a skill match doesn't end up "owning" a
    # request in the audit trail.
    #
    # Only fires on `matched_skills` (trigger/tag hit). Semantic
    # suggestions don't auto-propose — fuzzy match isn't strong enough
    # signal to commit the owner to an install.
    auto_proposed: list[dict] = []
    auto_propose_deferred: int = 0
    if matched_skills:
        from . import installer as _installer
        seen_pairs: set[tuple[str, str]] = set()
        proposes_fired = 0
        requester_label = (
            f"{speaker_id} (auto-skill-match)"
            if speaker_id else "auto:skill-match"
        )
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
                # I2 cap: stop firing new propose calls past the limit.
                # Already-pending entries still get listed (no DM is
                # sent for those) so the LLM sees the full state.
                already_pending = _installer.has_pending([pkg_name], mgr)
                if not already_pending and proposes_fired >= AUTO_PROPOSE_CAP:
                    auto_propose_deferred += 1
                    continue
                if already_pending:
                    auto_proposed.append({
                        "name": pkg_name, "manager": mgr,
                        "skill": sk.name, "code": None,
                        "status": "already-pending",
                    })
                    continue
                try:
                    req = _installer.propose(
                        packages=[pkg_name],
                        manager=mgr,
                        reason=f"auto-proposed: required by skill {sk.name}",
                        requester=requester_label,
                    )
                except Exception as e:
                    log.warning(
                        "auto-propose install failed for %s (%s): %s",
                        pkg_name, mgr, e,
                    )
                    auto_proposed.append({
                        "name": pkg_name, "manager": mgr,
                        "skill": sk.name, "code": None,
                        "status": f"propose-error: {type(e).__name__}",
                    })
                    continue
                proposes_fired += 1
                if req is None:
                    auto_proposed.append({
                        "name": pkg_name, "manager": mgr,
                        "skill": sk.name, "code": None,
                        "status": "propose-returned-none",
                    })
                else:
                    auto_proposed.append({
                        "name": pkg_name, "manager": mgr,
                        "skill": sk.name, "code": req.code,
                        "status": "proposed",
                    })
        if auto_proposed or auto_propose_deferred:
            agent.progress(
                "install",
                "auto-proposed: " + ", ".join(
                    f"{p['name']}({p['manager']}/{p['status']})"
                    for p in auto_proposed
                ) + (
                    f"; deferred={auto_propose_deferred} (cap={AUTO_PROPOSE_CAP})"
                    if auto_propose_deferred else ""
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

    # Skill catalog (cheap, ~one line per skill) + full body of each
    # auto-matched skill (richer, but only on actual trigger hits).
    try:
        catalog = _SKILLS.catalog_block()
    except Exception:
        catalog = ""

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
    if auto_proposed or auto_propose_deferred:
        # H1-rev: tell the LLM what auto-installs are in flight. This
        # is a hint, not a command — the model decides whether to call
        # the dependent skill anyway (might pivot) and whether to tell
        # the user to tap Approve in Telegram.
        ap_lines = [
            "# AUTO-PROPOSED INSTALLS",
            "(Skills matched this turn need tools that aren't installed. "
            "I've auto-fired `propose_install` for each one — the owner "
            "gets a Telegram DM with [Show][Approve][Reject] per request. "
            "Those tools will NOT be available this turn — either tell "
            "the user to tap Approve and retry next turn, or pivot to an "
            "approach that doesn't need them.)",
        ]
        for p in auto_proposed:
            tag = p.get("status") or "?"
            line = f"- `{p['name']}` via `{p['manager']}` (skill `{p['skill']}`) — {tag}"
            if p.get("code"):
                line += f" — request code `{p['code']}`"
            ap_lines.append(line)
        if auto_propose_deferred:
            # I2: per-turn cap exceeded — tell the LLM how many tools
            # still need attention so it can ask the user about a
            # batched approval instead of expecting all of them.
            ap_lines.append(
                f"- _…and {auto_propose_deferred} more missing tool(s) "
                f"deferred (per-turn cap = {AUTO_PROPOSE_CAP}). Ask the "
                f"user to approve the first batch in Telegram, then "
                f"trigger the skill again to surface the rest._"
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
    system_parts.append(f"---\n\n{_UNIFIED_RULES}")
    system_parts.append(f"---\n\n{perms}")
    system_prompt = "\n\n".join(system_parts)

    # Tool surface = whole registry. The LLM picks.
    # MUST be after SKILLS.ensure_loaded() above so skill-provided
    # tools (handler.py register) make it into the schema.
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
    # TSP-2: catch the "I can't do this" / "у меня нет инструмента"
    # opener when the agent literally didn't try (< 2 distinct tools).
    # Same shape as the XML-dump rewriter — surface the violation in
    # plain language with a recovery prompt instead of letting a
    # bare refusal reach the user.
    answer = _rewrite_refusal_without_attempts(answer, agent)

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
            "n_llm_calls": len(agent._llm_calls or []),
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
    )
