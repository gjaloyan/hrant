---
name: skill_creator
description: Post-task review — decide if the workflow you just finished should become a reusable skill, and call propose_skill with a well-formed body. Load this AFTER finishing a non-trivial composed workflow that future turns could reuse.
triggers: []
tags: []
when_to_use: |
  Load this at the END of a turn, AFTER a non-trivial workflow has
  completed successfully, to decide whether the workflow is worth
  saving as a reusable skill. NOT a chat-time match — the LLM
  explicitly invokes `load_skill("skill_creator")` when it judges
  the just-finished work to be a candidate.

  Do NOT load this for: casual chat, recall questions, single-tool
  one-shots (`read_file`, `web_search` alone), failed turns,
  refusals, or turns where the work was already in an existing skill.
---

# Skill Creator — post-task workflow review

You just finished a workflow. This skill is your structured review:
should the workflow become a reusable skill for future turns?

## Gate — apply all three, in order. Bail at first NO.

### Gate 1. Was the task non-trivial?

**YES** if any of:
- 3+ distinct tools called in this turn (not three `read_file` calls
  on the same file — three DIFFERENT tools),
- a composed workflow happened (read → transform → write → verify),
- the work required research (`search_package`, `web_search`,
  `fetch_url`) plus tool execution to converge on the answer,
- you had to handle a non-obvious edge case (encoding fallback,
  format probing, retry path).

**NO** if any of:
- the turn was pure chat / recall / acknowledgement,
- one tool answered the question (single `read_file`,
  `search_knowledge`, `analyze_image`),
- the task was already covered by an existing skill — re-using a
  skill isn't a candidate for a NEW skill,
- the user just asked for a fact you produced from context.

If NO — STOP. Reply with "no skill needed" and end.

### Gate 2. Did the work succeed?

**YES** if:
- the deliverable was produced (file written to `outbox/`, fact
  computed, conversion done) AND
- you verified it (re-probed the output, checked exit codes,
  spot-checked content) — not just "the command returned 0".

**NO** if:
- the turn ended in a refusal or "I couldn't get there",
- the deliverable is suspect / unverified,
- you patched over an error with a fallback that may have lost
  information.

If NO — STOP. A skill that captures a flawed workflow is worse than
no skill. Reply with "no skill needed — work was not verified
good".

### Gate 3. Is the shape recurring?

**YES** if a future user message of the same SHAPE (different
specifics) would benefit from this exact workflow:
- "remove the logo from this video" — YES, generalises.
- "convert pdf to text" — YES, generalises.
- "extract tabular data from xlsx" — YES, generalises.

**NO** if the task is one-off:
- "open this very specific spreadsheet I just sent" — NO,
  next turn's spreadsheet won't share the same structure.
- "what does my CLAUDE.md say about X" — NO, that's recall.
- "compose a tweet about this specific event" — NO, the event
  is the whole task.

If NO — STOP. Reply with "no skill needed — task is one-off".

## All three gates passed → check for an EXISTING near-match first

Before composing a new skill, scan the CURRENT SKILL CATALOG (which
the post-turn reflection always includes in its system prompt).
Look for an existing skill whose `description` / `triggers` / `tags`
cover the same problem shape as the one you just solved.

If a near-match exists:
  1. Call `load_skill(name)` to read its body.
  2. Decide if the new turn taught a **better** approach (a tool
     sequence that's cleaner, a workflow step the old body missed,
     a pitfall the old body didn't warn about, a corrected command).
  3. If yes, compose a **merged** body that:
     - keeps the useful parts of the old body (don't throw away
       working steps just because they aren't yours),
     - adds the new patterns / pitfalls,
     - drops only what's genuinely wrong or superseded.
  4. Call `propose_skill(name=<same_name>, ...)` with the merged
     content. `upsert_user_skill` writes by name, so a propose with
     the SAME name OVERWRITES the user-tier copy. The new version
     is DISABLED until the owner explicitly approves the replacement.
  5. The DM the owner gets references the existing skill so they
     understand this is an improvement, not a fresh skill.

If no near-match exists, write a fresh skill as below.

## Writing the proposal

Call `propose_skill(name, description, triggers, when_to_use, body)`
with these constraints:

- `name`: kebab-case, descriptive, 2-4 words.
  - good: `video-overlay-remove`, `xlsx-table-extract`,
    `pdf-form-fill`, `audio-noise-clean`.
  - bad: `do-stuff`, `helper`, `task-12`.
- `description`: ONE sentence stating what the skill does and
  the input/output shape. ~80-120 chars.
- `triggers`: 3-6 short phrases that would appear in a user's
  message asking for this. Substring-match — pick distinctive
  ones, not generic words like "file" or "process".
- `tags`: 5-10 broader topical keywords (word-boundary matched).
  More forgiving than triggers — include synonyms and related
  domains. Where the skill needs binaries, also list them as
  tags so a future user typing "ffmpeg ..." pulls this skill.
- `when_to_use`: short paragraph; emphasis on when NOT to use.
  - CAUTION (2026-06-15): do NOT scope a skill so narrowly that it
    excludes dimensions a domain expert would call essential. A
    self-learned `crypto-technical-analysis` skill once said "do not
    use for news sentiment" and so produced price-only analyses that
    ignored a pending ETF — a wrong-feeling-right answer. Capture the
    COMPLETE professional method, not the one mechanical slice you
    happened to run this turn.
- `body`: the step-by-step. Be explicit:
  - the FULL methodology a professional would use, covering every
    essential dimension of the task (e.g. an asset-analysis skill
    must include a news/catalyst/regulatory step, not price alone),
  - exact tool names (`run_python`, not "Python"),
  - exact commands with concrete arguments (not "run ffmpeg
    with appropriate flags"),
  - the output verification step (re-probe / spot-check),
  - common pitfalls you actually hit this turn (e.g. "if
    `--map-root-user` is rejected, drop it — kernel feature").
- If the skill calls binaries (`ffmpeg`, `libreoffice`, etc.) or
  Python modules (`pypdf`, `pillow`), include them in
  `required_tools` so future turns see `⚠️ [NEEDS: ...]` if a
  host is missing them.

## Safety reminders

- `propose_skill` writes the skill **DISABLED**. The owner gets
  a Telegram DM with `[👀 Show] [✅ Activate] [❌ Delete]`. Don't
  expect the skill to be live this turn — that's correct, and
  it's what makes auto-creation safe.
- Don't propose a skill that already exists. Check `list_skills()`
  first; if there's a near-match, prefer to leave the existing
  one alone (the owner can edit it in the WebUI).
- Don't write a skill body that quotes private user data
  (names, emails, file paths under `/home/<user>/`). Generalise.
- One `propose_skill` call per turn at most.

## Output format

End your turn with EITHER:
- A short prose sentence "no skill needed — <reason>" if any
  gate failed, OR
- One `propose_skill(...)` call followed by a one-line
  confirmation: "Proposed skill `<name>` — owner will see the
  activation prompt in Telegram."
