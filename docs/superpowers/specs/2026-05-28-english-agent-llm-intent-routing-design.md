# English Agent + Remove Dead Legacy Intent Routing — Design

**Date:** 2026-05-28
**Status:** approved (design, revised), pending implementation plan
**Sub-project:** 1 of 5 (see "Program context")

> **REVISED 2026-05-28:** The original design assumed keyword intent
> routing was live and planned to replace it with a strengthened LLM
> classifier. Exploration during planning showed the production agent
> already runs the **unified single-tool-loop** path (`run_unified`),
> where the LLM sees all tools and decides everything — there is NO
> separate intent classifier in the live path. The legacy classifier /
> regex routers still exist as **dead code** (kept "for
> reverter-friendliness", `agent.py:1727-1736`) but `Agent.run` never
> calls them. So "remove keyword intent routing" = **delete that dead
> legacy code**, not build a classifier.

## Program context

First sub-project of the initiative to make Hrant fully English and
remove keyword/regex matching in favor of LLM semantic decisions. Full
program (each piece shipped + tested on its own, in order):

1. **English agent + remove dead legacy intent routing** ← THIS SPEC
2. Remove keyword command parser from the interactive REPL
   (`commands.py`, `repl.py::handle_command`).
3. Replace verifier claim-detection regex (`verifier.py`,
   `endpoint_check.py`) with LLM judgment.
4. Language-agnostic search: drop stopword keyword lists
   (`knowledge_graph.py`, `skills.py`) for embedding retrieval.
5. Switch the prod agent's response-language directive to English.

**Hard constraint (user, 2026-05-28):** do NOT translate accumulated
data — core memory, user profiles, knowledge-base notes stay in their
original language. The agent reads them as-is and answers in English
(LLMs handle this cross-lingually). Only persona/behavior and code
become English.

## Current architecture (verified)

`Agent.run` (`backend/agent.py:1736-1750`) unconditionally sets
`self._mode = "unified"` and returns `unified_agent.run_unified(...)`.
The unified turn is a single `router.call_with_tools(...)` loop: the
LLM sees all tools and decides chat vs tool-use vs `save_user_fact` vs
refuse, etc. There is no intent classification, no preference branch,
no pipeline-tier selection, no regex routing in this path.

The legacy pipeline is **dead but still present**:
- `Agent` still inherits `IntentClassifierMixin` (`pipeline/intent.py`),
  `PreferenceHandlerMixin` (`pipeline/preferences.py`),
  `ThinkingMixin` (`pipeline/thinking.py`) — wired at
  `agent.py:739-748`.
- `agent.py` still defines the routing regexes (`_ARITHMETIC_*`,
  `_CHITCHAT_RE`, `_MICRO_ACK_RE`, `_PROFILE_RECALL_RE`,
  `_DIRECTIVE_VERBS_RE`, `_SYSTEM_ATTRIBUTE_RE`, `_SELF_QUESTION_RE`,
  `_SELF_ANALYSIS_HINT_RE`, `_DEEP_AGENT_HINT_RE`) + their
  `_looks_like_*` helpers + `_pick_pipeline_mode` + `_chat_fallback`.
- `prompts.py` still defines `INTENT_CLASSIFIER_SYSTEM` and
  `PREFERENCE_EXTRACTOR_SYSTEM`.
- None are reached from `run_unified`. The `agent.py:1727` comment
  explicitly schedules them for a "later cleanup sprint".

`SelfCriticMixin` (`pipeline/critic.py`, `_verify`) is **live** — the
unified path runs the verifier post-hoc. It is NOT in scope here
(its regexes are sub-project #3).

## Goal

The agent always responds in English, and the dead legacy
keyword/regex intent-routing code is removed — finishing the cleanup
the codebase already anticipates, so the LLM-decides architecture is
the only code that exists, not just the only code that runs.

## Approach (chosen, revised)

- **A1 — language switch:** new config flag `response_language`, read
  by `IdentityManager.preamble()` (the single system-prompt block all
  `run_unified` call sites use). Emits a high-priority English
  instruction that overrides the soul's "mirror the user's language"
  line. No memory/profile/soul.md edits.
- **B — delete dead legacy routing:** remove the unused mixins, regex
  constants, helper functions, and dead prompt strings — each gated on
  a grep proving no live caller remains.

## Components

### 1. Language switch (A1)

- Add a `response_language` property to `Config` (`backend/config.py`),
  reading `self._data.get("response_language")` and defaulting to
  `"en"`. Empty string / `"mirror"` means "mirror the user's language"
  (legacy behavior) for future flexibility.
- In `IdentityManager.preamble()` (`backend/identity.py`): when
  `CONFIG.response_language` resolves to a concrete language (e.g.
  `"en"`), append a final, highest-priority block:
  *"# RESPONSE LANGUAGE — respond ONLY in English, regardless of the
  language of the user's message. This overrides any soul-level rule
  about mirroring the user's language and any profile language pin."*
  When the flag is `mirror`/empty, preamble behaves exactly as today
  (including the existing profile LANGUAGE OVERRIDE block).
- Because the config flag is authoritative, when it is set the
  profile-derived LANGUAGE OVERRIDE is skipped to avoid a conflicting
  directive.

### 2. Delete dead legacy intent/preference/thinking routing

After confirming (via grep) that `run_unified` and all live callers do
not reference them:
- Remove `IntentClassifierMixin`, `PreferenceHandlerMixin`,
  `ThinkingMixin` from the `Agent` base list (`agent.py:739-748`) and
  delete the files `pipeline/intent.py`, `pipeline/preferences.py`,
  `pipeline/thinking.py`.
- Delete from `agent.py`: `_pick_pipeline_mode`, `_chat_fallback`, the
  routing regex constants and their `_looks_like_*` helpers, and the
  now-unused `PIPELINE_*` tier constants, after grep confirms no live
  use.
- Delete `INTENT_CLASSIFIER_SYSTEM` and `PREFERENCE_EXTRACTOR_SYSTEM`
  from `prompts.py` (and their `__all__` entries) once their only
  consumers (the deleted mixins) are gone.
- Update `pipeline/__init__.py` docstring + any `pipeline_profile` or
  cheatsheet references that name the removed mixins.

### 3. Keep (explicitly out of deletion)

`pipeline/critic.py` (`SelfCriticMixin._verify`) — live in the unified
post-hoc step; untouched here. `memory_extractor`, verifier, KG,
goals, autonomic — all live; untouched.

## Data flow (after change)

Unchanged from today's live behavior (single unified tool-loop). The
only runtime difference: every assembled system prompt now carries the
English-language directive. The deletions remove code that `run()`
already never executed, so live behavior is otherwise identical.

## Error handling

No new runtime paths. Deletions are pure removal of unreachable code;
the safety check is "tests + a full import still pass", not new
error handling. The language block is unconditional string assembly
in `preamble()` with no new failure mode.

## Testing

- **Language unit test:** with `CONFIG.response_language = "en"`,
  `IDENTITY.preamble()` contains the English-response directive; with
  `response_language = "mirror"`, it does not (and the legacy
  profile-language behavior is unchanged).
- **Dead-code removal safety:** a grep/import test — after deletion,
  `import backend` and `import backend.agent` succeed, and there are
  zero references to the deleted symbols (`_classify_intent`,
  `_pick_pipeline_mode`, `INTENT_CLASSIFIER_SYSTEM`, etc.) outside
  comments/specs.
- **Regression cleanup:** delete or rewrite tests that exercised the
  legacy classifier / regex routers / `_save_preference` /
  `_chat_fallback` — they test code that no longer exists. (These are
  legacy-path tests, not unified-path tests.)
- **Full suite:** `pytest -q` green (modulo the known Windows-timing
  flakies).
- **Prod smoke:** send an English and a Russian message; both get an
  English reply; a tool-requiring request ("run X") still triggers a
  tool call (confirming the unified loop is unaffected).

## Out of scope (other sub-projects)

REPL command parser (#2), verifier claim-detection regex (#3), search
stopwords (#4), prod soul.md/profile/notes (#5 — and per the hard
constraint prod data is never translated, only the language directive
is switched, which #1 already delivers via the config flag).

## Risks

- **Hidden live reference to "dead" code.** Mitigation: every deletion
  task greps for callers first; the import + full-suite tests catch any
  missed reference before commit.
- **Language directive not strong enough** to override a Russian soul
  line on prod. Mitigation: place it last with explicit "this
  overrides soul + profile" wording; verify with the prod smoke test
  (Russian input → English output).
- Negligible runtime risk otherwise — the change is a config-driven
  prompt line plus removal of unreachable code.
