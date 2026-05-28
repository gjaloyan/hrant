# English Agent + LLM-Only Intent Routing — Design

**Date:** 2026-05-28
**Status:** approved (design), pending implementation plan
**Sub-project:** 1 of 5 (see "Program context" below)

## Program context

This is the first sub-project of a larger initiative: make the Hrant agent
fully English and remove keyword/regex matching everywhere in favor of LLM
semantic decisions. The full program (each piece shipped and tested on its
own, in order):

1. **English agent + LLM-only intent routing** ← THIS SPEC
2. Remove keyword command parser from the interactive REPL (`commands.py`,
   `repl.py::handle_command`).
3. Replace verifier claim-detection regex (`verifier.py` `_IDENT_PATTERNS`,
   `_FALSE_ABSENCE_PATTERNS`; `endpoint_check.py` verb lists) with LLM
   judgment.
4. Language-agnostic search: drop stopword keyword lists
   (`knowledge_graph.py` `_QUERY_STOPWORDS`, `skills.py`
   `_SEMANTIC_STOPWORDS`) in favor of embedding-based retrieval.
5. Switch the prod agent's response-language directive to English.

**Hard constraint (user, 2026-05-28):** do NOT translate accumulated data —
core memory, user profiles, and knowledge-base notes stay in their original
language. The agent reads them as-is and answers in English (LLMs handle this
cross-lingually). Only persona/behavior and code become English.

## Goal

The agent always responds in English, and the intent decision
(chat / preference / task, plus task depth) is made solely by the LLM
intent classifier — no regex or keyword fast-path anywhere in the intent
path.

## Background: current architecture

`pipeline/intent.py::_classify_intent` already calls an LLM classifier
(`INTENT_CLASSIFIER_SYSTEM`) that returns `{chat, preference, task}` and is
the documented "source of truth." Before that call, several regex/keyword
fast-paths short-circuit it (a pure cost optimization that also patched
cases where the classifier was observed to misroute):

- `len(msg) > 300 → task` (length heuristic, not a keyword)
- arithmetic regex → task (classifier sometimes answered "2+2" from training
  data as chat)
- system-directive regex → task (classifier sometimes labelled "change voice"
  as preference → saved a fact, applied nothing)
- chitchat regex → chat
- profile-recall regex → chat (classifier sometimes routed "what's my color"
  to task → 4 LLM calls)

The regexes live in `agent.py` (`_ARITHMETIC_RE`, `_ARITHMETIC_WORDS_RE`,
`_ARITHMETIC_DIGIT_RE`, `_MICRO_ACK_RE`, `_CHITCHAT_RE`, `_PROFILE_RECALL_RE`,
`_DIRECTIVE_VERBS_RE`, `_SYSTEM_ATTRIBUTE_RE`, `_SELF_QUESTION_RE`,
`_SELF_ANALYSIS_HINT_RE`, `_DEEP_AGENT_HINT_RE`) with `_looks_like_*`
wrappers. Production callers: `pipeline/intent.py` and `agent.py`.

## Approach (chosen)

- **A1 — language switch:** a new config flag, read by the system-prompt
  assembler. No memory/profile/soul.md edits.
- **B1 — intent routing:** delete the regex fast-paths and the now-unused
  helpers; strengthen `INTENT_CLASSIFIER_SYSTEM` so the classifier reliably
  covers the cases the fast-paths patched; the classifier becomes the sole
  intent decider.

Rejected: B2 (fold intent into the main agent turn, no separate classifier)
— a much larger rewrite of the tiered pipeline; YAGNI for now.

## Components

### 1. Language switch (A1)

- Add `response_language: "en"` to config (`backend/config.py` + config.yaml
  default), with a typed accessor.
- The system-prompt assembler injects, at high priority, an instruction:
  *"Always respond in English, regardless of the language of the user's
  message. This overrides any soul-level rule about mirroring the user's
  language."*
- Placement must out-weight the soul's "mirror the user's language" line
  (end-of-prompt / explicit override, mirroring the existing LANGUAGE
  OVERRIDE block in `identity.py::preamble`).
- Default value is English; the flag exists so the behavior is configurable
  rather than hard-coded.

### 2. Remove keyword intent routing (B1)

- In `pipeline/intent.py::_classify_intent`, remove the arithmetic,
  system-directive, chitchat, and profile-recall fast-path branches. Keep the
  `len > 300 → task` length guard (not a keyword).
- Delete from `agent.py` the regex constants and `_looks_like_*` helpers that
  exist solely for intent routing, after confirming each has no remaining
  caller. Symbols whose only use was routing are removed; any with a
  non-routing caller are handled in §4.

### 3. Strengthen the classifier

In `INTENT_CLASSIFIER_SYSTEM` (`backend/prompts.py`), add explicit rules and
a few-shot block covering the cases the fast-paths used to patch:

- arithmetic / "compute this" → **task** (so the solver can run `calc` /
  `run_python` instead of answering from memory)
- system directive ("change voice to male", "switch model to X", "set
  language to Y") → **task** (apply via tools, not save as a preference)
- profile recall ("what is my favorite color", "do you remember my brother's
  name") → **chat** (answer from profile + recent context, no full pipeline)
- greetings / thanks / short acks → **chat**

Extend the classifier's JSON output with a `depth` field
(`"normal" | "deep"`) that replaces the `_DEEP_AGENT_HINT_RE` /
`_SELF_ANALYSIS_HINT_RE` keyword tier-selection. The orchestrator reads
`depth` to choose deep_agent vs task_mode for `task` intents.

Output schema: `{"intent": "chat|preference|task", "depth": "normal|deep",
"recall": true|false, "reason": "..."}`. `depth` is only meaningful for
`task`; `recall` is true when the turn is a profile-recall question (used in
§4 to skip memory extraction).

### 4. Decoupling (these uses are NOT plain routing)

- `_looks_like_profile_recall` is also used in `agent.py:1595` to **skip
  memory extraction** on recall turns (so recall answers don't get
  re-stored as duplicate facts). Replacement: the classifier's `recall`
  output field (see §3 schema) is threaded to the extraction step, which
  skips when `recall` is true — instead of calling the deleted regex.
- `_chat_fallback` (used when the classifier LLM is unavailable) currently
  keyword-matches the message. Replacement: when the LLM is down there is no
  semantics to use, so return a fixed generic reply / default to a safe
  chat response — no keyword guessing.

### 5. Data flow (after change)

```
user message
  → _classify_intent
       len>300 → task (depth from a cheap default or classifier)
       else → LLM classifier → {intent, depth, recall}
  → branch:
       chat        → fast_chat tier
       preference  → preference extractor (unchanged)
       task        → depth=="deep" ? deep_agent : task_mode
  → system prompt always carries the English-language instruction
  → memory extraction skipped when classifier marks the turn as recall
```

## Error handling

- Classifier `LLMError` propagates to `Agent.run` (unchanged), which chooses
  the graceful path. With fast-paths gone, a down classifier means no
  heuristic routing; the fallback returns a generic reply rather than
  keyword-guessing.
- Malformed classifier JSON → default `intent="task"`, `depth="normal"`
  (current behavior for unknown intent).

## Testing

- **Unit (`tests/`):** mock the classifier LLM; assert `_classify_intent`
  returns the right `{intent, depth}` for representative inputs — arithmetic,
  system directive, profile recall, greeting, stable fact, generic task — in
  both English and Russian inputs.
- **Regression cleanup:** remove/replace tests that asserted the deleted
  regex fast-paths or `_looks_like_*` helpers.
- **Language test:** with `response_language="en"`, a Russian user message
  yields an English answer (mock or live).
- **Prod smoke:** send EN and RU messages; confirm English replies and
  correct routing ("2+2" → task → calc; "привет"/"hi" → chat;
  "what's my favorite color" → chat).
- Cost note: every turn now incurs one classification call (acceptable —
  agent has free-work budget, no hard token limit).

## Out of scope (other sub-projects)

REPL command parser (#2), verifier claim-detection regex (#3), search
stopwords (#4), prod soul.md/profile/notes (#5, and per the hard constraint
the prod data is never translated — only the language directive changes).

## Risks

- The classifier may still misroute arithmetic or directives (the exact
  reason the fast-paths existed). Mitigation: explicit rules + few-shot in the
  prompt, plus the prod smoke test before relying on it. If misrouting
  persists, the fallback is to add classifier examples, not to reintroduce
  keyword gates.
- Latency: one extra classification call on messages that previously
  short-circuited. Accepted.
