# Self-Learning Agent — Architecture

This document describes how the agent works end-to-end. All diagrams
are [Mermaid](https://mermaid.js.org/) — they render natively on GitHub.
For local viewing, paste any block into the [Mermaid Live Editor](https://mermaid.live).

> **Source of truth.** When this doc and the code disagree, the code wins.
> Treat this as a map, not a contract — keep it in sync when you ship
> structural changes (new pipeline branch, new memory store, new LLM call site).

---

## 1. Overview

The agent is a single-process Python service that turns a user message
into a verified answer, while learning from every turn. It exposes:

- A **WebUI** (React + FastAPI SSE chat).
- A **Telegram bot** (long-poll, real-time progress streaming).
- A **REST API** under `/api/*` for status, attachments, knowledge,
  goals, sessions, providers, etc.

Reasoning lives in a **dual-model router** (Claude as model A,
Qwen / Ollama as model B). Memory is layered: **CORE** (always-on),
**KNOWLEDGE GRAPH** + **NOTES** (semantic + structured), **CONVERSATION**
(short-term sliding window), **ATTACHMENTS** (sha-deduped images / voice
/ files).

```mermaid
flowchart LR
    subgraph IO["I/O"]
        WUI["WebUI<br/>(React + Vite)"]
        TG["Telegram bot<br/>(long-poll)"]
        API["REST + SSE<br/>(FastAPI)"]
    end

    subgraph CORE["Reasoning core"]
        AG["Agent.run()"]
        LR["DualModelRouter"]
        VRF["Verifier"]
        SC["Self-critic loop"]
    end

    subgraph MEM["Memory layers"]
        CM["CORE memory"]
        ID["Identity<br/>(soul / identity / user.md)"]
        KB["Notes (KB)"]
        KG["Knowledge graph"]
        VS["Vector store"]
        CONV["Conversation"]
        MEX["Memory facts<br/>(graph + extractor)"]
    end

    subgraph LEARN["Feedback loops"]
        ML["Meta-learner"]
        GOALS["Goals"]
        EVAL["Evaluator"]
        FT["Fine-tune queue"]
    end

    WUI --> API
    TG --> API
    API --> AG
    AG --> LR
    AG --> VRF
    VRF --> SC
    AG --> MEM
    AG --> LEARN
    LR --> CORE
```

---

## 2. Request lifecycle (`Agent.run`)

Every chat message — WebUI or Telegram — runs through the same pipeline.
After classification it splits into one of three branches: `chat`,
`preference`, or `task`. Only the `task` branch goes through the full
think → solve → verify cycle.

```mermaid
flowchart TD
    START(["Message arrives<br/>(text + optional attachments)"])
    CORE_LOAD["_load_core()<br/>read knowledge/core_memory.md"]
    CLASSIFY["_classify_intent()"]

    START --> CORE_LOAD --> CLASSIFY

    CLASSIFY -->|"arithmetic regex hit"| TASK_BRANCH
    CLASSIFY -->|"chitchat regex"| CHAT_BRANCH
    CLASSIFY -->|"LLM: chat"| CHAT_BRANCH
    CLASSIFY -->|"LLM: preference"| PREF_BRANCH
    CLASSIFY -->|"LLM: task / default"| TASK_BRANCH

    subgraph CHAT["Branch 1 — chat"]
        CHAT_BRANCH["_chat_reply()<br/>QUICK_ANSWER, no tools"]
        CHAT_REPLY[["one-shot reply"]]
        CHAT_BRANCH --> CHAT_REPLY
    end

    subgraph PREF["Branch 2 — preference"]
        PREF_BRANCH["_save_preference()<br/>extract → user.md or reject"]
        PREF_REPLY[["short ack"]]
        PREF_BRANCH --> PREF_REPLY
    end

    subgraph TASK["Branch 3 — full task"]
        THINK["_think()<br/>TASK_ANALYSIS<br/>question_type, tools, plan, topics"]
        SELF_AN{"self_analysis?"}
        ENSURE_KB["_ensure_knowledge()<br/>HYBRID.find_best per topic"]
        SKIP_KB["notes = []<br/>force read_file via tools"]
        SOLVE["_solve()<br/>COMPLEX_SOLVING + tool loop"]
        VERIFY["_verify()<br/>VERIFICATION<br/>+ deterministic detector"]
        CRITIC{"confidence < 50%?<br/>(critic_threshold)"}
        RETRY["inject critique →<br/>re-solve (max 2 retries)"]

        THINK --> SELF_AN
        SELF_AN -->|"yes"| SKIP_KB --> SOLVE
        SELF_AN -->|"no"| ENSURE_KB --> SOLVE
        SOLVE --> VERIFY --> CRITIC
        CRITIC -->|"yes"| RETRY --> VERIFY
        CRITIC -->|"no"| TASK_REPLY[["answer"]]
    end

    CHAT_REPLY --> POST
    PREF_REPLY --> POST
    TASK_REPLY --> POST

    subgraph POST_PROC["Post-processing"]
        POST["CONVERSATION.add_turn"]
        EXTRACT["MEMORY.extract_and_store<br/>(facts → graph)"]
        EVAL_LOG["EVALUATOR.log + finetune queue<br/>(if confidence ≥ 85)"]
        TICK["GOALS.tick_interaction<br/>+ proactive learning check"]
        CLEAN["_cleanup"]
        POST --> EXTRACT --> EVAL_LOG --> TICK --> CLEAN
    end

    POST --> RESPONSE(["AgentAnswer"])
```

### Branch decisions

| Branch | Trigger | What runs | Confidence |
|---|---|---|---|
| `chat` | regex chitchat OR classifier says "chat" | `_chat_reply` (1 LLM call, no tools) | always 100 |
| `preference` | classifier says "preference" | `_save_preference` (extractor + write to `user.md`) | always 100 |
| `task` | arithmetic regex OR classifier says "task" OR default | `_think → _ensure_knowledge → _solve → _verify → critic loop` | computed |

---

## 3. Identity preamble

Every chat / think / solve LLM call gets the same identity preamble at
the top of its system prompt. The order matters — `LANGUAGE OVERRIDE`
and `AGENT NAME OVERRIDE` are appended LAST so they outweigh any
conflicting rule from `# SOUL`.

```mermaid
flowchart TB
    subgraph IDENTITY["IdentityManager.preamble()"]
        S1["# SOUL<br/>knowledge/identity/soul.md<br/>(tone, character)"]
        S2["# IDENTITY<br/>knowledge/identity/identity.md<br/>(role, capabilities, name section)"]
        S3["# USER PROFILE<br/>knowledge/identity/user.md<br/>(facts about the user)"]
        S4["# AGENT NAME OVERRIDE<br/>(extracted from identity.md ## Имя)"]
        S5["# LANGUAGE OVERRIDE<br/>(extracted from user.md ## Язык общения)"]
        S1 --> S2 --> S3 --> S4 --> S5
    end
```

`LANGUAGE OVERRIDE` and `AGENT NAME OVERRIDE` are only emitted when their
source sections are non-empty. Soul-level rules (e.g., "mirror the user's
language") become defaults — overrides win on conflict because they're
positioned at the end of the preamble where attention weight peaks.

---

## 4. Per-turn user-message context

`_shared_context(task, core)` builds the per-turn block that sits BELOW
identity in the user message. Stable facts come first, ephemeral state
last:

```
# CORE MEMORY        ← long-term persistent facts
# CURRENT PROJECT    ← active workspace, if any
# GOALS              ← top N active goals
# SHORT-TERM MEMORY  ← semantic recall vs. current task
                       (REPLACED by # RECENT COMMITS on self_analysis)
# RECENT TURNS       ← last N conversation turns
```

**Self-analysis rewrite.** When `thinking.question_type == "self_analysis"`:

- `MEMORY.recall_block` is **dropped** — it's a snapshot of past
  observations and reproduces stale findings.
- `git log --oneline -50` is added as `# RECENT COMMITS` so the agent
  knows what its code looks like *now*.
- `_ensure_knowledge` is **bypassed entirely** — KB notes about the
  code are also snapshots; the only authoritative source is the file
  on disk via `read_file`.

This rewrite was the fix for the "agent finds bugs that were already
fixed last commit" pattern.

---

## 5. Memory hierarchy

The agent has five distinct stores, each with its own retrieval
strategy and lifetime:

```mermaid
flowchart LR
    subgraph STABLE["Stable / curated"]
        CORE["CORE memory<br/>core_memory.md<br/>~always loaded"]
        ID["Identity<br/>soul/identity/user.md<br/>~always loaded"]
        NOTES["Notes (KB)<br/>knowledge/profession/*.md<br/>loaded by topic match"]
    end
    subgraph SEMI["Auto-extracted"]
        KG["Knowledge graph<br/>graph.json<br/>entities + relations"]
        FACTS["Memory facts<br/>memory_facts.jsonl<br/>extracted from chats"]
    end
    subgraph EPH["Ephemeral"]
        CONV["Conversation<br/>conversation.json<br/>last 20 turns"]
        ATT["Attachments<br/>knowledge/attachments/<br/>(sha256-deduped)"]
    end

    CORE --> AGENT
    ID --> AGENT
    NOTES -->|"HYBRID.find_best<br/>(min_raw_score=0.4)"| AGENT
    KG -->|"BFS hops + decay"| HYBRID
    FACTS -->|"semantic recall<br/>via embeddings"| AGENT
    CONV --> AGENT
    ATT --> AGENT

    AGENT["Agent's per-turn context"]
    HYBRID["HybridSearcher"]
```

**Hybrid search** combines three signals when looking up a topic:
fuzzy keyword (rapidfuzz, threshold 0.6), graph traversal (BFS with
`graph_score_floor=0.10`), vector cosine (embedder, `vector_score_floor=0.30`).
When a signal is unavailable, weights re-normalize across the
remaining ones. Below the per-signal raw floor → noise → dropped.

---

## 6. LLM router

`DualModelRouter` picks model A (Claude / cloud) vs. model B
(Qwen / local) per `TaskType`. Defaults can be shifted by schedule
or by daily budget; `VERIFICATION` is always pinned to A.

```mermaid
flowchart TD
    CALL["router.call(task_type, system, user)"]
    OVERRIDE{"User pinned<br/>active model?"}
    USE_PINNED["use the pinned model<br/>(no router logic)"]

    PICK{"Pick A or B"}
    A_TASKS["MODEL_A_TASKS<br/>TASK_ANALYSIS, LEARNING,<br/>COMPLEX_SOLVING, VERIFICATION,<br/>NOTE_CREATION → A"]
    B_TASKS["MODEL_B_TASKS<br/>SIMPLE_LOOKUP, KEYWORD,<br/>NOTE_SEARCH, QUICK_ANSWER,<br/>CLASSIFICATION → B"]

    SHIFT{"shift_schedule<br/>pct → B?"}
    BUDGET{"A daily budget<br/>exhausted?"}
    HEALTH{"target reachable?"}
    FALLBACK["fallback to other side<br/>(if available)"]

    CALL --> OVERRIDE
    OVERRIDE -->|"yes"| USE_PINNED --> END(["LLM response"])
    OVERRIDE -->|"no"| PICK
    PICK --> A_TASKS
    PICK --> B_TASKS
    A_TASKS --> SHIFT
    SHIFT -->|"yes + B up"| BUDGET
    SHIFT -->|"no"| BUDGET
    BUDGET -->|"yes + B up"| FALLBACK
    BUDGET -->|"no"| HEALTH
    HEALTH -->|"yes"| END
    HEALTH -->|"no"| FALLBACK --> END
```

**Provider-agnostic.** Both A and B are configured through
`backend/providers.py`. A can be Anthropic, Codex (OpenAI ChatGPT
subscription), Bedrock, Cohere, Copilot, or any
OpenAI-compatible endpoint. B defaults to `llama-server` /
Ollama on `100.124.210.21:8015`-ish.

---

## 7. Tool loop

`COMPLEX_SOLVING` calls go through `complete_with_tools`. The model
emits `text + tool_use` blocks; the agent executes each tool and
re-feeds results until the model returns text without further
tool calls — or hits `max_iterations=6`.

```mermaid
sequenceDiagram
    participant S as Solver
    participant LLM as LLM (A or B)
    participant T as ToolRegistry

    S->>LLM: system + user + tools
    loop until end_turn or max_iterations
        LLM-->>S: text? + tool_use?
        alt has tool_use
            S->>T: execute(name, args)
            T-->>S: result_text
            Note over S: capture tool_call detail<br/>for thinking_trace
            S->>LLM: assistant block + tool_results
        else end_turn
            Note over S: return final_text
        end
    end
    Note over S: hit cap → forced<br/>tool-less synthesis call<br/>(synth_max = 6000)
    S-->>LLM: same messages, no tools
    LLM-->>S: synthesized answer
```

**Why the forced synthesis at max-iterations.** Anthropic models
emit `text + tool_use` in the same turn — the text is "preamble"
narration ("Now I will check the source"), not a final answer. If
we returned the last preamble, the user would see a promise-of-action
instead of an answer. The forced synthesis call drops `tools` and
asks for a real answer.

**`final_text` capture rule.** Only when there are NO tool_uses in the
response, otherwise we'd treat the preamble as the answer.

---

## 8. Verifier + self-critic loop

Verification pings the LLM (always model A) with the answer + notes +
tool outputs and asks it to bucket every claim. Confidence is computed
**deterministically in Python** from the bucket counts:

```
confidence = 100 × verified / (verified + unverified + 2 × contradictions)
```

Contradictions weight 2× because they're evidence *against*, not just
absent evidence. Then the result passes through a deterministic
false-absence detector — a regex sweep that catches "add X" / "missing X"
when X is in `EXTRACTED IDENTIFIERS` from tool output. Any hit is
promoted to a contradiction. This is belt-and-suspenders for the
LLM-side rule because Sonnet has been observed missing these even
with explicit prompt guidance.

```mermaid
flowchart TD
    ANS["solver answer + tool_context"]
    EXT["_extract_code_identifiers(tool_context)"]
    LLM["VERIFIER_SYSTEM<br/>+ EXTRACTED IDENTIFIERS list<br/>verifier LLM call (A)"]
    LLM_OUT["{verified[], unverified[], contradictions[]}"]
    DET["detect_false_absence_contradictions(<br/>answer, identifiers)"]
    MERGE["merge auto-detected contradictions<br/>(dedup against LLM)"]
    CONF["compute confidence<br/>(deterministic Python)"]
    CHK{"confidence < 50?"}
    OK[["return result"]]
    RETRY["inject CRITIQUE block:<br/>contradictions + unverified +<br/>previous answer<br/>→ _solve again"]

    ANS --> EXT --> LLM --> LLM_OUT --> MERGE
    ANS --> DET --> MERGE
    MERGE --> CONF --> CHK
    CHK -->|"≥50"| OK
    CHK -->|"<50, retry < 2,<br/>not stuck at 0"| RETRY --> ANS
    CHK -->|"<50, retry == 2"| OK
```

**Stop conditions:**

- Confidence ≥ `critic_threshold` (50%) → done.
- `retry == max_retries` (2) → done with whatever we have.
- Confidence stuck at 0% across two iterations → break (signal it's
  a structural problem, more retries won't help).
- `LLMError` mid-retry → break, return current best.

**Skip path.** For `question_type ∈ {"creative", "meta", "self_analysis"}`
*without* tool_context, verification is skipped entirely (there's
nothing to verify against — the verifier would just produce
unverified-noise).

---

## 9. Identifier extraction (verifier helper)

The detector and the LLM verifier both rely on a list of identifiers
present in the source code at this turn. Pre-extraction makes the
"already in code" check a keyword match instead of a 12k-char read.

```mermaid
flowchart LR
    TC["tool_context (file dumps)"]
    P1["regex: class X / def Y"]
    P2["regex: SCREAMING_CONST = ..."]
    P3["regex: self.attr = ..."]
    SET["sorted, deduped, capped at 200"]
    TC --> P1 & P2 & P3 --> SET
```

The detector then normalises both candidate (from answer) and
identifier (from extraction) via `s.replace("_", "").lower()`, so
`FILE_CACHE` ↔ `_file_cache` ↔ `fileCache` collapse to the same key.

---

## 10. Feedback loops

Two loops run in the background and shape future behaviour:

```mermaid
flowchart TD
    subgraph TURN["End of turn"]
        VR["VerificationResult"]
        EXTR["MEMORY.extract_and_store<br/>(LLM-extracted facts → graph)"]
        EVAL["EVALUATOR.log<br/>+ finetune_queue if conf ≥ 85"]
        FAIL{"confidence < 60?"}
        ANL["META_LEARNER.analyze_failure<br/>→ root_cause + fix_action<br/>→ goal in goals.json"]
        TICK["GOALS.tick_interaction()"]
    end

    VR --> EXTR
    VR --> EVAL
    VR --> FAIL
    FAIL -->|"yes"| ANL
    ANL -->|"every Nth (5)"| EXTRACT["META_LEARNER.extract_patterns()<br/>recurring patterns → high-priority goal"]
    TICK --> PROACTIVE{"every 10<br/>interactions?"}
    PROACTIVE -->|"yes"| GAPS["GOALS.suggest_from_gaps<br/>(KM.open_gaps → learning goals)"]
```

**Goals dedup.** All goal-creation paths normalize description through
`re.sub(r'\W+', ' ', s.lower()).strip()`, so punctuation/whitespace
variants of the same goal collapse. Distinct topics stay distinct.

**Auto-fix actions** that meta-learner can take based on failure
analysis:

- `learn_topic` → goal "Learn: \<topic\>" priority ≤ severity
- `add_core_fact` → goal to add a fact to `core_memory.md`
- `update_note` → goal to refresh an existing KB note
- `improve_prompt` → goal flagged as prompt engineering work

---

## 11. Channels — WebUI vs Telegram

Both go through the same `Agent.run()`. They differ only in transport
and progress UX.

**WebUI (`/api/chat`)** — Server-Sent Events:

```mermaid
sequenceDiagram
    participant FE as Chat.tsx
    participant API as /api/chat
    participant AG as Agent

    FE->>API: POST {message, project, attachments[]}
    API->>AG: Agent(progress=cb).run(...)
    loop progress events
        AG-->>API: progress("event", "msg")
        API-->>FE: SSE {type:"progress",...}
    end
    AG-->>API: AgentAnswer
    API-->>FE: SSE {type:"answer", data: {...}}
    Note over FE: render answer +<br/>verification + thinking trace +<br/>tool calls (collapsible)
```

**Telegram (`backend/channels.py`)** — placeholder + edit-in-place:

```mermaid
sequenceDiagram
    participant TG as User (Telegram)
    participant BOT as TelegramBot
    participant AG as Agent

    TG->>BOT: text / photo / voice
    BOT->>BOT: download attachments,<br/>transcribe voice if needed
    BOT->>TG: 🧠 Thinking… (placeholder)
    BOT->>AG: run_in_executor(agent.run)
    loop progress events (executor thread)
        AG-->>BOT: progress("event","msg")
        BOT->>BOT: _TgProgressStream.push (rate-limited)
        BOT-->>TG: edit placeholder ↑
    end
    AG-->>BOT: AgentAnswer
    BOT-->>TG: ✅ Done · summary footer (edits placeholder)
    BOT-->>TG: <answer> (new message, chunked at 4000 chars)
```

The placeholder edit is throttled to one per ~1.2s (Telegram rate
limit) and coalesces bursts into a single deferred flush.

---

## 12. Self-modifier (gated)

Optional path: the agent can analyse its own modules and propose
patches, but **never applies them without explicit user approve**.

```mermaid
flowchart LR
    REQ["analyze_module(module_name)"]
    READ["read_text() + head/tail truncate at 30k"]
    LLM["ANALYZE_SYSTEM<br/>LLM proposes diffs"]
    PROP["Proposal{old_code, new_code}"]
    USR{"user reviews"}
    APP["apply: write file<br/>(no auto-rollback)"]
    REJ["proposal.status = rejected"]
    REQ --> READ --> LLM --> PROP --> USR
    USR -->|"approve"| APP
    USR -->|"reject"| REJ
```

This module is read-only by default — `apply()` only runs on explicit
user action via the WebUI.

---

## 13. Useful entry points (file:line cheatsheet)

| Want to find | Look at |
|---|---|
| Pipeline entry | `backend/agent.py:Agent.run` |
| Intent classifier | `backend/agent.py:_classify_intent` |
| Thinking step | `backend/agent.py:_think` |
| Solver + tool loop call | `backend/agent.py:_solve` |
| Verification + critic loop | `backend/agent.py:run` (search `critic_threshold`) |
| Self-analysis context rewrite | `backend/agent.py:_shared_context` (look for `for_self_analysis`) |
| Identity preamble assembly | `backend/identity.py:IdentityManager.preamble` |
| Router selection | `backend/llm.py:DualModelRouter._pick` |
| Tool loop (Anthropic) | `backend/llm.py:AnthropicLLM.complete_with_tools` |
| False-absence detector | `backend/verifier.py:detect_false_absence_contradictions` |
| Hybrid search | `backend/hybrid_searcher.py:HybridSearcher.search` |
| Failure analysis + auto-extract | `backend/meta_learner.py:analyze_failure` |
| Goal dedup normalization | `backend/goals.py:_normalize_description` |
| Telegram realtime stream | `backend/channels.py:_TgProgressStream` |
| WebUI chat SSE | `backend/api/chat.py` |

---

*Last verified against commit `a8b0cbe`.*
