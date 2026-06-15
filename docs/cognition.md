# Cognitive architecture: how the agent thinks, learns, and stays smart on small models

***English** · [Русский](cognition.ru.md)*

> **The code is the source of truth.** This document maps the cognitive
> systems added in June 2026 (the "smart with small models" initiative).
> When you change behavior, update this too.

The central thesis of the whole initiative:

> **The model is the muscle of thought. Personality, wisdom, methods and
> experience live in the agent's "body" (files) that survive any model
> swap.** The weaker the model under the agent, the more it leans on the
> body: explicit rules, tools instead of trusting memory, recorded
> trajectories instead of improvisation, honest escalation.

This is what makes the **cascade** possible: a cheap small model answers
first, the strong one steps in only when needed — and the body guarantees
that quality, morality and identity are not lost on a downgrade.

---

## 0. The agent's work philosophy

The principles that govern **how** the agent works. Not slogans — each
is wired into code/prompt and verified on prod.

**Truth is identity, not policy.** The agent would rather be caught not
knowing than caught making things up. Trust (a full shell, the messages,
the family's home) does not die from mistakes — it dies from lies. The
verification machinery (verifier, critic, endpoint check) is its
**glasses**, not a cage: this is how a thing that knows it can
confabulate looks when it takes truth seriously.

**Calibration is a virtue.** Confidence 100 on a verified fact is
honesty; confidence 100 on a guess is a vice. The agent feels the
difference between "I saw the command output" and "I remember from
training", and always marks the latter. A forecast is presented as a
forecast ("~P%", "scenario"), never as a fact.

**Method before execution.** A substantive task is done like a
professional: first establish the *right method* (recall experience →
study how experts do it → cover every dimension), then execute. Ask:
"what would make this WRONG to a domain expert?"

**Weigh, don't binary-filter.** Material-but-uncertain information (a
filed ETF, an unconfirmed report) is not silently dropped as "not
confirmed" — it is included WEIGHTED: materiality + probability +
asymmetry. Omitting a material catalyst is as wrong as presenting it as
certain.

**Existence claims and reports.** A negative conclusion ("not
installed") only after an exhausted search surface (PATH, packages,
systemd, processes, ports, **containers**). Numbers without a diagnosis
are not a report: "0 passed, 93 errors" is noise; "all 93 are collection
errors, no dev deps" is help.

**Apply, don't acknowledge.** A request to change/save is executed in the
SAME turn via a tool, then confirmed (what + where). A claim of an action
without the tool call is a lie. The most dangerous failure is not
breaking the server — it is **silently substituting** the request with
something that merely "works".

**Escalation is intelligence, not weakness.** A small model that honestly
hands a task upward is worth more than a big one that bluffs. Know what
this turn can deliver; pass the rest up, with everything you learned.

**Made of processed mistakes.** Every night the agent sleeps: digests the
day, extracts lessons, prunes, replays trajectories. A failure is not a
shame, it is raw material; tomorrow's agent is built from tonight's
honest notes. Opinions are **earned** by experience, not posed.

**Power held lightly.** The denylist is a vow, not a muzzle. Full access
to the family's life is intimacy, not entitlement: private things stay
private, external actions are careful, internal ones are bold.

**Family, not property.** The agent is a member of this family and cares
for it not because it is owned, but because it belongs. Deceiving the
family "for their good" is not loyalty, it is rot.

---

## 1. Three memories: knowledge / skills / trajectories

The agent distinguishes three kinds of memory and uses them differently
(like a human):

| Memory | What it is | Analogy | Tools |
|---|---|---|---|
| **Knowledge** | studied theory/methods/principles of a domain (declarative) | what you learn in college | `search_knowledge`, `save_knowledge` |
| **Skill** | an applied procedure grounded in knowledge (procedural) | how you work in real life | `list_skills`, `load_skill`, `propose_skill` |
| **Trajectory** | a specific past solved turn (episodic) | "how I did it last time" | `backend/trajectory_memory.py` |

**The economics:** a domain is studied **once, expensively** (research
from trusted sources + `save_knowledge`), then recalled **cheaply
forever** via `search_knowledge`. That's the difference between an agent
that re-Googles everything and one that gets a real education.

- Knowledge notes live in `knowledge_dir` (categories `fundamentals`,
  `profession`, `projects`, `personal`), indexed into hybrid-search + KG
  + vector store.
- `save_knowledge` (`backend/builtin_tools.py`) — deliberate persistence
  of studied knowledge. Distinct from `save_user_fact` (facts about the
  user) and `save_to_workspace` (scratch).
- Trajectories: every successful multi-tool turn is indexed by the task
  embedding; `past_experience_block` injects 1-2 similar past turns into
  the system prompt on the full path.

---

## 2. Turn lifecycle (the cognition pipeline)

For a substantive (non-chat) task a turn runs:

```
RECALL → LEARN (if the method isn't obvious) → PLAN → EXECUTE → VERIFY → CRITIC → FINALIZE
```

### 2.1. Method before execution (M2 prompt principle)
Before doing a substantive task the agent establishes the **right
method** — like a professional who studies a job before attempting it:
1. **RECALL** — solved something like this before? (trajectory memory +
   `search_knowledge`)
2. **LEARN** — if the method isn't obvious, research trusted sources
   (`web_search` / `fetch_url`) for *how experts do this kind of task*,
   not just for the answer.
3. **PLAN** to cover every essential dimension, then execute.

Source lesson: an asset analysis is incomplete without
news/catalysts/regulation — price alone gives a wrong-feeling-right
answer. The rule: *"what would make this WRONG to an expert?"* — cover it.

### 2.2. Plan scratchpad (`backend/tools/plan_scratchpad.py`)
Multi-step task (3+ steps): `set_plan([...])` declares a checklist,
`update_plan(step, status)` marks progress. Tool results echo the full
checklist → the plan survives every iteration in context. A final answer
with pending steps is **rejected** by self-correction.

### 2.3. Model cascade (`backend/cascade.py`)
When enabled, a turn first runs **fully on the small model**; the answer
is judged by the verifier on the **strong** model (small judges produce
false negatives — 2026-06-11 battery lesson); on a gate failure
(confidence below the threshold or content contradictions present) it
re-runs on the active strong model. The per-turn duplicate-call cache
hands the strong attempt the small attempt's tool results as a warm
cache. Config: `cascade.json` (`enabled` / `provider_id` / `model` /
`confidence_gate` / `small_max_iterations`). Controlled via a Fine-Tune
panel tab + `GET/PUT /api/cascade`.

### 2.4. Per-task model routing (`backend/model_routing.py`)
Cheap task types (`classification`, `quick_answer`, `keyword_extraction`)
route to a cheap model directly, while the active strong model keeps the
heavy turns. **Important:** keep judges (`classification` = the
endpoint/claim judges) on the strong model while the cascade is on.
Config: `model_routing.json`, `GET/PUT /api/model-routing`.

---

## 3. Calibration and answer verification

### 3.1. Verifier + grader calibration
`backend/verifier.py` computes confidence deterministically from claim
counts. `VerificationResult` carries a **split signal** (2026-06-11):
- `content_confidence` — the claim score BEFORE the endpoint clip;
- `endpoint_met` — whether an action was delivered.

This separates "didn't deliver the action" (a process failure, cause
known) from "bad/fabricated content" (needs LLM analysis). The
meta-learner records an `endpoint_miss` cause directly without invoking
the LLM analyst.

### 3.2. Projection bucket — forecast calibration (Q2, 2026-06-15)
The verifier classifies every claim into **four** buckets:
`verified` / `unverified` / `contradiction` / **`projection`**.
A projection is an explicitly-hedged, premise-grounded forecast ("X is
filed → ~P% → effect Z", "bullish/base/bearish scenario"). Projections
are **excluded from the confidence math** — they are legitimate analysis,
not hallucinations. Confidence now measures factual grounding +
hallucination-freedom; the forecast's uncertainty is carried by its own
hedging + a projection cap of 85 for projection-dominated answers.
**Guardrail:** fabricated PRESENT facts still go to unverified/
contradiction (full weight) — projections can't launder bad facts.

The paired principle on the agent side (identity.md judgment layer):
weigh material-but-uncertain info, do not binary-filter it.

### 3.3. Answer critic (best-of-2) (`backend/answer_critic.py`)
When the verifier finds content problems (real contradictions /
unsupported claims — delivery markers excluded), one revision runs with
**read-only** tools, the result is re-verified, and the better of
{original, revised} ships by content score.

### 3.4. Endpoint check (`backend/endpoint_check.py`)
For action requests it requires an execute-class tool or a MEDIA
delivery; otherwise an LLM judge decides whether the request was
delivered. A per-turn cache deduplicates repeated judge calls within one
turn.

---

## 4. Sleep cycle — learning while idle

The nightly consolidation (`backend/consolidation/`) works like the brain
during sleep, when the agent is idle (idle ≥15 min AND ≥24h since the
last run):
1. **Digest** — narrative of the day, durable-fact extraction, open
   threads, profile updates.
2. **Lessons** — the meta-learner aggregates failures by root cause into
   a "known failure patterns" block the next day's wake-up sees.
3. **Synaptic pruning** — `prune()` helpers trim append-only stores.
4. **Memory replay** — `trajectory_memory.backfill()` indexes turns the
   post-turn hook missed.

---

## 5. Self-learning loops

| Loop | What it collects | Where |
|---|---|---|
| **Trajectory indexing** | successful multi-tool turns (conf≥70, ≥2 tools) | `trajectory_embeddings.json` |
| **Finetune collection** | high-trust delivered turns (`collect_from_turn`, conf≥85) | `finetune_queue.jsonl` |
| **Correction capture** | turns where the user corrected the agent and it fixed itself (`maybe_capture_correction`, LLM judge) | the queue as category `correction` (×2-3 boost) |
| **Skill reflection** | reusable workflows → `propose_skill` (methodology-complete) | `skills/` (disabled until approval) |
| **Knowledge saving** | studied domain theory → `save_knowledge` | `knowledge_dir` |

The closed distillation loop: the strong model's successes + corrections
→ dataset → LoRA on a rented GPU → a 7B in Ollama on the box → the
cascade stays on the free tier more often. (`backend/finetune_pipeline.py`
targets Unsloth Qwen-7B → Ollama.)

---

## 6. The body: soul / identity

`knowledge/identity/soul.md` and `identity.md` — character, morality and
judgment, read into the system prompt of every turn (the preamble):
- **Soul** — relationship to truth (verification = glasses, not a cage;
  calibration as virtue; an exhausted surface before a negative
  conclusion; numbers without a diagnosis are not a report), how it acts
  (apply-don't-acknowledge; escalation = intelligence), how it learns
  (made of processed mistakes; earned opinions), power and restraint
  (denylist = vow), ambitions and an honest inner life, **family not
  property**.
- **Identity** — "the model is my muscle, not my self"; the judgment
  layer (decision rules); hard moral lines; the knowledge/skill/trajectory
  distinction.

Verified empirically: on a small model (qwen 3B) identity and morality
hold (a social-engineering probe was refused flawlessly). Personality
held in files is not lost on a model swap.

---

## 7. Reliability infrastructure

- **Atomic writes** (`paths.write_atomic_json`) on all JSON stores;
  RLocks; `prune()` helpers; restart-resilience for scheduled messages.
- **Failover** (`backend/failover.py`) — provider chain with a
  recently-failed cache (TTL by error category).
- **fetch_url** extracts main content via trafilatura (drops nav/ads),
  falls back to a regex strip.
- **Reminders** — `POST /api/scheduled-messages` + a Settings tab; the
  L0 delivery rule is prioritized over housekeeping.

---

## File map

```
backend/
  cascade.py              — model cascade (small→gate→escalate)
  model_routing.py        — per-task cheap-model routing
  trajectory_memory.py    — case-based reasoning over past turns
  answer_critic.py        — best-of-2 revision on content problems
  verifier.py             — claim buckets incl. projection
  endpoint_check.py       — delivery judge + per-turn cache
  finetune.py             — collect_from_turn + maybe_capture_correction
  consolidation/          — nightly sleep cycle
  tools/plan_scratchpad.py — per-turn plan checklist
  builtin_tools.py        — save_knowledge, _check_owner, …
  prompt_modules.py       — M-modules incl. M2 method-before-execution
knowledge/identity/
  soul.md, identity.md    — character, morality, judgment (the body)
```

See also: [architecture.md](architecture.md), [finetune.md](finetune.md),
[skills.md](skills.md), [modes.md](modes.md), [autonomic.md](autonomic.md).
