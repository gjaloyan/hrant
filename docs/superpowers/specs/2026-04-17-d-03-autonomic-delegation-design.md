# D-03 — Claude-delegation + 3 autonomic levers (design)

**Status:** Design (no implementation)
**Date:** 2026-04-17
**Parent:** [Model X spec, Section 11 — phased delivery](./2026-04-16-model-x-autonomic-design.md)
**Delivers:** Claude-delegation mechanics + `FIRE_INTEGRITY_HEARTBEAT`, `FIRE_MEMORY_CONSOLIDATION`, `FIRE_GOAL_PROPOSE`.

---

## 0. Context

D-01 (foundation) and D-02 (Layer 0 + immune levers) are merged. The autonomic scheduler runs a single-tempo 30s tick, Layer 0 has 4 reactive rules and 4 immune levers. D-03 adds the first **autonomic-cycle** levers — the ones that delegate work to the cortex (main LLM) instead of being pure-Python.

**Goal:** land three green-safety autonomic levers without growing the scheduler architecture. Every new lever must fit the existing `Lever` / `Layer0Engine` / `LeverExecutor` contracts from D-01/D-02. No multi-cadence scheduler, no new abstractions for cortex delegation.

**Non-goals:** multi-cadence tick tracks, cortex-wrapper module, auto-fix in IntegrityHeartbeat, any yellow/red-safety work.

---

## 1. Claude-delegation mechanics

A lever that needs the cortex imports existing singletons directly inside its `run()` method. There is **no** new `backend/autonomic/cortex.py` wrapper.

Concretely:

- `FIRE_MEMORY_CONSOLIDATION` imports `backend.memory_extractor.MEMORY` and `backend.llm.router`. The lever assembles a daily-consolidation prompt, calls `router.call_task(TaskType.TASK_ANALYSIS, ...)`, parses JSON, writes to the three memory tiers (Section 3).
- `FIRE_GOAL_PROPOSE` imports `backend.goals.GOALS`. Internally `GOALS.suggest_from_gaps()` already calls the cortex — the lever is a thin wrapper with dedup.
- `FIRE_INTEGRITY_HEARTBEAT` does not use the cortex at all (pure filesystem walk).

**Why no wrapper:** Section 5 of the parent spec says "рычаг вызывает cortex через `backend/llm.py` → `backend/providers.py`". D-03 has only one lever that directly calls Claude (`MEMORY_CONSOLIDATION`); building a shared helper for one consumer is premature. When D-04 adds `FIRE_NOTE_CURATION` and `FIRE_SELF_STUDY`, we will see what the common contract should be and extract it then.

**Cost accounting:** each delegating lever fills `LeverReport.cost` with `TokenUsage` from `router.call_task` return value. No new infrastructure.

**Error handling:** each lever wraps its cortex call in a broad `try/except` and returns `LeverStatus.FAILURE` with `reason="exception:<msg>"`. `LeverExecutor` already retries nothing (D-02 behavior) — that is unchanged.

---

## 2. Tick cadence — one-tempo scheduler + per-rule cooldowns

The D-01 scheduler fires a single tick every `AUTONOMIC_TICK_SECONDS` (default 30s). D-03 does **not** add fast/medium/slow/nightly tick tracks. Instead each new `LayerZeroRule` declares a `cooldown_seconds` that gates its own firing:

| Rule | Lever | `cooldown_seconds` | Effective cadence |
|---|---|---|---|
| `integrity_tick` | `FIRE_INTEGRITY_HEARTBEAT` | 300 | every 5 min |
| `goal_propose_tick` | `FIRE_GOAL_PROPOSE` | 3600 | hourly |
| `consolidation_tick` | `FIRE_MEMORY_CONSOLIDATION` | 86400 | daily |

`Layer0Engine.evaluate()` already handles per-rule cooldowns (see `backend/autonomic/layer0.py` cooldown test — D-02). No code changes to the engine are needed for cadence.

**Rationale:** multi-cadence tick tracks are ~200 lines of new code (separate tick timers, separate loops, separate `on_tick` callbacks) for zero functional benefit at this scale. Cooldowns on a 30s base tick are exact up to 30s granularity, which is fine for a daily cycle. Revisit in D-05+ if the lever count or tick load makes a single loop too hot.

---

## 3. Levers

### 3.1 `FIRE_INTEGRITY_HEARTBEAT`

**Category:** autonomic. **Safety:** green. **Executor:** python. **Cadence:** 5 min.

**Purpose:** detect discrepancies between `knowledge/index.json` and files on disk. Read-only for D-03 — report problems, do not fix.

**Behavior (`run`):**
1. Load `knowledge/index.json` into memory.
2. Walk `knowledge/**/*.md` files, excluding `knowledge/_history/`, `knowledge/autonomic/`, `knowledge/immune/`, `knowledge/identity/`.
3. Compute:
   - `orphan_files` — md file exists on disk, not in index.
   - `dead_entries` — entry in index, file missing on disk.
   - `index_count`, `file_count`.
4. Return `LeverReport` with `outcome` = `{index_count, file_count, orphan_files: [...], dead_entries: [...]}` and `reason` = `integrity_ok` or `integrity_drift:{N}_issues`.

**Preconditions:** always True. The cooldown is the only gate.

**Out of scope for D-03:** auto-fix (delete dead entries, re-add orphans), duplicate detection in `user.md`, graph consistency. These are either yellow-safety or belong to later levers.

### 3.2 `FIRE_MEMORY_CONSOLIDATION`

**Category:** autonomic. **Safety:** green. **Executor:** claude-delegate. **Cadence:** daily.

**Purpose:** review the last 24h of sessions, extract durable information, and route it to the correct memory tier.

**Memory tiers written to (all green-safety):**

| Tier | File | Content |
|---|---|---|
| Identity / user profile | `knowledge/identity/user.md` | user-specific facts: role, location, preferences, ongoing projects |
| Memory facts | `knowledge/memory_facts.jsonl` | world-facts: prices, technical specs, events, dates, non-user entities |
| Session summary | `knowledge/sessions.json` (new field `summary`) | 2–3 sentence human-readable digest of the session |

**Tiers explicitly NOT written:**
- `knowledge/identity/soul.md`, `knowledge/identity/identity.md` — agent identity, yellow safety.
- `knowledge/core_memory.md` — user-curated only, yellow safety.
- `knowledge/{fundamentals,profession,projects,personal}/*.md` — the domain of `FIRE_NOTE_CURATION` (D-04).
- `knowledge/patterns.json`, any `meta_learner` state — the domain of `FIRE_SELF_REFLECTION` (D-05).

**Behavior (`run`):**
1. Load `knowledge/sessions.json`. Structure: `{"current_id": "...", "sessions": [{"id", "started", "ended", "title", "archived", "turns": [...]}, ...]}`. Filter to sessions where `consolidated != True` AND session has at least one turn.
2. For each such session (cap at `max_sessions=5` per tick to bound cost):
   1. Build a consolidation prompt: session transcript + current `user.md` (for dedup context).
   2. Call `router.call_task(TaskType.TASK_ANALYSIS, ...)` with a JSON-output contract (see prompt spec in Section 4).
   3. Parse response into three lists: `user_profile_facts`, `durable_facts`, `topic_threads`.
   4. For each `user_profile_fact` with `confidence ≥ 0.8`: normalize, diff against existing `user.md` content, append new lines under a date-stamped section.
   5. For each `durable_fact` with `confidence ≥ 0.8`: write as a `MemoryFact` to `memory_facts.jsonl` using the existing `MemoryFact.to_dict()` format. Dedup by `summary` string against last N (`dedup_window=200`) entries in the file.
   6. Set `session.consolidated = True` and `session.summary = <claude-provided summary>`.
3. Save `sessions.json` once at end. Return LeverReport with counts: `{sessions_processed, profile_added, facts_added, threads_queued}` and `follow_ups = topic_threads[:10]`.

**Preconditions:** at least one session in `sessions.json` has `consolidated != True` and at least one turn.

**Dedup rules:**
- `user.md`: case-insensitive substring match on normalized fact summary. If any existing line contains the summary, skip.
- `memory_facts.jsonl`: exact match on `summary` field in the last 200 entries.

**Prompt contract (returned JSON):**
```json
{
  "session_summary": "2-3 sentence digest",
  "user_profile_facts": [
    {"summary": "...", "confidence": 0.0-1.0, "category": "role|location|preference|project|general"}
  ],
  "durable_facts": [
    {"summary": "...", "triples": [["e1","r","e2"]], "tags": [...], "category": "price|technical|event|location|preference|relationship|rule|general", "confidence": 0.0-1.0}
  ],
  "topic_threads": ["short topic phrase", ...]
}
```

### 3.3 `FIRE_GOAL_PROPOSE`

**Category:** autonomic. **Safety:** green. **Executor:** python (delegates to `GOALS` which may call cortex internally). **Cadence:** hourly.

**Purpose:** read current knowledge gaps, propose learning goals, dedup against existing goals.

**Behavior (`run`):**
1. Load `knowledge/gaps.json`. If the file is missing or empty, return `LeverStatus.SKIPPED` with `reason="no_gaps"`.
2. Build `gaps` list in the shape `GOALS.suggest_from_gaps` expects.
3. Get current active proposals: `existing = {g.description for g in GOALS.active_goals() if g.goal_type == "proactive"}`.
4. Call `GOALS.suggest_from_gaps(gaps, max_goals=3)`.
5. For each suggested goal: skip if `g.description in existing`. Otherwise persist via `GOALS.add(g)` or whatever the existing API is.
6. Return LeverReport with `outcome = {proposed: N, skipped_dup: M, gap_count: K}` and `reason = "proposed_{N}_goals"`.

**Preconditions:** `Path("knowledge/gaps.json").exists()` AND file is non-empty.

**Safety note:** `GoalManager.suggest_from_gaps` itself is defined as calling Claude to generate descriptions. That cortex call is green because it only writes structured goal records — it never executes a learning step. Actual execution is `FIRE_SELF_STUDY` (D-04) or user-triggered `learn_topic`.

---

## 4. Layer 0 rule extension

`backend/autonomic/layer0.py::default_rules()` grows from 4 to 7 rules. The 4 existing rules (disk/memory/cpu/errors) are unchanged. The 3 new rules have trivial predicates — they rely on cooldowns for scheduling, not on state inspection:

```python
LayerZeroRule(
    name="integrity_tick",
    predicate=lambda s: True,
    lever="FIRE_INTEGRITY_HEARTBEAT",
    params={},
    cooldown_seconds=300.0,
),
LayerZeroRule(
    name="goal_propose_tick",
    predicate=lambda s: True,
    lever="FIRE_GOAL_PROPOSE",
    params={},
    cooldown_seconds=3600.0,
),
LayerZeroRule(
    name="consolidation_tick",
    predicate=lambda s: True,
    lever="FIRE_MEMORY_CONSOLIDATION",
    params={},
    cooldown_seconds=86400.0,
),
```

**Ordering note:** rules are evaluated in declaration order, and the first match wins (D-02 `test_first_matching_rule_wins`). Reactive rules (`errors_present`, `disk_low`) must stay **before** the new schedule-driven rules so that an ongoing error doesn't get blocked by `consolidation_tick` firing first.

Final order in `default_rules()` for D-03:
1. `disk_low` (reactive)
2. `memory_low` (reactive)
3. `cpu_high` (reactive)
4. `errors_present` (reactive)
5. `integrity_tick` (scheduled)
6. `goal_propose_tick` (scheduled)
7. `consolidation_tick` (scheduled)

`register_default_immune_levers()` in `backend/autonomic/levers/__init__.py` grows to register the 3 new levers alongside the existing 4 immune ones. Same file, same pattern.

---

## 5. File layout

**New files:**
```
backend/autonomic/levers/
├── integrity_heartbeat.py        # FIRE_INTEGRITY_HEARTBEAT
├── memory_consolidation.py       # FIRE_MEMORY_CONSOLIDATION
└── goal_propose.py               # FIRE_GOAL_PROPOSE

tests/autonomic/
└── test_autonomic_levers.py      # covers all 3 levers, mirrors test_immune_levers.py
```

**Modified files:**
- `backend/autonomic/layer0.py` — extend `default_rules()` with 3 scheduled rules.
- `backend/autonomic/levers/__init__.py` — rename `register_default_immune_levers` → `register_default_levers` (includes autonomic too), keep old name as deprecated alias for one cycle to avoid breaking `backend/autonomic/startup.py`. **Or** add a parallel `register_default_autonomic_levers()` and call both from `startup.py`. The plan will pick one; recommend the parallel function.
- `backend/autonomic/startup.py` — call the new registration function.
- `knowledge/sessions.json` — runtime data; new fields `consolidated: bool` and `summary: str` added by the first consolidation run. No migration needed; missing fields default to `False`/empty.
- `README.md` — mention the 3 new levers in the autonomic section.

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, existing immune levers, `backend/main.py` (router already extracted).

---

## 6. Testing strategy

Three test files, mirroring D-02 patterns:

**`tests/autonomic/test_autonomic_levers.py`** — unit tests per lever:
- `integrity_heartbeat`: metadata; empty knowledge dir returns zero counts; orphan file detection; dead entry detection; honours exclusion list (`_history`, `autonomic`, `immune`, `identity`).
- `memory_consolidation`: metadata; preconditions False when all sessions consolidated; monkey-patch `router.call_task` to return canned JSON; verify writes to `user.md` (tmp), `memory_facts.jsonl` (tmp), and `sessions.json`; dedup against existing `user.md` lines; dedup against recent `memory_facts.jsonl` entries; cap at `max_sessions=5`.
- `goal_propose`: metadata; preconditions False when `gaps.json` missing; monkey-patch `GOALS.suggest_from_gaps` to return 3 canned goals; verify dedup against existing active proactive goals.

**`tests/autonomic/test_layer0.py`** — extend:
- `default_rules()` now returns 7 rules in the correct order.
- Each scheduled rule has the right `cooldown_seconds` and lever name.

**`tests/autonomic/test_d03_integration.py`** — end-to-end:
- Start scheduler with `default_rules()`. Put one unconsolidated session in a tmp `sessions.json`. Monkey-patch `router.call_task`. Run for 200ms. Verify `lever_log.jsonl` shows `FIRE_MEMORY_CONSOLIDATION` (first cooldown window fires immediately), `user.md` and `memory_facts.jsonl` got appended to, session is now `consolidated=True`.

**Monkey-patch strategy for cortex calls:** the levers import `backend.llm.router` at module top; tests patch `backend.memory_consolidation.router.call_task` with a fake that returns pre-built JSON. No real network calls in tests.

**Test count target:** ~20 new tests. Combined autonomic suite post-D-03: ~130 tests.

---

## 7. Open questions (not blocking)

1. **`user.md` write format** — append under a `## YYYY-MM-DD` heading or inline? Recommend heading per day — keeps history visible and makes dedup easier. Resolve in the implementation plan.
2. **`sessions.json` atomic write** — the file is 1000+ lines and read by other modules. Consolidation writes once per tick (5 sessions max), so write-lock-write is fine in a single-process app. Add a backup file `.bak` before write. Resolve in the plan.
3. **`gaps.json` shape** — need to read one real `gaps.json` during plan-writing to confirm the exact key names for the `gaps: list[dict]` parameter of `GOALS.suggest_from_gaps`. If the file is absent at plan time, use the shape declared in `backend/goals.py`.

---

## 8. What comes after D-03

Next sub-project: **D-04** per parent spec Section 11. `FIRE_SELF_STUDY`, `FIRE_CAPABILITY_SCAN`, `FIRE_NOTE_CURATION`, `FIRE_GRAPH_MAINTENANCE`. D-04 absorbs `backend/background.py` into `FIRE_SELF_STUDY` and retires it.
