# D-06 — Telemetry cohort (design)

**Status:** Design (no implementation)
**Date:** 2026-04-18
**Parent:** [Model X spec, Section 11 — phased delivery](./2026-04-16-model-x-autonomic-design.md)
**Delivers:** `FIRE_MODEL_EVAL`, `FIRE_SESSION_ARCHIVE`, `FIRE_COST_AUDIT` — three pure-Python observation levers.

---

## 0. Context

D-01 through D-05 are merged. The autonomic scheduler runs 12 levers on per-rule cooldowns (4 immune, 8 autonomic). D-06 is the telemetry cohort: periodic observation of system state, quality metrics, cost burn, and session lifecycle. None of the three levers need the cortex — they aggregate data that was already logged during normal operation.

**Scope adjustment note:** Section 11 of the parent spec originally listed D-06 as a six-lever cohort (`MODEL_EVAL`, `SESSION_ARCHIVE`, `COST_AUDIT`, `SELF_REFLECTION`, `FINETUNE_QC`, `GAP_DETECTION`). During brainstorming on 2026-04-18 we split it along the spec-Section-3 taxonomy: **telemetry** goes to D-06 (this), **reflection** goes to D-07, and the original D-07 body+UI cohort shifts to D-08. Three-lever cohorts are the sweet spot we validated across D-03/D-04/D-05.

**Goal:** three green-safety pure-Python levers that produce append-only JSONL logs under `knowledge/autonomic/` (for `MODEL_EVAL` and `COST_AUDIT`) and move old sessions into `knowledge/_history/` (for `SESSION_ARCHIVE`). Logs become the substrate for D-07's reflection cohort, which asks the cortex to analyze patterns.

**Non-goals:** A/B model evaluation against a held-out test set (that is `backend/model_evaluator.py::ModelEvaluator`, run by the human at version upgrades, not by the scheduler). Auto-shutdown on budget overrun (yellow safety — D-08 AutonomicPanel). Per-provider cost breakdown (`router_state.json` does not track it today). Archived-session browsing UI (frontend work, not needed at 3 sessions total).

---

## 1. Three levers — all pure-Python

| Lever | Safety | Executor | Cadence | Reads | Writes |
|---|---|---|---|---|---|
| `FIRE_MODEL_EVAL` | green | python | daily (86400s) | `knowledge/eval_log.jsonl` via `EVALUATOR` | `knowledge/autonomic/model_eval_log.jsonl` |
| `FIRE_SESSION_ARCHIVE` | green | python | daily (86400s) | `knowledge/sessions.json` | `knowledge/_history/<session_id>.json` + mutates `sessions.json` |
| `FIRE_COST_AUDIT` | green | python | hourly (3600s) | `knowledge/router_state.json` | `knowledge/autonomic/cost_audit_log.jsonl` |

All three are autonomic-category levers registered via `register_default_autonomic_levers()`. `default_rules()` grows 12 → 15; registrations grow 8 → 11.

**Why all python, no cortex:** the cortex already did its work during normal operation — `eval_log.jsonl` is populated turn-by-turn by the chat pipeline, `router_state.json` is updated by each API call, `sessions.json` is written by the session manager. These levers aggregate and rotate, they do not analyze. Analysis (pattern extraction, regression interpretation) is D-07's job.

---

## 2. `FIRE_MODEL_EVAL`

**Purpose:** produce a daily snapshot of answer-quality metrics + detected regressions + suggested priorities, appended as one line per day to `model_eval_log.jsonl`.

**Dependencies:** `backend.evaluator.EVALUATOR` (singleton `SelfEvaluator`) exposes `daily_report(date)`, `detect_regression()`, `suggest_priorities()`, `weekly_trend()`, `stats()`.

**Behavior (`run`):**

1. Compute `target_date = yesterday` (`(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")`).
2. Call `EVALUATOR.daily_report(target_date)` — returns a dict with `{date, total, avg_confidence, intents, contradictions_rate, avg_response_time_ms, ...}`.
3. If `daily_report["total"] == 0`: return `LeverStatus.SKIPPED` with `reason="no_eval_entries"` (no chat traffic yesterday — nothing to audit).
4. Call `EVALUATOR.detect_regression()` — returns a list of `{metric, yesterday, 7d_avg, delta}` where current metric has degraded.
5. Call `EVALUATOR.suggest_priorities()` — returns a list of `{topic, reason, priority}` suggestions.
6. Build snapshot dict: `{ts: utcnow().isoformat(), date: target_date, daily_report: {...}, regressions: [...], priorities: [...]}`.
7. Append one JSONL line to `knowledge/autonomic/model_eval_log.jsonl`.
8. Return `LeverReport` with `outcome = {date, total, avg_confidence, regressions_count, priorities_count}` and `reason = f"evaluated_{total}_entries"`.

**Preconditions:** always True. Cooldown gates cadence; the "no entries" branch returns SKIPPED.

**Error handling:** each EVALUATOR call is wrapped in try/except. If `detect_regression` or `suggest_priorities` fails, the snapshot still writes with those fields as empty lists and the lever returns SUCCESS. A failure reading `eval_log.jsonl` returns FAILURE with `reason="eval_log_read_failed"`.

**Idempotence note:** if the lever runs twice for the same date (e.g. after a crash+restart), it appends two lines for the same date. That is acceptable — the log is audit-grade, append-only, and downstream readers deduplicate by taking the latest entry per date.

---

## 3. `FIRE_SESSION_ARCHIVE`

**Purpose:** move old, already-consolidated sessions out of active `sessions.json` into per-session files under `knowledge/_history/`. Keeps `sessions.json` small and prevents replay of already-processed data by D-03's `MEMORY_CONSOLIDATION`.

**Policy:** a session is archivable when **all** of:

- `ended` parses to a datetime older than **30 days** (`SESSION_ARCHIVE_DAYS`).
- `consolidated == True` (set by `FIRE_MEMORY_CONSOLIDATION` in D-03). Un-consolidated old sessions are left in place so consolidation can process them first.
- `archived != True` already.

**Behavior (`run`):**

1. Load `knowledge/sessions.json`.
2. Build candidate list from the policy above. Sort by `ended` ascending (oldest first).
3. Cap at `max_per_tick = 10`. For each candidate:
   1. Write the full session dict to `knowledge/_history/<session_id>.json` (pretty-printed JSON). Create parent dir if missing.
   2. Remove that entry from `sessions.json["sessions"]`.
4. Save `sessions.json` via the same atomic pattern used by `FIRE_MEMORY_CONSOLIDATION`: write to `.tmp` file, then `os.replace`.
5. If no candidates: return `LeverStatus.SKIPPED` with `reason="no_old_sessions"`.
6. Return `LeverReport` with `outcome = {archived, remaining_active, cutoff_date}`, `reason = f"archived_{archived}_sessions"`.

**Preconditions:** always True.

**Safety invariant:** the full session record is written to `_history/` **before** removal from `sessions.json`. If the second write fails, the session appears in both files — redundant but never lost. On next tick, `archived != True` check still matches (we never set it on the active record), so the lever will retry the removal.

**`current_id` handling:** if the active session (`sessions_blob["current_id"]`) is older than 30 days and consolidated, we still do **not** archive it. Policy is the normal filter **plus** `session.id != sessions_blob.get("current_id")`. Edge case, but safer.

---

## 4. `FIRE_COST_AUDIT`

**Purpose:** snapshot the running router-state cost counters hourly so we have a time series for cost trend analysis (which D-07's `FIRE_SELF_REFLECTION` can read). Flag daily-budget overruns in the outcome.

**Behavior (`run`):**

1. Read `knowledge/router_state.json`. If missing: return `LeverStatus.SKIPPED` with `reason="no_router_state"`.
2. Extract `{date, api_calls_today, api_cost_today, model_b_calls_today, total_a_calls, total_b_calls, last_reason}` with graceful defaults (0 / empty string).
3. Compute `issues: list[str]`. If `api_cost_today > daily_budget_usd` (param, default **10.0**): append `f"over_budget:{api_cost_today:.2f}_usd_>_{daily_budget_usd}"`.
4. Build snapshot: `{ts: utcnow().isoformat(), date, api_calls_today, api_cost_today, model_b_calls_today, total_a_calls, total_b_calls, last_reason, issues}`.
5. Append to `knowledge/autonomic/cost_audit_log.jsonl`.
6. Return `LeverReport` with `outcome = {date, api_calls_today, api_cost_today, issues}` and `reason` = `"cost_audit_ok"` if no issues, else `f"cost_audit:{len(issues)}_issues"`.

**Preconditions:** always True.

**Cadence rationale:** hourly (not daily) because daily ticks miss intra-day spikes (e.g. a runaway loop at 3pm could spend the budget before midnight). 24 snapshots/day × 365 days ≈ 9K lines/year — file growth is trivial.

**Parametrization:** `daily_budget_usd` param is overridable via `LayerZeroRule.params`. Default is 10.0. Value flows through `SafetyGate.evaluate` unchanged (the gate doesn't inspect params beyond `lever.safety`).

---

## 5. Layer 0 rule extension (12 → 15)

`default_rules()` appends three rules, in this order:

```python
LayerZeroRule(
    name="model_eval_tick",
    predicate=lambda s: True,
    lever="FIRE_MODEL_EVAL",
    params={},
    cooldown_seconds=86400.0,  # daily
),
LayerZeroRule(
    name="session_archive_tick",
    predicate=lambda s: True,
    lever="FIRE_SESSION_ARCHIVE",
    params={},
    cooldown_seconds=86400.0,  # daily
),
LayerZeroRule(
    name="cost_audit_tick",
    predicate=lambda s: True,
    lever="FIRE_COST_AUDIT",
    params={},
    cooldown_seconds=3600.0,  # hourly
),
```

**Ordering rationale:** `model_eval_tick` and `session_archive_tick` both want to fire on the first tick of each day; put `model_eval` first so that if it SKIPPEDs on "no_eval_entries", `session_archive` can still run. `cost_audit_tick` has the shortest cooldown (hourly) and goes last — cooldown fall-through from D-03 handles the cascade.

All three are `predicate=True` schedule-driven; reactive rules (4) still preempt, and the earlier D-03/D-04/D-05 scheduled rules (8) still win over D-06 when their cooldowns elapse simultaneously.

---

## 6. File layout

**New files:**

```
backend/autonomic/levers/
├── model_eval.py             # FIRE_MODEL_EVAL (~90 lines)
├── session_archive.py        # FIRE_SESSION_ARCHIVE (~110 lines)
└── cost_audit.py             # FIRE_COST_AUDIT (~90 lines)

tests/autonomic/
├── test_telemetry_levers.py   # ~15 unit tests
└── test_d06_integration.py    # 3 integration tests
```

**Modified files:**

- `backend/autonomic/layer0.py` — `default_rules()` 12 → 15.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers()` 8 → 11 registrations.
- `README.md` — list D-06 levers; add "telemetry logs" path section.

**New runtime files (created by levers on first run):**

- `knowledge/autonomic/model_eval_log.jsonl`
- `knowledge/autonomic/cost_audit_log.jsonl`
- `knowledge/_history/<session_id>.json` (one per archived session)

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `api.py`, `startup.py`, `backend/evaluator.py`, `backend/main.py`, existing D-01..D-05 levers, frontend.

---

## 7. Testing strategy

**Unit tests — `tests/autonomic/test_telemetry_levers.py`:**

- **`FIRE_MODEL_EVAL`** (5 tests):
  - metadata (name, category, safety, executor).
  - SKIPPED when `daily_report["total"] == 0`.
  - writes snapshot to `model_eval_log.jsonl` with date + total + regressions + priorities.
  - tolerates `detect_regression` raising (snapshot still writes, regressions empty).
  - preconditions always True.

- **`FIRE_SESSION_ARCHIVE`** (6 tests):
  - metadata.
  - SKIPPED when no sessions match policy.
  - archives sessions older than 30 days AND `consolidated=True`; writes to `_history/<id>.json`.
  - skips sessions older than 30 days but NOT consolidated.
  - skips the `current_id` session even when old+consolidated.
  - caps at `max_per_tick = 10` and leaves the rest for the next tick.

- **`FIRE_COST_AUDIT`** (4 tests):
  - metadata.
  - SKIPPED when `router_state.json` is missing.
  - writes snapshot with no issues when under budget.
  - flags `over_budget` issue when `api_cost_today > daily_budget_usd`.

**Integration test — `tests/autonomic/test_d06_integration.py`** (3 tests):

- three consecutive ticks (with only D-06 scheduled rules active) fire in order: `MODEL_EVAL` → `SESSION_ARCHIVE` → `COST_AUDIT`.
- reactive rule (`errors_present`) preempts all three.
- after D-06 runs on seeded data, `knowledge/autonomic/model_eval_log.jsonl` and `knowledge/autonomic/cost_audit_log.jsonl` both have exactly one line.

**Mock strategy:**
- `patch("backend.autonomic.levers.model_eval.EVALUATOR")` — replace singleton with a Mock that returns canned dicts.
- `SESSION_ARCHIVE` and `COST_AUDIT` have no external dependencies; use `tmp_path` for fake sessions.json / router_state.json.

**Test count target:** 18 new tests. Combined autonomic suite post-D-06: ~204 tests.

---

## 8. Open questions (not blocking)

1. **`daily_budget_usd` default** — is 10.0 USD/day the right trigger? The user's current `api_cost_today` is 0.0 (using local Ollama). Leave the default and expose via rule params so it can be tuned in `default_rules()` later.
2. **`eval_daily.json` cache** — `SelfEvaluator.__init__` references a `daily_path = kb_dir / "eval_daily.json"` that the codebase does not currently write or read. Not relevant to D-06; just noting during exploration.
3. **`current_id` policy** — safer to never archive the active session even if it's old+consolidated. Edge case but worth a test. Covered in Section 3 and Section 7.

---

## 9. What comes after D-06

Next sub-project: **D-07** per updated parent spec Section 11. Reflection cohort: `FIRE_SELF_REFLECTION` + `FIRE_FINETUNE_QC` + `FIRE_GAP_DETECTION`. Those levers read the D-06 telemetry logs (`model_eval_log.jsonl`, `cost_audit_log.jsonl`) plus existing artifacts (`gaps.json`, `finetune_queue.jsonl`) and ask the cortex to analyze patterns and suggest actions. D-07 is where the telemetry D-06 collected becomes actionable.

After D-07: **D-08** body cohort (yellow-safety `FIRE_TOOL_INSTALL`, Linux-only OS inventory extras, AutonomicPanel frontend, `backend/autonomic/api.py` expansion).
