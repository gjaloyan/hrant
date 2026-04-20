# D-07 — Reflection cohort (design)

**Status:** Design (no implementation)
**Date:** 2026-04-20
**Parent:** [Model X spec, Section 11 — phased delivery](./2026-04-16-model-x-autonomic-design.md)
**Delivers:** `FIRE_SELF_REFLECTION`, `FIRE_FINETUNE_QC`, `FIRE_GAP_DETECTION`.

---

## 0. Context

D-01 through D-06 are merged. The autonomic scheduler runs 15 levers on per-rule cooldowns (4 immune, 11 autonomic). D-06 gave the agent a time series of quality/cost/session telemetry. D-07 is the reflection cohort: it looks at what D-06 and the normal chat flow have recorded, and asks (sometimes the cortex, sometimes algorithmically) what it means.

**Scope adjustment note:** parent Section 11 originally declared D-07 as "all claude-delegate". During brainstorming on 2026-04-20 we looked at the underlying modules: only `META_LEARNER.extract_patterns()` genuinely uses the cortex today (via `PATTERN_EXTRACTION_SYSTEM`). `FinetuneDataCurator` is algorithmic. `gaps.json` is a counter file. For v0 data sizes (10 finetune pairs, 5 gaps) any cortex call on FINETUNE_QC/GAP_DETECTION would be premature — the inputs are too small to synthesize useful patterns. We ship pure-Python aggregators and leave a cortex upgrade for later when the corpus justifies it.

**Goal:** three green-safety autonomic levers that convert raw telemetry and training-data artifacts into actionable snapshots. One (`SELF_REFLECTION`) lets Claude look at recent failures; two (`FINETUNE_QC`, `GAP_DETECTION`) compute distributions and publish them for downstream readers.

**Non-goals:** cortex analysis of finetune queue diversity (too small to be useful at 10 pairs — reconsider at 100+), cortex clustering of gaps by theme (too small at 5 gaps), automatic migration of legacy `finetune_queue.jsonl` entries (yellow safety — belongs in D-08), auto-fixing error patterns (already partially handled by `META_LEARNER` creating improvement goals — we just observe).

---

## 1. Three levers with mixed executors

| Lever | Safety | Executor | Cadence | Wraps |
|---|---|---|---|---|
| `FIRE_SELF_REFLECTION` | green | claude | nightly (86400s) | `META_LEARNER.extract_patterns()` + `stats()` |
| `FIRE_FINETUNE_QC` | green | python | daily (86400s) | `FinetuneDataCurator.score_all()` + aggregate |
| `FIRE_GAP_DETECTION` | green | python | daily (86400s) | `knowledge/gaps.json` aggregate |

All three are autonomic-category levers. `default_rules()` grows 15 → 18; `register_default_autonomic_levers()` 11 → 14.

**Why mixed executors:** the cohort is united by its *purpose* (reflect on collected data to drive improvement), not by its *implementation*. `SELF_REFLECTION` needs Claude because failure analysis requires natural-language understanding. The other two aggregate numeric fields; Claude would not add value at current data sizes.

---

## 2. `FIRE_SELF_REFLECTION`

**Purpose:** periodically ask the cortex to summarize recurring failure patterns from the last ~30 logged agent failures, save the patterns, and audit-log what it saw.

**Dependencies:** `backend.meta_learner.META_LEARNER` — singleton `MetaLearner` exposing `extract_patterns()` and `stats()`. Internally `extract_patterns` reads `knowledge/error_log.jsonl`, needs at least 3 entries whose `analysis` field is populated (the analysis comes from the live chat-flow call to `META_LEARNER.analyze_failure()` inside `/api/chat`), calls Claude via `PATTERN_EXTRACTION_SYSTEM`, writes `knowledge/error_patterns.json`, and creates `improvement`-type goals for patterns where `priority >= 7`.

**Behavior (`run`):**

1. Call `META_LEARNER.stats()` — returns `{total_failures, by_root_cause, by_domain, avg_severity, patterns_count, patterns}`.
2. If `stats["total_failures"] < 3` — return `LeverStatus.SKIPPED` with `reason="insufficient_failures"` (no new signal; skip the cortex call to save tokens).
3. Call `META_LEARNER.extract_patterns()` — returns the resulting patterns list. `extract_patterns` internally skips the cortex call if `< 3` failures carry an `analysis` payload, so the lever may get back the old cached patterns.
4. Append a snapshot to `knowledge/autonomic/self_reflection_log.jsonl`:
   ```json
   {"ts": "...", "total_failures": N, "by_root_cause": {...}, "by_domain": {...}, "avg_severity": X, "patterns_count": P, "patterns": [...]}
   ```
5. Return `LeverReport` with `outcome = {total_failures, avg_severity, patterns_count}` and `reason = f"reflected_on_{total_failures}_failures"`.

**Preconditions:** always True. The `< 3` guard runs inside `run`.

**Error handling:** wrap `extract_patterns()` in try/except — if the cortex call inside fails (`LLMError`), `META_LEARNER` already catches it and returns the old patterns list. If something else raises, log and return `LeverStatus.FAILURE` with `reason=f"reflect_failed:{exc}"`.

**Idempotence:** safe. Multiple runs in the same day just append multiple snapshots; `error_patterns.json` gets rewritten each time with the latest clustering. Redundant but not harmful.

---

## 3. `FIRE_FINETUNE_QC`

**Purpose:** produce a daily audit of the fine-tune queue — score distribution, category mix, curated count, legacy-format count — so the human can see at a glance whether the queue is growing and whether it is biased.

**Dependencies:** `backend.finetune_curator.FinetuneDataCurator` (pure algorithmic scorer — no cortex). `backend.models.FinetunePair` (Pydantic model).

**Data format caveat:** `knowledge/finetune_queue.jsonl` contains two formats in the same file:
- **Legacy** (early entries): `{"instruction": "...", "response": "...", "sources": [...], "confidence": N, "timestamp": "..."}`.
- **Current** (post-migration): full `FinetunePair` JSON — `{id, messages: [...], metadata: {...}}`.

The lever tolerates both: lines that fail `FinetunePair.model_validate_json(line)` are counted under `legacy_entries` and skipped for scoring. Migration of legacy entries is deferred (Section 8).

**Behavior (`run`):**

1. Read `knowledge/finetune_queue.jsonl` line by line. For each line:
   - Try `FinetunePair.model_validate_json(line)`. Success → append to `pairs`. Failure → `legacy_entries += 1`.
2. If the file is missing or `pairs` is empty (only legacy or nothing):
   - Return `LeverStatus.SKIPPED` with `reason="no_valid_pairs"` and `outcome = {total: 0, legacy_entries}`.
3. `scored = FinetuneDataCurator().score_all(pairs)` — list of `ScoredPair(pair, score)`.
4. Compute distribution buckets: `low = sum(s.score < 0.5)`, `medium = sum(0.5 <= s.score < 0.7)`, `high = sum(s.score >= 0.7)`.
5. Compute `by_category`: dict mapping category → count (e.g. `{"factual_qa": 5, "correction": 2, ...}`).
6. Compute `boosted_count`, `verified_count` from `pair.metadata`.
7. Compute `avg_score = sum(scores) / len(scores)`.
8. `curated = FinetuneDataCurator().curate(pairs)` — list of pairs passing `MIN_SCORE = 0.7` AND dedup.
9. Append to `knowledge/autonomic/finetune_qc_log.jsonl`:
   ```json
   {"ts": "...", "total": N, "legacy_entries": L, "low": x, "medium": y, "high": z, "curated": C, "boosted": B, "verified": V, "avg_score": A, "by_category": {...}}
   ```
10. Return `LeverReport` with `outcome = {total, curated, avg_score, legacy_entries}` and `reason = f"qc_{total}_pairs"`.

**Preconditions:** always True.

**Observational only:** the lever **does not** mutate `finetune_queue.jsonl`. No deletion, no migration, no reformatting. If the human later wants automatic legacy-migration, that is a yellow-safety change (rewrites a persistent artifact) and belongs in D-08 with `AutonomicPanel` approval, not here.

---

## 4. `FIRE_GAP_DETECTION`

**Purpose:** produce a daily snapshot of the knowledge-gap tracker — total gaps, actionable gaps (those a proactive goal could be made for), stale gaps, and the top-5 hot topics. Complements D-03's `FIRE_GOAL_PROPOSE` which acts on individual gaps; `GAP_DETECTION` gives a time series of the whole tracker.

**Dependencies:** `knowledge/gaps.json` — dict keyed by topic slug, values `{topic, count, last}`. Read directly; no cortex, no `KM` wrapper.

**Behavior (`run`):**

1. Read `knowledge/gaps.json`. If missing or empty dict — return `LeverStatus.SKIPPED` with `reason="no_gaps"`.
2. Parse entries. For each dict-valued entry:
   - Count toward `total_gaps`.
   - If `count >= 2` → count toward `actionable_gaps` (matches `GOALS.suggest_from_gaps` threshold).
   - If `last` parses to datetime older than 30 days → count toward `stale_gaps`.
3. Sort by `count` descending, take top 5 → `hot_gaps` list of `{topic, count, last}`.
4. Append to `knowledge/autonomic/gap_detection_log.jsonl`:
   ```json
   {"ts": "...", "total": N, "actionable": A, "stale": S, "hot": [{"topic": "...", "count": K, "last": "..."}, ...]}
   ```
5. Return `LeverReport` with `outcome = {total_gaps, actionable_gaps, stale_gaps, hot_count}` and `reason = f"detected_{total}_gaps"`.

**Preconditions:** always True.

**Parametrization:** `gaps_path` and `log_path` overridable via params; defaults are the real paths. `actionable_threshold` defaults to 2 (matches `GOALS.suggest_from_gaps`).

**Observational only:** never rewrites `gaps.json`. The tracker is maintained by the chat flow (`KM._track_gap()` during topic lookups).

---

## 5. Layer 0 rule extension (15 → 18)

`default_rules()` appends three rules, in this order:

```python
LayerZeroRule(
    name="self_reflection_tick",
    predicate=lambda s: True,
    lever="FIRE_SELF_REFLECTION",
    params={},
    cooldown_seconds=86400.0,
),
LayerZeroRule(
    name="finetune_qc_tick",
    predicate=lambda s: True,
    lever="FIRE_FINETUNE_QC",
    params={},
    cooldown_seconds=86400.0,
),
LayerZeroRule(
    name="gap_detection_tick",
    predicate=lambda s: True,
    lever="FIRE_GAP_DETECTION",
    params={},
    cooldown_seconds=86400.0,
),
```

**Ordering rationale:** reactive rules and the 11 earlier scheduled rules (D-03 through D-06) still preempt. Among the three D-07 rules, `self_reflection_tick` runs first because it is the costliest (cortex call); if the cooldown fall-through lets multiple D-07 rules fire in consecutive ticks, the reflection snapshot is present before the finetune/gap reports reference it.

---

## 6. File layout

**New files:**

```
backend/autonomic/levers/
├── self_reflection.py         # FIRE_SELF_REFLECTION (~90 lines, claude)
├── finetune_qc.py             # FIRE_FINETUNE_QC (~140 lines, python)
└── gap_detection.py           # FIRE_GAP_DETECTION (~100 lines, python)

tests/autonomic/
├── test_reflection_levers.py  # ~18 unit tests
└── test_d07_integration.py    # 3 integration tests
```

**Modified files:**

- `backend/autonomic/layer0.py` — `default_rules()` 15 → 18.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers()` 11 → 14.
- `README.md` — list D-07 levers + new log paths.

**New runtime files (created by levers on first run):**

- `knowledge/autonomic/self_reflection_log.jsonl`
- `knowledge/autonomic/finetune_qc_log.jsonl`
- `knowledge/autonomic/gap_detection_log.jsonl`

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `api.py`, `startup.py`, `backend/meta_learner.py`, `backend/finetune_curator.py`, `backend/knowledge_manager.py`, `backend/main.py`, existing D-01..D-06 levers, frontend.

---

## 7. Testing strategy

**Unit tests — `tests/autonomic/test_reflection_levers.py`:**

- **`FIRE_SELF_REFLECTION`** (5 tests):
  - metadata (name, category, safety, executor=claude).
  - SKIPPED when `stats.total_failures < 3`.
  - calls `extract_patterns` and appends snapshot when enough failures.
  - tolerates `extract_patterns` raising (FAILURE status with exception reason).
  - preconditions always True.

- **`FIRE_FINETUNE_QC`** (7 tests):
  - metadata (executor=python).
  - SKIPPED when `finetune_queue.jsonl` is missing.
  - SKIPPED when file has only legacy entries.
  - scores valid pairs and buckets them into low/medium/high.
  - counts `legacy_entries` alongside valid pairs.
  - `curated` count matches `MIN_SCORE=0.7` + dedup behavior.
  - writes snapshot with `by_category` distribution.

- **`FIRE_GAP_DETECTION`** (6 tests):
  - metadata.
  - SKIPPED when `gaps.json` is missing or empty.
  - counts `actionable_gaps` (count >= 2) correctly.
  - counts `stale_gaps` (`last > 30 days ago`) correctly.
  - `hot` list sorted by count desc, capped at 5.
  - writes snapshot to `gap_detection_log.jsonl`.

**Integration test — `tests/autonomic/test_d07_integration.py`** (3 tests):

- three consecutive ticks (D-07 rules only) fire `SELF_REFLECTION` → `FINETUNE_QC` → `GAP_DETECTION` in order.
- reactive rule (`errors_present`) preempts all three.
- after one run of each, the three JSONL log files each have exactly one line.

**Mock strategy:**
- `patch("backend.autonomic.levers.self_reflection.META_LEARNER")` — replace with Mock that returns canned `stats()` and `extract_patterns()`.
- `FINETUNE_QC` and `GAP_DETECTION` are pure-python — use `tmp_path` for fake queue/gaps files.

**Test count target:** 21 new tests. Combined autonomic suite post-D-07: ~227 tests.

---

## 8. Open questions (not blocking)

1. **Legacy finetune_queue migration** — 1 of the 10 current entries is legacy format. Out of scope for D-07; D-08 with AutonomicPanel approval is the right place.
2. **Cortex upgrade for FINETUNE_QC/GAP_DETECTION** — revisit when finetune queue > 100 entries or gaps > 20. Both levers would ask Claude to identify diversity gaps / cluster themes. Not blocking D-07.
3. **Goal auto-creation from gap clustering** — D-03's `FIRE_GOAL_PROPOSE` already creates goals from individual gaps. Cluster-based goals (e.g. "learn Python async family") would be a natural D-07+ extension once gap_detection_log shows persistent clusters.

---

## 9. What comes after D-07

Next sub-project: **D-08** per updated parent spec Section 11. Body + UI cohort: `FIRE_TOOL_INSTALL` (yellow safety — requires user approval via pending_approvals.jsonl) + `AutonomicPanel.tsx` frontend panel (visualizes lever history, pending approvals, immune signatures, kill switch) + `backend/autonomic/api.py` expansion (additional status endpoints beyond `/status`) + optional Linux-only OS inventory extras for `FIRE_CAPABILITY_SCAN`.

With D-08, Model X reaches the full 19-lever catalog from Section 3 of the parent spec (we are at 15 + 3 = 18 after D-07; D-08 adds `FIRE_TOOL_INSTALL` for 19), plus first-class UI observability.
