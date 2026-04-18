# D-05 — Knowledge curation cohort (design)

**Status:** Design (no implementation)
**Date:** 2026-04-18
**Parent:** [Model X spec, Section 11 — phased delivery](./2026-04-16-model-x-autonomic-design.md)
**Delivers:** `FIRE_NOTE_CURATION`, `FIRE_GRAPH_MAINTENANCE`, `FIRE_PROACTIVE_LEARN` — and retirement of `backend/background.py`.

---

## 0. Context

D-01 through D-04 are merged. The autonomic scheduler runs 9 levers with per-rule cooldowns on a single-tempo 30s tick. D-04 added self-knowledge levers (`CAPABILITY_SCAN`, `SELF_STUDY`) that read the agent's own code. D-05 turns the focus outward — to the notes and graph the agent builds about the world and the user's interests.

**Scope adjustment note:** Section 11 of the parent spec originally listed D-05 as two levers (`NOTE_CURATION` + `GRAPH_MAINTENANCE`) plus retirement of `backend/background.py`. During brainstorming on 2026-04-18 we decided to make the retirement concrete by introducing a dedicated third lever `FIRE_PROACTIVE_LEARN` that absorbs `background.py::learn_topic_bg`. The parent spec Section 11 is updated alongside this document.

**Goal:** keep the knowledge base fresh, the graph clean of dead references, and turn proactive learning goals into notes without the side-channel of `background.py`.

**Current corpus scale (grounding):** 13 notes in `knowledge/profession/`, 0 in other categories; graph holds 413 entities / 793 edges / 14 notes indexed. Every scope decision in this spec is sized for this corpus: we do the minimum that helps now and leaves obvious extension points for when the corpus grows.

**Non-goals (explicit):** duplicate-note merging, automatic `[[wiki-link]]` insertion, entity normalization via LLM, embedding indexing, deletion of any note (yellow safety), event-bus integration of `follow_ups` from other levers. Each of these has a pointer in Section 8 to the later sub-project that will tackle it.

---

## 1. Three levers

| Lever | Safety | Executor | Cadence | Absorbs |
|---|---|---|---|---|
| `FIRE_NOTE_CURATION` | green | claude | weekly (604800s) | — |
| `FIRE_GRAPH_MAINTENANCE` | green | python | daily (86400s) | — |
| `FIRE_PROACTIVE_LEARN` | green | claude | hourly (3600s) | `backend/background.py::learn_topic_bg` |

All three are autonomic-category levers (shipped via `register_default_autonomic_levers`). `default_rules()` grows 9 → 12. Order: reactive rules (4) → D-03 scheduled (3) → D-04 scheduled (2) → D-05 scheduled (3) = 12.

---

## 2. `FIRE_NOTE_CURATION`

**Purpose:** refresh notes that look stale, so the knowledge base does not drift away from reality.

**Behavior:**

1. Load `knowledge/index.json`. Build candidate list of notes matching **any** of:
   - Frontmatter `confidence` is `partial` or `unverified` (initial creation lacked good sources).
   - Frontmatter `updated` is older than 30 days AND `access_count >= 5` (hot but aging).
2. Filter: exclude notes whose path lives under `knowledge/personal/` or `knowledge/projects/` — those can contain user-specific content the web cannot refresh without damage.
3. Rank candidates: lowest-confidence first, then oldest `updated`, then highest `access_count`.
4. Take the top **N = 2** per tick. For each:
   1. Call `note_creator.learn_topic(topic=<note.topic>, depth="quick", category=<note.category>)`.
   2. `KM.save_note` inside `learn_topic` automatically snapshots the previous version to `_history/` (invariant preserved — no new rollback logic needed).
   3. After save, extraction writes new graph edges via `GRAPH.add_relations` (existing behavior).
5. Return `LeverReport` with `outcome = {candidates, refreshed, skipped, errors}` and `reason = f"curated_{refreshed}_notes"`.

**Preconditions:** always True. The cooldown gates cadence; empty candidate list returns `LeverStatus.SKIPPED` with `reason = "no_stale_notes"`.

**Cost cap:** 2 `learn_topic` calls per week ≈ 8/month. Each call does one web search + one cortex call (`NOTE_CREATION`). Acceptable.

**Out of scope (deferred):**
- Merging near-duplicate notes: needs embeddings. Listed in Section 8 → D-07 or later.
- Deletion of nothingburger notes: yellow safety — future `AutonomicPanel` approval.
- Cross-link filling (auto `[[wiki-links]]`): needs corpus-wide semantic pass.

---

## 3. `FIRE_GRAPH_MAINTENANCE`

**Purpose:** prune dead references from `knowledge/graph.json` — edges whose source note was deleted, entities with no surviving edges. Pure Python, no cortex.

**Behavior:**

1. Load `knowledge/graph.json`. Build set of known note slugs from `knowledge/index.json`.
2. Iterate `edges` dict. For each entity's edge list:
   - Drop any edge whose `note` field does NOT appear in the known-slugs set.
   - Count dropped edges.
3. After edge pruning, iterate entities. If an entity's edge list is empty AND no surviving edge anywhere references it as `target`, remove the entity.
4. Save `graph.json` via the same atomic pattern used in `_save` (overwrite with JSON). If nothing changed, skip the write to avoid unnecessary mtime churn.
5. Return `LeverReport` with `outcome = {edges_before, edges_after, edges_removed, entities_before, entities_after, entities_removed}` and `reason = f"graph_pruned:{edges_removed}_edges"`.

**Preconditions:** always True. Missing `graph.json` or empty graph returns `SKIPPED` with `reason = "empty_graph"`.

**Idempotence:** a second tick with no changes returns `edges_removed == 0` and does not rewrite the file.

**Thread/process safety:** acceptable to do a read-mutate-write cycle in a single process — the autonomic scheduler is the only writer to `graph.json` besides `learn_topic` (which only appends via `add_relations`). Concurrency is not a concern in v0.

**Out of scope (deferred):**
- Entity normalization via LLM (`python_async` vs `asyncio_python`) — Section 8 → D-07.
- Edge weight decay by age/access — premature optimization at 793 edges.
- Embedding index construction — separate infra, D-06 or D-07.

---

## 4. `FIRE_PROACTIVE_LEARN` + retirement of `backend/background.py`

**Purpose:** one proactive goal per hour becomes a note. Replaces the `BACKGROUND.process_proactive_goals()` path that was triggered ad-hoc from the chat flow.

**Behavior (`run`):**

1. Read `knowledge/goals.json` via `backend.goals.GOALS.active_goals()`.
2. Filter: `goal_type == "proactive"` AND `status == "active"` AND `description.startswith("Learn about: ")`.
3. If no candidate: return `SKIPPED` with `reason = "no_proactive_goals"`.
4. Pick the first candidate. Extract topic: `goal.description.removeprefix("Learn about: ").strip()`.
5. Call `note_creator.learn_topic(topic, depth="quick", category="profession")`.
6. On success: `GOALS.complete_goal(goal.id, f"Learned: {note.frontmatter.topic}")`.
7. On failure: `goal.add_progress(f"Lever failed: {exc}")` and return `LeverStatus.FAILURE` with `reason = f"learn_failed:{exc}"`. Do not mark the goal failed — next tick can retry.
8. Return LeverReport with `outcome = {topic, note_topic, category}`, `reason = f"learned_{topic}"`.

**Preconditions:** always True. Cooldown and the "no candidate" branch gate cadence.

### 4.1 Retirement plan for `backend/background.py`

Executed in the same plan as the lever lands so master never has two proactive-learning paths coexisting.

1. **Delete** `backend/background.py`.
2. **Delete** `tests/` references to `BACKGROUND` — there are no dedicated tests; only indirect uses in `main.py`.
3. **Update** `backend/main.py`:
   - Remove `from .background import router as background_router` import.
   - Remove `app.include_router(background_router)` call.
   - Remove the chat-flow call `await BACKGROUND.process_proactive_goals()` (currently around line 137).
4. **HTTP surface change:** the four endpoints `/api/background`, `/api/background/learn`, `/api/background/cancel`, `/api/background/process-goals` disappear. This is a breaking change for any frontend code that calls them.
   - Check `frontend/src/**` for usages. If any exist, delete or stub them in the same plan (document in README).
   - After retirement, `/api/autonomic/status` remains the surface for introspecting autonomic activity; a future D-07 `AutonomicPanel` will show lever run history.
5. **Update** `README.md`: remove the "background tasks" mention (if present) and add the D-05 lever description.
6. **Update** the parent Model X spec Section 11 note about `background.py` — mark it retired under D-05.

**Why retire now, not keep the shim:** background.py duplicates what `FIRE_PROACTIVE_LEARN` does. Keeping both means two paths can fire on the same goal, doubling cost and corrupting `goal.status` via race.

---

## 5. Layer 0 rule extension (9 → 12)

`backend/autonomic/layer0.py::default_rules()` appends three rules in this order:

```python
LayerZeroRule(
    name="graph_maintenance_tick",
    predicate=lambda s: True,
    lever="FIRE_GRAPH_MAINTENANCE",
    params={},
    cooldown_seconds=86400.0,
),
LayerZeroRule(
    name="proactive_learn_tick",
    predicate=lambda s: True,
    lever="FIRE_PROACTIVE_LEARN",
    params={},
    cooldown_seconds=3600.0,
),
LayerZeroRule(
    name="note_curation_tick",
    predicate=lambda s: True,
    lever="FIRE_NOTE_CURATION",
    params={},
    cooldown_seconds=604800.0,
),
```

**Ordering rationale:**
- `graph_maintenance_tick` first — cheap, daily, runs before any new notes come in via the other two.
- `proactive_learn_tick` second — hourly, higher frequency than curation, more likely to match on any given tick.
- `note_curation_tick` last — weekly, runs when nothing else needs to fire.

All three are `predicate=True` schedule-driven; cooldown fall-through from D-03 lets all three take turns on consecutive ticks. Reactive rules and D-03/D-04 scheduled rules still preempt.

`register_default_autonomic_levers()` grows from 5 registrations to 8.

---

## 6. File layout

**New files:**

```
backend/autonomic/levers/
├── note_curation.py          # FIRE_NOTE_CURATION (~140 lines)
├── graph_maintenance.py      # FIRE_GRAPH_MAINTENANCE (~120 lines)
└── proactive_learn.py        # FIRE_PROACTIVE_LEARN (~110 lines)

tests/autonomic/
├── test_curation_levers.py   # unit tests for all 3 levers
└── test_d05_integration.py   # end-to-end + retirement verification
```

**Modified files:**

- `backend/autonomic/layer0.py` — `default_rules()` 9 → 12.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers()` 5 → 8 registrations.
- `backend/main.py` — remove `background_router` import + include_router call + `BACKGROUND.process_proactive_goals()` from chat flow.
- `README.md` — list D-05 levers, remove any background-section mention, note the HTTP surface change.
- Parent spec Section 11 — update D-05 to three-lever cohort; mark `background.py` retired.

**Deleted files:**

- `backend/background.py` — absorbed by `FIRE_PROACTIVE_LEARN`.

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `api.py`, `startup.py`, existing D-01..D-04 levers, `backend/goals.py`, `backend/note_creator.py`, `backend/knowledge_graph.py`, `backend/knowledge_manager.py`.

---

## 7. Testing strategy

**Unit tests — `tests/autonomic/test_curation_levers.py`:**

- **`FIRE_NOTE_CURATION`** (6 tests):
  - metadata.
  - empty index returns SKIPPED with `no_stale_notes`.
  - picks `confidence=partial` note first.
  - picks 30-day-old note with `access_count >= 5` second.
  - excludes `personal/` and `projects/` paths from candidates.
  - caps at N=2 per tick (with 5 candidates, only 2 processed).

- **`FIRE_GRAPH_MAINTENANCE`** (5 tests):
  - metadata.
  - empty graph → SKIPPED with `empty_graph`.
  - prunes edges whose `note` is missing from index.
  - prunes entities with no remaining edges after edge prune.
  - idempotent second run reports zero changes.

- **`FIRE_PROACTIVE_LEARN`** (6 tests):
  - metadata.
  - no proactive goals → SKIPPED with `no_proactive_goals`.
  - picks first `proactive + active` goal; ignores `user`/`improvement`/`completed`.
  - calls `learn_topic` with the extracted topic and marks goal completed.
  - on `learn_topic` exception, goal stays active and lever returns FAILURE.
  - ignores goals whose description doesn't start with `"Learn about: "`.

**Integration test — `tests/autonomic/test_d05_integration.py`** (4 tests):

- three consecutive ticks fire `GRAPH_MAINTENANCE`, then `PROACTIVE_LEARN`, then `NOTE_CURATION` in that order (cooldown fall-through).
- reactive rule (`errors_present`) still preempts the three new D-05 rules.
- after retirement, `from backend.main import app` does not import `BACKGROUND`, and `app.routes` contains no `/api/background/*` paths.
- after retirement, `backend/background.py` file does not exist.

**Mock strategy:**
- `patch("backend.autonomic.levers.note_curation.learn_topic")` and `patch("backend.autonomic.levers.proactive_learn.learn_topic")` to avoid real web calls.
- `patch("backend.autonomic.levers.proactive_learn.GOALS")` for goal lifecycle assertions.
- `FIRE_GRAPH_MAINTENANCE` is pure-python — no mocks needed; uses `tmp_path` for fake graph + index.

**Test count target:** 21 new tests. Combined autonomic suite post-D-05: ~183 tests.

---

## 8. Open questions (not blocking)

1. **Candidate ranking in NOTE_CURATION** — three sort keys (confidence, updated, access_count) combined. Simple lexicographic sort. If it proves too aggressive on `partial` notes, switch to weighted score in the plan.
2. **`personal/` / `projects/` exclusion list** — currently hard-coded. If the user creates custom categories later, this list may need widening. Revisit when new categories ship.
3. **`/api/background/*` frontend callers** — the plan must grep `frontend/src/**` to find any. If there are frontend references, delete or stub them alongside the backend retirement.
4. **Retry policy on `learn_topic` failure in `PROACTIVE_LEARN`** — current design keeps the goal active and returns FAILURE. If the same goal keeps failing, it will retry hourly forever. Acceptable for v0 (rare path); if it becomes noisy, add a fail-count field to the goal in D-06 or D-07.

---

## 9. What comes after D-05

Next sub-project: **D-06** per updated parent spec Section 11. Telemetry cohort: `FIRE_MODEL_EVAL`, `FIRE_SESSION_ARCHIVE`, `FIRE_COST_AUDIT`, plus the remaining autonomic-category levers (`FIRE_SELF_REFLECTION`, `FIRE_FINETUNE_QC`, `FIRE_GAP_DETECTION`).

The items deferred from D-05:
- Near-duplicate note merging — needs embeddings. Candidate for D-07 after AutonomicPanel lets users review yellow-safety merges.
- Auto `[[wiki-link]]` filling — needs corpus-wide semantic pass. D-07 candidate.
- Entity normalization in graph via LLM — D-07 candidate.
- `follow_ups` event integration from other levers (e.g. `MEMORY_CONSOLIDATION`'s `topic_threads` → goals) — D-06 or D-07, possibly via a shared event handler in `backend/autonomic/events.py`.
- bge-m3 embedding index + hybrid searcher integration — D-06 or D-07 infra work.
