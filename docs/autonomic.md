# Autonomic Subsystem (Model X)

Hrant includes an autonomic controller that runs in the background
alongside the cortex. It's modelled after the human autonomic nervous
system: reflexes (L0 rules), routing (L1 classifier, v1+), diagnosis
(L2 small LLM, v1+), and escalation to cortex (L3).

Design doc: [docs/superpowers/specs/2026-04-16-model-x-autonomic-design.md](superpowers/specs/2026-04-16-model-x-autonomic-design.md) (section 11 — phased delivery D-01..D-09).

## Levers (D-01 → D-09 delivered)

### Immune levers (D-02, react to ongoing errors/load)
- `FIRE_SERVER_HEALTH` — disk / memory / CPU threshold check (green).
- `FIRE_ERROR_TRIAGE` — classifies `error_log.jsonl` entries by severity (green).
- `FIRE_SELF_HEAL` — looks up an immune signature and returns its fix plan (green).
- `FIRE_SERVICE_REPAIR` — whitelist-gated `systemctl restart` with `max_attempts`, POSIX only (green, skipped on non-POSIX).

### Autonomic levers (D-03, scheduled self-maintenance)
- `FIRE_INTEGRITY_HEARTBEAT` — every 5 min, read-only check of `knowledge/index.json` vs files (green).
- `FIRE_GOAL_PROPOSE` — hourly, wraps `GOALS.suggest_from_gaps(gaps.json)` (green).
- `FIRE_MEMORY_CONSOLIDATION` — daily, reviews recent sessions and routes facts to `identity/user.md`, `memory_facts.jsonl`, and `sessions.json` summary field (green, delegates to cortex).

### Self-knowledge levers (D-04)
- `FIRE_CAPABILITY_SCAN` — every 6h, inventories `backend/tools/`, `backend/skills/`, `knowledge/channels.json`, and the host via psutil into `knowledge/self/` (green, python).
- `FIRE_SELF_STUDY` — daily, reads up to 3 priority-ordered `backend/**/*.py` modules per tick and writes one markdown note per module to `knowledge/self/modules/` via cortex (green, claude).

### Knowledge curation levers (D-05)
- `FIRE_GRAPH_MAINTENANCE` — daily, prunes orphan edges and unreferenced entities from `knowledge/graph.json` (green, python).
- `FIRE_PROACTIVE_LEARN` — hourly, picks one `goal_type="proactive"` goal (`"Learn about: X"` description) and runs `learn_topic` to create the note (green, claude).
- `FIRE_NOTE_CURATION` — weekly, refreshes notes with `confidence="partial"/"unverified"` or 30+ days old with `access_count >= 5`, up to 2 per tick; excludes `personal/` and `projects/` categories (green, claude).

### Telemetry levers (D-06)
- `FIRE_MODEL_EVAL` — daily, aggregates yesterday's `knowledge/eval_log.jsonl` via `EVALUATOR` (daily_report + regressions + priorities) into `knowledge/autonomic/model_eval_log.jsonl` (green, python).
- `FIRE_SESSION_ARCHIVE` — daily, moves sessions older than 30 days and `consolidated=True` from active `knowledge/sessions.json` into `knowledge/_history/<session_id>.json`. Caps at 10 per tick; never archives the `current_id` (green, python).
- `FIRE_COST_AUDIT` — hourly, snapshots `knowledge/router_state.json` into `knowledge/autonomic/cost_audit_log.jsonl`. Flags `over_budget` when `api_cost_today > daily_budget_usd` (green, python).

### Reflection levers (D-07)
- `FIRE_SELF_REFLECTION` — nightly, wraps `META_LEARNER.extract_patterns()` which asks Claude to cluster recent failures in `error_log.jsonl` into patterns, saves them to `error_patterns.json`, and auto-creates improvement goals for high-priority patterns. Audit-snapshots into `knowledge/autonomic/self_reflection_log.jsonl` (green, claude).
- `FIRE_FINETUNE_QC` — daily, scores `knowledge/finetune_queue.jsonl` via `FinetuneDataCurator` (pure-python), aggregates distribution (low/medium/high), categories, boosted/verified counts, curated size. Observational; never mutates the queue (green, python).
- `FIRE_GAP_DETECTION` — daily, aggregates `knowledge/gaps.json` — total gaps, actionable (count >= 2), stale (last > 30 days), top-5 hot topics. Snapshots to `knowledge/autonomic/gap_detection_log.jsonl` (green, python).

### Body + yellow lever (D-08)
- `FIRE_TOOL_INSTALL` — yellow safety; supports `pip_install {package}`, `ollama_pull {model}`, `llama_cpp_pull {url→.gguf}`. Enqueued via `POST /api/autonomic/pending`, executed only after `POST /api/autonomic/pending/{id}/approve`. No uninstalls/removes.

## HTTP endpoints

```
GET  /api/autonomic/status                — kill switch, scheduler liveness, registered lever names
GET  /api/autonomic/ticks?limit=50        — recent tick_log entries, newest-first
GET  /api/autonomic/levers/{name}?limit=N — recent reports for one lever
GET  /api/autonomic/pending               — pending yellow approvals
POST /api/autonomic/pending               — enqueue yellow action (body: {lever, params})
POST /api/autonomic/pending/{id}/approve  — execute approved action, remove from pending
POST /api/autonomic/pending/{id}/reject   — remove without executing
GET  /api/autonomic/immune                — immune signatures
POST /api/autonomic/kill-switch           — toggle enabled (body: {enabled: bool})
GET  /api/autonomic/settings              — current tick interval + range
PUT  /api/autonomic/settings              — set tick_interval_seconds (1..3600, live)
```

## Frontend (D-09)

- `🦾 Autonomic` panel (`frontend/src/components/AutonomicPanel.tsx`) shows kill switch, tick-interval slider, pending approvals with Approve/Reject, recent ticks, lever grid, immune signatures, and a per-lever history drawer.
- `StatusBar` at the bottom of every tab shows `autonomic: N levers` with a health dot and `⚠ N pending` badge when yellow actions are queued.

## Paths

- Kill switch: `knowledge/autonomic/ENABLED` — set content to `false` to disable.
- Logs: `knowledge/autonomic/lever_log.jsonl`, `tick_log.jsonl`, `pending_approvals.jsonl`.
- Immune DB: `knowledge/immune/signatures.jsonl` (seed) + `knowledge/immune/fixes/` (markdown recipes).
- Self-knowledge: `knowledge/self/modules/`, `knowledge/self/tools/`, `knowledge/self/skills/`, `knowledge/self/mcp_servers/`, `knowledge/self/server_inventory.md` (written by D-04 levers).
- Telemetry logs (D-06): `knowledge/autonomic/model_eval_log.jsonl`, `knowledge/autonomic/cost_audit_log.jsonl`.
- Session history: `knowledge/_history/<session_id>.json` (written by `FIRE_SESSION_ARCHIVE`).
- Reflection logs (D-07): `knowledge/autonomic/self_reflection_log.jsonl`, `knowledge/autonomic/finetune_qc_log.jsonl`, `knowledge/autonomic/gap_detection_log.jsonl`.

## Env vars

Set before starting uvicorn to override defaults:
`AUTONOMIC_ENABLED_PATH`, `AUTONOMIC_TICK_SECONDS`, `AUTONOMIC_KNOWLEDGE_ROOT`, `AUTONOMIC_ERROR_LOG_PATH`, `AUTONOMIC_LEVER_LOG_PATH`, `AUTONOMIC_PENDING_PATH`, `AUTONOMIC_TICK_LOG_PATH`.

After Phase 5D, the tick interval is also editable via Settings → Autonomic in the WebUI (persists to `knowledge/autonomic_settings.json`, applies live).
