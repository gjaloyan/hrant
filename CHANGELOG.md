# Changelog

Notable user-facing changes per release / phase. Format roughly
follows [Keep a Changelog](https://keepachangelog.com/) but without
the strict version-tag discipline — Hrant ships continuously and
"phases" are how features were grouped in the dev process.

For the full commit history, see `git log`. This file focuses on
**what users should know** when running `hrant update`.

---

## AGI initiative — smart with small models (2026-06-10 → 2026-06-15)

A two-week push to make Hrant genuinely intelligent on cheap/small
models by moving wisdom, method and identity into the agent's "body"
(files that survive any model swap). Full conceptual map + the agent's
work philosophy: [docs/cognition.md](docs/cognition.md).

**Cognitive systems**

- **Model cascade** (`backend/cascade.py`) — small model answers
  first, a strong-model verifier gates it, escalate only on failure.
  Test-mode, owner opt-in via Fine-Tune panel / `GET/PUT /api/cascade`
  (gate + small-tier iteration budget configurable).
- **Per-task model routing** (`backend/model_routing.py`) — cheap
  task types (classification / quick-answer / keyword) run on a cheap
  model; judges + heavy turns stay strong. `GET/PUT /api/model-routing`.
- **Method before execution** (M2 prompt) — for a substantive task the
  agent establishes the proper methodology first (recall → research how
  experts do it → cover every dimension), then executes. Financial
  analysis now weighs news/catalysts/regulation, not price alone.
- **Knowledge vs skills vs trajectories** — three memories, used
  differently. New `save_knowledge` tool persists studied domain
  theory (study-once, recall-forever); skills are applied procedures;
  trajectories are past cases.
- **Trajectory memory** (`backend/trajectory_memory.py`) — case-based
  reasoning: similar past solved turns injected as PAST EXPERIENCE.
- **Plan scratchpad** — `set_plan` / `update_plan` checklist;
  self-correction rejects a final answer with pending steps.
- **Answer critic** (`backend/answer_critic.py`) — best-of-2 read-only
  revision when the verifier finds content problems.

**Calibration**

- **Grader calibration** — `VerificationResult` gains
  `content_confidence` + `endpoint_met`; the learning loop separates
  "didn't deliver the action" from "bad content".
- **Projection bucket** — verifier 4th claim type for explicitly-
  hedged, premise-grounded forecasts; excluded from the confidence
  penalty (a year-ahead forecast is no longer scored like a
  hallucination) with a projection cap of 85. Paired agent rule:
  weigh material-but-uncertain info, don't binary-filter it.

**Sleep cycle & learning loops** — nightly consolidation (digest →
lessons → pruning → trajectory replay); finetune auto-collection of
high-trust turns; correction auto-capture (user fixes the agent →
stored as a high-value `correction` pair); methodology-complete skill
creation.

**The body (soul / identity)** — rewrote `soul.md` + `identity.md`:
relationship to truth, judgment layer, ambitions and honest inner
life, **family-unit framing** (a member of the family, not property).
Verified on a 3B model — identity and morality hold across a model
swap. Work philosophy documented in [docs/cognition.md §0](docs/cognition.md).

**Reliability & UX fixes** — atomic writes + RLocks + pruning on all
JSON stores; failover recently-failed cache; `fetch_url` main-content
extraction via trafilatura; user timezone (local times, not UTC);
Reminders settings tab + reminder-delivery priority; honest token
display (fresh vs cached split); Codex refusal capture (refusals were
silently empty); three tool-call-dump leak formats in the chat
fast-path; owner-only helper refactor; per-task routing wins over the
provider chain.

---

## QA-audit-fix6 — closes the last 3 deferred refactors (2026-05-15)

Audit findings #21, #22, #25 were marked as "deferred (refactor,
not bug)" in the prior pass. This commit closes all three.

**#21 — `cli.py` split**

`cli.py` was 2417 lines and growing. Extracted the 5 largest
cohesive command groups into sibling modules:

  cli_jobs.py         5 cmds + row formatter
  cli_failover.py     6 cmds (status / enable / disable / add / remove / clear)
  cli_consolidate.py  4 cmds (Phase 16A daily consolidation)
  cli_graph.py        4 cmds (Phase 16C knowledge graph)
  cli_gateway.py      7 cmds + 5 platform helpers + `_gateway_service_running`

`cli.py` drops to **1516 lines (−37%)**. Argparse plumbing +
init/run/status/config/provider/update/rollback/discover/chat/
rebuild stay in `cli.py`. Test fixtures that monkeypatched
`_run_cmd` / `cmd_gateway_install` / `ROOT` on `cli_mod` now
patch on `cli_gateway` so the call sites see the mock.

**#25 — `init_wizard.py` provider-flow registry**

The provider dispatch was an if/elif chain — adding a new
provider meant editing both `_ascii_provider_menu` AND the
`run_wizard` body. Now there's a `PROVIDER_FLOWS` dict + a
`register_provider_flow(key, handler)` helper. Plugins / future
providers can register handlers from outside `init_wizard.py`.
The three built-in non-trivial flows (Codex, Copilot, Ollama)
register themselves at module load; everything else falls
through to the generic api-key flow on a registry miss.

**#22 — `llm.py` HTTP-retry helper**

Every provider client's `_post` did the same retry-with-backoff
dance with subtle variations. Extracted `post_with_retry(url,
*, payload, headers, provider_name, model, ...)` to consolidate:
exponential backoff, Retry-After header handling, retryable-status
allowlist, error formatting via an optional `parse_error` callback.

Migrated the two heaviest-traffic clients in this pass:
- `AnthropicLLM._post` (was ~65 lines → 17 lines)
- `OpenAICompatibleLLM._post` (was ~32 lines → 18 lines)

The other 6 clients keep their inline loops for now; they can
opt in to the helper at their own pace. Total lines cut: ~60.
More importantly, future retry-policy changes (e.g. adding 524
to the retryable set, supporting Retry-After-Ms) now touch one
function instead of eight.

Tests: 1279 still pass.

---

## QA-audit-fix5 — frontend code-splitting (2026-05-15)

Closes **audit #26**: Settings tabs are now lazy-loaded via
`React.lazy` + `Suspense`. Each tab becomes its own Vite chunk;
the kitchen-sink import in `SettingsPanel.tsx` no longer pulls
the entire Settings panel into the initial JS bundle.

**Result:**
- Initial bundle: **613 KB → 526 KB** (−87 KB)
- Initial gzipped: **171 KB → 152 KB** (−19 KB)
- 13 tab chunks now load on demand (largest: VoiceTab 12 KB,
  KnowledgeGraphTab 11 KB)

Kept eager: `IdentityEditor`, `UserProfileTab`, `StatusTab` —
small + used on almost every Settings open. Everything else
fetches its chunk when the user actually clicks the tab.

**Remaining audit findings (deferred — documented in QA report,
not silently skipped):**

- **#21** `cli.py` 2417 LOC split — true refactor, would move 80+
  functions across 5-10 new files. Risk of breaking the 23
  existing CLI tests. Better as a dedicated session.
- **#22** `llm.py` 3197 LOC base-class for the 8 LLM clients —
  each provider has subtle HTTP retry differences; consolidating
  needs careful comparison. Real work, not mechanical.
- **#25** `init_wizard.py` 742 LOC provider-flow registry —
  refactor for future-proofing; not a bug fix.

Tests: 1279 still pass.

---

## QA-audit-fix4 — smoke tests + embedder path fix (2026-05-15)

Closes **audit #8** by adding smoke tests for the largest previously-untested modules (~7K LOC of code that had no dedicated test file). 8 new test files, 92 new tests. Strategy: pin the public surface + happy paths without requiring live HTTP / Telegram / LLM calls.

**New test files:**
- `tests/test_providers_smoke.py` — 14 tests. Provider CRUD, default-key resolution, PKCE, ACTIVE_MODEL set/clear/resolve.
- `tests/test_channels_smoke.py` — 13 tests. Channel record CRUD, lazy `CHANNELS_PATH`, ChannelManager status, send-to-telegram-chat returns False (not crash) when no bot running, `_MAX_CONCURRENT_AGENT_RUNS` invariant.
- `tests/test_llm_smoke.py` — 15 tests. `TaskType` enum required members, `create_llm` dispatch for every provider type, `_parse_json_response` markdown-fence stripping + prose-around handling, `TokenTracker` reset + record round-trip, router singleton.
- `tests/test_tts_smoke.py` — 9 tests. Voice picker (Russian vs default by Cyrillic content), config round-trips, tolerates invalid JSON, status/reset don't crash.
- `tests/test_transcriber_smoke.py` — 6 tests. Same shape as TTS.
- `tests/test_embedder_smoke.py` — 5 tests. Acknowledges the project-wide conftest fixture that disables embedder during tests; pins the audit-relevant claim that `_config_path()` resolves via `paths.knowledge_dir()` at call time, not the CONFIG snapshot.
- `tests/test_memory_extractor_smoke.py` — 8 tests. recall/recall_block/stats shape, extract_and_store with mocked router (write-on-facts, no-write on empty, swallow LLMError).
- `tests/test_self_modifier_smoke.py` — 14 tests. Proposal CRUD + status transitions, `_validate_test_command` allow/reject matrix (pytest, python -m, bare python — accepted; bare paths + shell injection — rejected), persistence round-trip, stats.
- `tests/test_identity_smoke.py` — 8 tests. Default templates, set_user_profile persistence, per-speaker isolation (Telegram users keep separate profile files), legacy `webui:default` path.

**Bonus bug fix surfaced by writing the tests:**
- `backend/embedder.py:_config_path()` was reading `CONFIG.knowledge["base_dir"]` — captured at import time. Test overrides of `HRANT_DATA_DIR` were silently going to the dev's real `~/.hrant/data/`. Switched to `paths.knowledge_dir()` which re-reads on every call. The audit had flagged this same anti-pattern for `channels.py:CHANNELS_PATH` in #19; embedder had it too.

**Test isolation:** initial drafts used `importlib.reload()` to refresh module state; that pattern leaked into other tests' singletons and caused 23 cascade failures. Reworked all fixtures to `monkeypatch.setattr(_mod, "SINGLETON", fresh_instance)` instead.

1279 tests passing total (1187 → 1279, +92 smoke tests).

---

## QA-audit-fix3 (2026-05-15)

Closes 7 more findings from the original full-codebase audit:

**Important fixes:**
- **#9** Per-IP rate-limit on `POST /api/chat`. Defence-in-depth on top of the owner-role gate: 60 req/min, 10 req/5s burst, both tunable via env (`HRANT_CHAT_RATE_PER_MIN`, `HRANT_CHAT_RATE_BURST`). New `backend/api/_rate_limit.py` — single-process sliding-window deque, no external dep.
- **#10** Telegram group-chat isolation. Bot now refuses messages from groups/supergroups/channels unless the speaker is in `allowed_users`. Quiet drop (no reply spam) + log line for the owner. Set `HRANT_TELEGRAM_ALLOW_GROUPS=1` to revert.
- **#13** Failover chain now applies in the default A/B routing path too. Before: only pinned-model turns walked the chain; default routing did A→B then died. After: A→B→chain (chain runs as Tier 2 fallback when both A and B failed).
- **#14** `agent.run` is now safe under re-entrancy. Snapshots all instance state at entry, restores in `finally`. Today no tool handler calls back into `agent.run`, but if any future skill does, outer call state survives.
- **#16** Dedup cache for `_existing_fact_summaries`. Cache key is `(mtime, size)` of `memory_facts.jsonl` so concurrent appends from autonomic levers invalidate cleanly. Eliminates ~30ms of redundant I/O per consolidation when nothing changed on disk.

**Minor fixes:**
- **#20** `/api/notes-graph/*` alias added for the legacy `/api/graph/*` triples graph. Both URLs serve the same data; the new name is clearer about what's being queried (notes graph vs Phase 16C memory graph at `/api/kgraph/*`).
- **#23** Dead-code audit done. `ANALOGIES`, `EVALUATOR`, `META_LEARNER`, `embedding_backfill` are all in real use — nothing to remove.
- **#28** Stricter `require_owner_strict(request, action=...)` helper added next to the existing lenient `require_owner_for_writes`. Refuses the request when the speaker ContextVar is unset within an HTTP request context. Not applied to any endpoint yet — opt-in for endpoints that can't tolerate a "ContextVar setter bug silently passes" failure mode.

All 1187 tests pass.

---

## QA-audit-fix2 (2026-05-15)

Second pass on the full-codebase QA audit, after honest recheck:
"are these all fixed?" → no, not all. The prior commit closed the
Critical auth gap + 3 of the 12 Important findings. This commit
closes 4 more Important + 1 Minor.

**Closed in this commit:**
- **#15 memory_facts.jsonl growth bound** — rotates to a dated `.archive` file at 50,000 lines (~27 years of typical use). Active file stays small, dedup scan stays cheap, history preserved as separate archive files. Configurable via `HRANT_MEMORY_FACTS_ROTATE`.
- **#17 jobs/ auto-cleanup** — daily consolidation now opportunistically purges completed/cancelled jobs older than 30 days. Failed and interrupted jobs are kept indefinitely (audit log). No more 30k-file `jobs/` after a year.
- **#18 recover_interrupted in background** — was synchronous inside FastAPI startup; with 30k+ jobs the port stayed closed for ~15s. Now runs as an `asyncio.create_task`, port opens immediately. The Telegram interrupted-job notification awaits the same task, so users still get their "I got interrupted" message.
- **#27 ANSI-aware f-string padding** — instead of patching the 27 individual `:<14`-style call sites in `cli.py`, fixed the root cause: `c.muted()` / `c.success()` / etc. now return an `_AnsiAware` str subclass whose `__format__` honours visible (non-escape) length. Every existing colored-column table in the CLI lines up automatically; no per-site changes needed.

**Still open (deferred or by-design):**

Important:
- **#8 missing tests** for `providers.py` (1467 LOC) / `channels.py` (1021 LOC) / `llm.py` (3197 LOC) clients. Designing fixtures for HTTP mocking + Telegram bot lifecycle is a session each.
- **#9 `/api/chat` rate-limit** — auth gate added, but no per-IP rate-limit. For a personal-agent single-WebUI deployment this is fine; for shared deployments add a `slowapi` middleware.
- **#10 Telegram group-chat isolation** — the bot answers every group member equally. Phase 11's role-gate restricts MUTATIONS but not chat consumption. Design decision: do we want guests to talk to the agent at all, or only owner+trusted?
- **#13 Failover not applied in default A/B path** — user previously chose to leave this; only the pinned-model path uses the failover chain.
- **#14 `agent.run` re-entrant state brittleness** — instance attrs (`_speaker_id`, `_channel`, `_t0`, ...) would clobber on re-entry. Not triggered today (no tool handler calls back into agent.run). Worth a refactor before that pattern ever appears.

Minor:
- **#20 two graphs at `/api/graph` (note triples) vs `/api/kgraph` (memory graph)** — coexistence documented; merge is design work.
- **#21 cli.py 2417 LOC split** — refactor.
- **#22 llm.py 3197 LOC base-class** — refactor.
- **#23 dead-code audit** (EmbedTracker, meta_learner, analogy_engine, evaluator usage check).
- **#25 init_wizard.py provider-flow registry** — refactor.
- **#26 frontend code-splitting** — 613 KB bundle; lazy-load Settings tabs would cut ~40%.
- **#28 require_owner_for_writes trusts ContextVar=None** — by design for CLI/tests; documented.

**Compatibility:** all 1187 tests still pass.

---

## QA-audit-fix (2026-05-15)

Major security + operational pass after the full-codebase QA audit. **Closes the auth gap that affected ~80 mutation endpoints across 14 API modules** — before this commit, anyone who could reach the API (notably anyone on the LAN/Tailnet when the gateway was bound to 0.0.0.0 via `hrant gateway start --gateway`) could:

- Promote themselves to owner via `PUT /api/roles/{id}`
- Apply self-modifier patches (`POST /api/self-modifier/proposals/{id}/apply`) — remote code execution on the agent process
- Register an LLM provider with their own API key + base URL (`POST /api/providers`) — cost amplification or prompt exfiltration
- Rewrite `soul.md` / `identity.md` (`PUT /api/identity`) — prompt-injection persistence
- Spawn a Telegram bot with an attacker token (`POST /api/channels`)
- Swap STT / TTS / embedding URLs (`PUT /api/transcribe|tts|embeddings/config`)
- Read arbitrary files via `POST /api/finetune/import-gguf`

**Auth gates added (modules → mutation count):**
- `intel.py` (15 mutations, all gated): graph reindex, meta-learner extract, memory recall, embeddings backfill/config/reset, every self-modifier endpoint, every self-mods endpoint
- `providers.py` (14): Ollama pull/delete, active-model set/clear, provider CRUD + test, every OAuth endpoint
- `finetune.py` (11): every example edit, correction, pipeline start, switch/rollback, export-cloud, import-gguf, add-from-chat, compare
- `goals.py` (7): add, complete, pause, resume, fail, delete, priority
- `channels.py` (6): create, update, delete, start, stop, test
- `projects.py` (5): create, end, context, decision, issue
- `knowledge.py` (5): learn, delete, core add/delete, quick-note
- `attachments.py` (4): upload, transcribe, transcriber config, transcriber reset
- `sessions.py` (3): new, delete, archive
- `roles.py` (3): role set, relationships PUT, scheduled-message cancel
- `voice.py` (2): TTS config, TTS reset
- `identity.py` (2): conversation clear, identity files PUT
- `engine.py` (2): config PUT, config reset
- `chat.py` (1): chat

All use the shared `backend/api/_auth.py:require_owner_for_writes` helper. Read endpoints (GET) remain open — they don't change state or cost money.

**Other operational fixes:**

- **Attachment + audio upload size cap (50 MB).** Pre-fix, `await file.read()` slurped the whole request body into memory. A 10 GB upload OOMed the agent. Now chunk-reads with a running total; over-cap returns 413 instead of dying.
- **Telegram concurrency bound (default 3 concurrent runs per bot).** A user spamming the bot (or a group with many active members) used to spawn one executor thread per message → N concurrent `agent.run` → N concurrent LLM streams. The semaphore queues the excess. Tune via `HRANT_TELEGRAM_MAX_CONCURRENCY` if you have many trusted group members.
- **Lazy `CHANNELS_PATH` resolution.** `backend/channels.py` captured `paths.knowledge_dir() / "channels.json"` at import time, so a test that monkeypatched `HRANT_DATA_DIR` after import silently wrote to the dev's real ~/.hrant/data/channels.json. `_load_channels` / `_save_channels` now re-resolve on every call, prefer test-overridden `CHANNELS_PATH` when set.

**Compatibility:** existing tests pass unchanged because `require_owner_for_writes` is a no-op when there's no speaker ContextVar set (CLI, tests, autonomic ticks). All 1187 tests still pass, 3 skipped (Windows pty).

**Not in this commit (deferred from the audit):**
- Smoke tests for `providers.py` / `channels.py` / `llm.py` (the three biggest untested modules — separate session)
- `cli.py` (2417 LOC) split into per-subcommand modules
- `llm.py` (3197 LOC) base-class refactor for the 8 client classes
- `init_wizard.py` provider-flow registry
- Lazy frontend bundle code-splitting

---

## Phase 16-audit-fix (2026-05-15)

Cleanup pass after the post-Phase-16 audit. Fixes 1 critical + 10 important + 11 minor findings across consolidation, knowledge graph, REST surfaces, WebUI, and CLI.

**Critical fix**
- **`runForceLayout` was O(N × E × iterations)** because `positioned.indexOf(node)` ran inside the edge-attraction loop. At ~200 nodes / 400 edges / 150 iterations that was ~2 billion comparisons synchronously inside `useMemo` — the Graph view froze the browser tab. Replaced with a `Map<nodeId, index>` for O(1) lookup; same workload now <100ms.

**Important fixes**
- **Scheduler exceptions outside `pipeline.run` now record `status=failed` state.** Pre-fix: a disk-write or state-save failure left `last_run_status` lying as "success" from a prior run indefinitely.
- **Profile path now matches the user's spec:** Telegram speakers → per-chat profile files, everything else (WebUI, voice, future channels) → global `user.md`. Pre-fix: only `webui:default` mapped to the global file, so future non-default speakers would have silently diverged into per-speaker files.
- **`_should_fire` no longer scans `jobs/` every minute.** Default `MIN_JOBS_FOR_RUN=1` now uses the single-job-existence check from `last_activity_ts()` instead of a full directory scan. Only `MIN >= 2` triggers the expensive path.
- **Knowledge graph saves once per consolidation, not per fact.** Pre-fix: 10 promoted facts → 10 full graph-file rewrites. One flush at end of step 4 reduces I/O 10×.
- **`Digest.links_added` entries now carry a `kind` field** (`"is_about"` for step 4, `"relates_to"` for step 5.5) so the WebUI can render them with the right styling instead of sniffing the dict shape.
- **Owner-role gate on all new write endpoints.** Added to `POST /api/consolidation/run`, `POST /api/jobs/{id}/retry|cancel`, `DELETE /api/jobs/{id}`, `POST /api/jobs/_/cleanup`, `PUT /api/failover`, `POST /api/failover/toggle|reorder`, `POST /api/kgraph/rebuild`. Read endpoints stay open. Pre-fix: anyone reachable on a `--gateway`-bound (0.0.0.0) install could trigger LLM-costing actions or wipe state.
- **Dedup window bumped 500 → 5000 lines.** At 5 facts/day the previous cap covered only ~100 days of history.
- **Force layout runs asynchronously after first paint.** `setTimeout(0)` trampoline keeps the browser responsive; the "Computing layout…" placeholder shows immediately.
- **GraphView wrapped in an `ErrorBoundary`.** A render-time crash (NaN positions from corrupt data) no longer blanks out the whole Settings panel.
- **`start_scheduler` is idempotent.** A re-entrant call (hot reload, test rerun) cancels the previous task before creating a new one rather than leaking it.

**Minor fixes**
- Stale comment about `MIN_JOBS_FOR_RUN=0` default updated.
- TOCTOU comment in `_fire_one` corrected (second caller waits, doesn't skip).
- `_resolve_to_fact_id` replaced with an O(1) `{label → fact_id}` index built once.
- Documented `MAX_NEW_FACTS_IN_PROMPT=15` as defence-in-depth despite the pipeline cap of 12.
- Proposer pre-filters pairs already connected by `relates_to` before the LLM call — saves tokens.
- `_G.save()` failure inside the pipeline batch no longer crashes consolidation (best-effort log).
- Graph store surfaces a `load_error` field on `/api/kgraph/stats` when `graph.json` is corrupt or schema-newer; WebUI renders a "rebuild required" banner instead of silently showing 0 nodes.
- `stop_scheduler` awaits the cancellation after a timeout so uvicorn doesn't tear down with a half-cancelled task.
- `ResizeObserver` replaces the window-resize listener so Graph view sizes correctly when the Settings panel was hidden at mount.
- New `pad_visible` helper in `cli_colors.py` for ANSI-aware column alignment. `hrant jobs list` and similar tables now line up whether colors are on or off.

**Tests** added: +1 (proposer pre-filter regression). 1187 passed total, 3 skipped (Windows pty), 8 autonomic reactive-rule tests flaky on suite-order (pre-existing).

---

## Phase 16C.1 (2026-05-15)

**LLM-proposed `relates_to` edges between facts.**

A new step in the daily consolidation pipeline — runs after fact promotion + profile updates, before open-threads detection. The LLM is shown today's newly-promoted facts plus the top ~20 most-connected existing facts in the graph, and asked to identify pairs that are semantically related but don't already share a topic tag.

**What this catches that topic edges miss:**
- "User uses Tailscale" + "Whisper STT runs on 100.124.210.21" — both about home network infra but tagged with different topics
- A new fact about a project + an older fact about the same project that's drifted to a different tag set
- Cross-domain connections the user might not have realised the agent could see

**Symmetry handling:** `relates_to` is conceptually undirected. The proposer canonicalises pairs by sorting node ids before upserting — so `A↔B` and `B↔A` hit the same edge key and accumulate weight rather than creating duplicates across runs.

**Cost / safety:**
- Skipped in `--dry-run` mode (no preview-burn)
- Skipped when there's only one new fact and zero existing facts to relate against
- LLM failures swallowed — graph keeps its existing state, digest just records `links_added=[]` for this run
- Cap: max 6 links per run, max 15 new + 20 existing facts in the prompt → ~1k tokens input, cheap

**Surfaces:**
- Edges land in `knowledge/graph.json` as `relates_to` with `{reason, source: "consolidation:<date>", proposed_at}` metadata
- Visible in the WebUI Knowledge Graph tab: Explorer view's neighborhood pane shows the edge + its reason; Graph view renders the cross-link
- Recorded in `Digest.links_added[]` alongside the is_about edges from step 4

**Tests:** +12 (10 proposer-unit covering empty/single-fact short-circuits, edge canonicalisation, whitespace drift in fact-text resolution, hallucinated-text dropping, cap enforcement, disk persistence, LLM-failure swallowing; 2 pipeline integration covering the new step firing on real runs and being skipped under `--dry-run`).

---

## Phase 16C (2026-05-15)

**Knowledge graph — see what the agent knows + how it connects.**

A small graph over the agent's persistent knowledge — facts, topics, skills, projects, named entities — automatically grown by the daily consolidation pipeline and viewable in the WebUI.

**Data model:**
- Nodes: `fact`, `topic`, `skill`, `project`, `entity`
- Edges: `is_about` (fact→topic), `uses` (skill→topic), `mentions` (fact→entity, with predicate metadata), `relates_to` (fact↔fact, LLM-proposed in 16C.1), `continues` (project→fact)
- Stored as a single `~/.hrant/data/knowledge/graph.json` with atomic-ish writes

**Sources:**
- `memory_facts.jsonl` → fact nodes + topic edges + RDF triples
- Skills registry (Phase 12) → skill nodes + topic edges via skill triggers
- `goals.json` → project nodes (best-effort, shape-tolerant)

**Integration with consolidation (Phase 16A):**
After each consolidation promotes a new fact, the pipeline mirrors it into the graph: `add_fact()` upserts a fact node, walks the related_topics list creating topic nodes + `is_about` edges, processes any RDF triples. Failures are caught and logged; the graph is derivable from sources, so a bad add doesn't break consolidation.

**Surfaces:**
- REST: `GET /api/kgraph`, `GET /stats`, `GET /search?q=…&kind=…`, `GET /node/{id}`, `POST /rebuild`
- CLI: `hrant graph stats / search <q> [--kind …] / show <node_id> / rebuild`
- WebUI: `Settings → Knowledge Graph` tab with two views:
  - **Explorer:** search box + filtered list of nodes, click any node to see its neighbourhood pane (incoming + outgoing edges with direction arrows)
  - **Graph view:** SVG node-link diagram laid out by a built-in force-directed simulation (no extra deps). Color-coded by kind. Topic/skill/project labels shown; fact labels hidden to avoid clutter at >50 nodes.

**Why namespaced `/api/kgraph` not `/api/graph`:** an older notes-derived triples graph already owns `/api/graph` in `backend/api/intel.py`. The two coexist; this knowledge graph (the broader memory graph) is incremental, the old one is derived from note frontmatter. They may unify later.

**Tests:** +27 — id dedup (case/whitespace), fact_id collision check at 100 facts, store round-trip + upsert merge semantics (weight max for nodes, sum for edges), schema-version guard for forward-compat, builder idempotency, neighborhood direction tagging, search ranking by degree + kind filter, REST surface.

**Config tweak:** `MIN_JOBS_FOR_RUN` default changed from 0 → 1. Consolidation now skips truly empty 24h periods. Set `HRANT_CONSOLIDATION_MIN_JOBS=0` to keep firing on quiet days.

---

## Phase 16A (2026-05-15)

**Daily memory consolidation — like sleep, but for the agent.**

While the agent is idle, a background scheduler periodically runs through the past ~24h of activity and:

1. Writes a **narrative** of what happened
2. Extracts **durable facts** worth promoting to long-term memory
3. Updates the per-speaker **profile** (global `user.md` for WebUI, per-Telegram-user files for Telegram speakers)
4. Surfaces **open threads** — unresolved questions / abandoned projects

Each run produces a **Digest** stored at `~/.hrant/data/knowledge/memory_digests/<YYYY-MM-DD>.json`. Inspectable from CLI + WebUI.

**Scheduler (adaptive):**
- Fires when the agent has been idle for ≥15 min AND ≥24h since last run
- Min activity threshold: 0 — runs even on idle days (empty digest)
- Cost cap: unlimited (token usage tracked + reported, never blocks)
- Lives in the FastAPI lifespan, ticks every 60s

**Pipeline runs through:**
- Main LLM via the failover chain (Phase 15B) — so a 429 doesn't kill a consolidation
- 4 LLM calls per run (narrative, facts, per-speaker profile updates, open threads)
- ~30–60s wallclock for a typical day

**Surfaces:**
- REST: `GET /api/consolidation/status`, `POST /run`, `GET /digests`, `GET /digests/{date}`
- CLI: `hrant consolidate status / run [--dry-run] / list / show <date>`
- WebUI: `Settings → Memory Digests` tab — status banner + Run/Dry-run buttons + per-digest detail pane

**Safety:**
- `--dry-run` mode: pipeline runs but skips memory_facts/profile writes
- Zero-activity windows short-circuit before any LLM calls
- Pipeline failures are caught + recorded in the digest's `error` field; never crashes the scheduler
- Per-speaker isolation: WebUI sessions update global `user.md`, Telegram speakers each get their own `profiles/telegram_<id>.md`

**NOT in 16A** (coming in 16B/16C):
- Pruning (auto-trash with rollback window)
- Knowledge graph (cross-link inference + visualization)
- Adaptive multi-pass scheduling

---

## Phase 15B-fix2 (2026-05-15)

Two fixes that together solve "I ran `hrant update` but don't see the new WebUI tabs":

**Fixes**
- **`frontend_changed` heuristic was always returning False after pull.** The previous code ran `git diff HEAD..origin/<branch>` AFTER `git pull` succeeded, at which point HEAD and origin are the same → diff is empty → "frontend unchanged" → rebuild skipped. Phase 15A/B users hit this directly: the JobsTab and FailoverPanel files were in the pulled commits, but the npm build ran zero times. Fixed by walking each commit via `git show --name-only` — works regardless of whether the pull has happened.
- **`hrant update` now auto-restarts the gateway service.** After a successful update, detects whether `hrant.service` / `ai.hrant.agent` / `HrantAgent` is running and restarts it so the freshly-built frontend bundle is served immediately. Opt out with `hrant update --no-restart`. If you're running in foreground via `hrant run`, you'll get a clear "no service to restart — Ctrl-C and re-run" message instead.

So `hrant update` now actually does the right thing end-to-end: pull → pip → rebuild (when needed, correctly detected) → restart service → WebUI shows new code. No more `hrant rebuild && hrant gateway restart` chase.

---

## Phase 15B-fix (2026-05-14)

Cleanup pass after the Phase 15A/B audit. Behavior fixes.

**Fixes**
- **Failover attribution.** When the failover chain delivers via a non-primary provider (e.g. Anthropic 429 → OpenAI), the daily usage breakdown now records the call under the provider that actually answered. Pre-fix: every failover-delivered call was attributed to the pinned primary, misleading the cost dashboard.
- **Failover chain no longer stops on a misconfigured entry.** A chain entry pointing at a provider with no API key used to halt the whole walk (classified as "unknown" → non-retryable). The classifier now recognises "no/missing API key" as an `auth_error`, so failover keeps walking past broken entries.
- **`/api/jobs` `total` respects filters.** When filtering by status/channel the response now returns the matching count, not the global count. Adds `total_all` for the global figure if you need both.
- **`SESSIONS.add_turn` records `job_id`.** Conversation entries now link to their Job record so the WebUI can deep-link Conversation → Jobs.
- **`AgentAnswer.job_id` exposed in TypeScript.** Frontend can now access the job id without casting.
- **JobsTab auto-refresh interval no longer thrashes.** Previously the 5s timer was re-created on every refresh; now uses a stable `useRef` and runs at a steady 5s tick while jobs are active.
- **`hrant config` arrow keys no longer cancel the menu.** The Unix `_read_key` path now reads from the raw FD via `os.read` instead of through Python's buffered stdin, so arrow-key burst sequences (`\x1b[A`) aren't split between the buffer and the kernel.
- **`hrant config` menu glyphs fall back to ASCII on legacy Windows code pages.** No more `UnicodeEncodeError` on cp1252.
- **`cli_menu` Enter no longer leaves a gap.** Padding line replaces the cleared hint slot.
- **`cli_menu` ignores multi-byte UTF-8 keystrokes** (Cyrillic, accented Latin) instead of looping byte-by-byte.

**Additions**
- **`hrant jobs cleanup` + `Settings → Jobs → Cleanup` button** — purge old completed/cancelled jobs (kept by default: failed/interrupted, as audit log).
- **Telegram restart notification.** If the server dies mid-turn, the next boot sends a "I was interrupted earlier" message to the original Telegram chat so users on the bot don't experience silent message loss.
- **Failover error scrubber.** API keys / Bearer tokens accidentally echoed in provider error messages are now redacted before persisting to a Job record.
- **`hrant failover add` validates the model name** against the provider's declared models. Pass `--force` for fresh providers whose model list isn't discovered yet.
- **Failover `on_success` callback** in `try_call` — let callers attribute the call to the provider that actually answered.

**Internal**
- `Router.call` / `Router.call_with_tools` now share a `_call_with_failover_chain` helper. Less duplication; failover behavior is single-sourced.
- Chain fallback LLMs build **lazily** inside their closure — no httpx/auth setup for entries never reached.
- Pre-existing audit findings #18 (interval-print micro-perf) and #20 (`datetime.utcnow` deprecation in `test_workspace.py`) are tracked separately and intentionally deferred.

---

## Phase 15B (2026-05-14)

**Multi-provider failover chain.** When the active LLM returns a retryable error (rate limit, 5xx, timeout, auth, connection), fall through a user-configured chain of (provider, model) pairs.

- `~/.hrant/data/knowledge/failover_config.json` — `{enabled, chain[], retry_on[], max_attempts}`. Default off.
- REST: `GET/PUT /api/failover`, `POST /toggle`, `POST /reorder`.
- CLI: `hrant failover status / enable / disable / add / remove / clear`.
- WebUI: `Settings → Providers → Failover chain` panel.
- Every attempt — success or failure — logs to the active `Job.attempts[]`.

---

## Phase 15A (2026-05-14)

**Durable Job records.** Every user turn (WebUI, Telegram, voice) now gets a persistent record at `~/.hrant/data/jobs/<id>.json`. Survives crashes — on next boot any job left `running` or `queued` is marked `interrupted` so users can retry.

- States: `queued → running → completed | failed | interrupted | cancelled`.
- REST: `GET /api/jobs`, `POST /retry`, `POST /cancel`, `DELETE`, `GET /_/stats`.
- CLI: `hrant jobs list / show / retry / cancel / delete`.
- WebUI: `Settings → Jobs` tab with status-filter chips, details pane, auto-refresh.

---

## Phase 14D (2026-05-13)

**`hrant config` — friendly knob surface.** Modelled on `openclaw config`. Surfaces only the configs most users actually change (API keys, voice, Telegram, autonomic). Interactive arrow-key menu via `hrant config`; non-interactive get/set via `hrant config get/set <key> <value>`. Warm-orange CLI palette, ASCII fallback on legacy Windows code pages.

---

## Phase 14C (2026-05-13)

**`hrant gateway start/stop/restart/logs/install/status/uninstall`.** One subcommand group for running Hrant as a background service. Replaces the previous `hrant service install` group.

---

## Phase 14B (2026-05-13)

**Edge TTS as the default voice backend.** ~400 free Microsoft neural voices, no API key, no model files. `pip install edge-tts` is bundled in main deps. Falls through to Piper / OpenAI when offline.

---

## Phase 14A (2026-05-12)

**Interactive setup wizard** for `hrant init`. Replaces the flat Q&A with TTY-aware step-by-step provider selection, model picker, live API-key validation, optional Telegram / Tailscale flows.

---

## Phase 13 (2026-05-11)

**`hrant provider` CLI family** — `list / login / test / use / logout`. Mirror of the WebUI Providers tab.

---

## Earlier phases

See `git log --oneline` and the docstrings in `backend/` modules for everything before Phase 13. Notable earlier work:

- **Phase 12:** Autonomic heartbeat + Skills UI + external install
- **Phase 11:** Roles (owner/trusted/guest) + cross-speaker scheduled messages
- **Phase 10:** `speaker_id` partitions sessions/conversation/profile
- **Phase 9:** Archive self-mods on `hrant update`
- **Phase 8:** Engine/data split — `~/.hrant/data/` separate from the engine repo
