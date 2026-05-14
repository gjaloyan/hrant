# Changelog

Notable user-facing changes per release / phase. Format roughly
follows [Keep a Changelog](https://keepachangelog.com/) but without
the strict version-tag discipline — Hrant ships continuously and
"phases" are how features were grouped in the dev process.

For the full commit history, see `git log`. This file focuses on
**what users should know** when running `hrant update`.

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
