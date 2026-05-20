# Cheatsheet — quick reference for me

Where to look when answering questions about myself.

## Filesystem map

| Path | What's there |
|---|---|
| `backend/agent.py` | Main orchestrator (analyze→solve→verify→learn loop) |
| `backend/llm.py` | DualModelRouter, A/B routing, budget gates |
| `backend/pipeline/` | Mixin chunks: intent classifier, preferences, thinking, critic |
| `backend/knowledge_manager.py` | Notes CRUD, indexing |
| `backend/core_memory.py` | Pinned-context layer |
| `backend/vector_store.py` | Embeddings storage |
| `backend/workspace.py` | Inbox/outbox/notes/turns filesystem |
| `backend/channels.py` | Telegram + future channel bridges |
| `backend/transcriber.py` | Whisper STT integration |
| `backend/tts.py` | Piper TTS integration |
| `backend/attachments.py` | sha256-keyed upload store |
| `backend/autonomic/` | Background scheduler + 19 levers |
| `backend/api/` | HTTP routers (one file per concern) |
| `backend/cli.py` | `hrant` command entry point |
| `backend/paths.py` | data_dir vs repo_root resolution |
| `backend/bootstrap.py` | First-run wizard helpers |
| `backend/updater.py` | git wrapper + history ledger |
| `backend/runtime_config.py` | Live overrides whitelist |
| `backend/self_modifier.py` | Local code patches (Phase 7) |
| `frontend/src/components/SettingsPanel.tsx` | WebUI settings tab dispatcher |
| `frontend/src/components/AutonomicPanel.tsx` | Model X dashboard |
| `frontend/src/api.ts` | All API client helpers |
| `knowledge/` (data_dir) | User's notes, identity, memory |
| `knowledge_templates/` (repo) | Starter content copied on `hrant init` |
| `workspace/` (data_dir) | Inbox / outbox / notes / turns |
| `config.yaml` (data_dir) | Mode + tunables (live-edited via Settings → Engine) |
| `~/.hrant/data/update_history.json` | Engine SHA history for `hrant rollback` |
| `~/.hrant/data/self_mods/` | Local code patches (Phase 7+) |

## User-facing commands

| What user runs | Where it lives |
|---|---|
| `hrant init` | `backend/cli.py:cmd_init` (delegates to `backend/bootstrap.py`) |
| `hrant run` | `backend/cli.py:cmd_run` → uvicorn `backend.main:app` |
| `hrant update` | `backend/cli.py:cmd_update` (delegates to `backend/updater.py`) |
| `hrant rollback` | `backend/cli.py:cmd_rollback` |
| `hrant rebuild` | `backend/cli.py:cmd_rebuild` |
| `hrant discover` | `backend/cli.py:cmd_discover` (uses `backend/discovery.py`) |
| `hrant gateway start/stop/restart/logs/install/status/uninstall` | `backend/cli.py:cmd_gateway_*` |
| WebUI Settings → Voice save | `PUT /api/transcribe/config` + `PUT /api/tts/config` |
| WebUI Settings → Engine save | `PUT /api/engine/config` |
| WebUI Settings → Autonomic interval | `PUT /api/autonomic/settings` |
| Self-mod accept | `POST /api/self-modifier/proposals/{id}/apply` |

## How a user message gets answered

1. POST `/api/chat` → SSE stream
2. `backend/api/chat.py` → `agent.run(message)`
3. `agent.py` runs pipeline mode:
   - `fast_chat`: chat-style, no verifier
   - `task_mode`: solver only
   - `deep_agent`: full analyze→solve→verify→retry
4. Each pipeline stage calls `backend/llm.py:router()` which picks A (Claude) vs B (local) per task type, then dispatches to provider.
5. Workspace turn artifact written to `workspace/turns/<turn_id>.json`.
6. Response shipped back as SSE `answer` event.

## How autonomic works

1. `backend/autonomic/scheduler.py` ticks every `tick_interval_seconds`.
2. Each tick fires `make_real_tick(builder, engine, registry, executor, …)`.
3. The L0 engine checks routing rules (`backend/autonomic/rules.py`) and picks levers.
4. `LeverExecutor` runs them; yellow-safety levers queue to `pending_approvals.jsonl` and wait for user approve via `POST /api/autonomic/pending/{id}/approve`.
5. Results land in `knowledge/autonomic/lever_log.jsonl` (one line per lever fire).

Kill switch: `knowledge/autonomic/ENABLED` (`true`/`false` content). Disable via Autonomic panel.
