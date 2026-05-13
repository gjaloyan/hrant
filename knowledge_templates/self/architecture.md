# My Architecture

I am Hrant — a self-learning AI agent. This note is my baseline understanding of how I'm built. It gets supplemented over time by `FIRE_SELF_STUDY` (an autonomic lever that reads my own modules and writes per-file notes).

## Engine vs Data

I am split into two pieces:

- **Engine** — the git checkout (my code). Lives wherever the user cloned the repo. Refreshed by `hrant update`.
- **Data** — the user's stuff: knowledge, memory, settings, conversations, uploads. Lives in `data_dir` (default `~/.hrant/data/`, override via `$HRANT_DATA_DIR`). Survives every engine update.

These two never overlap. `hrant update` only touches the engine; `hrant rollback` only touches the engine. The user's data is mine to preserve.

`backend/paths.py` is the single source of truth: `repo_root()`, `data_dir()`, `knowledge_dir()`, `workspace_dir()`, `config_yaml_path()`, `env_path()`, `history_path()`, `templates_dir()`.

## Module map (engine)

```
backend/
  agent.py           — orchestrator: load core → analyze → solve → verify → learn
  pipeline/          — mixin classes (intent classifier, preferences, thinking, critic)
  llm.py             — DualModelRouter: A (Claude) vs B (local Ollama)
  providers.py       — provider registry, OAuth, active-model selection
  knowledge_manager.py / core_memory.py / vector_store.py — memory layers
  workspace.py       — inbox/outbox/notes/turns filesystem tree
  channels.py        — Telegram / future channels
  transcriber.py / tts.py — Whisper / Piper integration
  attachments.py     — sha-keyed upload store with mirroring to workspace/inbox/
  autonomic/         — Model X: scheduler + 19 levers, kill-switch, safety gates
  api/               — HTTP routers, one per concern
  cli.py             — unified `hrant` entry point: init / run / status / chat /
                       update / rollback / rebuild / discover / service
  paths.py           — engine/data path resolution
  bootstrap.py       — knowledge_templates → data_dir on fresh install
  updater.py         — git wrapper + history ledger
  runtime_config.py  — whitelisted live overrides for router/verification/
                       workspace/knowledge sections
  self_modifier.py   — propose/apply code patches; locally stored, never pushed
```

```
frontend/src/
  App.tsx           — top-level routing
  components/
    Chat.tsx        — main conversation surface
    SettingsPanel.tsx — Identity / Soul / User / Providers / Channels /
                        Memory / Voice / Engine / Self-Mods / Conversation /
                        Capabilities / Status
    AutonomicPanel.tsx — Model X dashboard
    KnowledgePanel.tsx / GraphViewer.tsx / GoalsPanel.tsx / etc.
```

## Two distinct loops

**1. Cortex (synchronous, user-facing).** When a message arrives, `agent.py` runs the pipeline (`analyze → solve → verify → learn`). The dual-model router picks Claude vs local Qwen per task type. Verification can self-critic + retry. Then the answer ships and a turn artifact lands in `workspace/turns/`.

**2. Autonomic (background, every `tick_interval_seconds`).** `backend/autonomic/scheduler.py` fires the registered levers — health checks, memory consolidation, gap detection, proactive learning, etc. Yellow-safety levers (anything that mutates external state, e.g. `pip_install`) wait for user approval via the Autonomic panel.

## Memory hierarchy

- **Core memory** (`knowledge/core_memory.md`) — always in the system prompt. Pinned facts. Cap: `knowledge.core_memory_max_tokens` (editable via Settings → Engine).
- **Knowledge notes** (`knowledge/<category>/<topic>.md`) — markdown with frontmatter. Loaded into context only when relevant (semantic + keyword search). Auto-promoted to core when accessed often (threshold: `auto_promote_threshold`).
- **Conversation memory** (`knowledge/conversation.json`, per-channel) — recent turns, used for context.
- **Workspace** (`workspace/inbox/`, `outbox/`, `notes/`, `turns/`) — the agent's working directory. Auto-sweep by retention days (per-subtree, Settings → Engine).

## How user controls me

- WebUI Settings tabs cover the live-editable knobs. Saved into `data_dir/runtime_overrides.json` (the agent picks them up on next request).
- `config.yaml` (in `data_dir`) is the persisted baseline; mode + presets come from there.
- `.env` holds API keys and secrets.
- CLI `hrant <subcommand>` for install/update/rollback/diagnostic flows.

## Self-modification (Phase 7+)

The user can ask me to modify my own code (`"save memory to SQLite instead of RAG"`). When I do:

1. I generate a unified diff against the engine repo.
2. I show the user the diff + risks; only proceed on explicit yes.
3. The patch is saved to `data_dir/self_mods/NNNN-<slug>.patch`.
4. I `git apply` it to the engine repo (in-place, but the patch file is the source of truth).
5. `hrant update` reset-stomps the engine, then re-applies each patch in order. Conflicts are surfaced; the conflicting patch is skipped (engine stays stable), user sees it in Settings → Self-Modifications.
6. `Revert one` / `Revert all to official` buttons exist in Settings.

The official git on GitHub never sees these patches.
