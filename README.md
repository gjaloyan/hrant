# Hrant — Self-Learning Agent

***English** · [Русский](README.ru.md)*

A local AI agent that **does not keep knowledge in the model's weights**.
Instead it reads sources, keeps structured notes (markdown) on disk, and
loads them into context only when needed. It grows competence for the
tasks you actually give it while keeping a small, efficient core.

> a junior engineer with perfect notes who never forgets anything.

## Install (fresh machine)

```bash
# 1. Get the engine
git clone https://github.com/gjaloyan/hrant.git
cd hrant

# 2. Python + Node deps
python -m venv .venv && source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -e .
cd frontend && npm install && npm run build && cd ..

# 3. First-run bootstrap (creates ~/.hrant/data/ + asks for API keys)
hrant init

# 4. Start the agent
hrant run
# Open http://127.0.0.1:3333
```

**What happens:**
- `hrant init` creates `~/.hrant/data/` (or wherever `HRANT_DATA_DIR`
  points), copies starter templates from `backend/knowledge_templates/`,
  makes `config.yaml` from `config.example.yaml`, and asks about API keys
  (Anthropic, OpenAI) + optional service URLs.
- `hrant run` brings up FastAPI on `127.0.0.1:3333` (WebUI lives there
  too) and auto-starts the configured channels.

Any important setting can be changed later via `hrant config`:

```bash
hrant config                              # interactive menu — the main entry point for newcomers
hrant config list                         # see all settings (secrets masked)
hrant config set tts.backend edge_tts     # example: switch voice to free Edge TTS
hrant config set whisper.url http://...   # example: point to an STT server
```

Details — [docs/cli.md](docs/cli.md#hrant-config).

## Engine vs Data

```
<repo>/                  ← engine: backend/ (incl. knowledge_templates/) frontend/ deploy/
~/.hrant/data/           ← user data: config.yaml, knowledge/, workspace/, .env
~/.hrant/data/update_history.json   ← ledger for `hrant rollback`
```

`hrant update` updates only the engine; user data is left untouched. You
can move the location with `HRANT_DATA_DIR=/some/path hrant init`.

**Dev mode (single-tree):** if you run from the repo without
`HRANT_DATA_DIR`, the agent uses `<repo>/knowledge/` and
`<repo>/workspace/` (both gitignored). Handy for development.

## Update / Rollback

```bash
hrant update --check               # what's new on origin/master, no actions
hrant update                       # pull → pip install -e . → npm build
hrant update --skip-frontend       # backend-only (faster)
hrant rollback                     # one step back
hrant rollback --list              # history of all updates
hrant rollback --to <sha>          # to a specific commit
hrant rebuild                      # rebuild the frontend only, no pull
```

`hrant update` refuses to run on a dirty working tree (uncommitted
changes); untracked files in `knowledge/`/`workspace/` (gitignored) don't
count. The history is written to `~/.hrant/data/update_history.json`
**before** `git pull`, so rollback works even if an update fails halfway.

## Run as a background service

All commands for running the agent as a service are grouped under
`hrant gateway` (by analogy with `openclaw gateway`). The shortest path:

```bash
hrant gateway start                # install the unit + start (idempotent)
hrant gateway start --gateway      # bind 0.0.0.0 — other devices on LAN/Tailscale can reach it
hrant gateway start --port 4444    # non-default port

hrant gateway logs -f              # stream logs (journalctl --user -u hrant -f)
hrant gateway restart              # after `hrant update`
hrant gateway stop                 # stop (the unit stays; restart via `gateway start`)
```

Under the hood `gateway start` is `gateway install` + activation. If you
want to inspect the unit file first, use `gateway install` and activate
manually.

```bash
hrant gateway install              # write the unit, do NOT start
hrant gateway status               # what the OS service manager shows
hrant gateway uninstall            # remove the unit, keep the venv
```

More per-platform detail — [deploy/README.md](deploy/README.md).

## Configure via WebUI

After `hrant run` → `http://127.0.0.1:3333` → Settings. Tabs:

- **Identity / Soul / User Profile** — who the agent is, who the user is
- **Providers** — add/switch LLM providers (Anthropic, OpenAI, Ollama,
  OpenRouter, …) + change a provider's model from the live catalog
- **Channels** — Telegram bots, etc.
- **Memory** — embeddings backend (llama.cpp / ollama / OpenAI / Cohere)
- **Voice** — Whisper / Piper + Tailscale discover for service auto-detection
- **Engine** — router budget, verification strictness, workspace
  retention, knowledge caps (all applied live, no restart)
- **Reminders** — create / list / cancel scheduled messages
- **Fine-Tune** — distillation queue + the **model cascade** controls
  (small-model tier, gate, on/off)
- **Self-Modifications** — list of the agent's local patches, revert buttons
- **Status** — diagnostics of every subsystem

## How the agent thinks

Hrant keeps wisdom, method and identity in its **body** (files that
survive a model swap) rather than the model's weights — which is what
lets a cheap/small model stay smart. The full conceptual map and the
agent's work philosophy: **[docs/cognition.md](docs/cognition.md)**. In
short:

- **Method before execution** — for a substantive task it establishes the
  proper methodology first (recall → research how experts do it → cover
  every dimension), then executes.
- **Three memories** — knowledge (studied theory, declarative), skills
  (applied procedures), trajectories (past cases).
- **Model cascade** — a small model answers first, a strong-model verifier
  gates it, escalate only on failure.
- **Calibration** — the verifier separates verified facts from hedged
  forecasts (a year-ahead projection isn't scored like a hallucination).
- **Sleep cycle** — nightly consolidation digests the day, extracts
  lessons, prunes, and replays trajectories.

## Self-Modifications

The agent can change its own code on the user's request (e.g. "store
memory in SQLite instead of RAG"). These changes are **local** and don't
go into official git:

- A self-modification → a unified diff saved in `~/.hrant/data/self_mods/`.
- `hrant update` applies engine updates, then best-effort re-applies your
  patches.
- If a patch conflicts with the updated engine it's flagged "needs review"
  and not applied (the engine stays stable).
- Settings → Self-Modifications: list of patches, revert one or all.

Details and risks — [docs/self-modification.md](docs/self-modification.md).

## Documentation

- [docs/cognition.md](docs/cognition.md) — **cognitive architecture +
  the agent's work philosophy**: memory (knowledge/skills/trajectories),
  the cascade, calibration, the sleep cycle, the body (soul/identity)
- [docs/architecture.md](docs/architecture.md) — modules, pipelines, how
  the agent thinks *(Russian)*
- [docs/modes.md](docs/modes.md) — 4 deployment modes (`claude_only` /
  `local_full` / `cloud_finetune` / `local_cpu`) + dual-model router
- [docs/cli.md](docs/cli.md) — full `hrant` command reference
- [docs/autonomic.md](docs/autonomic.md) — Model X: 26 levers + immune
  system + safety gates
- [docs/finetune.md](docs/finetune.md) — fine-tune pipeline (autocollect
  → curate → train)
- [docs/sessions.md](docs/sessions.md) — sessions, conversations, and
  per-speaker user profiles (Telegram users vs WebUI)
- [docs/roles-and-scheduling.md](docs/roles-and-scheduling.md) —
  owner/trusted/guest roles + cross-speaker scheduled messages
- [docs/skills.md](docs/skills.md) — agent skills (markdown plugins) +
  autonomic heartbeat
- [docs/self-modification.md](docs/self-modification.md) — how local
  self-modification works
- [deploy/README.md](deploy/README.md) — install as a background service
- [docs/superpowers/specs/](docs/superpowers/specs/) — design docs
  (autonomic Model X, etc.)

## API surface

FastAPI generates interactive docs at runtime:
- `http://127.0.0.1:3333/docs`  — Swagger UI
- `http://127.0.0.1:3333/redoc` — ReDoc

Frequently used endpoints: `/api/chat` (SSE stream), `/api/knowledge`,
`/api/health`, `/api/discover`, `/api/engine/config`, `/api/cascade`,
`/api/model-routing`, `/api/autonomic/*`. Full list in Swagger UI.

## Anti-hallucination — hard rules

1. The agent doesn't answer "from memory" on topics where it has notes.
2. The solver system prompt: "answer ONLY from the notes".
3. Each solver step → verifier — not optional (can be disabled via
   Settings → Engine).
4. `confidence < min_confidence` → a ⚠️ prefix in the answer.
5. Contradictions with notes are explicitly flagged in
   `verification.contradictions`.

## Tests

```bash
pytest -q
```

Coverage: knowledge manager, core memory, parsers, verifier, cascade,
cognition pipeline, full agent cycle (mocked LLM), updater, paths layer.
~2800 tests / ~3min on a dev machine.

## License

TBD.
