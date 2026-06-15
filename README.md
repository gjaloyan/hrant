<div align="center">

# 🧠 Hrant

### A self-learning AI agent that keeps its knowledge in *notes*, not in the model's weights

*[English](README.md) · [Русский](README.ru.md)*

![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)
![Backend](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Frontend](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)
![Local-first](https://img.shields.io/badge/local--first-no%20cloud%20required-success)
![Tests](https://img.shields.io/badge/tests-2.8k-brightgreen)
![Model-agnostic](https://img.shields.io/badge/models-Claude%20%C2%B7%20GPT%20%C2%B7%20Qwen%20%C2%B7%20local-blueviolet)

> *A junior engineer with perfect notes who never forgets anything — and gets smarter every night while you sleep.*

</div>

---

Hrant **doesn't store knowledge in the model's weights.** It reads sources,
keeps structured markdown notes on disk, and loads them into context only
when needed. It grows competence for the tasks you actually give it while
keeping a small, efficient core — which is what lets it stay smart even on
cheap or local models.

```
   the model is the muscle.   the body — files on disk — is who it is.
   swap the model, the agent remains.
```

## ✨ What makes it different

|   | |
|---|---|
| 🧩 **Knowledge ≠ weights** | Studies a domain once (expensive), recalls it cheaply forever. Notes + knowledge graph + vector search. |
| 🪜 **Model cascade** | A cheap small model answers first; a strong-model verifier gates it; escalate only on failure. |
| 🔎 **Anti-hallucination by design** | Every answer is verified against sources; hedged forecasts are scored separately from facts. |
| 🌙 **Learns while idle** | Nightly consolidation digests the day, extracts lessons, prunes, and replays past solutions. |
| 🧭 **Method before execution** | For a real task it first researches *how experts do it*, then executes — not price-only analysis. |
| 🫀 **A body, not just a prompt** | Character, morality and judgment live in `soul.md` / `identity.md` — and hold even on a 3B model. |
| 🛠 **Owns the machine** | Full shell, background jobs, Telegram, self-modification, reminders — with hard safety rails. |

→ Full design + the agent's work philosophy: **[docs/cognition.md](docs/cognition.md)**

## 🚀 Install (fresh machine)

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
#    → open http://127.0.0.1:3333
```

`hrant init` creates `~/.hrant/data/` (or wherever `HRANT_DATA_DIR` points),
copies starter templates, makes `config.yaml`, and asks about API keys
(Anthropic, OpenAI) + optional service URLs. `hrant run` brings up FastAPI on
`127.0.0.1:3333` (WebUI included) and auto-starts the configured channels.

Change anything later through `hrant config`:

```bash
hrant config                              # interactive menu — the main entry point
hrant config list                         # see all settings (secrets masked)
hrant config set tts.backend edge_tts     # switch voice to free Edge TTS
```

Details — [docs/cli.md](docs/cli.md#hrant-config).

## 📦 Engine vs Data

```
<repo>/                  ← engine: backend/ (incl. knowledge_templates/) frontend/ deploy/
~/.hrant/data/           ← user data: config.yaml, knowledge/, workspace/, .env
~/.hrant/data/update_history.json   ← ledger for `hrant rollback`
```

`hrant update` updates only the engine; user data is untouched. Move it with
`HRANT_DATA_DIR=/some/path hrant init`. **Dev mode:** run from the repo without
`HRANT_DATA_DIR` and the agent uses `<repo>/knowledge/` + `<repo>/workspace/`
(both gitignored).

## 🔄 Update / Rollback

```bash
hrant update --check               # what's new on origin/master, no actions
hrant update                       # pull → pip install -e . → npm build
hrant update --skip-frontend       # backend-only (faster)
hrant rollback                     # one step back
hrant rollback --list              # history of all updates
hrant rollback --to <sha>          # to a specific commit
```

`hrant update` refuses a dirty working tree (gitignored `knowledge/` /
`workspace/` don't count). History is written **before** `git pull`, so
rollback works even if an update fails halfway.

## 🖥 Run as a background service

All service commands are grouped under `hrant gateway`:

```bash
hrant gateway start                # install unit + start (idempotent)
hrant gateway start --gateway      # bind 0.0.0.0 — reachable on LAN/Tailscale
hrant gateway logs -f              # stream logs
hrant gateway restart              # after `hrant update`
```

More per-platform detail — [deploy/README.md](deploy/README.md).

## ⚙️ Configure via WebUI

After `hrant run` → `http://127.0.0.1:3333` → **Settings**:

| Tab | What it does |
|---|---|
| **Identity / Soul / User Profile** | who the agent is, who you are |
| **Providers** | add/switch LLM providers (Anthropic, OpenAI, Ollama, OpenRouter…) + pick a model from the live catalog |
| **Channels** | Telegram bots, etc. |
| **Memory / Voice** | embeddings backend · Whisper / Piper + Tailscale auto-discovery |
| **Engine** | router budget, verification strictness, retention, caps — all live, no restart |
| **Reminders** | create / list / cancel scheduled messages |
| **Fine-Tune** | distillation queue + **model cascade** controls (small-tier, gate, on/off) |
| **Self-Modifications** | the agent's local patches, revert buttons |
| **Status** | diagnostics of every subsystem |

## 🤔 How the agent thinks

Wisdom, method and identity live in the agent's **body** (files that survive a
model swap) rather than the weights — which is what lets a small model stay
smart. Full map + work philosophy: **[docs/cognition.md](docs/cognition.md)**.
In short:

- **Method before execution** — research *how the job is properly done*, then do it.
- **Three memories** — knowledge (studied theory), skills (procedures), trajectories (past cases).
- **Model cascade** — small answers first, strong-model verifier gates, escalate on failure.
- **Calibration** — verified facts vs hedged forecasts scored separately (a year-ahead projection isn't a hallucination).
- **Sleep cycle** — nightly: digest → lessons → prune → replay.

## 🔧 Self-Modifications

The agent can change its own code on request (e.g. "store memory in SQLite
instead of RAG"). Changes are **local** and never enter official git: a
unified diff is saved in `~/.hrant/data/self_mods/`, re-applied best-effort
after `hrant update`, and flagged "needs review" if it conflicts with the
updated engine. Revert one or all in Settings.

Details and risks — [docs/self-modification.md](docs/self-modification.md).

## 📚 Documentation

| Doc | What's inside |
|---|---|
| **[cognition.md](docs/cognition.md)** | **cognitive architecture + the agent's work philosophy** |
| [architecture.md](docs/architecture.md) | modules, pipelines, how the agent thinks *(Russian)* |
| [modes.md](docs/modes.md) | 4 deployment modes + dual-model router |
| [cli.md](docs/cli.md) | full `hrant` command reference |
| [autonomic.md](docs/autonomic.md) | Model X: 26 levers + immune system + safety gates |
| [finetune.md](docs/finetune.md) | fine-tune pipeline (autocollect → curate → train) |
| [sessions.md](docs/sessions.md) | sessions, conversations, per-speaker profiles |
| [roles-and-scheduling.md](docs/roles-and-scheduling.md) | owner/trusted/guest roles + scheduled messages |
| [skills.md](docs/skills.md) | agent skills (markdown plugins) + autonomic heartbeat |
| [self-modification.md](docs/self-modification.md) | how local self-modification works |
| [deploy/README.md](deploy/README.md) | install as a background service |

## 🌐 API surface

FastAPI generates interactive docs at runtime — Swagger at
`http://127.0.0.1:3333/docs`, ReDoc at `/redoc`. Frequently used:
`/api/chat` (SSE stream), `/api/knowledge`, `/api/health`, `/api/cascade`,
`/api/model-routing`, `/api/autonomic/*`.

## 🛡 Anti-hallucination — hard rules

1. No answering "from memory" on topics where notes exist.
2. Solver prompt: answer **only** from the notes.
3. Each solver step → verifier (can be disabled via Settings → Engine).
4. `confidence < min_confidence` → a ⚠️ prefix on the answer.
5. Contradictions with notes are flagged in `verification.contradictions`.

## 🧪 Tests

```bash
pytest -q     # ~2800 tests / ~3min on a dev machine
```

Coverage: knowledge manager, core memory, parsers, verifier, cascade,
cognition pipeline, full agent cycle (mocked LLM), updater, paths layer.

## 📄 License

TBD.
