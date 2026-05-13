# `hrant` — full CLI reference

Top-level command structure:

```
hrant <subcommand> [options]
```

All subcommands work two ways: `hrant <subcommand>` after `pip install -e .`, or `python -m backend.cli <subcommand>` without installation.

## Subcommands

### `hrant init`

Bootstrap a fresh install or reconfigure an existing one.

```
hrant init [--reset]
```

On a fresh install (`data_dir` empty):
1. Creates `~/.hrant/data/` (or `$HRANT_DATA_DIR` if set).
2. Copies `knowledge_templates/` → `data_dir/knowledge/`.
3. Copies `config.example.yaml` → `data_dir/config.yaml`.
4. Asks for API keys (Anthropic, OpenAI) and optional service URLs (Tailscale host, Whisper, Piper) and writes `.env`.

On an existing install: only the Q&A step runs; templates and config.yaml are preserved.

`--reset` re-copies templates even over existing files. Destructive — use when you want to throw away identity/soul customisations and start over.

### `hrant run`

Start the FastAPI server.

```
hrant run [--host HOST] [--port PORT] [--reload] [--log-level LEVEL]
```

Defaults: `--host 127.0.0.1 --port 8000 --log-level info`. Without `--reload`, runs as a single process suitable for systemd / launchd.

### `hrant status`

Read-only diagnostic. Prints active model + provider, external services (Whisper / Piper / Ollama + ffmpeg probe), Telegram channels with running state, workspace counts.

```
hrant status
```

### `hrant chat [...args]`

Drops into the interactive REPL (legacy `python cli.py` behaviour). Any extra args are forwarded to the REPL as the initial prompt.

```
hrant chat
hrant chat "what's the difference between RS-485 and RS-232?"
```

REPL commands (Russian, inherited): `запомни …` / `забудь …` / `изучи …` / `что ты знаешь?` / `статус` / `exit`.

### `hrant version`

Print the version string.

### `hrant discover`

Probe a Tailscale or LAN host for the agent's external services.

```
hrant discover [--host HOST] [--services CSV] [--apply]
```

Without `--host`, uses `$TAILSCALE_HOST`. Without `--services`, probes all three known services (whisper, piper, ollama). With `--apply`, writes discovered URLs into the per-service config files.

### `hrant rebuild`

Run `npm install && npm run build` in `frontend/`. Saves typing during dev when you've edited TSX and need port 8000 to pick it up.

### `hrant update`

Pull the latest engine code from `origin/<branch>`, reinstall Python deps, rebuild frontend.

```
hrant update [--check] [--branch NAME] [--skip-frontend] [--skip-pip]
```

- `--check` — show what's available on `origin`, don't change anything.
- `--branch` — track a non-master branch.
- `--skip-frontend` — backend-only (faster when no frontend changes).
- `--skip-pip` — skip `pip install -e .` (only safe when `pyproject.toml` hasn't moved).

Refuses on a dirty working tree (uncommitted tracked changes). Records the previous SHA in `~/.hrant/data/update_history.json` BEFORE pulling, so rollback is always one command away.

### `hrant rollback`

Revert engine to a previous SHA from the update history.

```
hrant rollback [--to SHA] [--list] [--skip-frontend] [--skip-pip]
```

- Without `--to`: one step back (the entry recorded right before the last successful update).
- `--to SHA` — explicit target.
- `--list` — print the history, don't change anything.

### `hrant service install / status / uninstall`

Install Hrant as a background service.

```
hrant service install   [--platform linux|macos|windows] [--host HOST] [--port PORT]
hrant service status    [--platform ...]
hrant service uninstall [--platform ...]
```

Renders the unit file from `deploy/<platform>/` into the user-mode location for that OS, then prints the activation command (you copy-paste it; `hrant` never runs privileged ops itself). See [deploy/README.md](../deploy/README.md) for activation details.

## Legacy REPL commands

`hrant chat` drops into the REPL. Commands accepted there:

| Command | What it does |
|---|---|
| `запомни <факт>` | Add to core memory |
| `забудь про <текст>` | Remove from core memory |
| `изучи <тема>` / `изучи глубоко <тема>` | Force-learn a topic (writes a note) |
| `что ты знаешь?` | List all topics |
| `что ты знаешь о <тема>?` | Show one note |
| `удали знания о <тема>` | Delete the note |
| `начать проект <имя>` / `завершить проект` | Project lifecycle |
| `контекст проекта <текст>` | Add project context |
| `решили X потому что Y` | Add decision |
| `проблема: X → fix: Y` | Add issue |
| `статус` | Status snapshot |
| `help` | Help |
| `exit` | Quit |

For the curated fine-tune subcommands (`finetune status`, `finetune review`, …), see [docs/finetune.md](finetune.md).
