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
2. Copies `backend/knowledge_templates/` → `data_dir/knowledge/`.
3. Copies `config.example.yaml` → `data_dir/config.yaml`.
4. Asks for API keys (Anthropic, OpenAI) and optional service URLs (Tailscale host, Whisper, Piper) and writes `.env`.

On an existing install: only the Q&A step runs; templates and config.yaml are preserved.

`--reset` re-copies templates even over existing files. Destructive — use when you want to throw away identity/soul customisations and start over.

### `hrant run`

Start the FastAPI server.

```
hrant run [--host HOST] [--port PORT] [--reload] [--log-level LEVEL]
```

Defaults: `--host 127.0.0.1 --port 3333 --log-level info`. Without `--reload`, runs as a single process suitable for systemd / launchd.

### `hrant status`

Read-only diagnostic. Prints active model + provider, external services (Whisper / Piper / Ollama + ffmpeg probe), Telegram channels with running state, workspace counts.

```
hrant status
```

### `hrant chat [...args]`

Drops into the interactive REPL. Any extra args are forwarded as the initial prompt.

```
hrant chat
hrant chat "what's the difference between RS-485 and RS-232?"
```

REPL commands (Russian, inherited): `запомни …` / `забудь …` / `изучи …` / `что ты знаешь?` / `статус` / `exit`.

### `hrant version`

Print the version string.

### `hrant config`

Friendly surface over the configs most users actually want to change. Modelled on `openclaw config <get|set|...>` — flat, colored, with redacted secrets and an interactive wizard.

```
hrant config                    # interactive menu — main entry point for noob users
hrant config list               # print every known setting, grouped, secrets redacted
hrant config get  <key>         # print one value (secrets redacted)
hrant config set  <key> <value> # change a value; coerces to int/float/choice per key
hrant config unset <key>        # remove a value
hrant config files              # show the paths to .env / config.yaml / *.json
hrant config edit               # open .env in $EDITOR (escape hatch)
```

**Surfaced keys** — only the configs that matter for first-time users. Advanced knobs (engine sliders, autonomic levers, full provider table) stay in the WebUI Settings tabs.

| Key | Backing file |
|---|---|
| `anthropic.api_key` | `.env:ANTHROPIC_API_KEY` (redacted) |
| `openai.api_key` | `.env:OPENAI_API_KEY` (redacted) |
| `telegram.bot_token` | `knowledge/channels.json` — the first `type=telegram` channel's `config.bot_token` (redacted) |
| `tailscale.host` | `.env:TAILSCALE_HOST` |
| `whisper.url` | `knowledge/transcriber_config.json:local_whisper.url` |
| `tts.backend` | `knowledge/tts_config.json:backend` — choice: `auto / edge_tts / local_piper / openai_tts / disabled` |
| `tts.piper_url` | `knowledge/tts_config.json:local_piper.url` |
| `tts.edge_voice` | `knowledge/tts_config.json:edge_tts.voice` |
| `tts.edge_voice_ru` | `knowledge/tts_config.json:edge_tts.voice_ru` |
| `autonomic.heartbeat_seconds` | `knowledge/autonomic_settings.json:tick_interval_seconds` (1..3600) |

Secrets are redacted on display — `list` and `get` show only the first 4 + last 4 chars. The actual file on disk is the source of truth; `hrant config` is just a friendlier face on it.

**Navigation:** `hrant config` (no args) opens an arrow-key menu — `↑/↓` move the selection, `Enter` picks, `q` / `Esc` go back. On non-TTY runs (cron, scripts, piped stdin) the menu degrades to a numbered prompt automatically.

**Colors:** warm orange accent (think openclaw's Lobster palette). Disabled automatically when stdout is not a TTY or `NO_COLOR` is set. On legacy Windows code pages, Unicode glyphs fall back to ASCII (`✓` → `[ok]`, `→` → `->`).

**Examples:**

```bash
hrant config                                  # opens the interactive wizard
hrant config set tts.backend edge_tts         # change TTS to free Microsoft voices
hrant config set whisper.url http://100.64.0.5:8016
hrant config set autonomic.heartbeat_seconds 60
hrant config get anthropic.api_key            # prints `sk-a…xxxx`
```

### `hrant provider`

Manage LLM providers from the command line — same surface as the
WebUI Providers tab, with the bonus that Codex / Copilot
subscription handoffs are easier to drive interactively.

```
hrant provider list                       # registered providers + active
hrant provider login <type>               # interactive sign-in
hrant provider test <provider_id>         # connectivity check
hrant provider use <provider_id> [--model X]   # set active model
hrant provider logout <provider_id>       # clear stored credentials
```

#### `hrant provider login <type>`

Picks the right flow for the provider type:

- **`codex` / `openai_codex`** — reads `~/.codex/auth.json` written
  by the upstream Codex CLI's `codex login`. If the file's missing,
  prints install + sign-in steps and exits non-zero so a script
  can retry.
- **`copilot` / `github_copilot`** — same idea for VS Code / `gh
  auth login --scopes copilot`. Reads `~/.config/github-copilot/`
  (or the Windows equivalent).
- **`ollama`** — probes `http://localhost:11434` (or a custom URL),
  lists installed models, lets you pick the default, registers
  the provider.
- **Anything else** — generic API-key flow. Prints the official
  signup URL + the steps from `PROVIDER_CONNECT_INFO`, then asks
  for the key + any extra fields (`base_url` for Azure /
  openai_compatible, AWS creds for Bedrock, etc.).

`hrant provider login` without a type prints the full list of
supported types from `backend.providers.PROVIDER_CONNECT_INFO`.

### `hrant discover`

Probe a Tailscale or LAN host for the agent's external services.

```
hrant discover [--host HOST] [--services CSV] [--apply]
```

Without `--host`, uses `$TAILSCALE_HOST`. Without `--services`, probes all three known services (whisper, piper, ollama). With `--apply`, writes discovered URLs into the per-service config files.

### `hrant jobs`

Durable per-turn records. Every user message (WebUI, Telegram, voice) gets a Job: an ID, status, prompt, response, timing, tool-call trace. Survives crashes — on `hrant run` startup, any job still in `running` or `queued` state is marked `interrupted` so you can retry it.

```
hrant jobs list   [--status running|failed|interrupted|...] [--channel webui|telegram|...] [--limit N]
hrant jobs show   <id>
hrant jobs retry  <id>      # clone the prompt into a new queued job
hrant jobs cancel <id>      # mark a non-terminal job cancelled
hrant jobs delete <id>      # purge the record
```

**State machine:**

```
queued → running → completed
                ↘ failed         (LLM down, tool crash, exception)
                ↘ interrupted    (server died mid-run — set on next boot)
                ↘ cancelled      (user clicked Cancel)
```

**Storage:** one JSON file per job at `~/.hrant/data/jobs/<id>.json`. Flat layout — under ~1k jobs it's fine. WebUI **Settings → Jobs** tab is the primary surface (status filter chips with live counts, details pane with prompt/response/tool trace, retry/cancel buttons).

**Boot recovery:** at `hrant run` startup, before binding the port, the recovery hook scans the jobs directory and flips anything `running` / `queued` to `interrupted`. Those were jobs in flight when the previous process died. The Jobs tab shows them with a yellow `interrupted N×` badge so you can retry.

**Failover trace (Phase B preview):** each Job has an `attempts[]` list. When auto-failover lands, every provider tried for that turn appends one entry — you'll see "Anthropic 429 → OpenAI ok" in the WebUI without needing log files.

### `hrant failover`

Multi-provider failover chain. When the active LLM returns a retryable error (rate limit, 5xx, timeout, auth, connection), the agent tries the next provider in the chain instead of failing the turn.

```
hrant failover status                          # show chain + flags
hrant failover enable                          # turn auto-failover on
hrant failover disable                         # turn off (keeps chain)
hrant failover add <provider_id> <model>       # append to the chain
hrant failover remove <1-based-index>          # drop one entry
hrant failover clear                           # empty the chain
```

**Storage:** `~/.hrant/data/knowledge/failover_config.json`:

```json
{
  "enabled": true,
  "chain": [
    {"provider_id": "openai-default", "model": "gpt-4o"},
    {"provider_id": "ollama-default", "model": "llama3.2:latest"}
  ],
  "retry_on": ["rate_limit", "server_error", "timeout", "auth_error", "connection"],
  "max_attempts": 4
}
```

**How it works:** the active pinned model is ALWAYS tried first. If it fails with a retryable error (classified by error message — `429` / `5xx` / timeout / `401`/`403` / connection refused), the agent walks the chain top-to-bottom and tries each entry. Each attempt — success or failure — appends one entry to the active Job's `attempts[]` so you can see the full trace in `Settings → Jobs → PROVIDER ATTEMPTS`.

**What's NOT retried:** `400` (bad request), content-policy violations, context-length errors. Those are the prompt's fault — trying another provider just wastes API quota. The `retry_on` config controls this.

**WebUI:** Settings → Providers → "Failover chain" panel — toggle enabled, drag-up/down to reorder, pick provider+model from dropdowns, toggle which error categories trigger failover, set max attempts.

### `hrant rebuild`

Run `npm install && npm run build` in `frontend/`. Saves typing during dev when you've edited TSX and need port 3333 to pick it up.

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

### `hrant gateway start / stop / restart / logs / install / status / uninstall`

Manage Hrant as a background service. Modelled on `openclaw gateway <action>` — one subcommand group for everything related to running the agent in the background.

```
hrant gateway start     [--host HOST] [--port PORT] [--gateway] [--platform ...]
hrant gateway stop      [--platform ...]
hrant gateway restart   [--platform ...]
hrant gateway logs      [-f|--follow] [--lines N] [--platform ...]
hrant gateway install   [--host HOST] [--port PORT] [--platform ...]
hrant gateway status    [--platform ...]
hrant gateway uninstall [--platform ...]
```

#### Lifecycle (the everyday commands)

- **`hrant gateway start`** — renders the platform unit file with the current install's paths, enables linger so the service survives logout (Linux), and starts the service. Idempotent — safe to re-run after `hrant update`.
  - `--gateway` is shorthand for `--host 0.0.0.0` so other devices on your LAN / Tailscale can reach it.
- **`hrant gateway stop`** — stop the service. Keeps the unit file (use `hrant gateway uninstall` for full teardown).
- **`hrant gateway restart`** — restart in place. Most common use: after `hrant update` so the new engine code is loaded.
- **`hrant gateway logs`** — tail the service's stdout/stderr:
  - Linux: `journalctl --user -u hrant`
  - macOS: `tail` of `logs/hrant.out.log`
  - Windows: `Get-ScheduledTaskInfo` (Scheduled Tasks doesn't stream stdout)
  - `-f` follows live; `--lines N` controls history (default 200).

#### Lower-level surface (review-before-activate)

`hrant gateway start` calls `install` internally + runs the activation step. If you'd rather inspect the unit file first, use these directly:

- **`hrant gateway install`** — render the unit file from `deploy/<platform>/` into the user-mode location (`~/.config/systemd/user/hrant.service` / `~/Library/LaunchAgents/ai.hrant.agent.plist` / `deploy/windows/install-service.rendered.ps1`). Prints the activation command but never runs privileged ops itself.
- **`hrant gateway status`** — wraps the platform's native status command (`systemctl --user status hrant` / `launchctl print …` / `Get-ScheduledTask`).
- **`hrant gateway uninstall`** — remove the unit file. Prints the matching disable command so the user can disable + remove from the service manager.

See [deploy/README.md](../deploy/README.md) for the underlying unit-file templates.

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
