# Hrant — Self-Learning Agent

Локальный AI-агент, который **не хранит знания в весах модели**. Вместо этого он читает источники, ведёт структурированные заметки (markdown) на диске и подгружает их в контекст только когда нужно. Растёт в компетенции под конкретные задачи, сохраняя маленькое эффективное ядро.

> junior engineer с идеальными конспектами, который никогда ничего не забывает.

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

**Что произойдёт:**
- `hrant init` создаст `~/.hrant/data/` (или туда, куда указывает `HRANT_DATA_DIR`), скопирует стартовые шаблоны из `knowledge_templates/`, сделает `config.yaml` из `config.example.yaml`, и спросит про API-ключи (Anthropic, OpenAI) + опциональные URL сервисов.
- `hrant run` поднимет FastAPI на `127.0.0.1:3333` (WebUI там же) и автоматически запустит сконфигурированные channels.

## Engine vs Data

```
<repo>/                  ← engine: backend/ frontend/ deploy/ knowledge_templates/
~/.hrant/data/           ← user data: config.yaml, knowledge/, workspace/, .env
~/.hrant/data/update_history.json   ← ledger for `hrant rollback`
```

`hrant update` обновляет только engine; пользовательские данные не трогаются. Можно сменить расположение через `HRANT_DATA_DIR=/some/path hrant init`.

**Dev mode (single-tree):** если запускаешь из репо без `HRANT_DATA_DIR`, агент использует `<repo>/knowledge/` и `<repo>/workspace/` (всё в .gitignore). Удобно для разработки.

## Update / Rollback

```bash
hrant update --check               # что нового на origin/master, без действий
hrant update                       # pull → pip install -e . → npm build
hrant update --skip-frontend       # backend-only (быстрее)
hrant rollback                     # шаг назад
hrant rollback --list              # история всех обновлений
hrant rollback --to <sha>          # к конкретному коммиту
hrant rebuild                      # только пересборка фронта без pull
```

`hrant update` отказывается работать при dirty working tree (есть незакоммиченные изменения); untracked файлы в `knowledge/`/`workspace/` (.gitignore'd) не считаются. История пишется в `~/.hrant/data/update_history.json` **до** `git pull`, так что rollback доступен даже если update упал на половине.

## Run as a background service

Все команды для запуска агента как сервиса собраны в группе `hrant gateway` (по аналогии с `openclaw gateway`). Самый короткий путь:

```bash
hrant gateway start                # установить unit + стартовать (idempotent)
hrant gateway start --gateway      # bind 0.0.0.0 — другие устройства в LAN/Tailscale достанут
hrant gateway start --port 4444    # нестандартный порт

hrant gateway logs -f              # стрим логов (journalctl --user -u hrant -f)
hrant gateway restart              # после `hrant update`
hrant gateway stop                 # остановить (unit остаётся, перезапуск через `gateway start`)
```

Под капотом `gateway start` это `gateway install` + activation. Если хочется сначала посмотреть unit-файл — используй `gateway install` и активируй вручную.

```bash
hrant gateway install              # положить unit, НЕ стартовать
hrant gateway status               # что показывает OS service manager
hrant gateway uninstall            # удалить unit, оставить venv
```

Подробнее по платформам — [deploy/README.md](deploy/README.md).

## Configure via WebUI

После `hrant run` → `http://127.0.0.1:3333` → Settings. Вкладки:

- **Identity / Soul / User Profile** — кто такой агент, кто такой пользователь
- **Providers** — добавлять/переключать LLM-провайдеры (Anthropic, OpenAI, Ollama, …)
- **Channels** — Telegram-боты и др.
- **Memory** — embeddings backend (llama.cpp / ollama / OpenAI / Cohere)
- **Voice** — Whisper / Piper + Tailscale discover для авто-обнаружения сервисов
- **Engine** — router budget, verification strictness, workspace retention, knowledge caps (всё применяется live, без рестарта)
- **Self-Modifications** — список локальных патчей агента, кнопки revert (см. ниже)
- **Status** — диагностика всех подсистем

## Self-Modifications

Агент умеет менять собственный код по запросу пользователя (например, «сохраняй память в SQLite вместо RAG»). Эти изменения **локальны** и не уходят в official git:

- Самомодификация → unified diff сохраняется в `~/.hrant/data/self_mods/`.
- `hrant update` применяет апдейты engine, потом best-effort переприменяет твои патчи.
- Если патч конфликтует с обновлённым engine, он помечается «needs review» и не применяется (engine остаётся стабильным).
- Settings → Self-Modifications: список патчей, revert по одному или всех.

Детали и риски — [docs/self-modification.md](docs/self-modification.md).

## Documentation

- [docs/architecture.md](docs/architecture.md) — модули, пайплайны, как агент думает
- [docs/modes.md](docs/modes.md) — 4 deployment modes (`claude_only` / `local_full` / `cloud_finetune` / `local_cpu`) + dual-model router
- [docs/cli.md](docs/cli.md) — полная справка по командам `hrant`
- [docs/autonomic.md](docs/autonomic.md) — Model X: 19 levers + immune system + safety gates
- [docs/finetune.md](docs/finetune.md) — fine-tune pipeline (autocollect → curate → train)
- [docs/sessions.md](docs/sessions.md) — sessions, conversations, и per-speaker user profiles (Telegram users vs WebUI)
- [docs/roles-and-scheduling.md](docs/roles-and-scheduling.md) — owner/trusted/guest roles + cross-speaker scheduled messages (Phase 11)
- [docs/skills.md](docs/skills.md) — agent skills (markdown plugins) + autonomic heartbeat (Phase 12)
- [docs/self-modification.md](docs/self-modification.md) — как работает локальная самомодификация (Phase 7+)
- [deploy/README.md](deploy/README.md) — установка как фоновый сервис
- [docs/superpowers/specs/](docs/superpowers/specs/) — design docs (autonomic Model X и др.)

## API surface

FastAPI генерирует интерактивную доку при запуске:
- `http://127.0.0.1:3333/docs`  — Swagger UI
- `http://127.0.0.1:3333/redoc` — ReDoc

Часто используемые endpoints: `/api/chat` (SSE-стрим), `/api/knowledge`, `/api/health`, `/api/discover`, `/api/engine/config`, `/api/autonomic/*`. Полный список — в Swagger UI.

## Anti-hallucination — hard rules

1. Агент не отвечает «из головы» по темам, где есть заметки.
2. Системный промпт solver'а: «отвечай ТОЛЬКО по заметкам».
3. Каждый шаг solver → verifier — не опционален (можно отключить через Settings → Engine).
4. `confidence < min_confidence` → в ответе ⚠️ префикс.
5. Противоречия заметкам явно помечаются в `verification.contradictions`.

## Tests

```bash
pytest -q
```

Coverage: knowledge manager, core memory, parsers, verifier, full agent cycle (mocked LLM), updater, paths layer. ~1130 tests / <2min on dev machine.

## License

TBD.
