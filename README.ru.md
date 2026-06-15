<div align="center">

# 🧠 Hrant

### Самообучающийся AI-агент, который хранит знания в *заметках*, а не в весах модели

*[English](README.md) · [Русский](README.ru.md)*

![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)
![Backend](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Frontend](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)
![Local-first](https://img.shields.io/badge/local--first-без%20облака-success)
![Tests](https://img.shields.io/badge/tests-2.8k-brightgreen)
![Model-agnostic](https://img.shields.io/badge/models-Claude%20%C2%B7%20GPT%20%C2%B7%20Qwen%20%C2%B7%20local-blueviolet)

> *Junior engineer с идеальными конспектами, который никогда ничего не забывает — и умнеет каждую ночь, пока ты спишь.*

</div>

---

Hrant **не хранит знания в весах модели.** Он читает источники, ведёт
структурированные markdown-заметки на диске и подгружает их в контекст только
когда нужно. Растёт в компетенции под конкретные задачи, сохраняя маленькое
эффективное ядро — именно это позволяет ему оставаться умным даже на дешёвых
или локальных моделях.

```
   модель — это мускул.   тело — файлы на диске — это кто он есть.
   смени модель, агент останется.
```

## ✨ Чем отличается

|   | |
|---|---|
| 🧩 **Знания ≠ веса** | Изучает домен один раз (дорого), достаёт его дёшево навсегда. Заметки + knowledge graph + векторный поиск. |
| 🪜 **Каскад моделей** | Дешёвая малая модель отвечает первой; верификатор на сильной модели судит; эскалация только при провале. |
| 🔎 **Анти-галлюцинации by design** | Каждый ответ проверяется по источникам; хеджированные прогнозы оцениваются отдельно от фактов. |
| 🌙 **Учится в простое** | Ночная консолидация переваривает день, извлекает уроки, чистит и переигрывает прошлые решения. |
| 🧭 **Метод до исполнения** | Для реальной задачи сначала изучает, *как делают эксперты*, потом исполняет — не price-only анализ. |
| 🫀 **Тело, а не просто промпт** | Характер, мораль и суждение живут в `soul.md` / `identity.md` — и держатся даже на 3B-модели. |
| 🛠 **Владеет машиной** | Полный shell, фоновые задачи, Telegram, самомодификация, напоминания — с жёсткими предохранителями. |

→ Полный дизайн + философия работы агента: **[docs/cognition.ru.md](docs/cognition.ru.md)**

## 🚀 Установка (с нуля)

```bash
# 1. Получить движок
git clone https://github.com/gjaloyan/hrant.git
cd hrant

# 2. Зависимости Python + Node
python -m venv .venv && source .venv/bin/activate     # или .venv\Scripts\activate на Windows
pip install -e .
cd frontend && npm install && npm run build && cd ..

# 3. Первичный bootstrap (создаёт ~/.hrant/data/ + спросит API-ключи)
hrant init

# 4. Запуск агента
hrant run
#    → открой http://127.0.0.1:3333
```

`hrant init` создаст `~/.hrant/data/` (или туда, куда указывает
`HRANT_DATA_DIR`), скопирует стартовые шаблоны, сделает `config.yaml` и
спросит про API-ключи (Anthropic, OpenAI) + опциональные URL сервисов.
`hrant run` поднимет FastAPI на `127.0.0.1:3333` (WebUI там же) и запустит
сконфигурированные channels.

Менять настройки потом — через `hrant config`:

```bash
hrant config                              # интерактивное меню — главный вход
hrant config list                         # все настройки (секреты замаскированы)
hrant config set tts.backend edge_tts     # переключить голос на бесплатный Edge TTS
```

Подробности — [docs/cli.md](docs/cli.md#hrant-config).

## 📦 Engine vs Data

```
<repo>/                  ← engine: backend/ (incl. knowledge_templates/) frontend/ deploy/
~/.hrant/data/           ← user data: config.yaml, knowledge/, workspace/, .env
~/.hrant/data/update_history.json   ← журнал для `hrant rollback`
```

`hrant update` обновляет только engine; пользовательские данные не трогаются.
Сменить расположение — `HRANT_DATA_DIR=/some/path hrant init`. **Dev mode:**
запуск из репо без `HRANT_DATA_DIR` → агент использует `<repo>/knowledge/` +
`<repo>/workspace/` (всё в .gitignore).

## 🔄 Update / Rollback

```bash
hrant update --check               # что нового на origin/master, без действий
hrant update                       # pull → pip install -e . → npm build
hrant update --skip-frontend       # только backend (быстрее)
hrant rollback                     # шаг назад
hrant rollback --list              # история всех обновлений
hrant rollback --to <sha>          # к конкретному коммиту
```

`hrant update` отказывается работать при dirty working tree (gitignore'нутые
`knowledge/` / `workspace/` не считаются). История пишется **до** `git pull`,
так что rollback доступен даже если update упал на половине.

## 🖥 Запуск как фоновый сервис

Все команды сервиса собраны в группе `hrant gateway`:

```bash
hrant gateway start                # установить unit + стартовать (idempotent)
hrant gateway start --gateway      # bind 0.0.0.0 — достанут устройства в LAN/Tailscale
hrant gateway logs -f              # стрим логов
hrant gateway restart              # после `hrant update`
```

Подробнее по платформам — [deploy/README.md](deploy/README.md).

## ⚙️ Настройка через WebUI

После `hrant run` → `http://127.0.0.1:3333` → **Settings**:

| Вкладка | Что делает |
|---|---|
| **Identity / Soul / User Profile** | кто такой агент, кто такой ты |
| **Providers** | добавлять/переключать LLM-провайдеров (Anthropic, OpenAI, Ollama, OpenRouter…) + выбрать модель из живого каталога |
| **Channels** | Telegram-боты и др. |
| **Memory / Voice** | embeddings backend · Whisper / Piper + Tailscale auto-discovery |
| **Engine** | router budget, строгость verification, retention, caps — всё live, без рестарта |
| **Reminders** | создать / список / отменить отложенные сообщения |
| **Fine-Tune** | очередь дистилляции + управление **каскадом моделей** (малый ярус, гейт, on/off) |
| **Self-Modifications** | локальные патчи агента, кнопки revert |
| **Status** | диагностика всех подсистем |

## 🤔 Как агент думает

Мудрость, метод и идентичность живут в **теле** агента (файлы, переживающие
смену модели), а не в весах — именно это позволяет малой модели оставаться
умной. Полная карта + философия работы: **[docs/cognition.ru.md](docs/cognition.ru.md)**.
Кратко:

- **Метод до исполнения** — изучить, *как работа делается правильно*, потом делать.
- **Три памяти** — knowledge (изученная теория), skills (процедуры), trajectories (прошлые случаи).
- **Каскад моделей** — малая отвечает первой, верификатор на сильной судит, эскалация при провале.
- **Калибровка** — проверяемые факты vs хеджированные прогнозы оцениваются отдельно (прогноз на год — не галлюцинация).
- **Sleep-цикл** — ночью: digest → уроки → чистка → переигрывание траекторий.

## 🔧 Self-Modifications

Агент умеет менять собственный код по запросу (например, «храни память в SQLite
вместо RAG»). Изменения **локальны** и не уходят в official git: unified diff
сохраняется в `~/.hrant/data/self_mods/`, переприменяется best-effort после
`hrant update`, помечается «needs review» при конфликте с обновлённым engine.
Revert по одному или всех — в Settings.

Детали и риски — [docs/self-modification.md](docs/self-modification.md).

## 📚 Документация

| Документ | Что внутри |
|---|---|
| **[cognition.ru.md](docs/cognition.ru.md)** | **когнитивная архитектура + философия работы агента** |
| [architecture.md](docs/architecture.md) | модули, пайплайны, как агент думает |
| [modes.md](docs/modes.md) | 4 deployment modes + dual-model router |
| [cli.md](docs/cli.md) | полная справка по командам `hrant` |
| [autonomic.md](docs/autonomic.md) | Model X: 26 levers + immune system + safety gates |
| [finetune.md](docs/finetune.md) | fine-tune pipeline (autocollect → curate → train) |
| [sessions.md](docs/sessions.md) | sessions, conversations, per-speaker профили |
| [roles-and-scheduling.md](docs/roles-and-scheduling.md) | owner/trusted/guest роли + scheduled messages |
| [skills.md](docs/skills.md) | agent skills (markdown plugins) + autonomic heartbeat |
| [self-modification.md](docs/self-modification.md) | как работает локальная самомодификация |
| [deploy/README.md](deploy/README.md) | установка как фоновый сервис |

## 🌐 API surface

FastAPI генерирует интерактивную доку при запуске — Swagger на
`http://127.0.0.1:3333/docs`, ReDoc на `/redoc`. Часто используемые:
`/api/chat` (SSE-стрим), `/api/knowledge`, `/api/health`, `/api/cascade`,
`/api/model-routing`, `/api/autonomic/*`.

## 🛡 Анти-галлюцинации — жёсткие правила

1. Никаких ответов «из головы» по темам, где есть заметки.
2. Системный промпт solver'а: отвечай **только** по заметкам.
3. Каждый шаг solver → verifier (отключается через Settings → Engine).
4. `confidence < min_confidence` → ⚠️ префикс в ответе.
5. Противоречия заметкам помечаются в `verification.contradictions`.

## 🧪 Тесты

```bash
pytest -q     # ~2800 тестов / ~3 мин на dev-машине
```

Покрытие: knowledge manager, core memory, parsers, verifier, cascade,
cognition pipeline, полный цикл агента (mocked LLM), updater, paths layer.

## 📄 License

TBD.
