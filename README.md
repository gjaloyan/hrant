# Self-Learning Agent

Локальный AI-агент, который **не хранит знания в весах модели**. Вместо этого он читает источники, ведёт структурированные заметки (markdown) на диске и подгружает их в контекст только когда нужно. Растёт в компетенции под конкретные задачи, сохраняя маленькое эффективное ядро.

> junior engineer с идеальными конспектами, который никогда ничего не забывает.

## Deployment Modes

Главная настройка в `config.yaml` — `mode:`. Выбор режима задаёт sensible defaults для всех подсистем. Любой ключ можно переопределить явно.

| mode | GPU нужен | Ollama нужен | Где тренируется | Когда выбирать |
|---|---|---|---|---|
| **`local_full`** | ✅ ≥12GB VRAM | ✅ | локально (Unsloth LoRA) | Полный стек на одной машине с мощной видеокартой. |
| **`cloud_finetune`** | ❌ | ✅ | арендованная GPU (RunPod/Vast.ai/Colab Pro) | Нет своего GPU, но готов платить за аренду для тренировки. Inference локально. |
| **`local_cpu`** | ❌ | ✅ (CPU-mode) | локально на CPU (экспериментально) | Слабая машина без видеокарты. Использует **Qwen 1.5B** вместо 7B, тренировка возможна но очень медленная. |
| **`claude_only`** | ❌ | ❌ | **отключено** | Только Claude API, без локальных моделей. Заметки и finetune_queue собираются, но обучение недоступно. |

### Что именно делает каждый режим

**`local_full`** — полная реализация CLAUDE_CODE_PROMPT.md:
- `model_b`: `qwen2.5:7b-instruct` через Ollama
- `router.auto_shift_after_finetune: true` — доля Qwen растёт по v0→v5
- `finetune start` → Unsloth LoRA на локальной GPU → регистрация в Ollama

**`cloud_finetune`** — сбор локально, тренировка в облаке:
- Всё как в `local_full`, но `training_location: cloud`
- `finetune start` блокируется с подсказкой использовать `finetune export-cloud`
- `finetune export-cloud` генерирует папку `models/cloud_export_v{N}/` с:
  - `train.jsonl` / `val.jsonl` (после курации)
  - `train_script.py` (готов к запуску `python train_script.py` на арендованной GPU)
  - `config.json` с hyperparams
  - `README_CLOUD.md` с пошаговой инструкцией
- Заливаешь пакет на RunPod/Vast.ai/Colab Pro → запускаешь `train_script.py` → получаешь `unsloth.Q4_K_M.gguf`
- Возвращаешь файл локально → `finetune import-gguf <path> --tag v1` → Ollama регистрирует модель

**`local_cpu`** — для машин без GPU:
- `model_b`: `qwen2.5:1.5b-instruct` (помещается в ≤4GB RAM)
- `router.auto_shift_after_finetune: false` — не пытаться грузить 1.5B сложными запросами
- `training_location: local_cpu` — Unsloth запускается без CUDA (медленно, но работает)
- Агент покажет warning перед стартом тренировки: "10+ часов на 50 примерах"

**`claude_only`** — чистый клиент Claude API:
- `model_b: None` — Ollama не нужна, роутер помечает B как постоянно недоступную
- `router.fallback_to_local: false`, `auto_shift_after_finetune: false`
- `finetune.enabled: false` — `finetune start` и `export-cloud` блокируются
- Агентский цикл (analyze → solve → verify) работает полностью через Claude
- Заметки, core memory, проекты — всё локально, как обычно
- **Finetune queue копится** на случай если позже переключишься на другой mode

### Переключение режимов

В `config.yaml`:
```yaml
mode: "cloud_finetune"     # ← замени на нужный
```
Перезапуск — и всё. Переопределения поверх пресета:
```yaml
mode: "local_full"
model_a:
  model: "claude-opus-4-6"     # использовать Opus вместо Sonnet
router:
  daily_api_budget_usd: 20.0    # увеличить бюджет
```

### UI / CLI

- **CLI:** `mode` или `режим` — показать текущий режим и доступные.
- **CLI:** `status` — первая строка показывает `mode: X (training: Y)`.
- **Web UI:** в StatusBar внизу — цветной бейдж с режимом.
- **FinetunePanel:** кнопки автоматически меняются под режим:
  - `local_full` / `local_cpu` → **Start Fine-Tune (local)**
  - `cloud_finetune` → **📦 Export for Cloud GPU** + поле `import-gguf`
  - `claude_only` → мягкое уведомление "fine-tune недоступен"

## Dual-Model Architecture

```
         USER TASK
             │
             ▼
  ┌────────────────────────────────┐
  │      DualModelRouter           │
  │                                │
  │  Для каждого подзапроса        │
  │  выбирает провайдера по:       │
  │                                │
  │  1. TaskType (analyze/verify → A, lookup → B)
  │  2. verification.always_use_model_a (hard override)
  │  3. shift_schedule (v0→v5: доля B растёт)
  │  4. daily_api_budget_usd → fallback B
  │  5. API availability → fallback B
  └──────┬───────────────────┬─────┘
         │                   │
         ▼                   ▼
  ┌────────────────┐  ┌──────────────────────────┐
  │ Model A        │  │ Model B                  │
  │ Claude Sonnet  │  │ Qwen 2.5 7B              │
  │ (API)          │  │ (Ollama + LoRA fine-tune)│
  │                │  │                          │
  │ • task_analysis│  │ • simple_lookup          │
  │ • learning     │  │ • keyword_extraction     │
  │ • complex_solve│  │ • note_search            │
  │ • verification │  │ • quick_answer           │
  │ • note_creation│  │ • classification         │
  └────────┬───────┘  └──────────┬───────────────┘
           │                     │
           └──────────┬──────────┘
                      ▼
         knowledge/ + finetune_queue.jsonl
         router_state.json (учёт вызовов, стоимость)
         model_versions.json (v0, v1, v2, ...)
```

### Эволюция (auto-shift after fine-tune)

| Версия Qwen | A → Claude | B → Qwen | Когда |
|---|---|---|---|
| v0 | 90% | 10% | base модель, до первого fine-tune |
| v1 | 60% | 40% | после первого LoRA обучения |
| v2 | 40% | 60% | второе обучение |
| v3 | 20% | 80% | третье обучение |
| v5 | 10% | 90% | зрелый агент — Claude только на новых задачах |

По мере дообучения Qwen на собранном опыте, всё больше A-задач случайным образом маршрутизируются на B, пока Qwen не покрывает почти все запросы локально и приватно.

### Hard constraints

- **Verification** (`config.verification.always_use_model_a: true`) — всегда Claude, независимо от shift. Критическая проверка ответов не должна деградировать.
- **Budget** (`router.daily_api_budget_usd`) — при превышении → fallback B (если Ollama поднят).
- **API down** (`router.fallback_to_local: true`) — при недоступности Claude → B (если Ollama поднят).

### Работа без Ollama

Проект изначально работает на чистом Claude — Qwen/Ollama **опциональны до того момента, когда ты накопишь 50+ Q&A и захочешь запустить первый fine-tune**.

Роутер детектит `ollama/api/tags` (cached ping 60 сек) и поведение когда B недоступен:

| Ситуация | Что делает роутер |
|---|---|
| A-задача, Ollama down | всё идёт на Claude (shift_schedule игнорируется) |
| A-задача, Claude down + Ollama down | `LLMError` с явным сообщением "обе модели недоступны" |
| A-задача, over-budget + Ollama down | идёт на Claude с пометкой `"no B available"` в last_reason |
| B-задача (`simple_lookup` etc.), Ollama down | **эскалация на Claude** — "Ollama down, escalate A" |
| B-задача, обе down | `LLMError` |
| Runtime падение Claude | fallback на B только если Ollama реально отвечает |

**Главный invariant:** агентский цикл (`TASK_ANALYSIS → COMPLEX_SOLVING → VERIFICATION`) состоит только из A-задач. Пока ключ Claude валиден, агент работает полностью без Ollama.

Индикаторы в UI:
- 🟢 зелёный dot рядом с `🧠 A:` / `📖 B:` = модель отвечает на ping
- 🔴 красный = не отвечает (tooltip "НЕДОСТУПЕН")

CLI: `status` → строка `last routing: ...` покажет причину последнего выбора провайдера.

**Чтобы полностью отключить Qwen-ветку** (например, нет GPU и никогда не будет):
```yaml
router:
  auto_shift_after_finetune: false   # не пытаться гнать shift на B
  fallback_to_local: false           # при падении Claude → LLMError, не идти на B
verification:
  always_use_model_a: true           # (уже так)
```

### State

[knowledge/router_state.json](knowledge/router_state.json) хранит per-day счётчики:
```json
{
  "date": "2026-04-07",
  "api_calls_today": 12,
  "api_cost_today": 0.12,
  "model_b_calls_today": 3,
  "total_a_calls": 145,
  "total_b_calls": 27,
  "last_reason": "verification: forced A"
}
```
Счётчики today сбрасываются при смене даты.

## Архитектура

4 слоя:

1. **LLM Core (мозг)** — Claude API или локальная Ollama. Только рассуждение.
2. **Orchestrator** — цикл из 7 шагов: load core → analyze → learn → solve → verify → experience → cleanup.
3. **Local Memory** — 3 уровня:
   - **core_memory.md** — всегда в контексте;
   - **knowledge/** заметки — подгружаются по запросу;
   - **finetune_queue.jsonl** — пары для будущего fine-tune.
4. **Tools** — web search, file reader, code executor.

Подробная схема — в `SPEC.md` (или в исходной спеке проекта).

## Установка

**Основа (CLI + Claude reasoning):**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

copy .env.example .env
# впиши ANTHROPIC_API_KEY
```

**Локальная Qwen 7B (для fine-tune memory):**
```bash
# Ollama для inference
winget install Ollama.Ollama        # или скачай с ollama.com
ollama pull qwen2.5:7b-instruct     # базовая модель (v0)

# Unsloth для training (требует GPU ≥12GB VRAM)
pip install unsloth trl transformers datasets
```

Без Qwen-части агент работает чисто на Claude — fine-tune queue копится, но `finetune start` упадёт пока не установишь Unsloth + Ollama.

## Быстрый старт — CLI

```bash
python cli.py                                 # интерактивный режим
python cli.py "как подключить DS18B20 к Arduino Uno?"
```

Команды в CLI:
- `запомни <факт>` — в core memory
- `забудь про <текст>` — удалить
- `изучи <тема>` / `изучи глубоко <тема>` — форсировать обучение
- `что ты знаешь?` — список всех тем
- `что ты знаешь о <тема>?` — показать заметку
- `удали знания о <тема>`
- `начать проект <имя>` / `завершить проект`
- `контекст проекта <текст>` / `решили X потому что Y` / `проблема: X → fix: Y`
- `статус`, `help`, `exit`

## Web-UI (FastAPI + React)

Бэкенд:
```bash
python -m backend.main
# → http://localhost:8000
```

Фронтенд (отдельная консоль):
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Dev-сервер Vite проксирует `/api/*` на `localhost:8000`.

## API

| метод | путь | назначение |
|---|---|---|
| POST | `/api/chat` | SSE-стрим ответа агента (progress + answer) |
| GET  | `/api/knowledge` | список всех тем (flat + by_category) |
| GET  | `/api/knowledge/{topic}` | получить заметку |
| POST | `/api/knowledge/learn` | форсировать обучение (`{topic, depth, category}`) |
| DELETE | `/api/knowledge/{topic}` | удалить заметку |
| GET / POST / DELETE | `/api/core-memory` | core memory |
| GET / POST | `/api/projects` | проекты |
| GET | `/api/projects/{name}` | overview проекта |
| GET | `/api/finetune/status` | статус очереди: total/curated/ready/by_category |
| GET | `/api/finetune/examples` | список с quality_score для курации |
| PUT | `/api/finetune/examples/{id}` | редактировать (`{assistant?, boosted?}`) |
| DELETE | `/api/finetune/examples/{id}` | удалить пример |
| POST | `/api/finetune/examples/{id}/boost` | пометить важным |
| POST | `/api/finetune/correction` | записать correction |
| POST | `/api/finetune/start` | запустить полный пайплайн (SSE) |
| POST | `/api/finetune/compare` | сравнить две последние версии |
| POST | `/api/finetune/switch` | переключить версию (`{tag}`) |
| POST | `/api/finetune/rollback` | откат на предыдущую версию |
| GET | `/api/finetune/export` | выгрузить jsonl |
| GET | `/api/model/versions` | список всех версий модели |
| GET | `/api/status` | сводка |

## Файловая структура знаний

```
knowledge/
├── core_memory.md          # всегда в контексте
├── index.json              # индекс тем: keywords, paths, access_count
├── access_log.json         # счётчики обращений
├── finetune_queue.jsonl    # пары instruction→response с confidence
├── fundamentals/           # физика, химия, математика...
├── profession/             # доменная экспертиза
├── projects/
│   └── <project_name>/
│       ├── overview.md
│       ├── decisions.md
│       ├── issues.md
│       └── hardware.md
└── personal/
```

Формат заметки (frontmatter + тело):
```markdown
---
topic: RS-485
category: profession
created: 2026-04-07 14:30
updated: 2026-04-07 14:30
keywords: rs485, serial, modbus
source: https://...
confidence: verified
access_count: 15
---

# RS-485

## Что это
## Ключевые параметры
## Практические заметки
## Частые ошибки
## Связанные темы
- [[MAX485]]
```

## Тесты

```bash
pytest
```

Покрывают: knowledge manager CRUD, core memory, парсер команд, verifier, полный цикл агента (с замоканным LLM).

## Конфигурация (`config.yaml`)

```yaml
# Model A — Claude Sonnet (brain)
model_a:
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  api_key_env: "ANTHROPIC_API_KEY"
  tasks: [task_analysis, learning, complex_solving, verification, note_creation]

# Model B — Qwen 7B via Ollama (apprentice + fine-tune target)
model_b:
  provider: "ollama"
  model: "qwen2.5:7b-instruct"
  base_url: "http://localhost:11434"
  tasks: [simple_lookup, keyword_extraction, note_search, quick_answer, classification]

# Smart Router
router:
  fallback_to_local: true
  daily_api_budget_usd: 5.0
  estimated_cost_per_call_usd: 0.01
  auto_shift_after_finetune: true
  shift_schedule:
    v0: { model_a_pct: 90, model_b_pct: 10 }
    v1: { model_a_pct: 60, model_b_pct: 40 }
    v2: { model_a_pct: 40, model_b_pct: 60 }
    v3: { model_a_pct: 20, model_b_pct: 80 }
    v5: { model_a_pct: 10, model_b_pct: 90 }

verification:
  enabled: true
  min_confidence: 70
  always_use_model_a: true

finetune:
  base_model: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  inference_model: "qwen2.5:7b-instruct"
  output_prefix: "qwen-agent"   # → qwen-agent-v1, v2, ...
```

## Анти-галлюцинации — жёсткие правила

1. Агент не отвечает «из головы» по темам, где есть заметки.
2. Системный промпт solver'a: «отвечай ТОЛЬКО по заметкам».
3. Каждый шаг solver → verifier — не опционален.
4. `confidence < min_confidence` → в ответе ⚠️ префикс.
5. Противоречия заметкам явно помечаются в `verification.contradictions`.

## Fine-Tuning Pipeline

Три этапа превращения опыта в знание модели (см. `FINETUNE_PIPELINE.md`):

### Этап 1 — автосбор
Агент сохраняет Q&A в `finetune_queue.jsonl` после каждого верифицированного ответа при:
- `confidence ≥ 85%` (порог в `config.yaml → finetune.confidence_threshold`);
- есть `source_notes` (ответ основан на заметках);
- `verified=True` (нет противоречий).

Формат записи — OpenAI chat-style с метаданными:
```jsonl
{"id":"a1b2c3...","messages":[{"role":"system","content":"..."},{"role":"user","content":"Q"},{"role":"assistant","content":"A"}],"metadata":{"source_notes":["RS-485"],"confidence":94,"project":null,"timestamp":"...","verified":true,"category":"procedure","boosted":false}}
```

Категории (автодетект): `factual_qa`, `troubleshooting`, `procedure`, `decision`, `correction`, `other`.

### Этап 2 — курация
- `FinetuneDataCurator` ([backend/finetune_curator.py](backend/finetune_curator.py)) фильтрует по quality score (0..1): длина, confidence, sources, категория, boost.
- Дедуп по fuzz.token_set_ratio ≥ 80%.
- Boosting: `correction`/`troubleshooting` повторяются 2-3 раза в датасете.
- Ручная курация — вкладка **Fine-Tune** в UI или команда `finetune review` в CLI.

### Этап 3 — обучение
`FineTunePipeline` ([backend/finetune_pipeline.py](backend/finetune_pipeline.py)): prepare_dataset → train_with_unsloth (LoRA + GGUF) → register_with_ollama → регистрация новой версии `v1/v2/...` в `knowledge/model_versions.json`.

Требует: `pip install unsloth trl transformers datasets` + GPU (минимум RTX 3060 12GB для 7B-моделей) + Ollama CLI.

### Corrections — учимся на ошибках
Если пользователь поправил ответ, пара сохраняется с `category=correction`, `confidence=100`, `original_wrong_answer=...`. Это самые ценные примеры (priority highest).

### Команды CLI
```
finetune status             — счётчики + готовность
finetune review             — список с quality score
finetune start              — запустить полный пайплайн
finetune compare            — сравнить две последние версии
finetune switch <tag>       — переключить версию (v0/v1/...)
finetune rollback           — откат
finetune export             — экспорт jsonl
model versions              — список версий
learn this to model         — добавить последний Q&A в очередь
неправильно, правильно: ... — записать correction
```

### Версионирование и Auto-Evolution
`ModelVersionRegistry` ведёт реестр в `knowledge/model_versions.json`. `ModelEvaluator` сравнивает ответы старой и новой модели на тестовом наборе `knowledge/eval_set.json` (формат: `[{"question":"...","expected":"..."}]`) и даёт рекомендацию upgrade/rollback.

## Autonomic subsystem (Model X)

The agent includes an autonomic controller ("Model X") that runs in the
background alongside the cortex. It is modelled after the human autonomic
nervous system: reflexes (L0 rules), routing (L1 classifier, v1+),
diagnosis (L2 small LLM, v1+), and escalation to cortex (L3).

**Delivered (D-01 → D-03):**

_Immune levers (D-02, react to ongoing errors/load):_
- `FIRE_SERVER_HEALTH` — disk / memory / CPU threshold check (green).
- `FIRE_ERROR_TRIAGE` — classifies `error_log.jsonl` entries by severity (green).
- `FIRE_SELF_HEAL` — looks up an immune signature and returns its fix plan (green).
- `FIRE_SERVICE_REPAIR` — whitelist-gated `systemctl restart` with `max_attempts`, POSIX only (green, skipped on non-POSIX).

_Autonomic levers (D-03, scheduled self-maintenance):_
- `FIRE_INTEGRITY_HEARTBEAT` — every 5 min, read-only check of `knowledge/index.json` vs files (green).
- `FIRE_GOAL_PROPOSE` — hourly, wraps `GOALS.suggest_from_gaps(gaps.json)` (green).
- `FIRE_MEMORY_CONSOLIDATION` — daily, reviews recent sessions and routes facts to `identity/user.md`, `memory_facts.jsonl`, and `sessions.json` summary field (green, delegates to cortex).

_Self-knowledge levers (D-04):_
- `FIRE_CAPABILITY_SCAN` — every 6h, inventories `backend/tools/`, `backend/skills/`, `knowledge/channels.json`, and the host via psutil into `knowledge/self/` (green, python).
- `FIRE_SELF_STUDY` — daily, reads up to 3 priority-ordered `backend/**/*.py` modules per tick and writes one markdown note per module to `knowledge/self/modules/` via cortex (green, claude).

**Paths:**
- Kill switch: `knowledge/autonomic/ENABLED` — set content to `false` to disable.
- Logs: `knowledge/autonomic/lever_log.jsonl`, `tick_log.jsonl`, `pending_approvals.jsonl`.
- Immune DB: `knowledge/immune/signatures.jsonl` (seed) + `knowledge/immune/fixes/` (markdown recipes).
- Self-knowledge: `knowledge/self/modules/`, `knowledge/self/tools/`, `knowledge/self/skills/`, `knowledge/self/mcp_servers/`, `knowledge/self/server_inventory.md` (written by D-04 levers).
- Design doc: `docs/superpowers/specs/2026-04-16-model-x-autonomic-design.md` (section 11 — phased delivery D-01..D-06).
- Implementation plans: `docs/superpowers/plans/`.

**HTTP:** `GET /api/autonomic/status` reports kill-switch, scheduler liveness, and registered levers. Router lives in `backend/autonomic/api.py`; extended in D-03 and D-06.

**Env vars** (set before starting uvicorn to override defaults):
`AUTONOMIC_ENABLED_PATH`, `AUTONOMIC_TICK_SECONDS`, `AUTONOMIC_KNOWLEDGE_ROOT`,
`AUTONOMIC_ERROR_LOG_PATH`, `AUTONOMIC_LEVER_LOG_PATH`, `AUTONOMIC_PENDING_PATH`,
`AUTONOMIC_TICK_LOG_PATH`.

**Relationship to `backend/background.py`:** the autonomic subsystem is reflex/immune-driven (L0 rules + signatures). `background.py` is goal-driven (user-proactive `learn_topic` from chat or gap tracker). They coexist through D-03; D-04's `FIRE_SELF_STUDY` absorbs `learn_topic` and retires `background.py`.
