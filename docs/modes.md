# Deployment Modes

Главная настройка в `config.yaml` — `mode:`. Выбор режима задаёт sensible defaults для всех подсистем. Любой ключ можно переопределить явно через WebUI Settings → Engine, либо вручную в yaml.

| mode | GPU нужен | Ollama нужен | Где тренируется | Когда выбирать |
|---|---|---|---|---|
| **`local_full`** | ✅ ≥12GB VRAM | ✅ | локально (Unsloth LoRA) | Полный стек на одной машине с мощной видеокартой. |
| **`cloud_finetune`** | ❌ | ✅ | арендованная GPU (RunPod/Vast.ai/Colab Pro) | Нет своего GPU, но готов платить за аренду для тренировки. Inference локально. |
| **`local_cpu`** | ❌ | ✅ (CPU-mode) | локально на CPU (экспериментально) | Слабая машина без видеокарты. Использует **Qwen 1.5B** вместо 7B, тренировка возможна но очень медленная. |
| **`claude_only`** | ❌ | ❌ | **отключено** | Только Claude API, без локальных моделей. Заметки и finetune_queue собираются, но обучение недоступно. |

## Что делает каждый режим

### `local_full` — полный локальный стек
- `model_b`: `qwen2.5:7b-instruct` через Ollama
- `router.auto_shift_after_finetune: true` — доля Qwen растёт по v0→v5
- `finetune start` → Unsloth LoRA на локальной GPU → регистрация в Ollama

### `cloud_finetune` — сбор локально, тренировка в облаке
- Всё как в `local_full`, но `training_location: cloud`
- `finetune start` блокируется с подсказкой использовать `finetune export-cloud`
- `finetune export-cloud` генерирует папку `models/cloud_export_v{N}/` с:
  - `train.jsonl` / `val.jsonl` (после курации)
  - `train_script.py` (готов к запуску `python train_script.py` на арендованной GPU)
  - `config.json` с hyperparams
  - `README_CLOUD.md` с пошаговой инструкцией
- Заливаешь пакет на RunPod/Vast.ai/Colab Pro → запускаешь `train_script.py` → получаешь `unsloth.Q4_K_M.gguf`
- Возвращаешь файл локально → `finetune import-gguf <path> --tag v1` → Ollama регистрирует модель

### `local_cpu` — для машин без GPU
- `model_b`: `qwen2.5:1.5b-instruct` (помещается в ≤4GB RAM)
- `router.auto_shift_after_finetune: false` — не пытаться грузить 1.5B сложными запросами
- `training_location: local_cpu` — Unsloth запускается без CUDA (медленно, но работает)
- Агент покажет warning перед стартом тренировки: «10+ часов на 50 примерах»

### `claude_only` — чистый клиент Claude API
- `model_b: None` — Ollama не нужна, роутер помечает B как постоянно недоступную
- `router.fallback_to_local: false`, `auto_shift_after_finetune: false`
- `finetune.enabled: false` — `finetune start` и `export-cloud` блокируются
- Агентский цикл (analyze → solve → verify) работает полностью через Claude
- Заметки, core memory, проекты — всё локально, как обычно
- **Finetune queue копится** на случай если позже переключишься на другой mode

## Переключение режимов

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
| A-задача, Claude down + Ollama down | `LLMError` с явным сообщением «обе модели недоступны» |
| A-задача, over-budget + Ollama down | идёт на Claude с пометкой `"no B available"` в last_reason |
| B-задача (`simple_lookup` etc.), Ollama down | **эскалация на Claude** — «Ollama down, escalate A» |
| B-задача, обе down | `LLMError` |
| Runtime падение Claude | fallback на B только если Ollama реально отвечает |

**Главный invariant:** агентский цикл (`TASK_ANALYSIS → COMPLEX_SOLVING → VERIFICATION`) состоит только из A-задач. Пока ключ Claude валиден, агент работает полностью без Ollama.

**Чтобы полностью отключить Qwen-ветку** (например, нет GPU и никогда не будет):
```yaml
router:
  auto_shift_after_finetune: false
  fallback_to_local: false
verification:
  always_use_model_a: true
```

## State

`router_state.json` хранит per-day счётчики:
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
