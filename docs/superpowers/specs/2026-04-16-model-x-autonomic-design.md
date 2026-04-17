# Model X — Autonomic Controller Design (D)

**Статус:** Design (no implementation)
**Дата:** 2026-04-16
**Область:** Под-проект D из декомпозиции D → A → B → C.
**Автор ведёт:** Gor (пользователь) + агент.

---

## 0. Контекст и цель

Проект: self-learning agent в `c:\Users\samsung\AGI` — Python CLI + FastAPI backend + React frontend. Production-deployment планируется на Linux-сервер пользователя (CPU-only: Intel i7-8700 6C/12T @ 3.2GHz, 24GB DDR4 RAM, без GPU).

**Цель D:** Задать архитектуру автономного (фонового) слоя агента — «вегетативной нервной системы» Model X, которая работает без команд пользователя, поддерживает гомеостаз агента, диагностирует и чинит сбои, растёт в самосознании. D — это дизайн-документ, а не реализация. Реализация планируется отдельно через writing-plans.

**Метафора, задающая архитектуру:** центральная нервная система человека. Рефлексы (спинной мозг), автономка (продолговатый мозг), адаптивные реакции (средний мозг), сознание (кора). Model X — всё, кроме коры. Кортекс — это главный LLM-провайдер, отвечающий за сознательное мышление и общение с пользователем.

### Кортекс = главный LLM (provider-agnostic)

В этом документе «Claude» и «кортекс» используются как синонимы, но **кортекс — это роль**, а не конкретный провайдер. Реальная модель настраивается через `backend/providers.py` и `config.yaml`:

- **Anthropic Claude** (Sonnet/Opus/Haiku) — дефолт сейчас, режим `claude_only`
- **OpenAI** (GPT-4o, GPT-4o-mini и др.) — через тот же `providers.py`
- **Локальный Qwen 14B/32B** в Ollama — если пользователь хочет offline-режим и у сервера хватает RAM
- **Любой другой провайдер**, добавляемый в `providers.py`

Везде ниже, где написано «Claude» в контексте архитектуры (escalation, делегация, cortex) — читать как «текущий главный LLM». Стоимость и скорости в Секции 6 — для Claude API по дефолту; для других провайдеров цифры будут другие, но архитектура не меняется.

**Ограничения:**
- CPU-only inference (через Ollama)
- 24GB RAM — хватает на Qwen 2.5-Coder-7B Q4_K_M + embeddings + ОС с запасом
- Никакого облачного GPU для старта — только для опциональных LoRA-тюнов позже
- Безопасность: автономка не должна ломать систему необратимыми действиями

---

## 1. Концепция и границы ответственности

### Два мозга агента

| Слой | Роль | Когда активен | Что исполняет |
|---|---|---|---|
| **Model X (ЦНС)** | гомеостаз, рефлексы, диагностика, роутинг | фоново на tick'ах + реактивно на события | 19 рычагов (levers) |
| **Cortex (главный LLM)** | сознательное мышление, диалог с пользователем, сложные задачи | в хот-пути разговора и при escalation из Model X | существующий `backend/agent.py` pipeline, провайдер настраивается |

### Три уровня памяти (остаются внешними, НЕ в весах)

- **Долговременная эпизодическая:** `knowledge/` — ноты, граф, core memory
- **Рабочая / сессия:** `backend/conversation.py`, `backend/sessions.py`
- **Процедурная (иммунная):** новая `knowledge/immune/` — сигнатуры ошибок → рецепты починки

### Границы Model X

**✅ Делает:**
- Решает *когда* запустить фоновую задачу — какой рычаг, с какими параметрами
- Диагностирует аномалии (ошибки, баги, сбои сервисов)
- Применяет known fixes из green-list (рестарт сервиса, ротация логов, чистка tmp)
- Обновляет self-knowledge, когда код/capabilities меняются
- Эскалирует сложное на Claude или пользователя

**❌ НЕ делает:**
- Не отвечает пользователю в чате (это cortex)
- Не принимает решения вне рычагового контракта (нет free-form действий)
- Не делает irreversible операций без подтверждения (green-list only)
- Не меняет свою own policy — это через cortex + пользователь

### Инвариант безопасности

Model X всегда действует **через рычаги**. Нет способа исполнить произвольный shell-вызов. Каждый рычаг имеет статически описанные эффекты и классификацию safety (green/yellow/red).

---

## 2. Слоистая архитектура

Все пути сходятся в единую точку `Lever Executor` — у автономки нет другого способа что-либо сделать. Но *как* выбран рычаг — зависит от сценария.

```
┌─────────────────────────────────────────────────────────────┐
│  Scheduler (tick каждые N сек) + Event bus (reactive hooks) │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
          ┌────────────────────────────────────┐
          │  State Snapshot Builder            │
          │  metrics, job timings, errors,     │
          │  presence, KB stats                │
          └────────────────┬───────────────────┘
                           ▼
╔══════════════════════════════════════════════════════════════╗
║  Layer 0 — РЕФЛЕКСЫ (Python, no LLM)                         ║
║  • rules: "if X and Y → fire lever Z"                        ║
║  • known-signature repairs из immune DB (short-circuit)      ║
║  • выход: (lever, params) | need_routing | need_diagnosis    ║
╚═══════┬═══════════════════╦═══════════════════╦══════════════╝
        │ lever chosen      │ need_routing      │ need_diagnosis
        │                   │                   │  (anomaly / novel)
        │                   ▼                   │
        │         ╔═════════════════════╗       │
        │         ║  L1 РОУТЕР          ║       │
        │         ║ embedding + logreg  ║       │
        │         ║ state → lever       ║       │
        │         ╚══════┬══════════┬═══╝       │
        │ high confidence │  low    │           │
        │     ◀───────────┘  conf.  ▼           ▼
        │                       ╔═══════════════════════════╗
        │                       ║ L2 ДИАГНОСТ               ║
        │                       ║ Qwen 2.5-Coder-7B         ║
        │                       ║ tool-use: read_file,      ║
        │                       ║   systemctl, journalctl   ║
        │                       ║ выход: lever OR escalate  ║
        │                       ╚══════┬════════════════╦═══╝
        │                              │ lever          │ escalate
        │                              │                ▼
        │                              │       ╔════════════════╗
        │                              │       ║ L3 КОРА (main LLM)║
        │                              │       ║ novel case      ║
        │                              │       ║ → lever OR      ║
        │                              │       ║   yellow-approval║
        │                              │       ╚════════┬════════╝
        │                              │                │
        ▼                              ▼                ▼
        ┌──────────────────────────────────────────────────┐
        │   Lever Executor                                 │
        │   выполняет lever, возвращает LeverReport        │
        │   → lever_log.jsonl + event bus                  │
        └──────────────────────────────────────────────────┘
```

**Отдельная делегация на Claude (не escalation):**
Когда рычаг `FIRE_MEMORY_CONSOLIDATION` / `FIRE_NOTE_CURATION` / `FIRE_SELF_REFLECTION` исполняется — его executor сам зовёт Claude как библиотеку для выполнения работы. Это не прохождение через L3; это делегация внутри Lever Executor. Model X уже решила *что* делать; Claude решает *как*.

**Почему несколько путей, а не один конвейер:**
- Большинство тиков — простые расписания (INTEGRITY_HEARTBEAT каждые 30 сек). L0 решает сразу, мс-время. Идти через L1/L2 — пустая трата.
- Immune path сразу даёт пару (error_signature → fix_lever) без необходимости в классификаторе. L0 short-circuit'ит.
- L1 вводится только когда L0 не уверен (редкие состояния, новые комбинации). Дешёвый классификатор.
- L2 — только при аномалии или low-confidence L1. Дорогая диагностика, не нужна в хот-пути.
- L3 — последний резорт. Дорогой и медленный, идёт только при полной невозможности решить ниже.

### Делегация vs Escalation

- **Делегация на Claude** — *нормальный* путь рычага. Например, `FIRE_MEMORY_CONSOLIDATION` всегда ходит в Claude, это его тело. Model X решила *что* делать; Claude — *как*.
- **Escalation в Claude** — *исключительный* путь из Layer 2, когда диагност не смог решить сам (novel error, новый bug). Здесь Claude решает **и что, и как**.

### Tick'и и события

**Периодические tick'и:**
- **Fast** (30 сек) — integrity, metrics
- **Medium** (5 мин) — error triage, cost audit
- **Slow** (1 час) — note curation, gap detection
- **Nightly** — memory consolidation, self-reflection, capability scan

**Событийные hooks (event bus):**
- Новая ошибка в `error_log.jsonl` → немедленный tick immune
- Пользователь завершил сессию → tick post-session (consolidation)
- Service crash detected → tick repair
- Код изменён (mtime files backend/) → tick capability_scan + self_study

### v0 → v1 → v2 прогрессия

- **v0 (запуск):** Layer 0 c правилами + Layer 1 пустой + Layer 2 stub → Layer 3 Claude. Работает без единого LoRA-тюна.
- **v1 (через 2–4 недели):** L1 получает embedding-классификатор, обученный на реальных (state → lever) парах из v0. L2 переключается на stock Qwen 2.5-Coder-7B (без тюна).
- **v2 (через 2–3 месяца):** L2 получает LoRA-тюн на реальных диагнозах и их исходах.

**Ключевое:** v0 работает без LLM в автономке. L1/L2 нужны только для снятия нагрузки с Claude — это оптимизация, не обязательное условие функционирования.

---

## 3. Контракт рычагов (Lever interface)

```python
class Lever:
    name: str                    # "FIRE_MEMORY_CONSOLIDATION"
    category: str                # "autonomic" | "telemetry" | "immune" | "body"
    safety: str                  # "green" | "yellow" | "red"
    executor: str                # "python" | "claude" | "small_llm"
    estimated_cost: Cost         # {tokens, seconds, $usd}
    required_context: list[str]  # какие части state snapshot нужны

    def preconditions(state) -> bool:
    def run(params, context) -> LeverReport:
    def rollback(report) -> None:   # если операция reversible
```

### Классификация safety

- **green** — автономно, безопасно. Примеры: integrity check, note curation, memory consolidation, log rotate, restart сервиса из whitelist, cache clear
- **yellow** — требует подтверждения пользователя. Примеры: install/uninstall пакета, апгрейд сервиса, изменение конфига, запись в identity/soul/core_memory
- **red** — заблокировано по умолчанию, только по прямому запросу пользователя. Примеры: `rm -rf`, `systemctl mask`, изменение firewall, production deploy, push кода

**Инвариант:** Model X автономно исполняет **только green**. Yellow уходит в очередь `pending_approvals.jsonl`, пользователь видит в UI и одобряет/отклоняет. Red — вообще не генерируется автономкой.

### LeverReport

```python
@dataclass
class LeverReport:
    lever: str
    params: dict
    started_at: datetime
    finished_at: datetime
    status: Literal["success", "failure", "skipped", "escalated"]
    outcome: dict                # what changed
    tokens_used: TokenUsage | None
    reason: str                  # человекочитаемое объяснение
    follow_ups: list[str]        # рекомендации на следующий tick
```

Все репорты → `knowledge/autonomic/lever_log.jsonl`. Это **готовый датасет** для тюна v1.

### Каталог из 19 рычагов

**Автономные циклы (7):**
1. `FIRE_MEMORY_CONSOLIDATION` — claude-delegate, nightly
2. `FIRE_GRAPH_MAINTENANCE` — embeddings + rules, claude only для merge
3. `FIRE_INTEGRITY_HEARTBEAT` — pure python, fast tick
4. `FIRE_NOTE_CURATION` — claude-delegate, slow tick
5. `FIRE_SELF_REFLECTION` — claude-delegate, nightly
6. `FIRE_FINETUNE_QC` — claude-delegate
7. `FIRE_GAP_DETECTION` — claude-delegate

**Телеметрия и оценка (4):**
8. `FIRE_MODEL_EVAL` — качество ответов
9. `FIRE_SESSION_ARCHIVE` — старые сессии в `_history/`
10. `FIRE_COST_AUDIT` — token budget watch
11. `FIRE_GOAL_PROPOSE` — вызов существующего `goals.py`

**Иммунная система (4):**
12. `FIRE_ERROR_TRIAGE` — парс `error_log.jsonl`, severity
13. `FIRE_SELF_HEAL` — known fix по error signature
14. `FIRE_SERVER_HEALTH` — disk/mem/proc/network check
15. `FIRE_SERVICE_REPAIR` — systemctl restart, log rotate, tmp cleanup

**Тело и среда (3):**
16. `FIRE_TOOL_INSTALL` — apt/pip/ollama pull (yellow safety)
17. `FIRE_CAPABILITY_SCAN` — пересканирование tools/MCP/skills → обновление `core_memory.md`
18. `FIRE_SELF_STUDY` — чтение N модулей своего кода → обновление `knowledge/self/`

**Мета (1):**
19. `IDLE(sleep_seconds)`

### Примеры рычагов

**`FIRE_INTEGRITY_HEARTBEAT`:**
```
category: autonomic, safety: green, executor: python
preconditions: last_run > 30s ago
run:
  1. проверить knowledge/index.json против файлов
  2. найти дубли в user.md (hash по normalized тексту)
  3. найти orphan ноты (файл есть, в index нет)
  4. авто-fix: удалить дубли, перестроить index
outcome: {checked: N, fixed: M, orphans: [...]}
```

**`FIRE_SERVICE_REPAIR`:**
```
category: immune, safety: green (для whitelisted сервисов), executor: python
preconditions: immune DB signature matched И service ∈ whitelist И cooldown > 5m
run:
  1. systemctl status {service}
  2. journalctl --since=5m -u {service} → в report
  3. systemctl restart {service}
  4. через 30 сек проверить status снова
  5. если опять failed → status=escalated, уходит в L3
outcome: {service, restart_attempts, final_status, journal_tail}
```

---

## 4. Self-knowledge: что Model X знает о себе

### Уровень 1 — Тождество (всегда в system prompt, ~500 токенов)

- `soul.md` + `identity.md` + цели (subset из `goals.json`)
- Каталог рычагов с кратким описанием (name, category, safety, one-line)
- Текущий режим (`claude_only`, версии моделей)

### Уровень 2 — Карта архитектуры (RAG, retrieve по запросу)

Новая директория `knowledge/self/` с авто-генерируемыми нотами:

```
knowledge/self/
├── modules/                       # одна нота на backend/*.py
│   ├── agent.md
│   ├── llm.md
│   ├── knowledge_graph.md
│   └── ... (37 нот)
├── tools/                         # описание каждого tool
├── mcp_servers/                   # из mcp_client.py
├── skills/                        # из backend/skills/
├── server_inventory.md            # Linux: OS, packages, services, disks
├── capability_map.md              # сводка возможностей
└── architecture_overview.md       # якорный файл, линкует всё
```

**Генерация (рычаг `FIRE_SELF_STUDY`):** Model X по расписанию читает исходники и через Claude генерирует/обновляет ноты модулей. Ноты индексируются bge-m3 embeddings → retrieval через `hybrid_searcher.py`.

**Почему RAG, а не system prompt:** 37 модулей + tools + skills не влезут в prompt, и меняются.

### Уровень 3 — Исходники on-demand (tool call)

L2 диагност вызывает `read_file(path)` напрямую, когда RAG-ноты недостаточно.

### Иммунная база знаний — `knowledge/immune/`

```
knowledge/immune/
├── signatures.jsonl      # error patterns → fix recipes
├── fixes/                # markdown рецепты
│   ├── ollama_not_responding.md
│   ├── disk_full_tmp.md
│   └── ...
└── incident_log.jsonl    # история всех инцидентов и исходов
```

**Пример сигнатуры:**
```json
{
  "id": "ollama_timeout_v1",
  "pattern": {"source": "error_log", "msg_regex": "ollama.*timeout", "service": "ollama"},
  "severity": "warn",
  "fix_lever": "FIRE_SERVICE_REPAIR",
  "fix_params": {"service": "ollama", "max_attempts": 2},
  "observed_count": 0,
  "success_rate": null
}
```

**Адаптивный иммунный ответ:**
1. L0 ищет match в `signatures.jsonl`
2. Match + хорошая `success_rate` → применяет `fix_lever` автоматически
3. Novel incident → L2 → L3 (Claude), предлагает рецепт → пользователь подтверждает → сигнатура добавляется
4. После каждой попытки обновляются `observed_count` + `success_rate`

### Server inventory

Рычаг `FIRE_CAPABILITY_SCAN` при первом запуске собирает:
- `uname -a`, `/etc/os-release`
- `systemctl list-units --type=service --state=running`
- `dpkg -l` / `pip list` / `ollama list` / `docker ps`
- Mount points, disk usage, memory, CPU info
- Open ports (`ss -tlnp`)

→ `knowledge/self/server_inventory.md`. Обновляется раз в сутки или по событию.

---

## 5. Интеграция с существующим кодом

### Новые модули в `backend/`

```
backend/autonomic/                  # новый пакет — вся Model X здесь
├── __init__.py
├── scheduler.py                    # tick loop + event bus       (~150 строк)
├── state.py                        # StateSnapshot builder        (~200)
├── layer0_reflexes.py              # rule engine                  (~300)
├── layer1_router.py                # embedding classifier wrapper (~150)
├── layer2_diagnoser.py             # Qwen-Coder + tool-use        (~250)
├── layer3_escalation.py            # Claude call c self-RAG       (~200)
├── lever.py                        # Lever base + LeverReport     (~100)
├── safety.py                       # green/yellow/red + approvals (~150)
├── immune.py                       # signature matching           (~250)
└── levers/                         # 19 файлов, один рычаг = один файл
    ├── __init__.py                 # registry
    ├── memory_consolidation.py
    ├── graph_maintenance.py
    ├── integrity_heartbeat.py
    ├── note_curation.py
    ├── self_reflection.py
    ├── finetune_qc.py
    ├── gap_detection.py
    ├── model_eval.py
    ├── session_archive.py
    ├── cost_audit.py
    ├── goal_propose.py
    ├── error_triage.py
    ├── self_heal.py
    ├── server_health.py
    ├── service_repair.py
    ├── tool_install.py
    ├── capability_scan.py
    └── self_study.py
```

### Новые директории знаний

```
knowledge/autonomic/
├── lever_log.jsonl           # все LeverReports (датасет для тюна)
├── pending_approvals.jsonl   # yellow-действия, ждут пользователя
├── tick_log.jsonl            # каждый tick: что решили, какой слой
├── metrics.jsonl             # агрегаты
├── dashboard_state.json      # текущий срез для UI
└── ENABLED                   # kill switch (файл с "true" | "false")

knowledge/self/               # авто-генерируется FIRE_SELF_STUDY
knowledge/immune/             # иммунная БД (см. Секция 4)
```

### Существующие модули — что меняется

| Модуль | Изменение | Причина |
|---|---|---|
| `backend/background.py` | удалить или поглотить в `autonomic/scheduler.py` | дублирует tick loop |
| `backend/meta_learner.py` | executor для `FIRE_SELF_REFLECTION` | уже делает это, просто через рычаг |
| `backend/evaluator.py` | executor для `FIRE_MODEL_EVAL` | — |
| `backend/goals.py` | executor для `FIRE_GOAL_PROPOSE` | — |
| `backend/memory_extractor.py` | executor для `FIRE_MEMORY_CONSOLIDATION` | — |
| `backend/self_modifier.py` | вызывается из L3 для серьёзных bug'ов | red-safety, всегда требует пользователя |
| `backend/main.py` | startup hook: запуск scheduler в background task | — |
| `backend/mcp_client.py` | L2 использует для tool-use | — |
| `backend/agent.py` | **не меняется** — cortex работает как раньше | Model X отдельная вселенная |

### Контракт с cortex (главный LLM)

- **Делегация** (нормальная): рычаг вызывает cortex через `backend/llm.py` → `backend/providers.py` с конкретным промптом. Какой провайдер используется — решает конфиг, не рычаг. Это существующий интерфейс, ничего нового.
- **Escalation** (аварийная): L2 отдаёт `EscalationContext` в L3, тот зовёт cortex. Ответ — либо новый fix recipe (в immune DB), либо yellow-action в pending_approvals.

### Event bus (cortex → Model X)

- Cortex после обработки сообщения шлёт `conversation_ended(session_id)` → очередь `FIRE_MEMORY_CONSOLIDATION`
- Новая ошибка в логе → `error_occurred` → immune tick
- Пользователь одобрил yellow-action → `approval_granted` → рычаг исполняется

### Изоляция от хот-пути

- Model X **никогда не блокирует** ответ пользователю
- Scheduler в отдельном asyncio task (запускается из `main.py` FastAPI startup)
- Если Model X падает — cortex продолжает работать; в UI появляется badge «autonomic offline»

### Frontend

- Новая панель `AutonomicPanel.tsx` — tick stream, недавние LeverReports, pending approvals, статус immune
- `StatusBar.tsx` добавляет индикатор здоровья Model X
- Существующие панели не меняются

### Удаляется

- LightRAG MCP dependency (пункт A — будет встроено внутри в отдельном spec)
- `backend/background.py`, если окажется чистым дубликатом scheduler'а

---

## 6. Дорожная карта v0 → v1 → v2

### v0 — «ЦНС без коры автономки» (день 1–14)

**Работает:**
- Layer 0 полностью: все rule-based рычаги
- Layer 1 — stub (всегда `undecided` → L2)
- Layer 2 — stub (всё эскалирует в L3)
- Layer 3 — Claude через существующий `llm.py`
- Все 19 рычагов имеют реальные executor'ы
- Immune — 5–10 seed-сигнатур (ollama timeout, disk full, ping failure, …)
- Scheduler: fast/medium/slow/nightly ticks + events
- Safety gates активны

**Собираем:**
- `tick_log.jsonl` — (state_snapshot, layer0_decision, final_lever, outcome)
- `lever_log.jsonl` — все LeverReports
- `incident_log.jsonl` — ошибки, диагнозы Claude, успешные фиксы

**Метрики:**
- Доля тиков, решённых на L0 (цель ≥70%)
- Доля тиков, дошедших до Claude (цель ≤30%)
- Incidents без signature match (должно падать)
- Tick-to-action latency

**Критерии перехода к v1:** ≥500 lever reports, ≥50 incidents с исходами, 2 недели стабильной работы.

### v1 — «Автономка получает свой маленький мозг» (неделя 3–8)

**Layer 1 — embedding classifier:**
- Датасет: из v0 `tick_log` → пары `(state_snapshot_json, correct_lever)`
- Инструмент: bge-m3 embeddings (уже есть через Ollama) + scikit-learn LogisticRegression
- Обучение локальное, секунды
- Синтетика для редких классов: Claude генерит 500 пар

**Layer 2 — stock Qwen 2.5-Coder-7B Q4_K_M:**
- Без тюна, просто загрузка в Ollama
- Тест: hold-out набор из incident_log → проверка, что 7B справляется с типичной диагностикой

**Экономика:**
- До v1: ~50–200 tick'ов в cortex/день. При Claude API — $0.01–0.05 за tick → $0.5–10/день. При локальном Qwen 14B — CPU-время (~60–120 сек на tick), денег $0
- После v1: 80% ловит L1 локально. Cortex только для escalation + делегации тяжёлых рычагов. Claude API оценка: $0.3–2/день. Локальный cortex: меньше CPU-нагрузки

### v2 — «Автономка учится на реальных исходах» (месяц 3+)

- Переобучение L1 на **реальных** данных v1
- Опционально LoRA тюн Qwen-Coder-7B на реальных диагнозах (если stock показал потолок)
- Reward signal: (lever, outcome_success_rate) — модель учится «что сработало», не только «что сказал Claude»
- Возможно DPO по парам (lever_that_worked, lever_that_failed) из одного и того же state
- Immune DB уже зрелая

**Non-goals:**
- Не тренируем одну гигантскую модель «знает всё». Узкие классификаторы/LoRA.
- Не применяем RL в классическом смысле (только offline LoRA + опц. DPO)
- Не меняем провайдер cortex в рамках D. Смена главного LLM (Anthropic ↔ OpenAI ↔ локальный Qwen 14B) настраивается через существующий `providers.py` / `config.yaml` и ортогональна этому дизайну
- Не делаем online fine-tune в продакшене — всегда batch + локально/cloud

---

## 7. Тестирование, наблюдаемость, границы

### Тестирование

```
tests/autonomic/
├── test_levers/                 # один тест на рычаг
│   ├── test_integrity_heartbeat.py
│   ├── test_service_repair.py
│   └── ...
├── test_layers/
│   ├── test_layer0_rules.py     # table-driven: state → expected_decision
│   ├── test_layer1_router.py    # после тюна: accuracy ≥80% на held-out
│   ├── test_layer2_diagnoser.py # известные incidents → корректный fix
│   └── test_escalation_flow.py
├── test_safety/
│   ├── test_green_list.py       # green исполняются автономно
│   ├── test_yellow_approval.py  # yellow → pending_approvals, не исполняются
│   └── test_red_blocked.py      # red невозможно запустить автономно
├── test_immune/
│   ├── test_signature_match.py
│   └── test_new_incident_learning.py
└── test_integration/
    ├── test_scheduler_tick.py
    ├── test_full_cycle_dry_run.py
    └── test_cortex_isolation.py  # Model X падает → cortex работает
```

**Dry-run mode:** каждый рычаг поддерживает `dry_run=True` — делает всё кроме финального write/systemctl/apt. Возвращает «что бы сделал». Используется в CI и при первой активации на новом сервере.

### Наблюдаемость

UI-панель `AutonomicPanel.tsx`:
- **Pulse** (сердцебиение) — tick rate, лента последних 10 рычагов
- **Levers** — таблица 19 рычагов: last_run, success_rate, avg_duration
- **Pending approvals** (yellow) — одобрить/отклонить одним кликом
- **Immune board** — сигнатуры с success_rate, недавние incidents, «learn this»
- **Health** — L1/L2/L3 status, Ollama availability, CPU/RAM/disk

**Alerts (из L0):**
- Claude escalation rate > 50% за час → баннер «Model X учится»
- Tick rate падает → «autonomic degraded»
- L2 timeout rate > 10% → «diagnoser slow»

### Failure modes

| Сбой | Что происходит | Защита |
|---|---|---|
| Ollama не отвечает | L1/L2 недоступны | fallback на L0+L3, баннер в UI |
| L2 зацикливается на tool-use | долгий hang | timeout 90s → force-escalate в L3 |
| Рычаг постоянно падает (3+ раз) | infinite retry | circuit breaker: блокировка на час, алерт |
| Model X crash | scheduler умирает | watchdog в `main.py` перезапускает; 3 крэша за 5 мин → выключение, cortex продолжает |
| Disk full | риск сломать систему | приоритетный `FIRE_SELF_HEAL[disk_full]` при >90% |
| Рекурсивный tick (livelock) | — | max_depth=3 для event-chain; далее дроп с логом |

**Kill switch:** `knowledge/autonomic/ENABLED` = `false` → scheduler не запускается. Пользователь может выключить в UI или удалением файла.

### Метрики успеха D

- **v0 (день 14):** ≥500 рычагов исполнено, ≥95% без участия пользователя, 0 incidents где Model X что-то сломала
- **v1 (день 45):** L0 ловит ≥70% тиков, L1 accuracy ≥80% на hold-out, cortex-стоимость автономки ≤$3/день при Claude API (или CPU-bound при локальном cortex)
- **v2 (день 90):** L0+L1+L2 закрывают ≥90% тиков, immune DB ≥50 сигнатур с success_rate ≥0.7

### Границы D с A/B/C

| Под-проект | Проектируется в D? | Реализуется в D? |
|---|---|---|
| **D** (этот spec) | Архитектура Model X + рычаги + v0/v1/v2 + тесты | Только документ. Никакого кода. |
| **A** (LightRAG внутри) | Упомянут как поставщик embeddings/graph для L2 RAG и self-knowledge retrieval | Код — отдельный spec для A |
| **B** (багфиксы) | Не проектируется | Отдельный spec. Но после D дубли в user.md и orphan ноты ловит `FIRE_INTEGRITY_HEARTBEAT` автоматически |
| **C** (рефакторинг) | Не проектируется | Отдельный spec. D добавляет новый пакет, не трогает main.py/llm.py/agent.py (кроме startup hook) |

### Non-goals D

- MCP-интеграция — делается в A (через встроенный LightRAG-подобный модуль)
- Production deployment на Linux — отдельная операционная задача
- Synthetic datasets — генерируются в v1, когда есть реальные паттерны

---

## 8. Hardware baseline

**Сервер:** Intel i7-8700 6C/12T @ 3.2GHz, 24GB DDR4 2666MHz, CPU-only.

**Ожидаемые скорости CPU inference (llama.cpp Q4_K_M):**

| Модель | Размер | RAM | tok/s | Латентность 200 tok |
|---|---|---|---|---|
| Qwen 2.5 0.5B | ~0.4GB | ~0.5GB | 30–50 | ~4–7s |
| Qwen 2.5 1.5B | ~1GB | ~1.5GB | 15–25 | ~8–13s |
| Qwen 2.5-Coder-7B | ~4.5GB | ~5GB | 4–7 | ~30–50s |
| Qwen 2.5 14B | ~8.5GB | ~10GB | 1.5–3 | ~70–140s |

**Выбор:**
- **L2 primary:** Qwen 2.5-Coder-7B Q4_K_M — оптимально для чтения кода и диагностики
- **L2 альтернатива nightly:** опционально Qwen 2.5 14B для deep self-reflection (приемлемо за ночь)
- **L1:** embedding classifier (bge-m3 уже доступен через Ollama)
- **embeddings:** bge-m3 через Ollama (~60 мс на эмбеддинг)

**Запас RAM:** 24GB - (~5GB Coder-7B + ~1GB bge-m3 + ~4GB OS/Python) = ~14GB свободно. Место для 14B модели (опц.) или Postgres/Redis в будущем.

---

## 9. Открытые вопросы (не блокирующие D)

Эти вопросы можно решить в плане реализации D или позже — они не меняют архитектуру:

1. **Частота `FIRE_SELF_STUDY`** — ежедневно или по событию `code changed`? Предложение: event + weekly full re-scan.
2. **Формат `knowledge/self/modules/*.md`** — какой шаблон использовать (imports, classes, functions, deps, one-liner)? Решается в implementation plan.
3. **Оповещение пользователя о `pending_approvals`** — только UI badge или ещё notification? Решается при build'е AutonomicPanel.
4. **Где хранятся LoRA артефакты** — пока ship'им stock Coder-7B, артефакты появятся в v2. Детали на v2.
5. **Default service whitelist для `FIRE_SERVICE_REPAIR`** — какие сервисы можно авто-рестартовать без подтверждения. Предложение: ship'ится с пустым списком, заполняется пользователем в UI или через `knowledge/autonomic/service_whitelist.json`. Кандидаты по умолчанию: `ollama`, `agent-backend` (сам агент). Любой systemd-сервис добавляется только вручную.
6. **Confidence threshold для L1 → L2** — какой порог уверенности классификатора считается «low» и триггерит L2. Предложение: 0.7 по умолчанию, тюнится по метрикам.

---

## 10. Что идёт после D

Следующий шаг — writing-plans для детального implementation плана под v0 D. План должен разложить:

1. Создание пакета `backend/autonomic/` с пустыми модулями и Lever interface
2. Scheduler + StateSnapshotBuilder + event bus
3. Safety gate + LeverRegistry + LeverReport
4. Реализация L0 с правилами + 19 executor'ов рычагов
5. L3 escalation путь + seed сигнатуры immune DB
6. Startup hook в `main.py`, kill switch
7. Тесты (dry-run для каждого рычага, safety gates, cortex isolation)
8. Frontend AutonomicPanel + StatusBar индикатор
9. v0 запуск на dev-машине, сбор метрик за 2 недели

---

## 11. Phased delivery (D-01 → D-06)

Секция 10 — полное видение v0. Чтобы не скатиться в "всё сразу и плохо", D декомпозирован на шесть под-проектов. Каждый самодостаточен, tests-first, сливается в master линейно. Видение (19 рычагов, L0→L3, UI-панель) остаётся полным; порядок поставки — ниже.

| Pid | Scope | Status |
|---|---|---|
| **D-01** | Foundation: scheduler, state snapshot, kill switch, safety gate, lever base, event bus, registry, 2 toy levers, FastAPI lifespan wire | ✅ merged (13 commits) |
| **D-02** | Layer 0 reflex engine + immune DB + 4 immune levers (`SERVER_HEALTH`, `ERROR_TRIAGE`, `SELF_HEAL`, `SERVICE_REPAIR`) + real tick + integration tests | ✅ d-02-layer0-immune (15 commits, 110 tests) |
| **D-03** | Claude-delegation executor + **3** autonomic levers: `FIRE_INTEGRITY_HEARTBEAT` (python), `FIRE_MEMORY_CONSOLIDATION` (claude), `FIRE_GOAL_PROPOSE` (wraps `goals.py`) | ⏳ next |
| **D-04** | `FIRE_SELF_STUDY` + `FIRE_CAPABILITY_SCAN` + `FIRE_NOTE_CURATION` + `FIRE_GRAPH_MAINTENANCE`; absorb `backend/background.py` into `FIRE_SELF_STUDY` and retire it | planned |
| **D-05** | Telemetry cohort: `FIRE_MODEL_EVAL`, `FIRE_SESSION_ARCHIVE`, `FIRE_COST_AUDIT` + remaining autonomic (`FIRE_SELF_REFLECTION`, `FIRE_FINETUNE_QC`, `FIRE_GAP_DETECTION`) | planned |
| **D-06** | Body cohort (`FIRE_TOOL_INSTALL` yellow, capability scan extras) + Frontend `AutonomicPanel.tsx` + StatusBar indicator + `backend/autonomic/api.py` expansion | planned |

**Why only 3 levers in D-03 (vs 7 "autonomic cycles" in Section 3):** to keep each sub-project reviewable inside a single plan-and-execute cycle. The remaining 4 autonomic-category levers are split across D-04 and D-05 by dependency — e.g. `SELF_STUDY` needs `knowledge/self/` infrastructure built first, `SELF_REFLECTION` benefits from telemetry signals from `MODEL_EVAL`, etc. The full catalog in Section 3 remains the target; only the order of arrival is staged.

**backend/background.py:** Kept as-is through D-03. Its sole job is user-proactive `learn_topic` triggered from chat flow or the gap tracker. This overlaps conceptually with `FIRE_SELF_STUDY` but not operationally until that lever lands. D-04 absorbs the `learn_topic` path into `FIRE_SELF_STUDY` and deletes `backend/background.py`. Header comment in `backend/background.py` records this intent.

**HTTP surface evolution:**
- D-02 (done): `/api/autonomic/status` — read-only kill-switch + scheduler liveness + registered lever names. Lives in `backend/autonomic/api.py`.
- D-03: add `/api/autonomic/ticks` (recent tick log tail), `/api/autonomic/levers/{name}` (last report).
- D-06: full panel API — pending approvals list, approve/reject, immune signatures, kill-switch toggle.

**main.py refactor policy (lightweight):** routers live next to their domain module (FastAPI-idiomatic). E.g. `backend/background.py` owns `router = APIRouter(prefix="/api/background")`; `backend/autonomic/api.py` owns its own. `backend/main.py` includes them. Big-bang rewrite of main.py is NOT a D-goal; extractions happen opportunistically when a D-NN touches a surface.

А/B/C получают свои spec'ы отдельно, в порядке: A → B → C.
