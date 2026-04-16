# AGI Agent — Roadmap

> Мастер-документ проекта. Показывает декомпозицию работ, статус каждой части, ссылки на spec и implementation plan'ы.

**Владелец:** Gor (`gjaloyan@glgroup.am`)
**Production target:** Linux server (Intel i7-8700 6C/12T, 24GB RAM, CPU-only)
**Dev:** Windows 11, `c:\Users\samsung\AGI`

---

## Контекст проекта

Self-learning AI agent — Python CLI + FastAPI backend + React frontend. Режим `claude_only` (главный LLM provider-agnostic через `backend/providers.py`).

**Архитектурный принцип:** знания вне весов. Модель маленькая и эффективная, ноты/граф/факты лежат на диске в markdown, подгружаются в контекст по релевантности.

**Слоистая когнитивная архитектура (задаётся под-проектом D):**

```
┌──────────────────────────────────────────────────────┐
│  CORTEX (главный LLM, provider-agnostic)             │
│  сознательное мышление, диалог, сложные задачи       │
└──────────────────────────┬───────────────────────────┘
                           │ escalation / delegation
                           ▼
┌──────────────────────────────────────────────────────┐
│  MODEL X — автономная нервная система                │
│  L0 рефлексы · L1 роутер · L2 диагност · L3 escalate │
│  управление телом агента через 19 рычагов            │
└──────────────────────────┬───────────────────────────┘
                           │ read / write
                           ▼
┌──────────────────────────────────────────────────────┐
│  KNOWLEDGE CORE (LightRAG-подобный, встроенный)      │
│  entity-relation граф + hybrid retrieval + embeds    │
└──────────────────────────────────────────────────────┘
```

---

## Декомпозиция работ

Проект разбит на четыре последовательных под-проекта:

### D — Model X (автономный контроллер) ⬅ сейчас тут

**Статус:** Design готов, implementation по плану D-01 в процессе.

**Что даёт:** агент получает «вегетативную нервную систему» — фоновый слой, который держит БЗ в порядке, реагирует на ошибки, сам изучает себя, эскалирует сложное на кортекс.

- 📄 **Spec:** [specs/2026-04-16-model-x-autonomic-design.md](specs/2026-04-16-model-x-autonomic-design.md)
- 📋 **Plans (5 штук, последовательно):**
  - D-01 Foundation — scheduler, lever interface, safety, registry. ⬅ первый, генерируется сейчас
  - D-02 Layer 0 + autonomic levers (7 штук)
  - D-03 Immune system + signatures + 4 immune levers
  - D-04 Self-knowledge + telemetry/body levers (7 штук)
  - D-05 Layer 3 escalation + AutonomicPanel UI

### A — Встроенный LightRAG-подобный knowledge core

**Статус:** не начато.

**Что даёт:** замена внешней зависимости `mcp__lightrag__*` на встроенный модуль. Entity/relation extraction, dual-level retrieval (local chunks + global graph), локальные embeddings через Ollama.

Почему после D: D создаёт autonomic слой, который будет использовать knowledge core для RAG в self-study. Если сначала сделать A, потом переписывать под потребности D — двойная работа.

- 📄 Spec: TBD
- 📋 Plans: TBD

### B — Багфиксы и очистка

**Статус:** не начато.

**Что даёт:** прогон по 37 модулям, вычистка дублей в `user.md`, мёртвого кода, нелогичных ветвей, TODO/XXX.

Почему после A: D внедрит integrity_heartbeat, который уже начнёт ловить часть проблем автоматически. B догребает остальное вручную.

- 📄 Spec: TBD
- 📋 Plans: TBD

### C — Рефакторинг раздутых файлов

**Статус:** не начато.

**Что даёт:** разбиение `main.py` (1669 строк), `llm.py` (1267), `agent.py` (1152), `providers.py` (767) по ответственностям.

Почему последним: рефакторить имеет смысл после того как новая функциональность устаканилась — иначе переделываем одно и то же дважды.

- 📄 Spec: TBD
- 📋 Plans: TBD

---

## Версионирование

| Версия | Что работает | Сроки |
|---|---|---|
| **v0** | D-01..05 foundation; Model X работает без LoRA-тюна, вся автономка идёт на rules + cortex escalation | 2–4 недели |
| **v1** | +L1 embedding classifier; +L2 stock Qwen 2.5-Coder-7B; собран датасет из v0 | +2–4 недели |
| **v2** | +LoRA тюн L1/L2 на реальных данных; адаптивная immune DB с историей success_rate | +2–3 месяца |

---

## Статус чеклист (обновляется при каждом изменении)

- [x] Декомпозиция на D/A/B/C согласована
- [x] Аппаратная база зафиксирована (i7-8700, 24GB, CPU-only)
- [x] Принцип provider-agnostic cortex зафиксирован
- [x] D spec написан и утверждён
- [ ] git init (перед стартом D-01 implementation)
- [ ] D-01 plan написан
- [ ] D-01 implementation
- [ ] D-02 plan
- [ ] D-02 implementation
- [ ] D-03 plan
- [ ] D-03 implementation
- [ ] D-04 plan
- [ ] D-04 implementation
- [ ] D-05 plan
- [ ] D-05 implementation
- [ ] v0 smoke test на dev Windows
- [ ] Deployment на Linux production
- [ ] v0 сбор метрик за 2 недели
- [ ] A spec + plans + implementation
- [ ] B spec + plans + implementation
- [ ] C spec + plans + implementation

---

## Принципы работы

Эти решения зафиксированы и применяются по умолчанию во всех под-проектах:

1. **TDD.** Каждая фича пишется через failing test → minimal code → pass → commit.
2. **English-only код.** Исходники, prompts, тесты — на английском. Документация и пользовательский UI — на русском.
3. **Provider-agnostic cortex.** Главный LLM не зашит на Claude, настраивается через `providers.py`.
4. **Intent separation.** chat / preference / task — разные пайплайны. Только task требует deep reasoning.
5. **Знания вне весов.** Markdown + graph + embeddings. Fine-tune только в cloud, опционально.
6. **Safety gates.** Любое автономное действие классифицируется green/yellow/red. Red не исполняется без прямого запроса.
7. **Kill switch.** Любой автономный слой может быть остановлен одним файлом (`knowledge/autonomic/ENABLED=false`).

---

## Структура документации

```
docs/superpowers/
├── ROADMAP.md                                         # этот файл
├── specs/                                             # дизайн-документы
│   └── 2026-04-16-model-x-autonomic-design.md        # D spec
└── plans/                                             # implementation планы
    └── 2026-04-16-d-01-foundation.md                 # D-01 plan
```

Spec описывает *что* и *почему*. Plan описывает *как* — конкретные файлы, шаги, код, тесты.
