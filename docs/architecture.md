# Архитектура self-learning агента

Документ описывает, как агент устроен от и до. Все диаграммы —
[Mermaid](https://mermaid.js.org/), они рендерятся прямо в GitHub
без плагинов. Локально можно вставлять блоки в [Mermaid Live Editor](https://mermaid.live).

> **Источник истины — код.** Если документ и код расходятся — прав код.
> Документ — это карта, не контракт. Когда меняешь структуру (новая ветка
> в pipeline, новое хранилище памяти, новая точка вызова LLM) — обнови
> и здесь.

---

## 1. Обзор

Агент — однопроцессный Python-сервис, который принимает сообщение
пользователя и возвращает **проверенный** ответ, попутно учась
с каждого хода. Доступ:

- **WebUI** (React + FastAPI SSE-чат).
- **Telegram-бот** (long-poll, real-time стрим прогресса).
- **REST API** под `/api/*` — статус, аттачменты, knowledge,
  цели, сессии, провайдеры, и т.д.

Рассуждение живёт в **dual-model router** (Claude как модель A,
Qwen / Ollama как модель B). Память слоистая: **CORE** (всегда
в контексте), **knowledge graph + notes** (семантика + структура),
**conversation** (короткое sliding-window), **attachments**
(sha-deduped картинки / голос / файлы).

```mermaid
flowchart LR
    subgraph IO["Ввод/вывод"]
        WUI["WebUI<br/>(React + Vite)"]
        TG["Telegram-бот<br/>(long-poll)"]
        API["REST + SSE<br/>(FastAPI)"]
    end

    subgraph CORE["Ядро рассуждения"]
        AG["Agent.run()"]
        LR["DualModelRouter"]
        VRF["Verifier"]
        SC["Self-critic loop"]
    end

    subgraph MEM["Слои памяти"]
        CM["CORE memory"]
        ID["Identity<br/>(soul / identity / user.md)"]
        KB["Notes (KB)"]
        KG["Knowledge graph"]
        VS["Vector store"]
        CONV["Conversation"]
        MEX["Memory facts<br/>(graph + extractor)"]
    end

    subgraph LEARN["Циклы обратной связи"]
        ML["Meta-learner"]
        GOALS["Goals"]
        EVAL["Evaluator"]
        FT["Fine-tune queue"]
    end

    WUI --> API
    TG --> API
    API --> AG
    AG --> LR
    AG --> VRF
    VRF --> SC
    AG --> MEM
    AG --> LEARN
    LR --> CORE
```

---

## 2. Жизненный цикл запроса (`Agent.run`)

Любое сообщение — из WebUI или Telegram — проходит через один и тот же
pipeline. После классификации — три ветки: `chat`, `preference`, `task`.
Только ветка `task` идёт в полный цикл think → solve → verify.

```mermaid
flowchart TD
    START(["Пришло сообщение<br/>(текст + опц. вложения)"])
    CORE_LOAD["_load_core()<br/>читает knowledge/core_memory.md"]
    CLASSIFY["_classify_intent()"]

    START --> CORE_LOAD --> CLASSIFY

    CLASSIFY -->|"арифметика regex"| TASK_BRANCH
    CLASSIFY -->|"chitchat regex"| CHAT_BRANCH
    CLASSIFY -->|"LLM: chat"| CHAT_BRANCH
    CLASSIFY -->|"LLM: preference"| PREF_BRANCH
    CLASSIFY -->|"LLM: task / по умолчанию"| TASK_BRANCH

    subgraph CHAT["Ветка 1 — chat"]
        CHAT_BRANCH["_chat_reply()<br/>QUICK_ANSWER, без инструментов"]
        CHAT_REPLY[["one-shot ответ"]]
        CHAT_BRANCH --> CHAT_REPLY
    end

    subgraph PREF["Ветка 2 — preference"]
        PREF_BRANCH["_save_preference()<br/>извлечь → user.md или reject"]
        PREF_REPLY[["короткий ack"]]
        PREF_BRANCH --> PREF_REPLY
    end

    subgraph TASK["Ветка 3 — полная задача"]
        THINK["_think()<br/>TASK_ANALYSIS<br/>question_type, tools, plan, topics"]
        SELF_AN{"self_analysis?"}
        ENSURE_KB["_ensure_knowledge()<br/>HYBRID.find_best по теме"]
        SKIP_KB["notes = []<br/>принудительно read_file через tools"]
        SOLVE["_solve()<br/>COMPLEX_SOLVING + tool loop"]
        VERIFY["_verify()<br/>VERIFICATION<br/>+ deterministic detector"]
        CRITIC{"confidence < 50%?<br/>(critic_threshold)"}
        RETRY["вшить critique →<br/>re-solve (макс. 2 retry)"]

        THINK --> SELF_AN
        SELF_AN -->|"да"| SKIP_KB --> SOLVE
        SELF_AN -->|"нет"| ENSURE_KB --> SOLVE
        SOLVE --> VERIFY --> CRITIC
        CRITIC -->|"да"| RETRY --> VERIFY
        CRITIC -->|"нет"| TASK_REPLY[["ответ"]]
    end

    CHAT_REPLY --> POST
    PREF_REPLY --> POST
    TASK_REPLY --> POST

    subgraph POST_PROC["Пост-обработка"]
        POST["CONVERSATION.add_turn"]
        EXTRACT["MEMORY.extract_and_store<br/>(факты → граф)"]
        EVAL_LOG["EVALUATOR.log + finetune queue<br/>(если confidence ≥ 85)"]
        TICK["GOALS.tick_interaction<br/>+ проверка proactive learning"]
        CLEAN["_cleanup"]
        POST --> EXTRACT --> EVAL_LOG --> TICK --> CLEAN
    end

    POST --> RESPONSE(["AgentAnswer"])
```

### Решение по веткам

| Ветка | Триггер | Что выполняется | Confidence |
|---|---|---|---|
| `chat` | regex chitchat ИЛИ классификатор сказал «chat» | `_chat_reply` (1 LLM-вызов, без tools) | всегда 100 |
| `preference` | классификатор сказал «preference» | `_save_preference` (extractor + запись в `user.md`) | всегда 100 |
| `task` | regex арифметики ИЛИ классификатор сказал «task» ИЛИ default | `_think → _ensure_knowledge → _solve → _verify → critic loop` | вычисляется |

---

## 3. Identity preamble

Каждый chat / think / solve LLM-вызов получает одинаковую identity-преамбулу
в начале system-промпта. Порядок важен: `LANGUAGE OVERRIDE` и
`AGENT NAME OVERRIDE` идут **последними**, чтобы перебить любое
конфликтующее правило из `# SOUL`.

```mermaid
flowchart TB
    subgraph IDENTITY["IdentityManager.preamble()"]
        S1["# SOUL<br/>knowledge/identity/soul.md<br/>(тон, характер)"]
        S2["# IDENTITY<br/>knowledge/identity/identity.md<br/>(роль, возможности, имя)"]
        S3["# USER PROFILE<br/>knowledge/identity/user.md<br/>(факты о пользователе)"]
        S4["# AGENT NAME OVERRIDE<br/>(извлечено из identity.md ## Имя)"]
        S5["# LANGUAGE OVERRIDE<br/>(извлечено из user.md ## Язык общения)"]
        S1 --> S2 --> S3 --> S4 --> S5
    end
```

`LANGUAGE OVERRIDE` и `AGENT NAME OVERRIDE` рендерятся только если
их секции-источники непустые. Soul-уровневые правила («повторяй язык
пользователя») становятся **значениями по умолчанию** — overrides
выигрывают конфликт, потому что лежат в конце preamble, где вес
attention максимален.

---

## 4. Per-turn user message context

`_shared_context(task, core)` собирает блок per-turn, который ложится
ПОД identity в user-сообщение. Стабильные факты сверху, эфемерные снизу:

```
# CORE MEMORY        ← постоянные факты долгой памяти
# CURRENT PROJECT    ← активный workspace, если есть
# GOALS              ← топ N активных целей
# SHORT-TERM MEMORY  ← семантический recall под текущий запрос
                       (ЗАМЕНЯЕТСЯ на # RECENT COMMITS на self_analysis)
# RECENT TURNS       ← последние N реплик
```

**Перестроение для self-analysis.** Когда `thinking.question_type == "self_analysis"`:

- `MEMORY.recall_block` **выкидывается** — это снапшот старых
  наблюдений, он воспроизводит уже-исправленные находки.
- Вместо него добавляется `git log --oneline -50` в виде
  `# RECENT COMMITS` — агент видит, как код выглядит **сейчас**.
- `_ensure_knowledge` **полностью обходится** — KB-заметки про код
  тоже снапшоты; единственный авторитет — файл на диске через
  `read_file`.

Это лечит паттерн «агент находит баги, которые уже починены
последним коммитом».

---

## 5. Иерархия памяти

У агента пять разных хранилищ, у каждого своя стратегия извлечения
и время жизни:

```mermaid
flowchart LR
    subgraph STABLE["Стабильное / курируемое"]
        CORE["CORE memory<br/>core_memory.md<br/>~всегда в контексте"]
        ID["Identity<br/>soul/identity/user.md<br/>~всегда в контексте"]
        NOTES["Notes (KB)<br/>knowledge/profession/*.md<br/>загружаются по match'у темы"]
    end
    subgraph SEMI["Авто-извлекаемое"]
        KG["Knowledge graph<br/>graph.json<br/>сущности + отношения"]
        FACTS["Memory facts<br/>memory_facts.jsonl<br/>извлечены из чатов"]
    end
    subgraph EPH["Эфемерное"]
        CONV["Conversation<br/>conversation.json<br/>последние 20 реплик"]
        ATT["Attachments<br/>knowledge/attachments/<br/>(sha256-deduped)"]
    end

    CORE --> AGENT
    ID --> AGENT
    NOTES -->|"HYBRID.find_best<br/>(min_raw_score=0.4)"| AGENT
    KG -->|"BFS прыжки + затухание"| HYBRID
    FACTS -->|"semantic recall<br/>через эмбеддинги"| AGENT
    CONV --> AGENT
    ATT --> AGENT

    AGENT["Per-turn контекст агента"]
    HYBRID["HybridSearcher"]
```

**Hybrid search** комбинирует три сигнала при поиске темы:
fuzzy keyword (rapidfuzz, threshold 0.6), graph traversal (BFS с
`graph_score_floor=0.10`), vector cosine (embedder, `vector_score_floor=0.30`).
Когда какой-то сигнал недоступен, веса перенормализуются по
оставшимся. Ниже сырого порога per-signal → шум → отбрасывается.

---

## 6. LLM router

`DualModelRouter` выбирает модель A (Claude / cloud) или B
(Qwen / local) per `TaskType`. Дефолты можно сместить расписанием
или дневным бюджетом; `VERIFICATION` всегда жёстко на A.

```mermaid
flowchart TD
    CALL["router.call(task_type, system, user)"]
    OVERRIDE{"Пользователь<br/>зафиксировал модель?"}
    USE_PINNED["использовать зафиксированную<br/>(без логики роутера)"]

    PICK{"Выбор A или B"}
    A_TASKS["MODEL_A_TASKS<br/>TASK_ANALYSIS, LEARNING,<br/>COMPLEX_SOLVING, VERIFICATION,<br/>NOTE_CREATION → A"]
    B_TASKS["MODEL_B_TASKS<br/>SIMPLE_LOOKUP, KEYWORD,<br/>NOTE_SEARCH, QUICK_ANSWER,<br/>CLASSIFICATION → B"]

    SHIFT{"shift_schedule<br/>pct → B?"}
    BUDGET{"дневной бюджет A<br/>исчерпан?"}
    HEALTH{"target доступен?"}
    FALLBACK["fallback на другую сторону<br/>(если доступна)"]

    CALL --> OVERRIDE
    OVERRIDE -->|"да"| USE_PINNED --> END(["LLM ответ"])
    OVERRIDE -->|"нет"| PICK
    PICK --> A_TASKS
    PICK --> B_TASKS
    A_TASKS --> SHIFT
    SHIFT -->|"да + B жив"| BUDGET
    SHIFT -->|"нет"| BUDGET
    BUDGET -->|"да + B жив"| FALLBACK
    BUDGET -->|"нет"| HEALTH
    HEALTH -->|"да"| END
    HEALTH -->|"нет"| FALLBACK --> END
```

**Provider-agnostic.** A и B настраиваются через
`backend/providers.py`. A может быть Anthropic, Codex (OpenAI ChatGPT
subscription), Bedrock, Cohere, Copilot, или любой
OpenAI-совместимый endpoint. B по умолчанию — `llama-server` /
Ollama на `100.124.210.21:8015`-ish.

---

## 7. Tool loop

`COMPLEX_SOLVING` идёт через `complete_with_tools`. Модель эмитит
`text + tool_use` блоки; агент исполняет каждый инструмент и
возвращает результаты, пока модель не выдаст текст без новых
tool-вызовов — либо пока не упрёмся в `max_iterations=6`.

```mermaid
sequenceDiagram
    participant S as Solver
    participant LLM as LLM (A или B)
    participant T as ToolRegistry

    S->>LLM: system + user + tools
    loop пока не end_turn или max_iterations
        LLM-->>S: text? + tool_use?
        alt есть tool_use
            S->>T: execute(name, args)
            T-->>S: result_text
            Note over S: захватываем tool_call detail<br/>для thinking_trace
            S->>LLM: assistant block + tool_results
        else end_turn
            Note over S: вернуть final_text
        end
    end
    Note over S: упёрлись в cap → форс<br/>tool-less synthesis call<br/>(synth_max = 6000)
    S-->>LLM: те же messages, без tools
    LLM-->>S: синтезированный ответ
```

**Зачем форс-синтез на max-iterations.** Anthropic-модели эмитят
`text + tool_use` в одном ходу — этот текст это «преамбула»
(«сейчас я проверю источник»), не финальный ответ. Если вернуть
последнюю преамбулу — пользователь увидит обещание-действия,
а не ответ. Forced synthesis сбрасывает `tools` и просит реальный
ответ.

**Правило захвата `final_text`.** Только когда в ответе **нет**
tool_use блоков, иначе мы записали бы преамбулу как ответ.

---

## 8. Verifier + self-critic loop

Верификатор пингует LLM (всегда модель A) с ответом + заметками +
tool outputs и просит разнести каждый claim по корзинам. Confidence
вычисляется **детерминистически в Python** из счётчиков корзин:

```
confidence = 100 × verified / (verified + unverified + 2 × contradictions)
```

Contradictions весят 2× потому что это evidence *против*, а не
просто отсутствие доказательств. Дальше результат проходит через
deterministic false-absence detector — regex-проход, который ловит
«add X» / «missing X», когда X есть в `EXTRACTED IDENTIFIERS`,
извлечённых из tool output. Любое попадание — promote в
contradiction. Это belt-and-suspenders для LLM-правила, потому что
Sonnet наблюдался как пропускающий это даже при явных инструкциях.

```mermaid
flowchart TD
    ANS["ответ solver'а + tool_context"]
    EXT["_extract_code_identifiers(tool_context)"]
    LLM["VERIFIER_SYSTEM<br/>+ список EXTRACTED IDENTIFIERS<br/>verifier LLM вызов (A)"]
    LLM_OUT["{verified[], unverified[], contradictions[]}"]
    DET["detect_false_absence_contradictions(<br/>answer, identifiers)"]
    MERGE["слить auto-detected contradictions<br/>(дедуп против LLM)"]
    CONF["вычислить confidence<br/>(детерминистически в Python)"]
    CHK{"confidence < 50?"}
    OK[["вернуть результат"]]
    RETRY["вшить CRITIQUE-блок:<br/>contradictions + unverified +<br/>прошлый ответ<br/>→ _solve снова"]

    ANS --> EXT --> LLM --> LLM_OUT --> MERGE
    ANS --> DET --> MERGE
    MERGE --> CONF --> CHK
    CHK -->|"≥50"| OK
    CHK -->|"<50, retry < 2,<br/>не застряли на 0"| RETRY --> ANS
    CHK -->|"<50, retry == 2"| OK
```

**Условия остановки:**

- `confidence ≥ critic_threshold` (50%) → готово.
- `retry == max_retries` (2) → отдаём что есть.
- Confidence застрял на 0% две итерации подряд → break (структурная
  проблема, retry не поможет).
- `LLMError` посреди retry → break, отдаём текущее лучшее.

**Skip path.** Для `question_type ∈ {"creative", "meta", "self_analysis"}`
*без* tool_context верификация полностью пропускается (нечего
проверять — верификатор просто наплодил бы unverified-шум).

---

## 9. Извлечение идентификаторов (helper для verifier)

И детектор, и LLM-верификатор оба опираются на список идентификаторов,
реально присутствующих в коде на этом ходу. Pre-extraction превращает
проверку «уже в коде» из «прочитай 12k символов» в keyword-match.

```mermaid
flowchart LR
    TC["tool_context (дампы файлов)"]
    P1["regex: class X / def Y"]
    P2["regex: SCREAMING_CONST = ..."]
    P3["regex: self.attr = ..."]
    SET["sorted, deduped, capped at 200"]
    TC --> P1 & P2 & P3 --> SET
```

Дальше детектор нормализует и кандидата (из ответа), и идентификатор
(из извлечения) через `s.replace("_", "").lower()` — `FILE_CACHE` ↔
`_file_cache` ↔ `fileCache` схлопываются в один ключ.

---

## 10. Циклы обратной связи

Два цикла крутятся в фоне и формируют будущее поведение:

```mermaid
flowchart TD
    subgraph TURN["Конец хода"]
        VR["VerificationResult"]
        EXTR["MEMORY.extract_and_store<br/>(факты от LLM → граф)"]
        EVAL["EVALUATOR.log<br/>+ finetune_queue если conf ≥ 85"]
        FAIL{"confidence < 60?"}
        ANL["META_LEARNER.analyze_failure<br/>→ root_cause + fix_action<br/>→ цель в goals.json"]
        TICK["GOALS.tick_interaction()"]
    end

    VR --> EXTR
    VR --> EVAL
    VR --> FAIL
    FAIL -->|"да"| ANL
    ANL -->|"каждый Nth (5)"| EXTRACT["META_LEARNER.extract_patterns()<br/>повторяющиеся паттерны → high-priority цель"]
    TICK --> PROACTIVE{"каждые 10<br/>взаимодействий?"}
    PROACTIVE -->|"да"| GAPS["GOALS.suggest_from_gaps<br/>(KM.open_gaps → learning цели)"]
```

**Дедуп целей.** Все пути создания целей нормализуют описание через
`re.sub(r'\W+', ' ', s.lower()).strip()`, поэтому punctuation/whitespace
варианты одной цели схлопываются. Разные темы остаются разными.

**Auto-fix actions**, которые meta-learner может предпринять
по результатам анализа провала:

- `learn_topic` → цель «Learn: \<topic\>» приоритет ≤ severity
- `add_core_fact` → цель добавить факт в `core_memory.md`
- `update_note` → цель обновить существующую заметку KB
- `improve_prompt` → цель помечается как prompt engineering работа

---

## 11. Каналы — WebUI vs Telegram

Оба идут через один `Agent.run()`. Различаются только транспортом
и UX прогресса.

**WebUI (`/api/chat`)** — Server-Sent Events:

```mermaid
sequenceDiagram
    participant FE as Chat.tsx
    participant API as /api/chat
    participant AG as Agent

    FE->>API: POST {message, project, attachments[]}
    API->>AG: Agent(progress=cb).run(...)
    loop события прогресса
        AG-->>API: progress("event", "msg")
        API-->>FE: SSE {type:"progress",...}
    end
    AG-->>API: AgentAnswer
    API-->>FE: SSE {type:"answer", data: {...}}
    Note over FE: рендерим ответ +<br/>verification + thinking trace +<br/>tool calls (collapsible)
```

**Telegram (`backend/channels.py`)** — placeholder + edit-in-place:

```mermaid
sequenceDiagram
    participant TG as Пользователь (Telegram)
    participant BOT as TelegramBot
    participant AG as Agent

    TG->>BOT: текст / фото / голос
    BOT->>BOT: скачать вложения,<br/>транскрибировать голос если нужно
    BOT->>TG: 🧠 Thinking… (placeholder)
    BOT->>AG: run_in_executor(agent.run)
    loop события прогресса (executor thread)
        AG-->>BOT: progress("event","msg")
        BOT->>BOT: _TgProgressStream.push (rate-limited)
        BOT-->>TG: edit placeholder ↑
    end
    AG-->>BOT: AgentAnswer
    BOT-->>TG: ✅ Done · summary footer (правит placeholder)
    BOT-->>TG: <ответ> (новое сообщение, чанками по 4000 символов)
```

Edit placeholder'а throttled до одного per ~1.2s (Telegram rate
limit), и burst'ы coalesce в один отложенный flush.

---

## 12. Self-modifier (gated)

Опциональный путь: агент может анализировать свои собственные
модули и предлагать патчи, но **никогда не применяет их без явного
approve пользователя**.

```mermaid
flowchart LR
    REQ["analyze_module(module_name)"]
    READ["read_text() + head/tail truncate at 30k"]
    LLM["ANALYZE_SYSTEM<br/>LLM предлагает diff'ы"]
    PROP["Proposal{old_code, new_code}"]
    USR{"пользователь смотрит"}
    APP["apply: writes file<br/>(без auto-rollback)"]
    REJ["proposal.status = rejected"]
    REQ --> READ --> LLM --> PROP --> USR
    USR -->|"approve"| APP
    USR -->|"reject"| REJ
```

Этот модуль read-only по умолчанию — `apply()` срабатывает только
по явному действию пользователя через WebUI.

---

## 13. Полезные точки входа (file:line cheatsheet)

| Что найти | Где смотреть |
|---|---|
| Точка входа pipeline | `backend/agent.py:Agent.run` |
| Intent classifier | `backend/agent.py:_classify_intent` |
| Шаг thinking | `backend/agent.py:_think` |
| Solver + вызов tool loop'а | `backend/agent.py:_solve` |
| Verification + critic loop | `backend/agent.py:run` (искать `critic_threshold`) |
| Перестроение контекста для self-analysis | `backend/agent.py:_shared_context` (искать `for_self_analysis`) |
| Сборка identity preamble | `backend/identity.py:IdentityManager.preamble` |
| Выбор роутера | `backend/llm.py:DualModelRouter._pick` |
| Tool loop (Anthropic) | `backend/llm.py:AnthropicLLM.complete_with_tools` |
| Детектор false-absence | `backend/verifier.py:detect_false_absence_contradictions` |
| Hybrid search | `backend/hybrid_searcher.py:HybridSearcher.search` |
| Анализ провала + auto-extract | `backend/meta_learner.py:analyze_failure` |
| Нормализация дедупа целей | `backend/goals.py:_normalize_description` |
| Telegram realtime stream | `backend/channels.py:_TgProgressStream` |
| WebUI chat SSE | `backend/api/chat.py` |
| Dev mode redactor | `backend/dev_capture.py:redact_prompt` |

---

*Последний раз сверено с коммитом `0f934d6`.*
