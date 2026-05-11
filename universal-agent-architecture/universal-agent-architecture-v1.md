# Universal Agent Architecture v1

## Goal

Построить универсального локального/гибридного агента, который:
- понимает задачу;
- умеет думать и планировать;
- использует инструменты;
- достаёт знания из внешних хранилищ;
- помнит прошлые кейсы;
- учится на завершённых задачах;
- при этом не упирается в context window на каждом шаге.

---

## Core principle

Это не одна «всезнающая» модель.

Это система из слоёв:

1. **Base Brain** — общий reasoning engine
2. **Orchestrator** — управляет циклом решения
3. **Tool & Skill Layer** — выполняет действия
4. **Knowledge Layer** — даёт факты и справки
5. **Case Memory** — даёт прошлый опыт
6. **Working Memory** — держит текущую рабочую сессию компактной
7. **Verifier / Critic** — проверяет качество и риск
8. **Learning Loop** — превращает опыт в память, датасеты и adapters
9. **Domain Adapters** — доменные стили мышления и эвристики
10. **Policy / Safety Layer** — ограничивает риск и управляет доступом

---

## Top-level architecture

```text
User / API / Chat
        |
        v
+-------------------+
|  Task Intake      |
|  + Normalizer     |
+-------------------+
        |
        v
+-------------------+
|  Orchestrator     |
|  Planner/Router   |
+-------------------+
   |      |      |
   |      |      |
   v      v      v
 Tools  Retrieval  Memory
   |       |         |
   +---+---+----+----+
       |        |
       v        v
   Base Brain   Verifier
       |           |
       +-----+-----+
             |
             v
        Final Answer
             |
             v
       Learning Loop
             |
   +---------+----------+
   |                    |
   v                    v
Case Store        Training/Adapters
```

---

## Main blocks

## 1. Task Intake

Входной слой.

Функции:
- принимает задачу от пользователя или API;
- нормализует язык, формат, вложения;
- определяет тип задачи: question / analysis / coding / engineering / health / research / automation;
- выставляет initial risk level;
- создаёт task id и session state.

Output:
- structured task object.

---

## 2. Orchestrator (Planner + Router)

Главный управляющий блок.

Функции:
- понимает, что именно надо сделать;
- делит задачу на шаги;
- решает, что делать моделью, а что инструментом;
- решает, какие знания нужны;
- выбирает domain adapter;
- следит, чтобы контекст не раздувался.

Подблоки:
- **Planner** — строит план;
- **Router** — выбирает tool / source / adapter / mode;
- **Budget Manager** — следит за context, latency, cost, RAM.

Deep dive:
- `knowledge/software/universal-agent-architecture/orchestrator.md`

---

## 3. Base Brain

Общая reasoning-модель.

Функции:
- анализирует задачу;
- читает retrieved context;
- делает выводы;
- пишет черновой ответ;
- формирует tool requests;
- умеет summarization/compression.

Хранит:
- общие языковые и reasoning способности.

Не должен хранить:
- всю предметную базу знаний;
- весь архив кейсов;
- всё project-specific knowledge.

---

## 4. Tool & Skill Layer

Слой действий.

Типы инструментов:
- search / web;
- file IO;
- calculator / python;
- code run / tests / lint;
- OCR / vision;
- DB access;
- domain tools (CAD, sensors, simulators, lab calculators, greenhouse telemetry);
- messaging / scheduling / automation.

Функции:
- выполнять конкретные операции вне модели;
- возвращать структурированные результаты;
- логировать side effects.

---

## 5. Knowledge Layer

Внешнее хранилище знаний.

Состоит из:
- **Document Store** — manuals, textbooks, docs, guides;
- **Structured Store** — tables, facts, ontologies, spec sheets;
- **Index Layer** — embeddings + keyword search + metadata filters;
- **Reranker** — выбирает лучшие куски.

Правило:
- в контекст попадают не все знания, а только top-K релевантных кусков.

---

## 6. Case Memory

Хранилище прошлого опыта.

Что хранить:
- задача;
- контекст;
- найденные источники;
- действия;
- ошибки;
- финальное решение;
- feedback;
- реальный outcome.

Зачем:
- reuse похожих кейсов;
- не повторять старые ошибки;
- собирать dataset для обучения.

Case memory != knowledge base.
- Knowledge base хранит факты.
- Case memory хранит опыт и прецеденты.

---

## 7. Working Memory

Краткоживущая рабочая память текущей задачи.

Содержит:
- цель;
- текущий план;
- активные гипотезы;
- ключевые факты;
- промежуточные результаты;
- короткий compressed history.

Назначение:
- не держать всю историю и все документы в context window;
- передавать в модель только рабочий набор данных.

---

## 8. Verifier / Critic Layer

Независимая проверка качества.

Функции:
- выявлять логические ошибки;
- проверять соответствие источникам;
- проверять расчёты и unit consistency;
- проверять код тестами/lint/run;
- проверять health/safety-risk;
- оценивать confidence.

Режимы:
- lightweight verify;
- strict verify;
- domain-specific verify.

---

## 9. Domain Adapters

Лёгкие доменные надстройки над базовой моделью.

Примеры:
- engineering adapter;
- coding adapter;
- botanics adapter;
- health adapter;
- legal/research adapter.

Хранят не «всю науку», а:
- стиль мышления;
- domain heuristics;
- типовые workflow;
- типовые ошибки;
- правила проверки;
- preferred output structures.

Используются выборочно, по решению router.

---

## 10. Learning Loop

Контур самообучения после завершения задач.

Этапы:
1. capture case;
2. extract lessons learned;
3. save to case memory;
4. update knowledge summaries/indexes;
5. build training examples;
6. train candidate adapter or prompt policy;
7. run eval;
8. promote only if better.

Критичный принцип:
- **не обучать веса после каждой задачи напрямую**;
- сначала копить опыт и чистить dataset;
- потом обучать кандидат и сравнивать.

---

## 11. Policy / Safety Layer

Функции:
- доступ к инструментам;
- ограничения по домену риска;
- human approval for dangerous actions;
- privacy boundaries;
- domain-specific guardrails.

Особенно важно для:
- health;
- legal;
- finance;
- industrial control.

---

## Runtime flow

### Normal solve flow

1. Intake принимает задачу.
2. Orchestrator классифицирует её.
3. Router выбирает adapter, tools и retrieval strategy.
4. Knowledge Layer и Case Memory отдают top-K контекст.
5. Working Memory собирает компактный рабочий пакет.
6. Base Brain решает текущий шаг.
7. Tool Layer выполняет действия при необходимости.
8. Verifier проверяет промежуточный или финальный результат.
9. Orchestrator либо делает следующий шаг, либо отдаёт ответ.
10. Learning Loop сохраняет кейс и уроки.

---

## Context window strategy

Чтобы не упираться в окно контекста:

- не грузить всё;
- retrieval только top-K;
- staged solving, а не one-shot giant prompt;
- compression/summaries;
- separate working memory;
- case retrieval only for nearest matches;
- procedural knowledge переносить в adapters, а не в prompt каждый раз.

---

## Data separation model

### In weights
- language;
- reasoning;
- general abstractions;
- tool-use patterns.

### In domain adapters
- domain heuristics;
- decision style;
- validation patterns;
- output style.

### In knowledge store
- facts;
- manuals;
- docs;
- standards;
- reference tables.

### In case memory
- solved tasks;
- outcomes;
- errors;
- fixes;
- lessons learned.

### In context window
- only current goal;
- current plan;
- top relevant facts;
- top relevant cases;
- current intermediate results.

---

## Minimal viable version (practical)

### Phase 1 — MVP
- one base model;
- orchestrator;
- tool layer;
- retrieval over local docs;
- case memory;
- verifier;
- manual lesson capture.

### Phase 2 — Strong system
- reranker;
- working memory manager;
- automatic case extraction;
- domain packs;
- initial adapters.

### Phase 3 — Self-improving system
- dataset builder;
- adapter training pipeline;
- benchmark/eval gate;
- candidate promotion;
- domain routing policies.

---

## Suggested first breakdown order

1. Orchestrator
2. Working Memory + context control
3. Knowledge Layer
4. Case Memory
5. Verifier
6. Domain Adapters
7. Learning Loop
8. Safety / policy

---

## Key design principle

Сильный универсальный агент строится не как «одна огромная модель, куда впихнули всё», а как:

**reasoning core + selective retrieval + compact working memory + tools + cases + verification + controlled learning**

---

## Brain-core candidate: recurrent-depth transformer

Один из сильных кандидатов на роль Base Brain:
- **Prelude** — один раз подготавливает вход;
- **Shared recurrent block** — один и тот же блок крутится T раз;
- **Latent reasoning** — внутреннее мышление идёт не через видимые токены, а в скрытом состоянии;
- **MoE inside loop** — внутренняя специализация экспертов;
- **Domain adapters / LoRA injection** — доменное поведение можно подмешивать внутрь цикла;
- **ACT halting** — простые случаи можно останавливать раньше, сложные думать дольше;
- **Coda** — финальная сборка ответа.

### Why this is promising

Плюсы:
- reasoning не засоряет context window;
- глубину мышления можно увеличивать без линейного роста числа уникальных слоёв;
- ближе к iterative reasoning, чем обычный single-pass transformer;
- лучше ложится на идею скрытого внутреннего рабочего пространства.

Ограничения:
- это всё ещё не заменяет retrieval, memory и tools;
- latent reasoning труднее интерпретировать и дебажить;
- циклы всё равно стоят compute/latency;
- tool-use и external actions должны управляться внешним orchestrator.

---

## Mapping: recurrent brain-core to universal agent

### What the recurrent schema covers

Эта схема покрывает в основном **Base Brain**:
- input encoding;
- hidden iterative reasoning;
- internal micro-specialization;
- adapter-conditioned thinking;
- adaptive halting;
- final answer synthesis.

### What it does NOT cover

Внешняя агентная оболочка всё равно нужна отдельно:
- Task Intake;
- Orchestrator;
- Tool Router;
- Knowledge Retrieval;
- Case Memory retrieval;
- Working Memory manager;
- Verifier;
- Learning Loop;
- Policy / Safety layer.

### Practical conclusion

Рекуррентный depth-brain, это не весь агент, а **очень хороший cognitive core** внутри большого agent shell.

---

## Domain adapters clarified

Domain adapter, это не «полная отдельная модель со всеми фактами домена».

Это лёгкая доменная надстройка над базой, которая переносит в систему:
- стиль мышления;
- эвристики домена;
- типовые workflow;
- типовые ошибки;
- паттерны проверки;
- preferred structure of outputs.

### Examples
- engineering adapter;
- coding adapter;
- botanics adapter;
- health adapter.

### Important boundary

Adapters лучше хранят **процедурный опыт**, а не весь массив фактов.

То есть:
- **facts/manuals/standards** лучше оставлять во внешнем knowledge store;
- **decision style / heuristics / validation habits** можно переносить в adapter.

---

## Context window problem, clarified

Да, если на каждую задачу просто загружать все знания и все кейсы в prompt, система быстро ломается.

Проблемы:
- context overflow;
- latency/cost explosion;
- degradation of focus;
- важное тонет среди нерелевантного;
- reasoning становится шумным.

### Correct strategy

Нельзя грузить всё. Нужно грузить только **рабочий набор на текущий шаг**.

### What should go into context window
- current goal;
- current plan;
- top relevant facts;
- top relevant cases;
- current constraints;
- current intermediate results.

### What should stay outside context window
- полная библиотека manuals;
- полные project archives;
- все прошлые кейсы;
- сырые логи;
- большие reference corpora.

### How to control context growth
- retrieval only top-K;
- reranking;
- staged solving instead of one giant prompt;
- working-memory summaries;
- case retrieval only for nearest matches;
- procedural compression into adapters;
- explicit context budget manager in orchestrator.

---

## Learning strategy, clarified

Идея «после каждой задачи дообучать систему» правильная по духу, но опасная, если делать её напрямую в веса после каждого кейса.

### Wrong pattern
- solve task;
- immediately fine-tune base model on one fresh example.

Это ведёт к:
- noise accumulation;
- overfitting;
- drift;
- catastrophic forgetting;
- unstable specialists.

### Better pattern
1. solve task;
2. save case;
3. extract lessons learned;
4. update case memory / summaries / datasets;
5. accumulate many good examples;
6. clean and validate dataset;
7. train candidate adapter;
8. run eval;
9. promote only if better.

### Key insight

Сначала система должна учиться в:
- **memory**,
- **case store**,
- **datasets**,

и только потом, периодически, переносить часть опыта в adapters.

---

## Two-level view of the whole system

### 1. Agent Shell
- Task Intake;
- Orchestrator;
- Retrieval;
- Tools;
- Working Memory;
- Verifier;
- Learning Loop;
- Safety / Policy.

### 2. Cognitive Core
- Prelude;
- recurrent shared block;
- latent reasoning;
- MoE experts;
- domain adapters;
- adaptive halting;
- Coda.

### Final formula

**Universal Agent = Agent Shell + Cognitive Core**

---

## Recommended step-by-step design order

Чтобы проектировать систему спокойно и без путаницы, разбирать её лучше в таком порядке:

1. **Orchestrator**
2. **Working Memory + context control**
3. **Knowledge Layer**
4. **Case Memory**
5. **Verifier**
6. **Domain Adapters**
7. **Learning Loop**
8. **Safety / Policy**
9. **Brain-core implementation choices**

### Why start with Orchestrator

Потому что именно он решает:
- когда думать внутри модели;
- когда идти в retrieval;
- когда звать tool;
- когда подключать adapter;
- какой budget дать по context/latency/compute;
- когда завершать цикл.
