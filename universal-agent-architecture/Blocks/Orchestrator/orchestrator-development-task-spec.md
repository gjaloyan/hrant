# Orchestrator Specification for Development Tasks

## Purpose

Этот документ задаёт **полную orchestration-схему** для задач разработки:
- coding;
- debugging;
- refactoring;
- test fixing;
- repo analysis;
- implementation planning;
- safe file changes.

Цель не в том, чтобы LLM сама «делала всё подряд», а в том, чтобы получить:
- управляемый workflow;
- предсказуемые state transitions;
- контролируемые tool calls;
- обязательную проверку результата;
- пригодность для слабых CPU nodes.

---

## Core design principle

### The orchestrator is not the coder

Orchestrator не является самим coding brain.

Он:
- управляет процессом решения;
- выбирает следующий шаг;
- решает, когда нужен code inspection, edit, test, search, verify;
- ограничивает хаос и риск.

Коротко:
- **runtime owns control**
- **general LLM proposes the next move**
- **tool layer executes**
- **verifier decides whether the result is acceptable**

---

## Scope of development tasks

Под development task понимаются:
- понять структуру проекта;
- найти нужный файл/модуль;
- понять причину ошибки;
- предложить fix;
- внести изменение;
- прогнать проверки;
- сделать replan, если fix не сработал;
- завершить задачу с explainable result.

Не входит в первый MVP:
- autonomous long-running branch management;
- mass refactors across many repos without supervision;
- auto-deploy в prod;
- self-modifying policy.

---

## High-level architecture

```text
User task
   |
   v
+--------------------------+
| Local Runtime Shell      |
| state + policy + memory  |
+--------------------------+
   |        |         |
   |        |         |
   v        v         v
 Repo IO   Exec      Verifier
   |        |         |
   +--------+---------+
            |
            v
     Observations / Artifacts
            |
            v
+--------------------------+
| General LLM Orchestrator |
| classify / plan / replan |
+--------------------------+
            |
            v
    Structured action proposal
            |
            v
     Runtime validation gate
            |
            v
 Execute / Replan / Answer / Escalate
```

---

## Component model

## 1. Local Runtime Shell

Владеет:
- task state;
- working memory;
- allowed tool registry;
- budgets;
- retry counters;
- policy rules;
- final state transitions.

Это deterministic слой.

---

## 2. General LLM Orchestrator

Отвечает за:
- initial classification;
- high-level plan;
- choosing next action type;
- deciding when more evidence is needed;
- replanning after failure;
- deciding when ready for verification.

General LLM не должна напрямую редактировать или исполнять без runtime approval.

---

## 3. Tool Layer

Для development tasks важны такие группы инструментов:

### Read tools
- file read;
- search/grep;
- repo tree discovery;
- config inspection.

### Change tools
- targeted edit;
- patch apply;
- file write.

### Execution tools
- run tests;
- lint;
- build;
- type-check;
- app startup check.

### Retrieval tools
- docs lookup;
- local project docs;
- prior case lookup.

---

## 4. Verifier Layer

Verifier отвечает не за «красоту текста», а за acceptance.

Для development tasks verifier должен проверять:
- syntax validity;
- tests pass/fail;
- lint/typecheck status;
- changed-files scope;
- regression signals;
- whether implementation satisfies task intent.

---

## Development task lifecycle

## State machine

### Main states
- `INTAKE`
- `CLASSIFY`
- `DISCOVER`
- `PLAN`
- `INSPECT`
- `EDIT`
- `RUN_CHECKS`
- `VERIFY`
- `REPLAN`
- `ANSWER`
- `ESCALATE`
- `ABORT`
- `LEARN`

---

## State semantics

### `INTAKE`
Нормализация задачи.

Runtime собирает:
- user request;
- repo root;
- sandbox constraints;
- risk level;
- known files if any.

### `CLASSIFY`
Определить тип dev task:
- bugfix;
- feature;
- refactor;
- test-only;
- analysis-only;
- docs-only.

### `DISCOVER`
Найти relevant files/modules before any edit.

Typical actions:
- grep;
- tree listing;
- read entry files;
- inspect configs.

### `PLAN`
Построить short execution plan.

Plan должен быть коротким и operational.

### `INSPECT`
Глубокое чтение только нужных файлов.

### `EDIT`
Внести минимально достаточные изменения.

### `RUN_CHECKS`
Запустить ровно те проверки, которые дают signal по задаче.

### `VERIFY`
Оценить:
- действительно ли решена проблема;
- не внесён ли очевидный regression;
- достаточно ли evidence.

### `REPLAN`
Сменить стратегию после failure.

### `ANSWER`
Вернуть пользователю:
- что сделано;
- какие файлы изменены;
- что проверено;
- какие ограничения остались.

### `ESCALATE`
Остановиться и запросить решение человека.

### `ABORT`
Прекратить задачу, если безопасного continuation нет.

### `LEARN`
Сохранить lessons/case summary после завершения.

---

## Main workflow for development tasks

### Standard bugfix flow

```text
INTAKE
 -> CLASSIFY
 -> DISCOVER
 -> PLAN
 -> INSPECT
 -> EDIT
 -> RUN_CHECKS
 -> VERIFY
 -> ANSWER
 -> LEARN
```

### If checks fail

```text
RUN_CHECKS -> VERIFY -> REPLAN -> INSPECT/EDIT -> RUN_CHECKS
```

### If evidence is insufficient

```text
VERIFY -> DISCOVER or INSPECT
```

### If task is analysis-only

```text
INTAKE -> CLASSIFY -> DISCOVER -> INSPECT -> ANSWER
```

---

## Decision rules by task type

## A. Bugfix
Default preference:
- discover cause first;
- patch minimal surface;
- run targeted checks;
- only then summarize.

### Required
- root-cause hypothesis;
- changed code evidence;
- at least one validation signal.

---

## B. Feature implementation
Default preference:
- inspect current architecture;
- define insertion point;
- implement incrementally;
- verify with tests or runtime signal.

### Required
- plan before edit;
- interface compatibility check;
- regression awareness.

---

## C. Refactor
Default preference:
- define non-functional goal;
- preserve behavior;
- run broader checks than for bugfix.

### Required
- scope control;
- explicit no-behavior-change assumption or note.

---

## D. Analysis-only task
Default preference:
- read, inspect, explain;
- no writes;
- no unnecessary execution.

---

## E. Test repair
Default preference:
- inspect failing tests;
- inspect implementation under test;
- decide whether code or tests are wrong;
- patch smallest correct location.

---

## Orchestration packet schema

Это то, что runtime отправляет general LLM.

```json
{
  "task_id": "dev-001",
  "task_type": "bugfix",
  "goal": "Fix failing local login flow",
  "repo_root": "/workspace/project",
  "current_state": "PLAN",
  "current_plan": [
    "inspect auth controller",
    "trace session cookie handling",
    "patch incorrect path setting",
    "run login tests"
  ],
  "constraints": [
    "do not edit unrelated modules",
    "prefer minimal diff"
  ],
  "known_files": [
    "src/auth/controller.ts",
    "src/auth/session.ts"
  ],
  "observations": [
    "test login_should_persist_session fails",
    "cookie path differs from expected route"
  ],
  "available_tools": [
    "read",
    "edit",
    "write",
    "apply_patch",
    "exec"
  ],
  "allowed_actions": [
    "discover",
    "inspect",
    "edit",
    "run_checks",
    "verify",
    "ask_user",
    "answer",
    "replan",
    "escalate",
    "abort"
  ],
  "budget": {
    "llm_calls_used": 2,
    "tool_calls_used": 5,
    "time_ms": 35000,
    "max_llm_calls": 10,
    "max_tool_calls": 30
  },
  "risk": "medium"
}
```

---

## Action proposal schema

Это то, что general LLM возвращает runtime.

```json
{
  "action": "inspect",
  "reason": "Need to confirm where session cookie path is set before editing",
  "target_files": [
    "src/auth/session.ts"
  ],
  "tool": "read",
  "tool_args": {
    "path": "src/auth/session.ts"
  },
  "expected_outcome": "Find cookie path construction logic",
  "next_if_success": "edit",
  "next_if_failure": "discover",
  "need_verification": false,
  "confidence": 0.81
}
```

---

## Allowed action enum

- `discover`
- `inspect`
- `edit`
- `run_checks`
- `verify`
- `ask_user`
- `answer`
- `replan`
- `escalate`
- `abort`

LLM не должна придумывать actions вне этого списка.

---

## Runtime validation rules

## 1. Schema validation
Проверить:
- корректный JSON;
- action входит в enum;
- нужные поля присутствуют;
- confidence в диапазоне 0..1.

## 2. Policy validation
Проверить:
- tool разрешён;
- file path допустим;
- действие соответствует sandbox policy;
- destructive commands запрещены без approval.

## 3. State validation
Проверить:
- допустим ли такой action из текущего state;
- например из `DISCOVER` нельзя сразу `ANSWER`, если нет evidence.

## 4. Budget validation
Проверить:
- не превышен лимит tool calls;
- не превышен лимит orchestration calls;
- не превышен time budget.

## 5. Scope validation
Проверить:
- proposed files относятся к задаче;
- diff scope не разрастается бесконтрольно.

---

## Transition rules

### `CLASSIFY -> DISCOVER`
Если repo/file context недостаточен.

### `DISCOVER -> PLAN`
Если найден минимальный набор relevant files.

### `PLAN -> INSPECT`
Если план сформирован и нужен file-level evidence.

### `INSPECT -> EDIT`
Если root-cause hypothesis достаточно подтверждена.

### `INSPECT -> DISCOVER`
Если inspected file не дал нужного evidence.

### `EDIT -> RUN_CHECKS`
Всегда, кроме чисто docs-only задач.

### `RUN_CHECKS -> VERIFY`
Когда появился объективный signal.

### `VERIFY -> ANSWER`
Если:
- acceptance criteria выполнены;
- checks good enough;
- no obvious regression found.

### `VERIFY -> REPLAN`
Если:
- tests failed;
- bug persists;
- implementation contradicts task;
- scope became wrong.

### `REPLAN -> INSPECT`
Если нужен новый diagnosis.

### `REPLAN -> EDIT`
Если root cause ясен и нужен другой fix.

### `REPLAN -> ESCALATE`
Если:
- несколько неудачных попыток;
- conflicting signals;
- risk rose too high.

---

## Verification policy for development tasks

## Minimum acceptable verification

### For bugfix
- targeted tests or executable signal;
- inspection confirms fix location;
- no obvious breakage in touched path.

### For feature
- implementation compiles or passes relevant tests;
- integration point checked;
- output matches requested behavior.

### For refactor
- broader tests than bugfix where possible;
- unchanged behavior assumption validated.

### For analysis-only
- no code modifications;
- answer backed by file evidence.

---

## Failure taxonomy

### 1. Discovery failure
Не удалось найти нужные файлы.

Action:
- broader search;
- inspect config/entrypoints;
- ask user for hint.

### 2. Diagnosis failure
Причина не подтверждается evidence.

Action:
- inspect adjacent modules;
- change hypothesis;
- avoid editing too early.

### 3. Edit failure
Patch applied, but behavior not fixed.

Action:
- rollback mentally;
- replan from root cause, not patch-on-patch.

### 4. Check failure
Tests/lint/build fail.

Action:
- classify whether failure is related;
- if unrelated but blocking, explain;
- if related, fix before answer.

### 5. Scope explosion
Task spreads into many modules.

Action:
- split into phases;
- stop and propose narrower change.

### 6. Risk escalation
Security, data loss, infra instability.

Action:
- ask approval;
- propose safer alternative;
- escalate.

---

## Budget policy

## Recommended budgets

### Low complexity
- llm orchestration calls: 2-4
- tool calls: 5-15
- edits: 1-3

### Medium complexity
- llm orchestration calls: 4-8
- tool calls: 10-30
- edits: 2-6

### High complexity
- should usually escalate to larger workflow or subagent

---

## Development workflow strategy

### Rule 1. Read before edit
Никогда не патчить файл, не прочитав relevant block.

### Rule 2. Discover before deep planning
Сначала понять структуру проекта, потом строить detailed plan.

### Rule 3. Minimal diff first
Предпочитать smallest correct change.

### Rule 4. Verify before claim
Не говорить “fixed”, пока нет validation signal.

### Rule 5. Replan from evidence, not from hope
После failure не лепить второй patch поверх первого без нового diagnosis.

### Rule 6. Prefer targeted checks over giant full-suite by default
Особенно на слабых машинах.

---

## Fallback mode

Если general LLM недоступен:
- runtime использует deterministic fallback;
- только safe workflows;
- no deep replanning;
- больше asks to user;
- no broad autonomous refactor.

Fallback flow:

```text
INTAKE -> DISCOVER -> INSPECT -> safe edit(optional) -> targeted checks -> ANSWER/ESCALATE
```

---

## MVP implementation plan

## Phase 1
- one runtime;
- one general LLM orchestrator;
- strict action JSON schema;
- read/edit/exec tools;
- targeted verification.

## Phase 2
- case memory for similar fixes;
- stronger verifier;
- budget-aware replanning;
- confidence calibration.

## Phase 3
- reusable workflow templates;
- multi-node orchestration;
- cached routing patterns;
- task family eval harness.

---

## Success criteria

Схема считается хорошей, если она:
- не редактирует вслепую;
- не теряет state между шагами;
- не даёт LLM прямой raw control;
- стабильно проходит через discover -> inspect -> edit -> verify;
- умеет останавливаться, когда evidence недостаточен;
- работает даже на слабом CPU node с внешним orchestration brain.

---

## Final principle

Для development tasks лучшая orchestration-схема такая:

**deterministic runtime + external general-LLM planner + controlled tool execution + mandatory verification**

Именно это даёт баланс между:
- качеством reasoning;
- безопасностью;
- стоимостью;
- переносимостью;
- пригодностью для universal agent.

---

## Related documents

- `knowledge/software/universal-agent-architecture/orchestrator-schema.md`
- `knowledge/software/universal-agent-architecture/development-workflow-templates.md`
