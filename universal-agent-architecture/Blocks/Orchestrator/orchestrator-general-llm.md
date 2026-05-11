# Orchestrator with General LLM

## Goal

Реализовать orchestrator так, чтобы даже маленький CPU server мог быть частью универсального агента.

Ключевая идея:
- **runtime/control shell живёт локально**;
- **general LLM вызывается как внешний orchestration brain**;
- **реальное управление системой остаётся у deterministic runtime**.

То есть:
- general LLM не управляет системой напрямую;
- general LLM предлагает следующий шаг;
- runtime валидирует, разрешает или отклоняет это предложение.

---

## Core principle

Правильная схема не такая:

```text
LLM decides everything
```

А такая:

```text
runtime owns state
runtime owns policy
runtime owns budgets
runtime calls general LLM for orchestration advice
runtime validates the result
runtime executes allowed action
```

Это превращает LLM из «контроллера системы» в **advisory orchestration engine**.

---

## Why this architecture fits small CPU servers

Маленький сервер плохо подходит для постоянного запуска сильной planning model.

Но он отлично подходит для:
- хранения state;
- working memory management;
- file/tool execution;
- local retrieval;
- verification;
- caching;
- policy enforcement;
- retry control.

Тяжёлая интеллектуальная часть выносится наружу в **general LLM API**.

---

## Responsibility split

## What stays in local runtime

### Deterministic responsibilities
- state machine;
- session/task lifecycle;
- tool registry;
- permission checks;
- budget counters;
- retry logic;
- escalation thresholds;
- memory write rules;
- side-effect control;
- safe fallbacks.

### Why
Это должно быть предсказуемым, inspectable и reproducible.

---

## What goes to general LLM

### LLM orchestration responsibilities
- task classification;
- mode selection;
- short plan draft;
- retrieval query rewrite;
- tool selection proposal;
- case adaptation hint;
- replanning after failure;
- answer strategy suggestion.

### Important boundary
LLM **предлагает**, но не **владеет** control state.

---

## High-level architecture

```text
User / API / Chat
        |
        v
+-----------------------+
| Local Runtime Shell   |
| state + policy + WM   |
+-----------------------+
    |        |        |
    |        |        |
    v        v        v
 Tools   Retrieval   Verifier
    |
    v
 Observations / Artifacts
    |
    v
+-----------------------+
| General LLM           |
| planner / router      |
| replan advisor        |
+-----------------------+
    |
    v
 Structured action proposal
    |
    v
Local Runtime Validation
    |
    v
Execution / Answer / Escalation
```

---

## Runtime loop

### Step 1. Intake
Runtime принимает задачу и создаёт task state.

### Step 2. Build compact orchestration context
Runtime собирает:
- goal;
- constraints;
- current plan;
- known facts;
- latest observations;
- available tools;
- risk level;
- current budget;
- allowed action types.

### Step 3. Call general LLM
LLM получает компактный orchestration packet и должен вернуть **только структурированный output**.

### Step 4. Validate result
Runtime проверяет:
- schema valid;
- action exists;
- tool allowed;
- arguments allowed;
- transition allowed;
- budget not exceeded;
- risk/policy satisfied.

### Step 5. Execute
Если всё допустимо, runtime:
- вызывает tool;
- делает retrieval;
- запускает verify;
- задаёт уточняющий вопрос;
- завершает ответ.

### Step 6. Observe and update working memory
Runtime записывает observation и обновляет state.

### Step 7. Re-enter loop if needed
Если задача не завершена, цикл повторяется.

---

## When to call general LLM

Не надо звать его на каждый микрошаг.

### Recommended call points

#### 1. Initial classification
Определить:
- direct answer;
- retrieval-first;
- tool-execution;
- ask clarification;
- complex planning.

#### 2. Initial plan
Сгенерировать короткий план на 1-5 шагов.

#### 3. After major observation
Например после:
- tool result;
- retrieval result;
- verifier verdict.

#### 4. After failure
Если:
- tool failed;
- contradiction found;
- verification failed;
- current plan collapsed.

#### 5. Before risky finalization
Для medium/high-risk trajectories.

---

## Working memory packet for orchestrator

Runtime не должен слать в LLM весь conversation dump.

Нужно отправлять только компактный рабочий пакет.

### Example packet

```json
{
  "task_id": "task-001",
  "goal": "Find why local access fails and propose safe fix",
  "mode": "tool_execution",
  "current_step": 2,
  "plan": [
    "inspect listener state",
    "check bind address",
    "validate LAN access"
  ],
  "constraints": [
    "avoid destructive changes",
    "prefer minimal exposure"
  ],
  "known_facts": [
    "LAN IP is 192.168.18.58",
    "service port is 8015"
  ],
  "observations": [
    "port refused on LAN",
    "tailnet endpoint works"
  ],
  "risk": "medium",
  "budget": {
    "llm_calls_used": 2,
    "tool_calls_used": 4,
    "time_ms": 18000
  },
  "available_tools": [
    "read_file",
    "run_shell",
    "restart_service",
    "http_check"
  ],
  "allowed_actions": [
    "answer",
    "ask_user",
    "retrieve",
    "call_tool",
    "verify",
    "replan",
    "escalate"
  ]
}
```

---

## Structured output schema from general LLM

LLM должен отвечать не prose, а строго структурированным решением.

### Example schema

```json
{
  "mode": "tool_execution",
  "reason": "Need fresh runtime evidence before final answer",
  "next_action": "call_tool",
  "tool_name": "run_shell",
  "tool_args": {
    "command": "ss -ltnp | grep 8015"
  },
  "need_retrieval": false,
  "need_verification": true,
  "need_user_clarification": false,
  "confidence": 0.84,
  "fallback_if_blocked": "ask_user",
  "notes_for_runtime": [
    "Check service bind host",
    "Avoid changing unrelated ports"
  ]
}
```

---

## Minimal action enum

Чтобы не дать LLM бесконтрольную свободу, лучше ограничить действия enum-списком:

- `answer`
- `ask_user`
- `retrieve`
- `call_tool`
- `verify`
- `replan`
- `escalate`
- `abort`

LLM не должен придумывать новые action types.

---

## Validation rules in runtime

Даже хороший LLM иногда ошибается.

### Runtime must validate

#### Schema validation
- обязательные поля присутствуют;
- enum values допустимы;
- JSON parseable;
- tool args type-correct.

#### Policy validation
- tool разрешён;
- action допустим в текущем state;
- dangerous action требует approval;
- memory write разрешён.

#### Budget validation
- не превышен лимит LLM calls;
- не превышен tool budget;
- не превышен time budget.

#### Transition validation
- из текущего state допустим такой переход;
- невозможные transition блокируются.

---

## Fallback behavior

Это обязательная часть architecture.

### If general LLM unavailable
Runtime переходит в fallback mode:
- only simple direct answers;
- only safe known tools;
- no deep replanning;
- ask user earlier;
- no risky autonomous decisions.

### If general LLM returns invalid schema
Runtime:
- retries with stricter prompt;
- or falls back to safe deterministic behavior.

### If LLM suggests forbidden action
Runtime:
- rejects proposal;
- logs violation;
- asks for alternative or escalates.

---

## Good sides of this design

### 1. Small-node friendly
Маленькие CPU nodes не обязаны держать сильную reasoning model локально.

### 2. Better orchestration quality
General LLM лучше решает:
- ambiguity;
- multi-step planning;
- replanning;
- tradeoff reasoning.

### 3. Strong safety boundary
Критичные decisions остаются под deterministic control.

### 4. Easier upgrades
Можно менять orchestration brain без переписывания всей системы.

### 5. Shared orchestrator possibility
Один strong orchestrator service может обслуживать много weak nodes.

---

## Weak sides and risks

### 1. Network dependence
Если нет доступа к external orchestrator, интеллект деградирует в fallback mode.

### 2. Cost
Если дёргать general LLM слишком часто, orchestration становится дорогой.

### 3. Nondeterminism
Одинаковые задачи могут иногда получать слегка разные orchestration decisions.

### 4. Tool hallucination risk
LLM может предложить несуществующий tool или странный аргумент.

### 5. Overthinking risk
General LLM может усложнять простые задачи.

---

## How to reduce the risks

### Keep temperature low
Orchestration should be disciplined, not creative.

### Use strict schema output
Prefer JSON/schema-only output.

### Limit action space
Only whitelisted actions.

### Send compact state
Не посылать весь мусор разговора.

### Use policy and budget gates
LLM never bypasses code-level constraints.

### Add caching
Повторяющиеся orchestration patterns можно кэшировать.

---

## Suggested MVP

### MVP components

#### 1. Local runtime service
Держит:
- task state;
- working memory;
- tool executor;
- validator;
- retry policy.

#### 2. Orchestration prompt template
Стабильный prompt с чёткой schema contract.

#### 3. JSON validator
Строгая runtime-проверка output.

#### 4. Minimal tools
Например:
- file read;
- command exec;
- retrieval query;
- verify result.

#### 5. Fallback mode
Без него система хрупкая.

---

## Suggested first implementation phases

### Phase 1
- one general LLM;
- local deterministic runtime;
- fixed JSON schema;
- no multi-agent complexity;
- simple tool routing.

### Phase 2
- better replan logic;
- verifier integration;
- case memory retrieval;
- budget-aware routing.

### Phase 3
- shared orchestration service for multiple nodes;
- confidence calibration;
- cached policy/routing patterns;
- richer eval harness.

---

## Practical implementation note

Лучше всего думать об этом как о двух слоях:

### Layer 1. Execution substrate
Это локальная системная часть:
- tools;
- files;
- services;
- memory;
- policy;
- verification.

### Layer 2. Orchestration intelligence
Это внешний advisory brain:
- classify;
- choose mode;
- draft plan;
- replan;
- decide next action candidate.

---

## Final design principle

Для маленьких CPU серверов лучший orchestration pattern такой:

**small local deterministic shell + remote general LLM planner + strict validation + safe fallback**

Это даёт хороший баланс между:
- portability;
- intelligence;
- safety;
- cost;
- robustness.

---

## Deep dives

- `knowledge/software/universal-agent-architecture/orchestrator-development-task-spec.md`
- `knowledge/software/universal-agent-architecture/orchestrator-schema.md`
- `knowledge/software/universal-agent-architecture/development-workflow-templates.md`
