# Orchestrator

## Purpose

Orchestrator, это control plane универсального агента.

Его задача не в том, чтобы быть вторым «мозгом», а в том, чтобы:
- классифицировать задачу;
- выбирать режим решения;
- собирать нужный рабочий контекст;
- маршрутизировать запросы в brain / retrieval / tools / verifier;
- следить за budget, риском и дисциплиной решения;
- переводить систему между состояниями до финального ответа или эскалации.

Коротко:
- **Base Brain = cognitive plane**
- **Orchestrator = control plane**

---

## Core responsibilities

- task classification;
- mode selection;
- planning and replanning;
- retrieval routing;
- tool routing;
- working memory assembly;
- verification gating;
- budget control;
- failure recovery;
- escalation or stop decision.

---

## Main operating modes

### 1. Direct Answer Mode
Используется, когда:
- задача простая;
- риск низкий;
- инструменты не нужны;
- factual grounding не критичен.

Flow:
`intake -> small context -> brain -> answer`

### 2. Retrieval-First Mode
Используется, когда:
- важны факты, документы, спецификации, manuals;
- нужен grounded answer.

Flow:
`intake -> retrieve -> rerank -> brain -> verify -> answer`

### 3. Tool-Execution Mode
Используется, когда:
- нужны внешние действия;
- расчёты, код, shell, API, DB, automation.

Flow:
`intake -> step plan -> tool call -> observe -> next step`

### 4. Deliberate Planning Mode
Используется, когда:
- задача многошаговая;
- есть зависимости;
- ошибка дорого стоит.

Flow:
`intake -> plan tree -> iterative execution -> milestone verification`

### 5. Case-Based Mode
Используется, когда:
- найден похожий прошлый кейс;
- важен reuse workflow, а не только facts.

Flow:
`intake -> retrieve similar cases -> adapt plan -> execute`

### 6. Strict Verification Mode
Используется, когда:
- health / finance / legal / industrial / production code / infra;
- нужен повышенный уровень проверки.

Flow:
`brain/tool output -> critic/verifier -> repair or approve -> answer`

### 7. Recovery Mode
Используется, когда:
- tool failed;
- verification failed;
- plan collapsed;
- exceeded budget;
- найдено противоречие.

Flow:
`detect failure -> classify failure -> retry / replan / escalate / abort`

---

## Routing policy

### Step 1. Classify task
Определить:
- task type;
- risk level;
- need for fresh facts;
- need for tool execution;
- need for prior cases;
- budget profile: fast / balanced / deep.

### Step 2. Choose primary mode

```text
if high_risk:
    mode = strict_verification

elif requires_external_action:
    mode = tool_execution

elif requires_factual_grounding:
    mode = retrieval_first

elif similar_case_found and case_similarity > threshold:
    mode = case_based

elif task_is_multi_step:
    mode = deliberate_planning

else:
    mode = direct_answer
```

### Step 3. Apply modifiers
Поверх основного mode применяются флаги:
- `verify = off | light | strict`
- `memory = off | case-only | full`
- `retrieval = none | kb | kb+case`
- `tool_policy = none | safe | full`
- `budget = fast | balanced | deep`

Это лучше, чем раздувать число режимов.

---

## State machine

### Main states
- `INTAKE`
- `CLASSIFY`
- `PLAN`
- `RETRIEVE`
- `LOAD_CASES`
- `PREPARE_WORKING_SET`
- `THINK`
- `ACT`
- `OBSERVE`
- `VERIFY`
- `REPLAN`
- `ANSWER`
- `LEARN`
- `ESCALATE`
- `ABORT`

### Happy path

```text
INTAKE
  -> CLASSIFY
  -> PLAN
  -> RETRIEVE
  -> PREPARE_WORKING_SET
  -> THINK
  -> VERIFY(optional)
  -> ANSWER
  -> LEARN
  -> END
```

### Tool loop

```text
THINK -> ACT -> OBSERVE -> THINK
```

### Failure loop

```text
ACT/VERIFY -> REPLAN -> THINK
```

### Escalation path

```text
REPLAN -> ESCALATE
REPLAN -> ABORT
```

---

## Transition rules

### `CLASSIFY -> DIRECT ANSWER`
Если:
- low risk;
- tools не нужны;
- retrieval не обязателен;
- estimated complexity small.

### `CLASSIFY -> RETRIEVE`
Если:
- важна factual correctness;
- вопрос ссылается на docs/specs/manuals;
- без grounding confidence низкий.

### `THINK -> ACT`
Если:
- brain выдал структурированный tool request;
- аргументы вызова полные;
- действие разрешено policy.

### `THINK -> RETRIEVE`
Если:
- не хватает ключевого факта;
- есть неопределённая гипотеза;
- перед действием нужен источник.

### `VERIFY -> ANSWER`
Если:
- проверки пройдены;
- confidence выше threshold;
- нет unresolved contradictions.

### `VERIFY -> REPLAN`
Если:
- test failed;
- source mismatch;
- unit inconsistency;
- safety concern;
- suspected hallucination.

### `REPLAN -> ESCALATE`
Если:
- repeated failures exceeded threshold;
- safe next action отсутствует;
- нужен выбор пользователя;
- budget almost exhausted.

---

## Working memory ownership

Working memory должна принадлежать orchestrator, а не самой модели.

### Why
- model остаётся более stateless;
- control state можно inspect/debug/replay;
- проще compression;
- проще budget-aware routing;
- проще сохранять и переиспользовать trajectory.

### Suggested working memory schema

```json
{
  "goal": "...",
  "mode": "retrieval_first",
  "plan": ["...", "..."],
  "current_step": 2,
  "constraints": ["...", "..."],
  "known_facts": ["...", "..."],
  "open_questions": ["...", "..."],
  "tool_observations": ["...", "..."],
  "retrieved_docs": ["doc_chunk_12", "doc_chunk_44"],
  "retrieved_cases": ["case_118"],
  "confidence": 0.71,
  "risk": "medium",
  "budget": {
    "tokens_used": 18200,
    "tool_calls": 4,
    "time_ms": 21000
  }
}
```

---

## Event model

Полезно строить orchestrator как event-driven систему.

### Event types
- `task_received`
- `classification_ready`
- `retrieval_ready`
- `case_match_found`
- `tool_requested`
- `tool_result_received`
- `verification_passed`
- `verification_failed`
- `budget_warning`
- `budget_exceeded`
- `user_clarification_needed`
- `final_answer_ready`

### Why this matters
- лучше debuggability;
- проще replay и audit;
- удобно для learning loop;
- удобно для eval и telemetry.

---

## Failure taxonomy

### 1. Knowledge failure
Не найден факт, источник слабый или retrieval пустой.

Typical action:
- broaden retrieval;
- change query;
- lower similarity threshold;
- ask user.

### 2. Tool failure
Tool упал, дал timeout или вернул плохую структуру.

Typical action:
- retry;
- switch tool;
- degrade gracefully;
- escalate.

### 3. Reasoning failure
Логика не сходится, противоречие между шагами.

Typical action:
- replan;
- reduce context;
- switch to stricter verification.

### 4. Verification failure
Провал тестов, mismatch с источником, unit error.

Typical action:
- repair;
- rerun;
- block final answer until fixed.

### 5. Policy failure
Действие запрещено или требует approval.

Typical action:
- stop;
- ask human;
- propose safe alternative.

### 6. Budget failure
Слишком много токенов, времени, tool calls или RAM.

Typical action:
- compress history;
- reduce top-K;
- disable optional retrieval;
- ask user whether to continue deeper.

### 7. Interaction failure
Нужны уточнения от пользователя.

Typical action:
- ask targeted clarification;
- pause execution.

### 8. Memory adaptation failure
Похожий кейс найден, но плохо ложится на текущую задачу.

Typical action:
- use case as hint only;
- fall back to fresh planning.

---

## Budget-aware control

У orchestrator должен быть Budget Manager.

Он отвечает за:
- token budget;
- latency budget;
- cost budget;
- tool-call budget;
- RAM/context budget.

### Typical reactions

```text
if token_budget_low:
    compress working memory
    reduce top_k
    disable non-critical case retrieval

if latency_budget_low:
    switch deep -> balanced -> fast

if tool_budget_exceeded:
    stop retries and escalate
```

---

## Practical pseudo-loop

```python
while not done:
    state = orchestrator.state

    if state == CLASSIFY:
        classify_task()
        choose_mode()

    elif state == RETRIEVE:
        fetch_docs_and_cases()

    elif state == PREPARE_WORKING_SET:
        build_compact_context()

    elif state == THINK:
        proposal = brain.solve_step(working_set)

        if proposal.need_tool:
            goto(ACT)
        elif proposal.need_retrieval:
            goto(RETRIEVE)
        elif proposal.ready_for_verify:
            goto(VERIFY)
        else:
            goto(ANSWER)

    elif state == ACT:
        result = run_tool(proposal.tool_call)
        observe(result)
        goto(THINK)

    elif state == VERIFY:
        verdict = verify(current_artifact)
        if verdict.pass:
            goto(ANSWER)
        else:
            goto(REPLAN)

    elif state == REPLAN:
        adjust_plan()
        if too_many_failures:
            goto(ESCALATE)
        else:
            goto(THINK)
```

---

## Design principle

Сильный orchestrator не пытается думать за модель.

Он:
- удерживает дисциплину;
- ограничивает хаос;
- дозирует контекст;
- заставляет систему проверять себя;
- не даёт тратить ресурсы бесконтрольно.

Коротко:

**brain produces candidate cognition, orchestrator enforces controlled problem solving**

---

## Deep dives

- `knowledge/software/universal-agent-architecture/orchestrator-general-llm.md`
- `knowledge/software/universal-agent-architecture/orchestrator-development-task-spec.md`
- `knowledge/software/universal-agent-architecture/orchestrator-schema.md`
- `knowledge/software/universal-agent-architecture/development-workflow-templates.md`
