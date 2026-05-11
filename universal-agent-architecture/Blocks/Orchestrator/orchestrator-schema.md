# Orchestrator Schema

## Purpose

Этот документ задаёт **жёсткий contract** между:
- local runtime shell;
- general LLM orchestrator;
- tool execution layer;
- verifier layer.

Это не brainstorm, а рабочая спецификация интерфейсов.

Цель:
- стандартизировать orchestration loop;
- сделать поведение inspectable и reproducible;
- исключить произвольные LLM actions вне allowed contract;
- дать разработчикам понятную основу для реализации.

---

## Design principles

1. **Runtime owns control**
2. **LLM returns proposals, not direct commands**
3. **Every proposal is schema-validated**
4. **Every transition is runtime-validated**
5. **Every side effect goes through tools/policy**
6. **Verification is separate from planning**
7. **Fallback mode must exist**

---

## Control boundary

## Runtime responsibilities

Runtime обязан владеть:
- task lifecycle;
- state machine;
- working memory;
- tool registry;
- permission policy;
- retry counters;
- budget counters;
- transition validation;
- fallback behavior;
- final answer emission.

## LLM orchestrator responsibilities

LLM orchestrator может делать:
- task classification;
- routing suggestion;
- plan draft;
- replan suggestion;
- retrieval suggestion;
- tool-use suggestion;
- verification trigger suggestion;
- escalation suggestion.

LLM orchestrator не должен:
- выполнять tools;
- менять runtime state напрямую;
- обходить policy;
- invent new action enums;
- emit unsafe shell or file mutations outside validated tool path.

---

## Canonical loop

```text
1. Runtime receives task
2. Runtime builds orchestration packet
3. Runtime sends packet to orchestrator LLM
4. LLM returns structured proposal
5. Runtime validates schema
6. Runtime validates transition/policy/budget
7. Runtime executes allowed action
8. Runtime records observation
9. Runtime updates working memory
10. Loop continues until answer / escalate / abort
```

---

## Canonical runtime states

- `INTAKE`
- `CLASSIFY`
- `DISCOVER`
- `PLAN`
- `INSPECT`
- `RETRIEVE`
- `EDIT`
- `ACT`
- `RUN_CHECKS`
- `VERIFY`
- `REPLAN`
- `ANSWER`
- `ASK_USER`
- `ESCALATE`
- `ABORT`
- `LEARN`

### State intent summary

- `INTAKE` , normalize incoming task
- `CLASSIFY` , determine task family and risk
- `DISCOVER` , find relevant files/modules/sources
- `PLAN` , create short operational plan
- `INSPECT` , read relevant code/docs deeply
- `RETRIEVE` , fetch external/local knowledge chunks
- `EDIT` , change artifacts
- `ACT` , run non-edit tool actions
- `RUN_CHECKS` , run tests/lint/build/other checks
- `VERIFY` , assess correctness / sufficiency
- `REPLAN` , adapt after failure or contradiction
- `ANSWER` , finalize result to user
- `ASK_USER` , request clarification or approval
- `ESCALATE` , stop autonomous loop and hand off decision
- `ABORT` , terminate due to hard failure or policy block
- `LEARN` , persist case summary / lessons

---

## Canonical action enum

LLM may only emit one of these action values:

- `discover`
- `inspect`
- `retrieve`
- `plan`
- `edit`
- `act`
- `run_checks`
- `verify`
- `ask_user`
- `answer`
- `replan`
- `escalate`
- `abort`

No other action strings are valid.

---

## Canonical tool categories

Runtime should map concrete tools into categories.

### Read category
- read file
- search
- grep
- list tree
- inspect config

### Change category
- edit file
- apply patch
- write file

### Exec category
- tests
- lint
- build
- run script
- start local service

### Retrieval category
- docs lookup
- case lookup
- structured KB lookup

### Verify category
- test verdict
- lint verdict
- diff scope review
- source-backed answer check

---

## Orchestration input packet

Это основной объект, который runtime передаёт LLM.

### Required fields

```json
{
  "schema_version": "1.0",
  "task_id": "string",
  "task_type": "bugfix|feature|refactor|analysis|test_repair|ops|general",
  "goal": "string",
  "current_state": "INTAKE|CLASSIFY|DISCOVER|PLAN|INSPECT|RETRIEVE|EDIT|ACT|RUN_CHECKS|VERIFY|REPLAN|ANSWER|ASK_USER|ESCALATE|ABORT|LEARN",
  "risk": "low|medium|high",
  "constraints": ["string"],
  "current_plan": ["string"],
  "current_step_index": 0,
  "known_facts": ["string"],
  "observations": ["string"],
  "candidate_files": ["string"],
  "available_tools": ["string"],
  "allowed_actions": ["string"],
  "budget": {
    "llm_calls_used": 0,
    "tool_calls_used": 0,
    "time_ms_used": 0,
    "max_llm_calls": 0,
    "max_tool_calls": 0,
    "max_time_ms": 0
  },
  "attempt_counters": {
    "replans": 0,
    "failed_checks": 0,
    "failed_actions": 0
  },
  "policy": {
    "allow_edits": true,
    "allow_exec": true,
    "allow_network": false,
    "approval_required_for_destructive": true
  }
}
```

---

## Optional input packet fields

```json
{
  "repo_root": "/workspace/project",
  "user_preferences": ["prefer minimal diff", "avoid dependency upgrades"],
  "retrieved_context": [
    {
      "source": "docs/auth.md#login-flow",
      "summary": "Session cookie path should be /auth"
    }
  ],
  "related_cases": [
    {
      "case_id": "case-118",
      "summary": "Similar session bug fixed by cookie path normalization"
    }
  ],
  "latest_tool_result": {
    "tool": "exec",
    "status": "failed",
    "summary": "login test still fails"
  },
  "verification_status": {
    "last_verdict": "failed",
    "reason": "targeted tests still red"
  }
}
```

---

## Orchestrator output proposal

LLM must return one JSON object only.

### Required fields

```json
{
  "schema_version": "1.0",
  "action": "inspect",
  "reason": "Need to inspect session handling before editing",
  "confidence": 0.84,
  "expected_outcome": "Identify where cookie path is set",
  "next_if_success": "edit",
  "next_if_failure": "discover",
  "need_verification": false
}
```

### Optional fields

```json
{
  "tool": "read",
  "tool_args": {
    "path": "src/auth/session.ts"
  },
  "target_files": [
    "src/auth/session.ts"
  ],
  "plan_update": [
    "inspect session file",
    "patch cookie path",
    "run auth tests"
  ],
  "questions_for_user": [
    "Should I preserve backward compatibility with legacy cookie names?"
  ],
  "verification_hint": "Run targeted auth login test",
  "fallback_if_blocked": "ask_user",
  "notes_for_runtime": [
    "Prefer minimal diff",
    "Avoid touching controller if session helper is enough"
  ]
}
```

---

## Output field semantics

### `action`
Следующий orchestration step. Must be from enum.

### `reason`
Короткое объяснение для audit/debug, why this action is selected.

### `confidence`
Float `0.0 .. 1.0`.
Used for escalation heuristics, not as truth.

### `expected_outcome`
Что runtime ожидает получить после успешного выполнения.

### `next_if_success`
Recommended next action after positive observation.

### `next_if_failure`
Recommended next action after negative observation.

### `tool`
Concrete runtime tool identifier, if action requires tool execution.

### `tool_args`
Arguments for selected tool. Must be validated by runtime schema.

### `target_files`
Primary files/modules relevant to current action.

### `plan_update`
Replacement or refinement of current short plan.

### `questions_for_user`
Only for `ask_user` or when approval/clarification is needed.

### `verification_hint`
Suggestion for what kind of check should be run.

### `fallback_if_blocked`
Suggested safe fallback if tool/policy/state blocks current action.

### `notes_for_runtime`
Non-authoritative hints. Runtime may ignore them.

---

## JSON schema constraints

### Hard constraints
- exactly one top-level JSON object
- no prose outside JSON
- `schema_version` required
- `action` required
- `reason` required
- `confidence` required
- `confidence` must be between `0.0` and `1.0`
- `need_verification` required
- if `tool` present, `tool_args` must be present
- if `action == ask_user`, `questions_for_user` should be non-empty
- if `action == edit`, either `target_files` or `tool_args` must specify edit target
- if `action == answer`, `tool` should usually be absent

---

## Runtime validation pipeline

## Phase 1. Parse validation
- JSON parseable?
- single object?
- required fields exist?
- enum fields valid?
- confidence in range?

## Phase 2. State validation
- is this action allowed from current state?
- is this action consistent with task type?
- is plan transition coherent?

## Phase 3. Tool validation
- tool exists?
- tool allowed by policy?
- tool args shape valid?
- file paths within allowed workspace?

## Phase 4. Budget validation
- llm_calls remaining?
- tool_calls remaining?
- time budget remaining?
- too many replans already?

## Phase 5. Risk validation
- does action raise risk level?
- is human approval required?
- does it attempt destructive or broad change?

If any validation fails, runtime rejects proposal and uses retry/fallback/escalation logic.

---

## Allowed transitions matrix

### Common allowed transitions
- `INTAKE -> CLASSIFY`
- `CLASSIFY -> DISCOVER|PLAN|ASK_USER|ABORT`
- `DISCOVER -> INSPECT|PLAN|ASK_USER|ESCALATE`
- `PLAN -> INSPECT|RETRIEVE|EDIT|ACT|ASK_USER`
- `INSPECT -> EDIT|RETRIEVE|DISCOVER|PLAN|VERIFY`
- `RETRIEVE -> PLAN|INSPECT|VERIFY|ASK_USER`
- `EDIT -> RUN_CHECKS|VERIFY|ASK_USER`
- `ACT -> OBSERVATION(update only in runtime) -> PLAN|VERIFY|REPLAN`
- `RUN_CHECKS -> VERIFY|REPLAN`
- `VERIFY -> ANSWER|REPLAN|ASK_USER|ESCALATE`
- `REPLAN -> DISCOVER|INSPECT|EDIT|RUN_CHECKS|ESCALATE|ABORT`
- `ASK_USER -> CLASSIFY|PLAN|ABORT`
- `ANSWER -> LEARN`

### Forbidden examples
- `DISCOVER -> ANSWER` with no evidence
- `CLASSIFY -> EDIT` before discovery/inspection in non-trivial tasks
- `VERIFY -> EDIT` without re-entering replan/edit flow
- `ABORT -> EDIT`

---

## Confidence and escalation rules

Runtime should not trust confidence blindly.
It should use confidence together with evidence.

### Suggested interpretation
- `0.00 - 0.39` , weak proposal, prefer ask_user / discover / re-evaluate
- `0.40 - 0.69` , usable but requires caution and evidence
- `0.70 - 0.85` , strong working proposal
- `0.86 - 1.00` , high-confidence proposal, still validate normally

### Suggested escalation triggers
- confidence < 0.45 and no strong evidence
- more than 2 replans
- more than 2 failed checks on same subgoal
- repeated contradiction between observations and plan
- policy/risk block on essential next action

---

## Retry policy

### LLM retry allowed when
- invalid JSON
- missing required field
- action/tool mismatch
- proposal too vague to execute

### LLM retry prompt should say
- what field failed
- what constraint was violated
- that only corrected JSON should be returned

### Stop retrying when
- 2 schema failures in a row
- budget low
- fallback available
- same invalid pattern repeats

---

## Fallback mode

When orchestrator LLM unavailable or repeatedly invalid:
- runtime enters `fallback_mode = true`
- allowed actions narrow to safe subset
- no broad edit plans
- prefer inspect, targeted edit, targeted checks, ask_user

### Safe fallback action subset
- `discover`
- `inspect`
- `edit`
- `run_checks`
- `ask_user`
- `answer`
- `abort`

---

## Event log schema

Runtime should record structured events.

### Example event object

```json
{
  "timestamp": "2026-04-29T12:00:00Z",
  "task_id": "dev-001",
  "state": "RUN_CHECKS",
  "event_type": "tool_result",
  "summary": "targeted auth tests failed",
  "metadata": {
    "tool": "exec",
    "exit_code": 1
  }
}
```

### Recommended event types
- `task_received`
- `classification_done`
- `plan_created`
- `file_inspected`
- `tool_called`
- `tool_result`
- `edit_applied`
- `verification_passed`
- `verification_failed`
- `replan_triggered`
- `user_clarification_requested`
- `answer_finalized`
- `task_aborted`

---

## Verifier integration contract

Verifier should return structured verdict too.

### Example verifier verdict

```json
{
  "verdict": "failed",
  "confidence": 0.91,
  "checks_run": [
    "targeted auth test"
  ],
  "findings": [
    "session cookie still not persisted"
  ],
  "recommended_next_action": "replan"
}
```

### Allowed verifier verdict values
- `passed`
- `failed`
- `inconclusive`

---

## Developer implementation guidance

## Runtime modules suggested
- `task_state.py`
- `working_memory.py`
- `orchestrator_client.py`
- `proposal_validator.py`
- `policy_guard.py`
- `tool_executor.py`
- `verifier.py`
- `event_log.py`
- `fallback_controller.py`

## Suggested implementation order
1. task state + enums
2. input/output JSON schemas
3. proposal validator
4. runtime transition engine
5. tool executor abstraction
6. verifier abstraction
7. fallback mode
8. telemetry/event log

---

## Completion criteria

Orchestration contract is acceptable when:
- LLM outputs are always validated
- runtime can reject malformed proposals safely
- state transitions are explicit
- tools never execute outside policy
- verification can block premature “done” claims
- fallback mode works when LLM is unavailable
- the same task family behaves consistently across runs

---

## Final principle

**The orchestrator schema must make LLM guidance useful, but never sovereign.**
