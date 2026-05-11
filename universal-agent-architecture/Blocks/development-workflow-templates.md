# Development Workflow Templates

## Purpose

Этот документ задаёт **готовые reusable workflow templates** для development tasks.

Нужен для того, чтобы orchestrator:
- не изобретал процесс с нуля на каждой задаче;
- быстрее выбирал стабильный workflow;
- держал predictable sequence of steps;
- легче проходил verification.

Это библиотека шаблонов поверх общей orchestrator architecture.

---

## Template design principles

1. **Template is a workflow skeleton, not a rigid script**
2. **Discover before deep action**
3. **Read before edit**
4. **Verify before claim**
5. **Replan from evidence**
6. **Prefer minimal correct change**

---

## Canonical template fields

Каждый template должен описываться одинаково.

### Template fields
- `name`
- `task_family`
- `use_when`
- `do_not_use_when`
- `default_states`
- `entry_conditions`
- `exit_conditions`
- `required_evidence`
- `default_checks`
- `failure_patterns`
- `replan_strategy`
- `notes`

---

## Template 1. Bugfix

### Name
`bugfix-minimal-safe`

### Task family
- bugfix
- regression fix
- local behavior mismatch

### Use when
- reported failure exists;
- expected behavior is known or inferable;
- a minimal targeted fix is plausible.

### Do not use when
- full feature redesign is needed;
- root cause spans many subsystems;
- architecture migration is required.

### Default states

```text
INTAKE -> CLASSIFY -> DISCOVER -> PLAN -> INSPECT -> EDIT -> RUN_CHECKS -> VERIFY -> ANSWER
```

### Entry conditions
- user describes broken behavior;
- repo/files accessible;
- edits allowed.

### Exit conditions
- bug no longer reproduced or strong equivalent validation signal;
- touched scope remains controlled;
- no obvious regression in touched path.

### Required evidence
- observed failure or clear failure description;
- root-cause hypothesis;
- evidence from relevant file/module;
- post-fix validation signal.

### Default checks
- targeted test;
- local repro command;
- lint/typecheck only if relevant to touched area.

### Failure patterns
- patch without confirmed diagnosis;
- too many speculative edits;
- passing unrelated checks but not fixing core issue.

### Replan strategy
- if check fails, go back to diagnosis;
- inspect adjacent module rather than stacking patches;
- narrow to smaller root cause before second edit.

### Notes
Best default for most real coding fixes.

---

## Template 2. Feature Implementation

### Name
`feature-incremental-compatible`

### Task family
- feature
- enhancement
- capability addition

### Use when
- requested behavior is additive;
- insertion point can be found in current architecture;
- backward compatibility matters.

### Do not use when
- request is mostly refactor;
- existing subsystem is fundamentally wrong;
- requirements too ambiguous.

### Default states

```text
INTAKE -> CLASSIFY -> DISCOVER -> PLAN -> INSPECT -> EDIT -> RUN_CHECKS -> VERIFY -> ANSWER
```

### Entry conditions
- feature goal understandable;
- codebase inspectable;
- implementation surface discoverable.

### Exit conditions
- requested behavior implemented;
- integration points validated;
- no obvious breakage introduced.

### Required evidence
- insertion point identified;
- interface/contract understood;
- implementation diff;
- success signal from test/build/manual execution.

### Default checks
- targeted feature tests if present;
- relevant build/typecheck;
- minimal runtime validation.

### Failure patterns
- coding before architecture inspection;
- touching too many unrelated files;
- adding behavior without validating contract compatibility.

### Replan strategy
- if integration breaks, inspect boundary contracts;
- split feature into smaller subgoals;
- if ambiguity too high, ask user before wide implementation.

### Notes
Prefer incremental implementation, not giant one-shot patch.

---

## Template 3. Refactor

### Name
`refactor-behavior-preserving`

### Task family
- refactor
- cleanup
- structure improvement
- deduplication

### Use when
- goal is improve structure, readability, maintainability;
- behavior should remain unchanged.

### Do not use when
- bugfix evidence is still missing;
- requested change includes new product behavior;
- test coverage is too weak and risk too high.

### Default states

```text
INTAKE -> CLASSIFY -> DISCOVER -> PLAN -> INSPECT -> EDIT -> RUN_CHECKS -> VERIFY -> ANSWER
```

### Entry conditions
- refactor objective clearly stated;
- preservation target identified.

### Exit conditions
- code structure improved;
- behavior-preservation evidence exists;
- checks pass at least at relevant scope.

### Required evidence
- baseline understanding of current behavior;
- clear explanation of intended structural improvement;
- post-change validation signal.

### Default checks
- broader tests than bugfix template;
- lint/typecheck strongly recommended;
- smoke run if applicable.

### Failure patterns
- hidden behavior changes;
- scope explosion;
- cleanup becoming architecture rewrite.

### Replan strategy
- if scope grows, split refactor into phases;
- if behavior risk appears, stop and downgrade ambition;
- if tests are weak, ask user whether to proceed conservatively.

### Notes
This template should be more conservative than feature work.

---

## Template 4. Test Repair

### Name
`test-repair-diagnose-first`

### Task family
- failing tests
- flaky tests
- mismatch between tests and implementation

### Use when
- one or more tests fail;
- unclear whether code or tests are wrong.

### Do not use when
- no failing signal available;
- task is only feature implementation with no broken tests.

### Default states

```text
INTAKE -> CLASSIFY -> DISCOVER -> INSPECT -> PLAN -> RUN_CHECKS -> VERIFY -> EDIT -> RUN_CHECKS -> VERIFY -> ANSWER
```

### Entry conditions
- failing test output available or reproducible.

### Exit conditions
- test intent understood;
- correct side patched (test or implementation);
- relevant tests pass or failure is explained.

### Required evidence
- failing test signal;
- diagnosis whether failure is in test, implementation, or environment;
- validation after fix.

### Default checks
- targeted failing tests first;
- only broaden scope if necessary.

### Failure patterns
- blindly changing tests to make green;
- patching implementation without understanding test intent;
- confusing flaky environment issue with product bug.

### Replan strategy
- rerun target test with extra inspection;
- inspect fixtures and mocks;
- compare test intent to current product contract.

### Notes
This template starts from evidence, not from patching.

---

## Template 5. Repo Analysis Only

### Name
`analysis-readonly-explainer`

### Task family
- understand codebase
- explain architecture
- trace behavior
- compare modules

### Use when
- user asks to explain/analyze;
- no code changes requested.

### Do not use when
- implementation/fix is expected immediately.

### Default states

```text
INTAKE -> CLASSIFY -> DISCOVER -> INSPECT -> ANSWER
```

### Entry conditions
- read access exists.

### Exit conditions
- explanation backed by source evidence;
- no unnecessary edits or execution performed.

### Required evidence
- relevant files inspected;
- claims traceable to code/docs.

### Default checks
- usually no exec needed;
- maybe grep/read-only inspection.

### Failure patterns
- over-executing tools for simple explanation;
- making speculative claims beyond inspected evidence.

### Replan strategy
- if architecture unclear, discover more files;
- if ambiguity remains, ask a targeted clarifying question.

### Notes
Fastest and safest template.

---

## Template 6. Config / Build / Environment Fix

### Name
`env-build-fix-safe`

### Task family
- broken build
- config mismatch
- local service startup issue
- dependency wiring issue

### Use when
- failure arises from environment, config, ports, services, startup, tooling.

### Do not use when
- issue is clearly in product logic only.

### Default states

```text
INTAKE -> CLASSIFY -> DISCOVER -> INSPECT -> PLAN -> ACT -> VERIFY -> EDIT(optional) -> RUN_CHECKS -> VERIFY -> ANSWER
```

### Entry conditions
- executable environment checks possible.

### Exit conditions
- service/build/config behaves as expected;
- changed config scope documented;
- verification signal exists.

### Required evidence
- failing startup/build/config signal;
- actual source of misconfiguration;
- post-fix validation.

### Default checks
- startup check;
- build command;
- config inspection;
- targeted health endpoint or local probe.

### Failure patterns
- editing config before proving root cause;
- changing many toggles at once;
- fixing symptom but not source.

### Replan strategy
- compare effective config with expected config;
- isolate one variable at a time;
- rollback mentally and test smaller hypothesis.

### Notes
Important for infra-adjacent development tasks.

---

## Template selection policy

Runtime or orchestrator should choose template by task family.

### Recommended mapping

```text
bug report / broken behavior          -> bugfix-minimal-safe
new capability / enhancement          -> feature-incremental-compatible
cleanup / simplify / split modules    -> refactor-behavior-preserving
failing tests                         -> test-repair-diagnose-first
explain repo / trace flow             -> analysis-readonly-explainer
startup/build/config problem          -> env-build-fix-safe
```

---

## Template override rules

Даже если template выбран, runtime должен уметь его сменить.

### Override when
- evidence contradicts initial task family;
- user clarifies a different goal;
- scope expands beyond template assumptions;
- risk increases;
- verification shows wrong workflow was selected.

### Example
Reported as bugfix, but inspection shows actually missing feature.
Then:
- `bugfix-minimal-safe` -> `feature-incremental-compatible`

---

## Template-specific budgets

### analysis-readonly-explainer
- llm orchestration calls: 1-3
- tool calls: 2-10
- edits: 0

### bugfix-minimal-safe
- llm orchestration calls: 2-6
- tool calls: 5-20
- edits: 1-4

### feature-incremental-compatible
- llm orchestration calls: 3-8
- tool calls: 8-25
- edits: 2-6

### refactor-behavior-preserving
- llm orchestration calls: 3-8
- tool calls: 8-30
- edits: 2-8

### test-repair-diagnose-first
- llm orchestration calls: 2-6
- tool calls: 5-20
- edits: 1-4

### env-build-fix-safe
- llm orchestration calls: 2-7
- tool calls: 5-20
- edits: 0-4

---

## Reusable micro-patterns

Эти паттерны можно вставлять внутрь templates.

### Micro-pattern A. Discovery burst
```text
list tree -> grep symbols -> read entry file -> narrow candidate files
```

### Micro-pattern B. Minimal fix cycle
```text
inspect -> edit -> targeted check -> verify
```

### Micro-pattern C. Failure diagnosis loop
```text
check fail -> inspect evidence -> adjust hypothesis -> replan
```

### Micro-pattern D. Safe clarification pause
```text
ambiguity detected -> ask user -> resume with narrowed goal
```

### Micro-pattern E. Verification ladder
```text
targeted test -> relevant lint/typecheck -> broader check if risk warrants
```

---

## Developer implementation guidance

Templates should be stored as machine-usable configs, not only markdown.

### Suggested machine form

```json
{
  "name": "bugfix-minimal-safe",
  "task_family": ["bugfix"],
  "default_states": [
    "INTAKE",
    "CLASSIFY",
    "DISCOVER",
    "PLAN",
    "INSPECT",
    "EDIT",
    "RUN_CHECKS",
    "VERIFY",
    "ANSWER"
  ],
  "required_evidence": [
    "root_cause_hypothesis",
    "relevant_file_evidence",
    "validation_signal"
  ],
  "default_checks": [
    "targeted_test",
    "local_repro"
  ]
}
```

---

## Completion criteria

Workflow template library is good enough when:
- orchestrator can select a template consistently;
- templates reduce unnecessary planning variance;
- templates still allow replanning;
- verification requirements are explicit;
- developers can implement them as configs or enums;
- templates improve reliability of development task execution.

---

## Final principle

**Templates should reduce orchestration entropy without turning the agent into a brittle script.**
