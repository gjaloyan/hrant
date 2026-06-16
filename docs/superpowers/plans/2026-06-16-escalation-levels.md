# Escalation Levels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make turn-routing weight explicit as levels (L0/L1/L2) in one auditable module, and skip the claim verifier on pure-action turns (save/set/schedule) where there is nothing to ground.

**Architecture:** A new pure module `backend/escalation.py` owns the `Level` enum, the curated `_PURE_ACTION_TOOLS` set, trace→tool-names extraction, and the level decision. `unified_agent.run_unified()` computes the level just before its verifier gate and runs `verifier.verify` only at L2. Both turn artifacts gain a `level` field. No new LLM call.

**Tech Stack:** Python 3.12, pytest. Spec: `docs/superpowers/specs/2026-06-16-escalation-levels-design.md`.

---

### Task 1: The `escalation` module (pure logic + unit tests)

**Files:**
- Create: `backend/escalation.py`
- Test: `tests/test_escalation_levels.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_escalation_levels.py`:

```python
"""Unit tests for backend.escalation — the level decision is pure logic."""
from __future__ import annotations

from backend.escalation import (
    Level, decide_level, should_run_verifier, tool_names_from_trace,
)


class _Step:
    """Minimal stand-in for a ThinkingStep with a ToolCallDetail."""
    def __init__(self, event, name):
        self.event = event
        self.tool_call = type("TC", (), {"name": name})() if name else None


def test_decide_level_fast_chat_is_l0():
    assert decide_level(was_fast_chat=True, tool_names=[]) is Level.L0_CHAT


def test_decide_level_pure_action_is_l1():
    assert decide_level(
        was_fast_chat=False, tool_names=["save_user_fact"]
    ) is Level.L1_ACTION
    assert decide_level(
        was_fast_chat=False, tool_names=["set_setting", "schedule_message"]
    ) is Level.L1_ACTION


def test_decide_level_info_tool_is_l2():
    # any non-pure-action tool drags the turn to L2
    assert decide_level(
        was_fast_chat=False, tool_names=["save_user_fact", "web_search"]
    ) is Level.L2_TASK
    assert decide_level(
        was_fast_chat=False, tool_names=["terminal_exec"]
    ) is Level.L2_TASK
    # sandbox_exec is an execute tool but PRODUCES verifiable output -> L2
    assert decide_level(
        was_fast_chat=False, tool_names=["sandbox_exec"]
    ) is Level.L2_TASK


def test_decide_level_no_tools_full_path_is_l2():
    # escalated off the fast path but used no tools -> verify the reasoning
    assert decide_level(
        was_fast_chat=False, tool_names=[]
    ) is Level.L2_TASK


def test_should_run_verifier_by_level():
    assert should_run_verifier(Level.L0_CHAT) is False
    assert should_run_verifier(Level.L1_ACTION) is False
    assert should_run_verifier(Level.L2_TASK) is True


def test_tool_names_from_trace_counts_completed_steps_only():
    trace = [
        _Step("tool_starting", "save_user_fact"),  # not yet run -> ignored
        _Step("tool", "save_user_fact"),
        _Step("tool", "web_search"),
        _Step("assistant", None),                  # not a tool step
    ]
    assert tool_names_from_trace(trace) == ["save_user_fact", "web_search"]


def test_tool_names_from_trace_dict_fallback():
    class _DictStep:
        event = "tool"
        tool_call = {"name": "fetch_url"}
    assert tool_names_from_trace([_DictStep()]) == ["fetch_url"]


def test_tool_names_from_trace_empty():
    assert tool_names_from_trace([]) == []
    assert tool_names_from_trace(None) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_escalation_levels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.escalation'`

- [ ] **Step 3: Create the module**

Create `backend/escalation.py`:

```python
"""Explicit turn escalation levels — one source of truth for how much
verification weight a turn carries.

See docs/superpowers/specs/2026-06-16-escalation-levels-design.md.
Pure logic: no I/O, no LLM call. The level is derived from the fast-path
decision plus the tool classes that actually ran.
"""
from __future__ import annotations

from enum import IntEnum


class Level(IntEnum):
    L0_CHAT = 0     # direct answer, no tools
    L1_ACTION = 1   # tool turn, pure state-mutation — nothing to fact-verify
    L2_TASK = 2     # tool turn that produced assertable information


# Tools whose ONLY effect is a state mutation / confirmation. A turn that ran
# exclusively these has no assertable factual claims, so the claim verifier
# has nothing to check. Deliberately NOT endpoint_check._EXECUTE_TOOLS, which
# includes information-producing executors (sandbox_exec / agent_browser /
# delegate) and omits terminal_exec. save_to_workspace / save_knowledge are
# intentionally absent — their CONTENT can be wrong, so they still get verified.
_PURE_ACTION_TOOLS: frozenset[str] = frozenset({
    "save_user_fact", "set_setting", "schedule_message",
    "start_background_job", "define_task_endpoint", "complete_supervisor",
    "kick_supervisor", "grant_telegram_access", "revoke_telegram_access",
    "approve_pairing", "propose_skill", "propose_self_modification",
    "ask_user",
})


def tool_names_from_trace(trace) -> list[str]:
    """Names of the tools that actually ran in a turn's thinking trace.
    Mirrors the endpoint-check extraction: `tool_call` is a ToolCallDetail
    pydantic model on ThinkingStep (attribute access) with a dict fallback
    for older trace formats. Only completed steps (event 'tool'/'tool_error')
    count — a 'tool_starting' step has no result yet."""
    names: list[str] = []
    for step in (trace or []):
        if getattr(step, "event", "") not in ("tool", "tool_error"):
            continue
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None)
        if name is None and isinstance(tc, dict):
            name = tc.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def decide_level(*, was_fast_chat: bool, tool_names: list[str]) -> Level:
    """Pick a turn's level from what actually happened.
    - was_fast_chat            -> L0 (the chat lane answered directly).
    - non-empty trace, EVERY tool pure-action -> L1 (skip the verifier).
    - anything else            -> L2 (an information tool ran, or a no-tool
      reasoned answer escalated off the fast path -> verify it)."""
    if was_fast_chat:
        return Level.L0_CHAT
    if tool_names and all(n in _PURE_ACTION_TOOLS for n in tool_names):
        return Level.L1_ACTION
    return Level.L2_TASK


def should_run_verifier(level: Level) -> bool:
    """The claim verifier runs only at L2 — L0/L1 have nothing to ground."""
    return level >= Level.L2_TASK


def should_verify(tool_outputs, trace) -> bool:
    """THE unified verifier gate, as one testable function the production
    code calls. Run the claim verifier only when there is grounding material
    (`tool_outputs`) AND the turn is L2 (an information tool ran)."""
    if not tool_outputs:
        return False
    level = decide_level(
        was_fast_chat=False, tool_names=tool_names_from_trace(trace),
    )
    return should_run_verifier(level)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_escalation_levels.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/escalation.py tests/test_escalation_levels.py
git commit -m "feat(escalation): Level model + pure level decision (L0/L1/L2)"
```

---

### Task 2: Wire the level into `run_unified` + artifacts

**Files:**
- Modify: `backend/unified_agent.py` (verifier gate ~2948-2965; fast-path artifact ~2019-2044; full-path `turn_record` ~3168-3191)
- Test: `tests/test_escalation_verifier_gate.py`

- [ ] **Step 1: Write the failing regression test**

Create `tests/test_escalation_verifier_gate.py`:

```python
"""Regression: the unified verifier gate (escalation.should_verify — the
SAME function run_unified calls) must skip the claim verifier on a pure-action
turn (save_user_fact) and run it on an information turn (web_search)."""
from __future__ import annotations

from backend.escalation import should_verify


class _Step:
    def __init__(self, name):
        self.event = "tool"
        self.tool_call = type("TC", (), {"name": name})()


def test_verifier_skipped_on_save_only_turn():
    trace = [_Step("save_user_fact")]
    assert should_verify(["ok: saved"], trace) is False


def test_verifier_runs_on_web_search_turn():
    trace = [_Step("web_search")]
    assert should_verify(["results..."], trace) is True


def test_verifier_skipped_when_no_tool_outputs():
    # no grounding material at all -> gate is False regardless of trace
    assert should_verify([], [_Step("web_search")]) is False
```

- [ ] **Step 2: Run the test to verify it passes against the module**

Run: `python -m pytest tests/test_escalation_verifier_gate.py -q`
Expected: PASS (3 passed). `should_verify` is the exact function `run_unified` will call (Step 4), so this regression tests production code, not a copy.

- [ ] **Step 3: Compute the level before the verifier gate**

In `backend/unified_agent.py`, find the verifier block (around line 2945):

```python
    # Post-hoc: optional verifier. Same threshold as legacy
    # task_mode — only fires when there's grounding material to
    # verify against (notes + tool outputs).
    vr = VerificationResult(confidence=85)
    if _cascade_prevr is not None:
```

Insert, immediately ABOVE `vr = VerificationResult(confidence=85)`:

```python
    # Escalation level (L0/L1/L2). Pure-action turns (only state-mutation
    # tools) are L1 and skip the claim verifier — endpoint_met already
    # confirmed delivery deterministically and there are no claims to ground.
    # `_level` is also stamped on the artifact below. See backend/escalation.py.
    from .escalation import decide_level, should_verify, tool_names_from_trace
    _level = decide_level(
        was_fast_chat=False,
        tool_names=tool_names_from_trace(agent._trace or []),
    )
```

- [ ] **Step 4: Gate the verifier via `should_verify`**

In the same block, change the verifier trigger line from:

```python
    elif tool_outputs:
        try:
            from .verifier import verify
```

to:

```python
    elif should_verify(tool_outputs, agent._trace or []):
        try:
            from .verifier import verify
```

- [ ] **Step 5: Stamp the level on the full-path artifact**

Find the `turn_record` dict (around line 3168). After the `"n_llm_calls": (...)` entry and before the closing `}` (around line 3190), add:

```python
            "level": _level.name,
```

- [ ] **Step 6: Stamp L0 on the fast-path artifact**

Find the fast-path `artifact` dict (around line 2031, the `"mode": "fast_chat",` line). Immediately after that line, add:

```python
                    "level": "L0_CHAT",
```

- [ ] **Step 7: Verify the wiring by inspection**

Run: `python -m pytest tests/test_escalation_levels.py tests/test_escalation_verifier_gate.py -q`
Expected: PASS (11 passed)

Run: `python -c "import ast; ast.parse(open('backend/unified_agent.py', encoding='utf-8').read()); print('syntax OK')"`
Expected: `syntax OK`

Run: `git grep -n "should_run_verifier(_level)" backend/unified_agent.py`
Expected: one hit on the `elif tool_outputs and should_run_verifier(_level):` line.

- [ ] **Step 8: Run the existing unified-agent suite (no regressions)**

Run: `python -m pytest tests/test_unified_agent.py tests/test_endpoint_check_turn_cache.py tests/test_answer_critic.py tests/test_fast_path_conversation_persistence.py -q`
Expected: PASS (all green — the change is additive and gated).

- [ ] **Step 9: Commit**

```bash
git add backend/unified_agent.py tests/test_escalation_verifier_gate.py
git commit -m "feat(escalation): gate verifier by level + stamp level on turn artifacts"
```

---

## Notes for the implementer

- **`_level` scope:** it is defined in `run_unified`'s body just before the verifier block and read again when `turn_record` is built later in the same function — both on the normal (non-fast-path) completion path, so it is always in scope there. The fast-path branch returns earlier and uses the literal `"L0_CHAT"`.
- **Deliberate duplication:** the endpoint block (~2979-2994) has its own inline trace→names extraction. Leave it — `tool_names_from_trace` mirrors its logic but rewiring the endpoint block is out of scope and higher-risk. The two are independently correct.
- **No new LLM call:** `decide_level` / `tool_names_from_trace` are pure; the only behavioral change is that `verifier.verify` is skipped at L0/L1.
