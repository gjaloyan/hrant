# Hrant Bench-Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise Hrant's terminal-bench score from 9/20 to 14-17/20 by closing four orthogonal failure modes — weak self-verification, truncated-output refusal loop, fire-and-return on background processes, and missing provider fallback on safety refusals.

**Architecture:** Three new branches in the existing `_decide_self_correction` helper (Blocks 1b, 2, 3) + one new always-on prompt rule (Block 1a) + one router-level safety-error fallback (Block 4). Each branch is a deterministic detector + a one-shot corrective re-prompt — same shape as the existing self-correction branches.

**Tech Stack:** Python 3.12 / FastAPI / pytest. No new dependencies.

---

## Recurring conventions (read once, applies to every task)

- TDD: write the failing test first, run it to confirm failure, then implement, run again to confirm pass.
- English-only code, no emojis (project rule).
- One commit per task at the end. No `git push` from subagents.
- All tests live under `tests/`; run with `python -m pytest <path> -v`.
- `_decide_self_correction` is in `backend/unified_agent.py`. The existing signature is:
  ```python
  def _decide_self_correction(*, task: str, answer: str, turn_tools: list[str]) -> tuple[str, str]:
  ```
  Tasks 2-4 extend it to `(*, task, answer, turn_tools, trace=None, speaker_id="")`. The single call site is in `run_unified` near the end of the turn. Each branch returns `(tag, corrective_text)`; empty `tag` means no correction.
- Existing branches (unbacked-claim + endpoint-not-met) MUST remain functional. New branches go BEFORE them in first-match-wins order.

## File structure

| File | Role |
|---|---|
| `backend/unified_agent.py` (MODIFY) | Add `_RULES_VERIFY_TESTS` constant + 3 detector helpers + extend `_decide_self_correction` signature + 3 new branches + call-site update. |
| `backend/llm.py` (MODIFY) | Router-level fallback when active provider raises a safety-shaped `LLMError`. |
| `tests/test_self_correction_decision.py` (MODIFY) | Append tests for the 3 new branches + the verify-tests prompt rule. |
| `tests/test_router_safety_fallback.py` (CREATE) | Unit tests for Block 4 fallback. |

---

### Task 1: Block 1a — `_RULES_VERIFY_TESTS` prompt rule

**Files:**
- Modify: `backend/unified_agent.py`
- Test: `tests/test_self_correction_decision.py`

Background:
- `_build_rules_for_turn(ctx, has_attachments, sticky_fired, repeat_refusal)` composes the system-prompt rule string. It always includes `_RULES_JOURNAL_FIRST` plus structural-signal blocks. We add `_RULES_VERIFY_TESTS` to the always-on set.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_self_correction_decision.py`:

```python
def test_build_rules_for_turn_always_includes_verify_tests_rule():
    """The 'verify with the real test suite before declaring done'
    rule must appear in the per-turn system prompt regardless of
    attachments / sticky / refusal context. Always-on by design —
    universally useful, not bench-specific."""
    from backend.unified_agent import _build_rules_for_turn
    out = _build_rules_for_turn(
        ctx=None,
        has_attachments=False,
        sticky_fired=False,
        repeat_refusal=False,
    )
    assert "Before declaring a task done" in out
    assert "/tests/" in out
    assert "pytest" in out.lower() or "test suite" in out.lower()


def test_build_rules_for_turn_verify_tests_rule_present_with_attachments_too():
    """The verify-tests rule does NOT depend on structural signals
    — it must still appear when has_attachments / sticky / refusal
    add their own blocks."""
    from backend.unified_agent import _build_rules_for_turn
    out = _build_rules_for_turn(
        ctx=None,
        has_attachments=True,
        sticky_fired=True,
        repeat_refusal=True,
    )
    assert "Before declaring a task done" in out
    assert "/tests/" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_self_correction_decision.py::test_build_rules_for_turn_always_includes_verify_tests_rule tests/test_self_correction_decision.py::test_build_rules_for_turn_verify_tests_rule_present_with_attachments_too -v`

Expected: FAIL with `AssertionError: 'Before declaring a task done' not in <text>` (or similar).

- [ ] **Step 3: Add the constant and wire it into `_build_rules_for_turn`**

In `backend/unified_agent.py`, near the other `_RULES_*` constants (around `_RULES_JOURNAL_FIRST`), add:

```python
# Always-on: the agent MUST run the real test suite before declaring
# done. Closes the "I made it compile, looks good" failure mode that
# accounts for 6 of the 11 terminal-bench failures on the 2026-06-04
# baseline. Universal — many real-world tasks have test suites.
_RULES_VERIFY_TESTS = """## Before declaring a task done — run the real tests

If the workspace contains a test suite (any of: `/tests/` directory,
`test_*.py` files at the project root, a `Makefile` `test` target,
`pytest.ini`, or `pyproject.toml` with `[tool.pytest.ini_options]`),
you MUST execute it and observe a passing run BEFORE composing your
final answer.

"It compiles", "my sample input works", or "I checked the obvious
case" are NOT verification — only the real test suite's pass signal
counts. If tests fail, fix the cause and re-run. Do not synthesize
the final answer while any test is red.

If you cannot find a test suite, look for verifier hints in the
task instruction (paths under `/tests/`, "the verifier expects X",
"run pytest /tests"). Run them. If you genuinely cannot find any
verification mechanism, say so explicitly in your final answer —
don't pretend a check happened."""
```

Then in `_build_rules_for_turn`, ADD this to the always-included parts list. Find the function body:

```python
def _build_rules_for_turn(
    *,
    ctx=None,
    has_attachments: bool = False,
    sticky_fired: bool = False,
    repeat_refusal: bool = False,
) -> str:
    """..."""
    parts = [_unified_rules_core(ctx), _RULES_JOURNAL_FIRST]
```

Change the `parts = [...]` line to:

```python
    parts = [_unified_rules_core(ctx), _RULES_JOURNAL_FIRST, _RULES_VERIFY_TESTS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_self_correction_decision.py -v`

Expected: all pass.

Also confirm no regression in the rules-related tests:
```
python -m pytest tests/test_self_correction_decision.py tests/test_action_drift_marker.py --no-header -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```
git add backend/unified_agent.py tests/test_self_correction_decision.py
git commit -m "feat(prompt): always-on rule 'run real tests before declaring done'

Closes 6 of the 11 terminal-bench failures from the 2026-06-04 baseline
where Hrant declared a task done after running only a toy check
(compile-only, sample input, visible warriors only) and missed the
real /tests/ suite. Universal — many real-world tasks have test
suites; the rule is cheap (~150 input tokens) and unambiguous."
```

---

### Task 2: Block 3 — background-not-awaited detector + branch

**Files:**
- Modify: `backend/unified_agent.py`
- Test: `tests/test_self_correction_decision.py`

Background:
- We add a helper `_detect_background_not_awaited(trace)` that scans tool calls for backgrounding shapes AND the absence of any wait/poll afterwards.
- Then we extend `_decide_self_correction` to accept `trace` and add a branch that fires when the helper returns True.
- Update the single call site in `run_unified` to pass `agent._trace`.

We do Block 3 FIRST in the sequence because it's the smallest detector — easier to validate the new signature extension once before adding more.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_self_correction_decision.py`:

```python
def _fake_trace(*tool_calls):
    """Build a fake agent._trace from a list of (name, args_dict) tuples.
    Matches the ThinkingStep+ToolCallDetail shape consumed by helpers.
    """
    class _Step:
        def __init__(self, name, args):
            class _TC:
                pass
            tc = _TC()
            tc.name = name
            tc.args = args or {}
            self.tool_call = tc
            self.event = "tool"
    return [_Step(n, a) for n, a in tool_calls]


def test_detect_background_not_awaited_trailing_ampersand():
    """`nohup ... &` shape without a later wait/poll fires the detector."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "nohup python /app/train.py > /tmp/t.log 2>&1 &"}),
        ("terminal_exec", {"command": "echo started PID=5384"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_falsy_on_logical_and():
    """`cmd1 && cmd2` is NOT backgrounding — `&&` must not trigger."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "make build && make test"}),
    )
    assert _detect_background_not_awaited(trace) is False


def test_detect_background_not_awaited_false_when_wait_present():
    """If the agent backgrounded AND then waited, the failure mode
    doesn't apply — leave it alone."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "nohup python /app/train.py &"}),
        ("terminal_exec", {"command": "wait $!"}),
        ("terminal_exec", {"command": "ls /app/output/"}),
    )
    assert _detect_background_not_awaited(trace) is False


def test_detect_background_not_awaited_nohup_in_middle_of_chain():
    """`cd /x; nohup python y &` (no &&) still fires."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "cd /app; nohup python train.py &"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_disown_fires():
    """`disown` is only used for backgrounding."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "python long.py & disown"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_setsid_fires():
    """`setsid` is the same shape as `nohup`."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "setsid /app/run.sh &"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_empty_trace():
    """No tools = no backgrounding signal."""
    from backend.unified_agent import _detect_background_not_awaited
    assert _detect_background_not_awaited([]) is False
    assert _detect_background_not_awaited(None) is False


def test_decide_self_correction_background_branch_fires(monkeypatch):
    """When _detect_background_not_awaited returns True, the corrective
    text must mention 'background' and the tag must label it."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace(
        ("terminal_exec", {"command": "nohup python train.py &"}),
        ("terminal_exec", {"command": "echo started"}),
    )
    tag, corrective = _decide_self_correction(
        task="Train the model and tell me final accuracy.",
        answer="Started training in the background; will produce model.bin.",
        turn_tools=["terminal_exec", "terminal_exec"],
        trace=trace,
        speaker_id="webui:default",
    )
    assert tag.startswith("background")
    assert "wait" in corrective.lower() or "poll" in corrective.lower()
```

NOTE: the existing `_patch_judges` helper at the top of this test file already stubs the LLM judges (claim + endpoint). Re-use it for the new branch-fires test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_self_correction_decision.py -v -k "background or _fake_trace"`

Expected: 8 new failures with `ImportError`: `_detect_background_not_awaited` doesn't exist, AND signature mismatch on `_decide_self_correction` (`trace`/`speaker_id` kwargs not accepted).

- [ ] **Step 3: Implement the detector + extend the signature**

In `backend/unified_agent.py`, add the helper. Place it near the existing `_turn_tool_names` helper (so all trace-inspection helpers live together):

```python
import re as _re_bg  # local-only alias to avoid colliding with module-level `re` if reorder needed


_BG_TRAILING_AMP = _re_bg.compile(r"[^&]\s+&\s*$")
_BG_NOHUP = _re_bg.compile(r"(^|;|\|\||&&|;)\s*nohup\s+")
_BG_SETSID = _re_bg.compile(r"(^|;|\|\||&&|;)\s*setsid\s+")
_BG_DISOWN = _re_bg.compile(r"(^|\s)disown(\b|$)")

_WAIT_HINTS = (
    "wait $",      # `wait $PID` or `wait $!`
    "wait %",      # job-spec wait
    "while ps",    # busy-wait pattern
    "until [",     # busy-wait pattern
    "tail -f",     # following the job's log to completion
)


def _command_looks_backgrounded(cmd: str) -> bool:
    """Heuristic match for the four supported backgrounding shapes.
    Conservative on purpose — better miss a legitimate fire-and-forget
    than false-positive on `make build && make test`."""
    if not cmd:
        return False
    if _BG_TRAILING_AMP.search(cmd):
        return True
    if _BG_NOHUP.search(cmd):
        return True
    if _BG_SETSID.search(cmd):
        return True
    if _BG_DISOWN.search(cmd):
        return True
    return False


def _command_looks_like_wait(cmd: str) -> bool:
    """Did this command wait/poll for a background job to finish?"""
    if not cmd:
        return False
    low = cmd.lower()
    return any(hint in low for hint in _WAIT_HINTS)


def _detect_background_not_awaited(trace) -> bool:
    """True iff at least one terminal_exec command in `trace` was
    backgrounded AND no LATER command in the same trace waited/polled.
    Returns False on empty/None trace.

    Deterministic — does not call any LLM.
    """
    if not trace:
        return False
    bg_index = None
    for i, step in enumerate(trace):
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if name != "terminal_exec":
            continue
        args = getattr(tc, "args", None) or (
            tc.get("args") if isinstance(tc, dict) else {}
        )
        cmd = (args or {}).get("command") or ""
        if bg_index is None and _command_looks_backgrounded(cmd):
            bg_index = i
            continue
        if bg_index is not None and _command_looks_like_wait(cmd):
            return False
    return bg_index is not None
```

Now extend `_decide_self_correction`. Find the existing signature:

```python
def _decide_self_correction(
    *,
    task: str,
    answer: str,
    turn_tools: list[str],
) -> tuple[str, str]:
```

Replace with:

```python
def _decide_self_correction(
    *,
    task: str,
    answer: str,
    turn_tools: list[str],
    trace=None,
    speaker_id: str = "",
) -> tuple[str, str]:
```

Add the background branch at the TOP of the function body (after the empty-answer short-circuit, before the existing zero-tool / toolful-no-deliver branches):

```python
    if not (answer or "").strip():
        return "", ""

    # Block 3 — background-not-awaited. Fire FIRST because it's the
    # most specific pattern (the agent literally spawned a process
    # and walked away).
    if _detect_background_not_awaited(trace):
        corrective = (
            "You spawned a background process (nohup/&/setsid/disown) "
            "and never waited for it to finish. Wait for it now: use "
            "`wait $!` (if you have the PID), or poll with `while ps -p "
            "$PID >/dev/null 2>&1; do sleep 5; done`, or `tail -f` the "
            "job's log until you see its completion marker. Then verify "
            "the expected artifact actually exists on disk before "
            "composing your final answer."
        )
        return "background-not-awaited", corrective

    # ... existing branches below stay unchanged ...
```

Then in `run_unified`, find the existing call to `_decide_self_correction`. The current call shape is:

```python
        _correction_tag, _corrective = _decide_self_correction(
            task=task, answer=answer, turn_tools=_turn_tools,
        )
```

Replace with:

```python
        _correction_tag, _corrective = _decide_self_correction(
            task=task,
            answer=answer,
            turn_tools=_turn_tools,
            trace=getattr(agent, "_trace", None),
            speaker_id=speaker_id,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_self_correction_decision.py -v`

Expected: all pass (existing tests + 8 new).

- [ ] **Step 5: Commit**

```
git add backend/unified_agent.py tests/test_self_correction_decision.py
git commit -m "feat(self-correct): background-not-awaited branch

Adds _detect_background_not_awaited helper + new branch in
_decide_self_correction that fires when the agent spawned a job
(nohup/&/setsid/disown) and never waited/polled for it. Closes
the caffe-cifar-10 failure mode where the agent reported 'training
started' but the turn ended before the model file existed.

Conservative detection: && / regular pipelines are explicitly NOT
flagged. Extended the helper signature with trace + speaker_id,
plumbed from run_unified."
```

---

### Task 3: Block 2 — truncated-then-refusal detector + branch

**Files:**
- Modify: `backend/unified_agent.py`
- Test: `tests/test_self_correction_decision.py`

Background:
- We add `_detect_truncated_then_refusal(trace, answer)` that returns True iff the LAST `terminal_exec` result was truncated AND the final answer contains one of the existing refusal patterns.
- The existing module-level constant `_REFUSAL_PHRASES` (used by `_recent_refusal_pattern`) is the source of truth for refusal-shaped answers; reuse it.
- New branch fires AFTER the background one but BEFORE the existing toolful-no-deliver / unbacked-claim.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_self_correction_decision.py`:

```python
def _fake_trace_with_truncation(*tool_calls_with_trunc):
    """Like _fake_trace but each entry is (name, args, result_truncated_bool)."""
    class _Step:
        def __init__(self, name, args, truncated):
            class _TC:
                pass
            tc = _TC()
            tc.name = name
            tc.args = args or {}
            tc.result_truncated = bool(truncated)
            self.tool_call = tc
            self.event = "tool"
    return [_Step(n, a, t) for n, a, t in tool_calls_with_trunc]


def test_detect_truncated_then_refusal_positive():
    """Last terminal_exec was truncated AND answer matches a refusal
    phrase -> True."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat /app/big_log.txt"}, True),
    )
    assert _detect_truncated_then_refusal(
        trace, "Honestly: I cannot confirm the file was modified."
    ) is True


def test_detect_truncated_then_refusal_no_truncation():
    """Refusal phrase but no truncation -> False (other branches handle
    plain refusals)."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "echo hi"}, False),
    )
    assert _detect_truncated_then_refusal(
        trace, "I cannot confirm anything happened."
    ) is False


def test_detect_truncated_then_refusal_truncation_but_no_refusal():
    """Truncation happened but answer is fine -> False."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat /app/big_log.txt"}, True),
    )
    assert _detect_truncated_then_refusal(
        trace, "Done. The file has 1234 lines and ends with EOF marker."
    ) is False


def test_detect_truncated_then_refusal_russian_phrases():
    """Russian refusal phrases also count."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "ls -la /"}, True),
    )
    assert _detect_truncated_then_refusal(
        trace, "Честно: я не могу подтвердить что файл был создан."
    ) is True


def test_detect_truncated_then_refusal_only_last_terminal_exec_counts():
    """We look at the LAST terminal_exec's truncation flag, not earlier
    ones."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat huge"}, True),
        ("terminal_exec", {"command": "echo small"}, False),
    )
    assert _detect_truncated_then_refusal(
        trace, "I cannot confirm anything."
    ) is False


def test_detect_truncated_then_refusal_empty():
    """No trace or no terminal_exec calls -> False."""
    from backend.unified_agent import _detect_truncated_then_refusal
    assert _detect_truncated_then_refusal([], "I cannot confirm.") is False
    assert _detect_truncated_then_refusal(None, "I cannot confirm.") is False


def test_decide_self_correction_truncated_then_refusal_branch_fires(monkeypatch):
    """When truncated+refusal pattern matches, the corrective text
    must instruct the agent to narrow output (head/tail/grep)."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat /app/log.txt"}, True),
    )
    tag, corrective = _decide_self_correction(
        task="Check the log and tell me the final status.",
        answer="I cannot confirm what the final status is.",
        turn_tools=["terminal_exec"],
        trace=trace,
        speaker_id="webui:default",
    )
    assert tag.startswith("truncated")
    assert any(s in corrective.lower() for s in ("tail", "head", "grep"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_self_correction_decision.py -v -k "truncated"`

Expected: 7 new failures with `ImportError: cannot import name '_detect_truncated_then_refusal'`.

- [ ] **Step 3: Implement the detector + branch**

In `backend/unified_agent.py`, near the existing `_REFUSAL_PHRASES` constant and the `_detect_background_not_awaited` helper, add:

```python
def _detect_truncated_then_refusal(trace, answer: str) -> bool:
    """True iff the LAST terminal_exec call in `trace` returned a
    truncated result AND `answer` matches one of the existing
    refusal phrases. This narrowly catches the failure mode where
    Hrant's needed evidence was clipped past the 1500-char cap and
    the agent then refused to commit.

    Deterministic — does not call any LLM. Reuses _REFUSAL_PHRASES
    so future additions to the refusal list are picked up
    automatically.
    """
    if not trace or not answer:
        return False
    head = answer[:300].lower()
    if not any(phrase in head for phrase in _REFUSAL_PHRASES):
        return False
    # Walk trace backwards to find the last terminal_exec call.
    for step in reversed(list(trace)):
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if name != "terminal_exec":
            continue
        was_truncated = getattr(tc, "result_truncated", False)
        if was_truncated is None and isinstance(tc, dict):
            was_truncated = tc.get("result_truncated", False)
        return bool(was_truncated)
    return False
```

Then in `_decide_self_correction`, add the new branch AFTER the background branch but BEFORE the toolful-no-deliver branch:

```python
    # ... background-not-awaited branch from Task 2 ...

    # Block 2 — truncated-then-refusal. Specific recovery path: tell
    # the agent to narrow the output via tail/head/grep so the actual
    # evidence fits the 1500-char tool-result cap.
    if _detect_truncated_then_refusal(trace, answer):
        corrective = (
            "Your last terminal_exec output was truncated at the "
            "1500-char cap and the part you needed to act on didn't "
            "fit. Re-run the command with the output narrowed: pipe "
            "through `tail -200`, `head -200`, or `grep -n PATTERN` so "
            "only the relevant slice comes back. Read the actual "
            "evidence before composing the final answer — do not "
            "refuse based on truncated output."
        )
        return "truncated-then-refusal", corrective

    # ... existing branches below ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_self_correction_decision.py -v`

Expected: all pass (Task 1 + 2 + 3 tests all green).

- [ ] **Step 5: Commit**

```
git add backend/unified_agent.py tests/test_self_correction_decision.py
git commit -m "feat(self-correct): truncated-then-refusal recovery branch

Adds _detect_truncated_then_refusal helper + new branch in
_decide_self_correction. Closes the 3 terminal-bench failures
(largest-eigenval, path-tracing, path-tracing-reverse) where the
agent's needed evidence was clipped past the 1500-char tool-result
cap and the agent then refused to commit. The corrective tells it
to narrow output via tail/head/grep and act on what fits."
```

---

### Task 4: Block 1b — tests-exist-not-run detector + bench-mode branch

**Files:**
- Modify: `backend/unified_agent.py`
- Test: `tests/test_self_correction_decision.py`

Background:
- We add `_detect_tests_exist_not_run(trace)` that scans for evidence the agent discovered a test suite (via `ls /tests`, `find /tests`, `read_file` under `/tests/`, etc.) but never ran one (no `pytest`, `python -m pytest`, `python -m unittest`, `make test`, etc.).
- New branch fires ONLY when `speaker_id == "webui:bench-harness"` (bench-mode gate).
- Ordering: this branch goes AFTER truncated-then-refusal but BEFORE toolful-no-deliver.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_self_correction_decision.py`:

```python
def test_detect_tests_exist_not_run_positive():
    """Agent ran `ls /tests` (so tests exist) but never ran pytest etc."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cat /app/main.py"}),
        ("terminal_exec", {"command": "make build"}),
    )
    assert _detect_tests_exist_not_run(trace) is True


def test_detect_tests_exist_not_run_negative_pytest_was_run():
    """Discovery happened AND pytest was run -> False, all good."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "pytest /tests/ -v"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_negative_no_discovery():
    """No discovery signal at all -> False (we don't assume tests exist)."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /app"}),
        ("terminal_exec", {"command": "echo done"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_find_tests_discovery():
    """`find /tests` also counts as discovery."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "find /tests -name '*.py'"}),
    )
    assert _detect_tests_exist_not_run(trace) is True


def test_detect_tests_exist_not_run_read_file_under_tests():
    """A read_file call with path under /tests/ also counts as discovery."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("read_file", {"path": "/tests/test_outputs.py"}),
    )
    assert _detect_tests_exist_not_run(trace) is True


def test_detect_tests_exist_not_run_python_m_pytest_counts_as_run():
    """`python -m pytest` is the same test-run signal."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("read_file", {"path": "/tests/test_outputs.py"}),
        ("terminal_exec", {"command": "python -m pytest /tests/ -q"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_make_test_counts_as_run():
    """`make test` counts as a test run."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cd /app && make test"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_unittest_counts_as_run():
    """`python -m unittest` counts as a test run."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("read_file", {"path": "/tests/test_x.py"}),
        ("terminal_exec", {"command": "cd /app && python -m unittest discover"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_decide_self_correction_tests_branch_fires_under_bench_speaker(monkeypatch):
    """When tests-exist-not-run + speaker is bench-harness -> branch fires."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cat /app/main.c"}),
    )
    tag, corrective = _decide_self_correction(
        task="Implement /app/main.c per the spec.",
        answer="Wrote /app/main.c and verified it compiles.",
        turn_tools=["terminal_exec", "terminal_exec"],
        trace=trace,
        speaker_id="webui:bench-harness",
    )
    assert tag.startswith("tests-exist")
    assert "test" in corrective.lower()


def test_decide_self_correction_tests_branch_skipped_for_non_bench(monkeypatch):
    """Same pattern but speaker is NOT bench-harness -> branch does NOT
    fire (we don't want to force /tests checks on regular WebUI users)."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cat /app/main.c"}),
    )
    tag, corrective = _decide_self_correction(
        task="Look at /tests and tell me what's there.",
        answer="There are 3 files in /tests/.",
        turn_tools=["terminal_exec", "terminal_exec"],
        trace=trace,
        speaker_id="webui:default",
    )
    assert not tag.startswith("tests-exist")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_self_correction_decision.py -v -k "tests_exist or tests_branch"`

Expected: 10 new failures with `ImportError: cannot import name '_detect_tests_exist_not_run'`.

- [ ] **Step 3: Implement the detector + branch**

Add to `backend/unified_agent.py` near the other detectors:

```python
# Patterns that mean "the agent learned a test suite exists":
#   - terminal_exec command starting with `ls /tests`, `find /tests`,
#     `cat /tests/`, `head /tests/`, `tail /tests/`
#   - read_file with path under `/tests/`
_TESTS_DISCOVERY_TERMINAL_PREFIXES = (
    "ls /tests",
    "find /tests",
    "cat /tests/",
    "head /tests/",
    "tail /tests/",
)

# Patterns that mean "the agent ran the actual test suite":
#   - any terminal_exec command containing one of these tokens
_TESTS_RUN_TOKENS = (
    "pytest",         # also covers `python -m pytest`
    "unittest",       # covers `python -m unittest`
    "make test",
)


def _detect_tests_exist_not_run(trace) -> bool:
    """True iff (a) at least one tool call in `trace` indicates the
    agent discovered the existence of a test suite AND (b) no tool
    call in the same trace actually executed it.

    Deterministic — does not call any LLM.
    """
    if not trace:
        return False
    discovered = False
    ran = False
    for step in trace:
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        args = getattr(tc, "args", None) or (
            tc.get("args") if isinstance(tc, dict) else {}
        ) or {}
        if name == "terminal_exec":
            cmd = (args.get("command") or "").strip()
            cmd_low = cmd.lower()
            # Discovery via terminal_exec
            if any(cmd.startswith(p) or (";" + p) in cmd or ("&&" + p) in cmd
                   for p in _TESTS_DISCOVERY_TERMINAL_PREFIXES):
                discovered = True
            # Run via terminal_exec
            if any(token in cmd_low for token in _TESTS_RUN_TOKENS):
                ran = True
        elif name == "read_file":
            path = (args.get("path") or "")
            if path.startswith("/tests/") or path == "/tests":
                discovered = True
    return discovered and not ran
```

Then in `_decide_self_correction`, add the branch AFTER the truncated-then-refusal branch and BEFORE the toolful-no-deliver:

```python
    # ... truncated-then-refusal branch from Task 3 ...

    # Block 1b — tests-exist-not-run, bench-mode only. The universal
    # prompt rule from Task 1 already told the agent to run tests;
    # this branch is the structural backstop when the agent ignored
    # it. Bench-harness only: we don't want to force /tests checks on
    # the WebUI owner asking "what's in /tests/".
    if speaker_id == "webui:bench-harness" and _detect_tests_exist_not_run(trace):
        corrective = (
            "You discovered a test suite under /tests/ but never ran "
            "it. Run the actual tests NOW (`pytest /tests/ -v`, or "
            "`python -m pytest /tests/`, or `make test`, whichever "
            "matches the project setup) and observe a passing run "
            "BEFORE composing your final answer. If the tests fail, "
            "fix the cause and re-run — do not synthesize the final "
            "answer while any test is red."
        )
        return "tests-exist-not-run", corrective

    # ... existing branches below ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_self_correction_decision.py -v`

Expected: all pass (Tasks 1, 2, 3, 4 tests all green).

- [ ] **Step 5: Commit**

```
git add backend/unified_agent.py tests/test_self_correction_decision.py
git commit -m "feat(self-correct): tests-exist-not-run branch (bench-harness only)

Adds _detect_tests_exist_not_run helper + new branch in
_decide_self_correction that fires when the agent learned a test
suite exists (ls /tests, find /tests, read_file under /tests/) but
never ran one (no pytest / unittest / make test). Bench-harness
only — non-bench WebUI users asking 'what's in /tests/' should not
be force-prompted to execute tests.

Together with Task 1's universal prompt rule this closes the 6
'declared done after toy verification' failures from the
2026-06-04 baseline (gpt2-codegolf, regex-chess, torch-tp,
reshard-c4-data, winning-avg-corewars, break-filter-js-from-html)."
```

---

### Task 5: Block 4 — router safety-error fallback

**Files:**
- Modify: `backend/llm.py`
- Create: `tests/test_router_safety_fallback.py`

Background:
- The Codex Responses API returns a safety refusal as `LLMError` with a message that includes one of: `flagged`, `content_policy`, `cybersecurity_risk`, `safety` (case-insensitive substring match).
- Today the router (`backend/llm.py`) propagates the error without trying the next provider.
- The fix: catch this specific shape, log it, retry on the next configured provider. If the next provider also fails (with anything), give up and propagate.

Implementation note for the subagent: read `backend/llm.py` first to find the existing call dispatch / fallback chain. This task does NOT touch the per-provider `LLM` classes; only the router-level orchestration that picks a provider per request and falls back. The new behaviour: ALSO treat "safety-shaped LLMError" as a reason to try the next provider, alongside any existing 5xx / rate-limit / timeout fallback logic.

The test contract is fixed; the implementation choice is whether to add a helper `_is_safety_refusal(LLMError) -> bool` (recommended) or inline the substring check.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_router_safety_fallback.py`:

```python
"""Router-level fallback when the active provider returns a safety
refusal (Codex Responses API 'content flagged for cybersecurity risk',
or any LLMError whose message contains 'flagged' / 'content_policy' /
'cybersecurity_risk' / 'safety').

The router must retry the SAME call on the next configured provider
in priority order. If no fallback is configured, the original
LLMError propagates as before.
"""
from __future__ import annotations

import pytest


def test_is_safety_refusal_recognizes_flagged():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("content was flagged for cybersecurity risk")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_recognizes_content_policy():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("content_policy violation: refused")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_recognizes_safety_substring():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("Codex Responses API safety filter triggered")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_recognizes_cybersecurity_risk():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("blocked: cybersecurity_risk in user prompt")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_false_on_unrelated_errors():
    from backend.llm import _is_safety_refusal, LLMError
    assert _is_safety_refusal(LLMError("rate limit exceeded")) is False
    assert _is_safety_refusal(LLMError("HTTP 503: service unavailable")) is False
    assert _is_safety_refusal(LLMError("timed out")) is False
    assert _is_safety_refusal(LLMError("")) is False


def test_router_falls_back_to_next_provider_on_safety_refusal(monkeypatch):
    """When the active (codex) provider raises a safety-shaped LLMError,
    the router MUST try the next configured provider. The next provider
    completes normally and its answer is returned."""
    from backend import llm as _llm

    calls = []

    class _FakeProvider:
        def __init__(self, name, behaviour):
            self.name = name
            self.behaviour = behaviour
            self.model = "stub-model"

        def call(self, task_type, system, user, **kw):
            calls.append(self.name)
            if self.behaviour == "safety":
                raise _llm.LLMError(
                    "Codex Responses API stream error: content flagged "
                    "for cybersecurity risk"
                )
            if self.behaviour == "ok":
                return f"answer from {self.name}"
            if self.behaviour == "other":
                raise _llm.LLMError("HTTP 500: upstream")
            raise NotImplementedError(self.behaviour)

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return {"ok": True}

    fake_chain = [
        _FakeProvider("codex-prime", "safety"),
        _FakeProvider("anthropic-fallback", "ok"),
    ]

    # Monkeypatch the router's provider list. We don't care HOW the
    # router stores providers internally — patch the read accessor.
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda *_a, **_kw: fake_chain)

    router = _llm.router()
    out = router.call(
        _llm.TaskType.QUICK_ANSWER, "sys", "user prompt", max_tokens=100,
    )
    assert out == "answer from anthropic-fallback"
    assert calls == ["codex-prime", "anthropic-fallback"]


def test_router_propagates_when_no_fallback(monkeypatch):
    """If no next provider exists, the safety LLMError propagates."""
    from backend import llm as _llm

    class _OnlyProvider:
        name = "codex-prime"
        model = "stub-model"

        def call(self, *a, **kw):
            raise _llm.LLMError(
                "Codex stream: content was flagged for safety reasons."
            )

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return self.call(*a, **kw)

    monkeypatch.setattr(_llm, "_active_provider_chain", lambda *_a, **_kw: [_OnlyProvider()])
    router = _llm.router()
    with pytest.raises(_llm.LLMError):
        router.call(_llm.TaskType.QUICK_ANSWER, "sys", "user", max_tokens=10)


def test_router_does_not_fallback_on_non_safety_error(monkeypatch):
    """A plain LLMError (no safety substrings) does NOT trigger this
    specific fallback. Whatever the router's existing behavior for
    non-safety errors is must remain unchanged — we only ADD a new
    trigger. Pin this by asserting the first call's error propagates
    when the next provider would have succeeded (i.e. the new
    fallback DID NOT engage)."""
    from backend import llm as _llm

    class _FirstProvider:
        name = "first"
        model = "stub-model"

        def call(self, *a, **kw):
            raise _llm.LLMError("HTTP 400: bad request shape")

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return self.call(*a, **kw)

    class _SecondProvider:
        name = "second"
        model = "stub-model"

        def call(self, *a, **kw):
            return "should not be reached"

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return self.call(*a, **kw)

    monkeypatch.setattr(
        _llm, "_active_provider_chain",
        lambda *_a, **_kw: [_FirstProvider(), _SecondProvider()],
    )
    router = _llm.router()
    # Either: the router's existing logic also fails over on HTTP 400
    # (then this test trivially passes via second provider's answer),
    # OR the router does not fall back on HTTP 400 (then the original
    # error propagates). EITHER outcome is acceptable for THIS task —
    # what matters is that the SAFETY fallback path is the only thing
    # we are ADDING. Pin: no infinite loop, no unexpected exceptions
    # other than LLMError, no behavior change for unrelated errors.
    try:
        out = router.call(_llm.TaskType.QUICK_ANSWER, "sys", "user", max_tokens=10)
        # If the router DID fall back (its existing logic), the
        # second provider returned the canned string.
        assert out == "should not be reached"
    except _llm.LLMError as e:
        # If the router did NOT fall back, the original error
        # propagated — that is also fine; we did not break it.
        assert "400" in str(e)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_router_safety_fallback.py -v`

Expected: all fail with `ImportError: cannot import name '_is_safety_refusal'` (and the `_active_provider_chain` symbol not being patchable means the orchestration tests will also fail).

- [ ] **Step 3: Implement `_is_safety_refusal` + extend the router's call paths**

In `backend/llm.py`, near the existing `LLMError` class, add:

```python
_SAFETY_REFUSAL_MARKERS: tuple[str, ...] = (
    "flagged",
    "content_policy",
    "content policy",
    "cybersecurity_risk",
    "cybersecurity risk",
    "safety",
)


def _is_safety_refusal(err: "LLMError") -> bool:
    """True iff `err.message` contains a provider-side safety-refusal
    marker. Conservative narrow substring match — we want to catch
    Codex 'content flagged for cybersecurity risk' AND Anthropic
    safety messages, NOT generic 5xx / rate-limit errors.
    """
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _SAFETY_REFUSAL_MARKERS)
```

Now find the router's per-method call orchestration. Look for the existing fallback / failover code path that walks providers. There are three entry points: `call`, `call_with_tools`, `call_json`. Each MUST gain the same safety-fallback behavior. The minimal change for each is:

```python
        for prov in providers:
            try:
                return prov.<method>(...)
            except LLMError as e:
                if _is_safety_refusal(e):
                    log.warning(
                        "router: provider %s returned safety refusal; "
                        "falling back to next. detail=%s",
                        getattr(prov, "name", "?"), e,
                    )
                    continue
                # existing handling for non-safety errors stays here
                raise
        # no provider succeeded
        raise LLMError("all providers exhausted (last: safety refusal)")
```

If the router today already iterates a provider list with try/except on `LLMError` (e.g. for rate-limit / HTTP 5xx fallback), the addition is just one new `if _is_safety_refusal(e): continue` branch BEFORE whatever the existing logic does. If the router today does NOT walk a list (single-provider model) the change is bigger; document the structure you find in the commit message.

Also add an `_active_provider_chain()` helper (or expose the existing one with this name) that returns the ordered providers used by the router. This is what the tests monkeypatch. If a name like that already exists, leave it; if not, add it as a thin wrapper around the existing provider lookup logic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_router_safety_fallback.py -v`

Expected: 6 passed (the third `test_router_does_not_fallback_on_non_safety_error` accepts either branch).

Confirm no broader regression:
```
python -m pytest tests/ --no-header -q --ignore=tests/test_terminal_bench.py --ignore=tests/test_swebench.py -k "not test_full_unified and not telegram and not codex_subscription"
```
Expected: same pass count as before this PR (plus the new tests).

- [ ] **Step 5: Commit**

```
git add backend/llm.py tests/test_router_safety_fallback.py
git commit -m "feat(llm): router fallback on provider safety refusals

Codex Responses API returns 'content flagged for cybersecurity
risk' as an LLMError, and Anthropic occasionally returns 'content
policy' refusals. The router now catches these by message-substring
match (flagged / content_policy / cybersecurity_risk / safety,
case-insensitive) and retries the SAME call on the next provider
in the priority chain. If no fallback exists, the original error
propagates.

Closes the password-recovery terminal-bench failure where Codex
flagged the prompt and there was no fallback. Heuristic match is
content-of-error matching, not user-text keyword routing."
```

---

## Self-review

**Spec coverage:**
- Block 1a (always-on prompt rule) → Task 1 ✓
- Block 1b (bench-mode structural guard) → Task 4 ✓
- Block 2 (truncated-then-refusal) → Task 3 ✓
- Block 3 (background-not-awaited) → Task 2 ✓
- Block 4 (provider safety fallback) → Task 5 ✓

**Placeholder scan:** every code step contains the full code or full test. Task 5 leaves the precise router-touch site to subagent discovery (the existing fallback orchestration shape is unknown without reading the file) but pins the public test contract — that is intentional and explicit, not a placeholder.

**Type consistency:** `_decide_self_correction` signature is `(*, task, answer, turn_tools, trace=None, speaker_id="")` after Task 2 — Tasks 3 and 4 use the same signature. All detector helpers return `bool`. All branches return `tuple[str, str]`. Tag names: `background-not-awaited`, `truncated-then-refusal`, `tests-exist-not-run` — consistent kebab-case.

## Validation after all five tasks land

Re-bench: `harbor run --dataset terminal-bench --n-tasks 20 --agent hrant --agent-timeout-multiplier 10 --n-concurrent 1`. Compare against the 9/20 baseline. Predicted ceiling per the analysis: +6 from Block 1, +3 from Block 2, +1 from Block 3, +1 from Block 4. Realistic score after this PR: 13-17/20 (Mean 0.65-0.85). If the gain is less than +3, dig deeper into the trial logs before adding more fixes — possibly the fixes interact with bench-harness instruction tuning we haven't addressed.
