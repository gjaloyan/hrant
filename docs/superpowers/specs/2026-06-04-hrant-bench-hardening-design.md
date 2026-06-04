# Hrant bench-hardening — design

**Date:** 2026-06-04
**Status:** Spec (awaiting plan)

## Problem

Hrant scored 9/20 (Mean 0.450) on the 2026-06-04 real `--agent hrant` terminal-bench run. Per the failure analysis on the 11 fails, four orthogonal weaknesses account for **all** of them:

| # | Failure mode | Tasks affected |
|---|---|---|
| A | Self-verified weakly: ran only `compile` / sample check, never executed the real `/tests/` test suite before declaring done. | 6 |
| B | After a truncated `terminal_exec` output, the agent emitted a "no evidence X happened" refusal-shaped answer; the existing `self_correct` re-prompted with the same toolset and got the same refusal. | 3 |
| C | Spawned a long-running job (`nohup … &`) and returned immediately in the same turn; the job hadn't produced its artifacts when the turn synthesized. | 1 |
| D | Provider raised `LLMError: content flagged for cybersecurity risk` (Codex safety filter) and the router had no fallback. | 1 |

This spec fixes all four as one logical project: "harden Hrant on bench-style hidden-test environments." Each block is independently shippable; together they raise the score ceiling.

## Goal

Convert as many of the 11 failures as the diagnosis predicts into passes, without regressing the 9 currently-passing tasks. Realistic post-fix ceiling: 14-17/20 (Mean 0.70-0.85). Theoretical max: 20/20.

## Non-goals

- Bench-only logic. Each fix is universal where it makes sense; only the strictest variant (Block 1 structural guard) is gated to bench-harness sessions.
- Improving the underlying LLM. Out of scope — addressed only by re-running with different providers later.
- Concurrency for `--n-concurrent > 1` Harbor runs. Still serial.
- New tools. All four blocks reuse existing tools (`terminal_exec`, `read_file`, etc.).

## Architecture: four independent blocks

```
                                     ┌─────────────────────────────────┐
                                     │  backend/prompt_modules.py      │
                                     │  + M-verify module (Block 1a)   │
                                     └────────────┬────────────────────┘
                                                  │
            speaker=webui:bench-harness?          ▼
                  ┌─────────────────────────────────────┐
                  │  backend/unified_agent.py            │
                  │  ├─ _decide_self_correction          │
                  │  │   + truncated_then_refusal (B)    │
                  │  │   + background_not_awaited  (C)   │
                  │  │   + tests_exist_not_run     (A2)  │
                  └─────────────────────────────────────┘
                                                  │
                                                  ▼
                  ┌─────────────────────────────────────┐
                  │  backend/llm.py (router)             │
                  │  + safety-refusal fallback     (D)  │
                  └─────────────────────────────────────┘
```

Each branch in `_decide_self_correction` is one independent re-prompt trigger; they compose by **first-match-wins** ordering (most specific first: C → B → A2, then existing unbacked-claim / endpoint). Block D is handled OUTSIDE `_decide_self_correction` — it's a router-level fallback in `backend/llm.py`, so the safety refusal never reaches the self-correction phase.

## Block 1 — Verifier-aware self-verification

### 1a. Prompt rule (universal)

Add a new prompt module `M-verify` (or extend the existing core via prompt_modules) with:

> Before declaring a task done, run its actual test suite if one exists. If the workspace contains `/tests/`, `test_*.py`, a `Makefile`'s `test` target, `pytest.ini`, or `pyproject.toml` with `[tool.pytest.ini_options]`, you MUST execute it and observe a passing run before composing your final answer. "It compiles", "my sample input works", or "I checked the obvious case" are NOT verification — only the real test signal counts. If tests fail, fix and re-run.

Trigger: always-on. Cost: ~150 input tokens per turn. The rule is universal (good for non-bench too — many real-world tasks have tests).

### 1b. Structural guard (bench-harness only)

In `_decide_self_correction`, when `speaker_id == "webui:bench-harness"`:

- Scan `agent._trace` for any `terminal_exec` whose command head matches `ls /tests`, `find /tests`, `cat /tests/`, `head /tests/`, `tail /tests/`, OR a `read_file` with path under `/tests/`.
- Separately scan for any `terminal_exec` whose command CONTAINS `pytest`, `python -m pytest`, `python -m unittest`, `make test`, OR `bash /tests/*.sh`.
- If discovery happened AND no test-run command was issued → re-prompt: `"You discovered /tests/ but never ran the suite. Run the actual tests now and verify they pass before composing your final answer. If they fail, fix the cause and re-run — do not synthesize until tests pass."`

Wins over prompt-only: agent can't sneak a "trust me I checked" past it.

Pure deterministic check — no LLM judge needed (and no keyword routing for ROUTING; pattern matching here is verification, not routing).

## Block 2 — Truncated-output refusal recovery

Today, after `terminal_exec` truncates output to 1500 chars, the agent sometimes lands a final answer like "I don't have evidence file X was created" because the tool result it actually needed was past the cap. The existing `self_correct` then catches this as an unbacked claim and re-prompts — but with no specific guidance, the agent re-emits the same refusal.

Add a new branch in `_decide_self_correction`, BEFORE the existing zero-tool unbacked-claim branch:

- Detect: last `terminal_exec` result in `agent._trace` had `result_truncated=True`, AND the final answer contains one of: `no evidence`, `cannot confirm`, `cannot verify`, `не могу подтвердить`, `по предоставленному` (the existing refusal patterns from `unified_agent._REFUSAL_PHRASES`).
- Action: re-prompt with explicit recovery: `"Your previous terminal_exec output was truncated at 1500 chars and the part you needed didn't fit. Re-run the command with output narrowed: pipe through 'tail -200' / 'head -200' / 'grep -n PATTERN'. Read the actual evidence before composing the final answer."`
- Same one-shot re-prompt mechanic as existing self_correct (no infinite loops).

This converts B (3 tasks) without touching the existing unbacked-claim / endpoint branches.

## Block 3 — Background-process await

In `_decide_self_correction`, add a branch:

- Detect: any `terminal_exec` in this turn's trace has a command matching one of these deterministic patterns (NOT an LLM judge):
  - Trailing ` &` at end of command (one or more chars of whitespace before; explicitly NOT `&&` — match `[^&]\s+&\s*$`).
  - Token `nohup` at the start of a command segment (regex `(^|;|\|\||&&|;)\s*nohup\s+`).
  - Token `setsid` in the same position.
  - Token `disown` anywhere (it's only used for backgrounding).
- AND no later command in the trace polls / waits (no `wait`, no `while ps`, no `tail -f`, no `sleep N && ls <artifact>` shape).
- Action: re-prompt: `"You spawned a background process but never waited for it. Either 'wait $PID' / 'while ps -p $PID >/dev/null; do sleep 5; done' / poll for the expected artifact path. Then verify it actually exists before composing the final answer."`

Conservative on detection: better miss-and-fall-through than false-positive on a legitimate fire-and-forget (e.g. starting a webhook listener).

## Block 4 — Provider safety-refusal fallback

When the Codex Responses API returns a safety refusal (`LLMError` with message containing `flagged`, `content_policy`, `cybersecurity_risk`, `safety`), the router currently propagates the error and the bench trial dies with an exception (visible as `password-recovery` failure).

Add a fallback path in `backend/llm.py` (the router): catch this specific `LLMError` shape, log it, and retry the SAME call on the next configured provider in priority order (skipping codex). If no fallback is configured, propagate as before.

Heuristic match for "safety refusal" stays narrow (those four substrings, case-insensitive, in error message). This is content-of-error matching — explicitly NOT user-text keyword routing.

## Data flow (combined)

For a `webui:bench-harness` turn with a `/tests/` directory:

```
1. Agent.run starts → unified loop
2. System prompt includes M-verify rule (Block 1a)
3. Agent does terminal_exec, read_file, ... (normal loop)
4. (if codex flags) router catches LLMError, fails over to Anthropic (Block 4)
5. Tool loop completes, answer composed
6. _decide_self_correction runs in order:
     D (safety) ← already handled in step 4
     C (background not awaited) ← check trace
     B (truncated + refusal) ← check trace + answer
     A2 (tests exist + not run, bench-harness only) ← check trace
     existing (unbacked claim / endpoint)
   First match wins. One re-prompt cycle. Goes back through call_with_tools with the corrective.
7. Final answer written to logs_dir/agent_output.txt
8. Endpoint returns {ok, answer, ...}
9. Adapter writes answer to its output, harbor verifier scores against container state
```

## Component file map

| Block | File | Change |
|---|---|---|
| 1a | `backend/prompt_modules.py` | New `M-verify` module body |
| 1b | `backend/unified_agent.py` | `_decide_self_correction` new branch `tests_exist_not_run` |
| 2 | `backend/unified_agent.py` | `_decide_self_correction` new branch `truncated_then_refusal` |
| 3 | `backend/unified_agent.py` | `_decide_self_correction` new branch `background_not_awaited` |
| 4 | `backend/llm.py` | Router-level safety-error fallback |

All branches in `_decide_self_correction` share the same return type `(tag, corrective_text)` and the same one-shot semantics as the existing branches.

## Failure modes

| Scenario | Behaviour |
|---|---|
| Block 1b fires but `/tests/` is actually empty or not a real test suite (false positive) | Re-prompt tells agent to "run the tests"; agent runs `pytest /tests/`, gets "no tests collected" → that IS the run, branch doesn't fire again. Wasted ~1 minute. |
| Block 2 fires but truncation wasn't actually the issue | Re-prompt tells agent to re-run with narrower output. Agent does, sees nothing changed, composes a clearer refusal. No regression vs current. |
| Block 3 fires but agent had legitimate fire-and-forget (e.g. listener) | Re-prompt tells agent to wait. Agent can't wait (no expected artifact) → composes a clearer status. Acceptable false positive. |
| Block 4 fires on a non-safety error | Heuristic substrings are narrow; mismatch means no fallback, original error propagates. No regression. |
| Re-prompt itself fails | Same as existing self_correct — keep original answer. |

## Testing strategy

Each block gets unit tests in `tests/`:

- Block 1b: helper that scans a fake trace; assert it returns `(tag, text)` when tests were discovered but not run, and `("","")` otherwise.
- Block 2: helper that takes (final_answer, last_truncated_flag) and returns `(tag, text)` when both signals fire.
- Block 3: helper that scans a fake trace for the trailing-`&` / `nohup` shapes; positive and negative cases.
- Block 4: monkeypatch the codex provider to raise safety-error LLMError; assert the router retries on the next provider.

Integration validation: re-bench on the same 20 tasks. Score should rise from 9/20 to ≥13/20 (the analysis predicts up to +6 from Block 1, +3 from Block 2, +1 each from Blocks 3 and 4, before model-side variance). If the gain is less than +4, that's a signal to dig deeper before celebrating.

## Out of scope (track separately)

- LLM-judge mode for `_decide_self_correction` instead of pattern matching (more accurate but cost-heavier).
- Bench harness instruction prompt tuning (the instruction passed to the agent — currently lives in the harbor adapter; could be tightened to mention `/tests/` explicitly).
- Tighter `/tests/` discovery heuristic (e.g. parse `pyproject.toml`).
- Reward >0.5 partial-credit interpretation (currently all-or-nothing 0/1 in this dataset).
