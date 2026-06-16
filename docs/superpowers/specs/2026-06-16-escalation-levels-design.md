# Escalation Levels — Design

**Date:** 2026-06-16
**Status:** approved (brainstorm), pending implementation plan

**Goal:** Make the agent's turn-routing weight *explicit* as a small set of
levels (L0/L1/L2 + an orthogonal async flag), consolidating today's scattered
conditional gates into one auditable source of truth, and skip the claim
verifier on pure-action turns where there is nothing to verify.

**Architecture:** A new `backend/escalation.py` owns the `Level` model and the
deterministic policy. `unified_agent.run_unified()` consults it instead of the
ad-hoc `elif tool_outputs:` verifier gate. No new LLM call is introduced; the
level is derived from the fast-path decision plus the tool classes that
actually ran.

**Tech stack:** Python, existing `unified_agent`/`verifier`/`endpoint_check`
modules, pytest.

---

## 1. Background — what the audit found

This design was scoped *after* auditing the real pipeline
(`unified_agent.py:2945–3078`, `verifier.py:474–558`, `endpoint_check.py:68–135`).
Key findings — the heavy machinery is **already mostly conditional**:

| Stage | Real firing condition (today) | LLM call on a plain `save_user_fact` turn |
|---|---|---|
| solver loop | always | 2 (decide → finalize) |
| **verifier** | `elif tool_outputs:` — any tool produced output | **1 — the only avoidable one** |
| endpoint_met | always; deterministic `return True` when an execute-class tool ran | 0 |
| answer_critic | `should_critique(vr)` — only on content problems | 0 |
| skill_reflection | `≥3 distinct tools` + confidence floor + endpoint met | 0 |

So the only avoidable inefficiency is the **verifier firing on pure-action
turns** (remember / set-setting / schedule), where `endpoint_met` already
confirmed delivery deterministically and there are no substantive claims to
ground. The prior "everything runs the full stack" premise was inflated.

The user's intent is therefore **clarity first**: turn the four scattered
gates into one explicit ladder, with the verifier-skip as the single real
behavioral change.

## 2. The level model

```python
class Level(IntEnum):
    L0_CHAT = 0     # direct answer, no tools
    L1_ACTION = 1   # tool turn, pure state-mutation, nothing to fact-verify
    L2_TASK = 2     # tool turn that produced assertable information
```

There is **no L3 policy row.** "Deep / background" is an *orthogonal async
dimension*, not a lighter tier:

- The **dispatch turn** (synchronous) only calls an execute-class job tool
  (`start_background_job`, `schedule_message`) — it is an ordinary **L1**
  action ("started job X", confirmed by `endpoint_met`).
- The **execution** runs later in a background thread that **re-enters
  `run_unified()`** with `supervisor_mode=True` (`job_supervisor.py:42`). It is
  a fresh turn that classifies on its own merits — deep research uses
  information tools, so it lands on **L2 with the full verification stack**.
  Async work is verified *more*, never less.

### Per-level policy (the single source of truth)

| Level | solver | verifier | endpoint | critic | reflection |
|---|---|---|---|---|---|
| L0 chat | 1 no-tool call | no | no | no | no |
| L1 action | tool loop | **no** | yes (deterministic) | no (self-gated) | no (self-gated) |
| L2 task | tool loop | **yes** | yes | self-gated | self-gated |

Only the **verifier** column is a new behavioral gate. `critic`, `reflection`
and `endpoint` keep their existing self-gates — those gates are already
level-appropriate, and the policy table only *documents* them.

## 3. Level assignment (hybrid, no extra LLM call)

- **L0** — the existing fast-path (`_try_chat_path`). If it answers directly →
  L0, done. If it escalates (`ESCALATE:`, tool-dump guard, save-claim guard) →
  the turn proceeds to the tool path at L1+.
- **L1 vs L2** — decided **deterministically from the tools that ran**, after
  the loop, with no extra call:
  - If the trace is **non-empty and every tool is a pure-action tool** (§4) →
    **L1**: skip the verifier. (Pure-action tools are execute-class, so
    `endpoint_met` is already True by construction — no need to read it.)
  - Otherwise (any information-producing tool ran) → **L2**: run the verifier
    as today.
- **Upward ratchet (one-way):** the L0→L1+ boundary is the ratchet — the
  fast-path's existing escalate guards (`ESCALATE:`, tool-dump, save-claim)
  push a turn off L0 into the tool path. Levels never ratchet down. (Future
  refinement: a solver mid-loop `ESCALATE` could force L2; not in v1, see §7.)

This is faithful to the approved "classifier up front (the L0 fast-path) +
deterministic signal by fact (L1↔L2) + ratchet up" decision. Note `endpoint_met`
is deliberately NOT an input to the level decision — it is computed *after* the
verifier block today (`unified_agent.py:2995`), and pure-action ⟹ endpoint-met
makes it redundant, so the decision avoids that ordering dependency entirely.

## 4. The pure-action tool set

The L1-skip signal needs a **curated set distinct from**
`endpoint_check._EXECUTE_TOOLS`. That set is the *delivery* set and wrongly
includes information-producing executors (`sandbox_exec`, `agent_browser`,
`delegate`) whose output carries verifiable claims; it also omits
`terminal_exec`. We define a deliberate set in `escalation.py`:

```python
# Tools whose ONLY effect is a state mutation / confirmation — a turn that
# ran exclusively these has no assertable factual claims, so the claim
# verifier has nothing to check. Deliberately NOT endpoint_check._EXECUTE_TOOLS
# (which includes info-producing executors like sandbox_exec / agent_browser /
# delegate, and omits terminal_exec).
_PURE_ACTION_TOOLS: frozenset[str] = frozenset({
    "save_user_fact", "set_setting", "schedule_message",
    "start_background_job", "define_task_endpoint", "complete_supervisor",
    "kick_supervisor", "grant_telegram_access", "revoke_telegram_access",
    "approve_pairing", "propose_skill", "propose_self_modification",
    "ask_user",
})
```

`save_to_workspace` / `save_knowledge` are intentionally treated as
information-bearing (their content can be wrong), so a turn that writes them
still gets verified.

## 5. Components & changes

- **New `backend/escalation.py`:**
  - `Level(IntEnum)` — L0/L1/L2.
  - `_PURE_ACTION_TOOLS` (§4).
  - `decide_level(*, was_fast_chat, tool_names) -> Level` — pure function, no
    I/O. `was_fast_chat=True` → L0; else L1 if `tool_names` is non-empty and
    every name is in `_PURE_ACTION_TOOLS`; else L2.
  - `should_run_verifier(level) -> bool` — `level >= Level.L2_TASK`.
- **`unified_agent.py`:**
  - The verifier gate at line ~2954 changes from `elif tool_outputs:` to
    `elif tool_outputs and should_run_verifier(level):`. `level` is computed
    just before the verifier block from `decide_level(was_fast_chat=False,
    tool_names=...)`. The `_trace_tool_names` extraction that today lives in
    the endpoint block (`:2979–2994`) is hoisted above the verifier gate (or
    factored into a small helper) so both blocks share it.
  - The turn artifact gains a `"level"` field (e.g. `"L1_ACTION"`) for the dev
    panel / audits. The fast-path artifact (line ~2024) sets `"L0_CHAT"`.
- **No change** to `answer_critic`, `_post_turn_skill_reflection`, or
  `endpoint_check` firing logic — only documented as part of the policy.

## 6. Data flow

```
turn → fast-path (_try_chat_path)
        ├─ answers → artifact.level = L0_CHAT  (verifier never considered)
        └─ escalates → tool loop runs
                         → collect trace tool_names + endpoint_met + escalate?
                         → level = decide_level(...)
                         → verifier runs iff should_run_verifier(level)
                         → endpoint cap / critic / reflection: unchanged self-gates
                         → artifact.level = L1_ACTION | L2_TASK
```

## 7. Out of scope (v1)

- **No extra LLM classifier call.** The level is the fast-path decision plus a
  deterministic post-hoc read of the trace.
- **No per-level solver iteration cap.** Splitting the cap (L1 cap~3 vs L2
  full) would require an *upfront* level, which needs the classifier we
  agreed not to add. The loop keeps its single cap; the level only gates the
  verifier in v1. (Possible future: have the fast-path call return a level
  hint at no extra cost.)
- **No new L3 machinery.** Background dispatch/execution already work; L3 is a
  label, not new code.
- `answer_critic` / `skill_reflection` / `endpoint_check` gates are unchanged.

## 8. Testing

- **Unit (`tests/test_escalation_levels.py`):**
  - `decide_level`: `["save_user_fact"]` → L1; `["save_user_fact",
    "web_search"]` → L2; `["terminal_exec"]` → L2; `["sandbox_exec"]` → L2
    (info-producing executor, NOT pure-action); `[]` with `was_fast_chat=True`
    → L0; `[]` with `was_fast_chat=False` → L2 (no tools but escalated off the
    fast path — verify the reasoned answer).
  - `should_run_verifier`: L0/L1 → False, L2 → True.
- **Regression (real-pipeline, mocked LLM):** a `save_user_fact`-only turn does
  NOT call the verifier (`verifier.verify` patched + asserted not-called); an
  information turn (`web_search`) DOES. Guards the one behavioral change.
- Full existing suite stays green (the change is gated and additive).

## 9. Risks

- **Under-verifying a bad action turn.** Mitigation: `endpoint_met` still runs
  on every turn and still caps confidence on delivery failures; only the
  *claim* verifier is skipped, and only when nothing claim-bearing ran.
- **A pure-action turn that also makes a side claim in prose.** Rare; if the
  answer asserts a fact beyond the confirmation it should be L2. v1 accepts the
  small risk (the answer of an action turn is a confirmation by construction);
  a future refinement can scan for claim-shaped prose.
- **Mis-curated `_PURE_ACTION_TOOLS`.** It is an explicit, tested list; adding a
  new mutation-only tool means adding it here, documented at the definition.
