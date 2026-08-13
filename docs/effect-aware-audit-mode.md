# Effect-aware audit mode

`POST /api/chat` accepts `"audit_mode": true` for diagnostic turns that must
inspect without changing agent or external state.

```json
{
  "message": "Inspect service health and report anomalies",
  "audit_mode": true
}
```

The boundary has two layers:

1. The tool schema exposes only tools declared audit-visible.
2. `ToolRegistry.execute()` resolves the concrete call effect and rejects
   anything other than an allowed read before invoking its handler.

Every tool has typed semantics in `backend/tool_registry.py`:

- `read`: observation only;
- `control`: in-turn orchestration;
- `write`: local/persistent state change;
- `external`: a call that can change state outside the process;
- `unknown`: undeclared, and therefore blocked in audit mode.

Dual-use tools may provide an argument resolver. `terminal_exec`, for example,
classifies a strict allowlist of commands such as `systemctl status`, `git
status`, and `journalctl` as reads. Redirects, shell composition, restarts,
installs, and ambiguous commands resolve to `write` and fail closed.
Mixed-purpose APIs are treated the same way: `soul_history(action="list")` is
a read, while `action="restore"` is blocked. Inspection helpers that perform
maintenance writes (`check_subagents`, pending-pairing garbage collection) are
not exposed in audit mode.

Audit turns also disable action-pressure nudges and skip cognitive persistence:
conversation history, extracted memory, sessions, turn/trajectory artifacts,
goals, evaluator rows, fine-tune collection, and meta-learning. They bypass the
durable Job wrapper as well. Provider token/cost counters and ordinary process
logs remain operational telemetry; audit mode is not a promise of zero I/O at
the infrastructure layer.

Tool traces and durable non-audit Job receipts carry the resolved `effect`, so
proof, framing, action-drift, debugging, and future policy decisions can use one
deterministic signal instead of independent tool-name lists.

## Capability broker and bounded audit execution

`backend/capability_broker.py` is the per-turn authority for both the schema
shown to the model and admission of each concrete call. It derives the current
base/bundle/runtime tool surface, applies the typed-effect audit filter, and
checks the turn budget before the handler can run. The registry retains its
own read-only guard as defense in depth.

Normal work keeps the owner's deliberately generous policy: 500 configurable
iterations, with tool-call and input-token limits disabled by default. Explicit
audits use a separate bounded profile:

- `router.audit_loop_max_iterations`: 32;
- `router.audit_loop_max_tool_calls`: 32;
- `router.audit_loop_input_budget`: 60000 accumulated input tokens.

The values are runtime-configurable through the existing engine-config API.
`router.tool_loop_max_tool_calls` is also available as an opt-in emergency cap
for normal turns; its default is `0` (disabled).

Once a hard audit budget is reached, the broker returns
`TURN_BUDGET_EXCEEDED` without invoking the handler and instructs the model to
produce an evidence-based partial report that names what remains unchecked.
Every unified tool-loop response includes an `execution_budget` receipt with
limits, attempted/allowed/denied calls, observed input tokens, and exhaustion
reason. Normal durable Jobs and turn artifacts persist the same receipt.
