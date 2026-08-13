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
