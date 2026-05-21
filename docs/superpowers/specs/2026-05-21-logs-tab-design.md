# Logs Tab — Design Spec

**Status:** approved 2026-05-21
**Author:** Claude (brainstorming session)
**Scope:** WebUI Logs tab — unified live event stream with filtering, search, download.
**Out of scope (future, separate spec):** Pipeline Settings (profiles, system prompt editor, section toggles, LLM validation, version history) — phased after this lands.

## Goal

Give the owner a single place in the WebUI to see what the agent is doing right now — Python logging output, tool calls, job status transitions, supervisor events, and the agent's own progress signals — all in one timeline, filterable by level/source, searchable, downloadable.

Existing surfaces (Jobs tab, Chat ToolCallCards) show pieces but not a unified live feed. systemd journal has Python logging but you need SSH access to read it.

## Data Model

```python
@dataclass
class LogEvent:
    ts: float          # unix timestamp, ms-precision (time.time())
    level: str         # debug | info | warning | error | critical
    source: str        # python | tool | job | supervisor | agent
    logger: str        # python logger name OR component identifier
    message: str       # main human-readable text
    meta: dict         # source-specific: tool_name+args, job_id, speaker_id, etc.
    request_id: str    # ties events to a turn — uses agent._last_turn_id when set
```

Single shape for all sources — the `source` field is both filter and color code.

## Sources

Five publishers push into one bus:

1. **python** — custom `logging.Handler` (`LogBusHandler`) attached to the root logger in `backend/bootstrap.py`. Every existing `log.info()` / `log.warning()` / `log.error()` call in the codebase becomes a `LogEvent` as a side effect — no per-call instrumentation required.
2. **tool** — `_on_tool_call` callback in `unified_agent.py` already fires for every tool invocation; add a `LogBus.publish(...)` line there. Carries `tool_name`, args preview, result preview, `is_error`.
3. **job** — `backend/jobs.py:JOBS.set_status` is the single chokepoint for job state transitions (queued→running→done/error). Add `LogBus.publish(...)` there. Carries `job_id`, `from_status`, `to_status`, `error_message`.
4. **supervisor** — `backend/job_supervisor.py` heartbeats + final-decision emissions. Carries `job_id`, decision (done/escalate/retry).
5. **agent** — `agent.progress(event, msg)` already emits SSE events into the chat stream. Add a side-publish to LogBus so the same event lands in the Logs tab without re-instrumentation. Carries `event` (think/solve/verify/synth/tool/etc), `request_id`.

## Storage

### In-memory ring

`collections.deque(maxlen=20000)` protected by `threading.RLock`. Cheap appends + cheap snapshot via `list()`. Bounded so memory is predictable.

### Rotating JSONL file

One file per day: `<data_dir>/logs/agent-YYYYMMDD.jsonl`. Each line is `LogEvent.to_dict()` JSON. Append-only — no in-place mutation. Daily file rolls at first publish after midnight (UTC).

### Retention

7 days. Cleanup hook in the existing `bg_job_watchdog._gc_sweep_if_due` daily sweep so we don't add a new scheduler.

## Backend Module Map

- `backend/log_bus.py` (new, ~250 LOC)
  - `LogEvent` dataclass
  - `LogBus` class: `publish(event)`, `subscribe() -> asyncio.Queue`, `unsubscribe(q)`, `tail(...)`, `gc_old(days)`
  - `LogBusHandler(logging.Handler)` — bridges Python logging into the bus
  - Module-level `BUS` singleton
  - File writer (JSONL, atomic append, daily rotation)

- `backend/bootstrap.py` — attach `LogBusHandler` to root logger after standard logging setup.

- `backend/unified_agent.py` — add 2 lines in `_on_tool_call` to publish a tool event.

- `backend/jobs.py` — add 1 line in `set_status` to publish a job event.

- `backend/job_supervisor.py` — add publish calls at heartbeat + decision points.

- `backend/agent.py` — `progress(...)` adds a side-publish to LogBus.

- `backend/api/logs.py` (new, ~150 LOC)
  - `GET /api/logs` — REST snapshot with query params: `level`, `source`, `search`, `limit` (default 500), `before_ts`
  - `GET /api/logs/stream` — SSE live feed via `EventSourceResponse` (same pattern as `/api/chat`)
  - `GET /api/logs/download?format=jsonl|txt` — dump current ring buffer
  - `GET /api/logs/sources` — static enum lists for UI dropdowns
  - All endpoints owner-gated via `require_owner_for_writes`/equivalent reader gate

## Frontend Module Map

- `frontend/src/components/settings/LogsTab.tsx` (new, ~400 LOC)
  - Lazy-loaded via `SettingsPanel.tsx` like other tabs
  - State: `events: LogEvent[]`, `streaming: boolean`, `autoScroll: boolean`, `level`, `source`, `search`
  - EventSource subscription on mount, closed on unmount
  - Virtualized rendering — slice last 250 visible events, recycle DOM nodes (no react-window dependency needed; simple sliding window)
  - Auto-scroll: enabled by default; if user scrolls up, disable auto-scroll and show "▼ Jump to live" button
  - Pause: stop appending to visible state; queue continues collecting; resume flushes the queue
  - Filters apply client-side for instant response + same filters passed on `/api/logs` for initial backfill
  - Search: case-insensitive substring across `message + logger + JSON.stringify(meta)`
  - Timestamps rendered as `HH:MM:SS.mmm` (local time, ms-precision)
  - Copy button: copies the currently-visible filtered view to clipboard as plain text (one line per event, same format as the file dump)

- `frontend/src/api.ts` — typed API client for the four endpoints + `LogEvent` type.

### Layout

```
┌─ toolbar ─────────────────────────────────────────────────┐
│ [▶/⏸ live] [↓ auto-scroll] level: [all▼] source: [all▼]  │
│ search: [____________] [clear] [⬇ download] [⋮ copy]      │
└────────────────────────────────────────────────────────────┘
┌─ log stream (virtualized, monospace) ─────────────────────┐
│ 10:42:17.123 INFO  unified_agent   Phase 1: routing...    │
│ 10:42:17.245 TOOL  read_file       (file=...) → 1.2KB     │
│ 10:42:17.401 WARN  llm.codex       rate limit 429 (retry) │
│ 10:42:17.890 ERROR job_supervisor  job 2b7d failed: ...   │
│ 10:42:18.005 INFO  agent           verify: 85% confidence │
└────────────────────────────────────────────────────────────┘
```

### Colors

Tailwind classes (match the rest of the WebUI):

- **Level:** debug → `text-slate-500` · info → `text-slate-200` · warning → `text-amber-300` + `bg-amber-950/30` · error → `text-rose-300` + `bg-rose-950/30` · critical → `text-rose-100` + `bg-rose-900/60` + subtle pulse
- **Source pill:** tool → `violet` · job → `emerald` · supervisor → `indigo` · python → `slate` · agent → `sky`

## Error Handling

- **Slow SSE client:** each subscriber owns its `asyncio.Queue(maxsize=1000)`. If the queue fills, the bus drops new events for that subscriber (with a "dropped N events" sentinel) instead of blocking the publisher. Production clients are local-net only so this is defensive against pause-tabs.
- **File write failure:** logged once at warning level (without recursing into the bus — guard via a private flag), in-memory ring still works.
- **Handler attached during shutdown:** `LogBusHandler.emit` catches all exceptions and silently drops — never let logging crash the agent.
- **Disk full / dir missing:** writer skips persistence, bus continues in-memory. Health endpoint surfaces a flag.

## Testing

- `tests/test_log_bus.py` — `publish`/`subscribe`/`tail`/`gc_old` + level/source/search filter
- `tests/test_log_bus_handler.py` — Python logging integration: `log.warning(...)` arrives in bus with correct level/logger
- `tests/test_log_bus_persistence.py` — JSONL writer + daily rotation + 7-day retention sweep
- `tests/test_api_logs.py` — REST snapshot + SSE stream + download formats + owner gate
- `tests/test_log_bus_concurrency.py` — concurrent publish + multiple subscribers, no race / no drops in normal load

## YAGNI (explicit non-goals)

- ❌ Server-side regex search (substring is enough)
- ❌ Per-user log isolation (single-tenant box)
- ❌ Log compression
- ❌ External sinks (Loki / Datadog / etc) — separate spec when needed
- ❌ Real-time histograms / charts (raw stream only)
- ❌ Editing / acknowledging events (read-only feed)

## Files Touched (summary)

- New: `backend/log_bus.py`, `backend/api/logs.py`, `frontend/src/components/settings/LogsTab.tsx`, 5 test files
- Modified: `backend/bootstrap.py` (attach handler), `backend/agent.py` (progress side-publish), `backend/unified_agent.py` (tool side-publish), `backend/jobs.py` (status side-publish), `backend/job_supervisor.py` (decision side-publish), `backend/bg_job_watchdog.py` (gc hook), `backend/main.py` (router include), `frontend/src/components/SettingsPanel.tsx` (lazy-load tab), `frontend/src/api.ts` (client + types)
