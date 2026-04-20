# D-08 — TOOL_INSTALL + API expansion (design)

**Status:** Design (no implementation)
**Date:** 2026-04-20
**Parent:** [Model X spec, Section 11 — phased delivery](./2026-04-16-model-x-autonomic-design.md)
**Delivers:** `FIRE_TOOL_INSTALL` (yellow, 19th lever), yellow-approval flow, `backend/autonomic/api.py` expansion with 7 new endpoints.

---

## 0. Context

D-01 through D-07 are merged. 18 of the 19 Model X levers are live (4 immune + 14 autonomic). The remaining lever, `FIRE_TOOL_INSTALL`, is the only **yellow-safety** one in the catalog — execution requires user approval via `pending_approvals.jsonl`. Until now the SafetyGate only queued yellow actions; D-08 adds the approve-and-execute flow that completes the safety contract defined in D-01.

**Scope split note:** parent Section 11 originally bundled `FIRE_TOOL_INSTALL` + AutonomicPanel frontend + API expansion into a single D-08. Brainstorming on 2026-04-20 split this into two sub-projects — D-08 (backend: lever + approval flow + API expansion, this spec) and D-09 (frontend: AutonomicPanel.tsx + StatusBar indicator, consumes D-08 APIs). Reasoning: backend and frontend have different test/review contexts; frontend cannot start until the APIs exist. Same decomposition pattern as D-04/D-05 and D-06/D-07.

**Goal:** finish the 19-lever catalog backend-first. Ship the yellow lever, the approve/reject flow through the `SafetyGate`, and the HTTP surface that a future AutonomicPanel (D-09) will consume.

**Non-goals:** AutonomicPanel UI (D-09), StatusBar indicator (D-09), Linux OS inventory extras (deferred to a future "Linux deploy support" project — Windows dev machine means no present value), L2 / L3 auto-triggering of TOOL_INSTALL (no scheduled rule in D-08; the lever is only enqueued by explicit user POST or future cortex hook).

---

## 1. `FIRE_TOOL_INSTALL` (yellow, python)

First yellow-safety lever in the project. When the scheduler or an API caller fires it, `SafetyGate.evaluate` queues it to `pending_approvals.jsonl` instead of running. Actual execution happens only on explicit user approve (Section 2).

**Whitelist** (hard-coded, not parameterised):

| Command | Params | Action |
|---|---|---|
| `pip_install` | `package: str` | `sys.executable -m pip install <package>` |
| `ollama_pull` | `model: str` | `ollama pull <model>` (graceful FAILURE if ollama binary missing) |
| `llama_cpp_pull` | `url: str` | HTTPS chunked download via `httpx.stream` → `models/llama_cpp/<filename>.gguf` |

**No uninstalls, no removes.** Destructive commands are deliberately out of scope — a user who accidentally approves a queued action should not lose packages or models.

**Validation in `preconditions` (before gate):**
- `command in {"pip_install", "ollama_pull", "llama_cpp_pull"}`.
- `package` / `model` / `url` present, non-empty string.
- For `llama_cpp_pull`: URL starts with `https://`, URL basename ends in `.gguf`, basename contains no path-traversal segments (`..`, `/`, `\` after basename extraction).

If validation fails → `LeverStatus.BLOCKED_BY_SAFETY` with descriptive reason. This fires before the gate, so a malformed queued entry still gets rejected at approve-time.

**Execution (`run`):**

1. `command == "pip_install"`: `subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True, text=True, timeout=300)`.
2. `command == "ollama_pull"`: `subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=1800)`. If `FileNotFoundError` → return `LeverStatus.SKIPPED` with `reason="ollama_not_installed"`.
3. `command == "llama_cpp_pull"`:
   - `dest_dir = Path(params.get("dest_dir") or "models/llama_cpp")`.
   - `filename = url.rsplit("/", 1)[-1]` — sanitized.
   - `dest = dest_dir / filename`.
   - Stream download via `httpx.stream("GET", url, timeout=httpx.Timeout(1800.0), follow_redirects=True)` → chunked write.
   - On success: record `dest` path + file size.
   - On HTTP non-2xx: `FAILURE` with `reason=f"http_{status_code}"`.

Return `LeverReport` with `outcome = {command, target, rc, stdout_tail: str[-500:], stderr_tail: str[-500:], dest_path?}`.

**Meta:**
- `safety = LeverSafety.YELLOW`
- `executor = "python"`
- `category = LeverCategory.BODY` (new-ish — the spec catalog puts TOOL_INSTALL under "Тело и среда").

**No scheduled rule in D-08.** The lever is only triggered by explicit `POST /api/autonomic/pending` (Section 3). Future cortex hooks can enqueue programmatically; for v0 the user drives it.

---

## 2. Approval flow — SafetyGate id + LeverExecutor bypass

The `SafetyGate` from D-01 currently queues yellow entries without an addressable identifier. D-08 adds a short UUID and wires the approve/reject endpoints to execute approved entries through a bypass path on `LeverExecutor`.

### 2.1 `SafetyGate` changes

Current `_queue` writes to `pending_approvals.jsonl`:
```json
{"lever": "...", "params": {...}, "requested_at": "...", "status": "pending"}
```

Change: prepend a hex-12 id (via `secrets.token_hex(6)`):
```json
{"id": "a3f29b0e5c1d", "lever": "...", "params": {...}, "requested_at": "...", "status": "pending"}
```

- `list_pending()` already returns these as dicts — id comes through automatically.
- `_queue` returns the id (change signature from `None` → `str`). `SafetyGate.evaluate` currently returns `SafetyDecision` — unchanged; the id is retrievable via `list_pending()` after evaluate.

**Backward compatibility:** the current `pending_approvals.jsonl` is empty in the repo (no real user data). Entries without `id` during early runtime would be handled by giving them a synthetic id on first `list_pending()` read — but since the file is empty, we don't need that path. Plain and simple: all new entries have id.

### 2.2 `LeverExecutor` changes

Add `bypass_safety: bool = False` kwarg to `execute`:

```python
def execute(
    self,
    lever: Lever,
    params: dict[str, Any],
    state: StateSnapshot,
    *,
    bypass_safety: bool = False,
) -> LeverReport | None:
    if not bypass_safety:
        decision = self._gate.evaluate(lever, params)
        if decision is SafetyDecision.BLOCK:
            ...
        if decision is SafetyDecision.QUEUE_FOR_APPROVAL:
            ...
    # preconditions + run + persist as today
```

When `bypass_safety=True`, skip `gate.evaluate` entirely. The rest of the path (preconditions → run → persist → event) is unchanged. No safety-level coupling inside `run`; the contract is "caller accepts responsibility."

**Who sets `bypass_safety=True`:** only the approve-endpoint after verifying the entry was actually queued. The scheduler tick path stays `bypass_safety=False`.

### 2.3 Approve flow (sketch, full endpoint spec in Section 3)

```python
def approve(id: str):
    entries = gate.list_pending()
    entry = next((e for e in entries if e.get("id") == id), None)
    if entry is None:
        raise HTTPException(404, "pending entry not found")
    lever = registry.get(entry["lever"])
    if lever is None:
        raise HTTPException(400, f"lever not registered: {entry['lever']}")
    state = builder.build()
    report = executor.execute(lever, entry["params"], state, bypass_safety=True)
    # remove entry from pending_approvals.jsonl
    gate.remove_pending(id)
    return report
```

**New `SafetyGate.remove_pending(id: str) -> bool`** — rewrites `pending_approvals.jsonl` excluding the matching id. Returns True if removed. Atomic: write to `.tmp`, `os.replace`.

### 2.4 Reject flow

Same shape as approve but no execution:

```python
def reject(id: str):
    removed = gate.remove_pending(id)
    if not removed:
        raise HTTPException(404, "pending entry not found")
    return {"ok": True, "rejected_id": id}
```

---

## 3. `/api/autonomic/*` expansion — 7 new endpoints

Existing router (`backend/autonomic/api.py`) has `GET /status`. D-08 adds:

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/autonomic/ticks?limit=50` | — | `{"ticks": [...]}` — last N lines from `tick_log.jsonl`, newest-first. Default limit=50, max=500. |
| `GET` | `/api/autonomic/levers/{name}?limit=10` | — | `{"lever": name, "reports": [...]}` — last N `LeverReport` dicts for that lever from `lever_log.jsonl`. 404 if lever name not registered. |
| `GET` | `/api/autonomic/pending` | — | `{"pending": [...]}` — full list from `pending_approvals.jsonl`. |
| `POST` | `/api/autonomic/pending` | `{"lever": str, "params": dict}` | Enqueue. 404 if lever not registered. 400 if lever is green (`"use direct POST instead"`) or red. Returns `{"id": "...", "status": "queued"}` on success. |
| `POST` | `/api/autonomic/pending/{id}/approve` | — | 404 if id missing. Otherwise executes with `bypass_safety=True` and returns the `LeverReport` dict. Entry removed from pending. |
| `POST` | `/api/autonomic/pending/{id}/reject` | — | 404 if id missing. Otherwise removes entry, returns `{"ok": true, "rejected_id": id}`. |
| `GET` | `/api/autonomic/immune` | — | `{"signatures": [...]}` — full list from `SignatureStore.load()`. |
| `POST` | `/api/autonomic/kill-switch` | `{"enabled": bool}` | Write `"true"` or `"false"` into kill-switch file via `KillSwitch.enable/disable`. Returns `{"enabled": bool}`. |

Total = 8 endpoints (existing `/status` plus 7 new).

**Request validation via Pydantic:** each POST body is a small BaseModel subclass in the same file. Path params validated by FastAPI.

**Dependency injection:** endpoints need access to:
- `SafetyGate` (for pending list + remove)
- `LeverRegistry` (for lever lookup)
- `LeverExecutor` (for approve execution path)
- `StateSnapshotBuilder` (for state at approve time)
- `KillSwitch` (for toggle)
- Log paths (tick_log.jsonl, lever_log.jsonl)

Current `api.py` reads `request.app.state.autonomic_scheduler` and the `KillSwitch(DEFAULT_ENABLED_PATH)`. Simplest pattern: expose these on `app.state` during `startup.build_scheduler` (we already stash `autonomic_scheduler` there). Add `app.state.autonomic_gate`, `app.state.autonomic_executor`, `app.state.autonomic_builder`. The api.py module reads them via `request.app.state`.

**Log-path discovery:** kept consistent with startup env vars — api module re-reads the same env vars (`AUTONOMIC_LEVER_LOG_PATH`, `AUTONOMIC_TICK_LOG_PATH`) with the same defaults. DRY: a tiny `_env_path()` helper lives in api.py or is imported from startup.py.

**Error handling:** FastAPI's standard `HTTPException(status_code, detail)` — no custom middleware. 404 for missing id / lever; 400 for bad safety or malformed body; 500 only on true infrastructure failures.

---

## 4. File layout + modifications

**New files:**

```
backend/autonomic/levers/
└── tool_install.py             # FIRE_TOOL_INSTALL (~150 lines)

tests/autonomic/
├── test_tool_install.py        # 8 lever-level tests
└── test_api.py                 # 13 API tests via FastAPI TestClient
```

**Modified files:**

- `backend/autonomic/safety.py` — `_queue` generates and writes id; `remove_pending(id)` new method; `SafetyGate.evaluate` return unchanged.
- `backend/autonomic/executor.py` — `execute(..., *, bypass_safety=False)` kwarg.
- `backend/autonomic/api.py` — 7 new endpoints.
- `backend/autonomic/startup.py` — stash `gate`, `executor`, `builder` on `app.state` for api.py.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers()` adds FIRE_TOOL_INSTALL (14 → 15).
- `backend/main.py` — pass the application instance into `build_scheduler` (or have startup write to `app.state` via the existing wiring). Implementation detail captured in the plan.
- `tests/autonomic/test_executor.py` — new test for `bypass_safety=True` path.
- `tests/autonomic/test_safety.py` — tests for id assignment + `remove_pending`.
- `tests/autonomic/test_registry.py` — assert `FIRE_TOOL_INSTALL` registered.
- `tests/autonomic/test_api.py` (new) — `/status` test migrates here alongside 13 new tests.
- `README.md` — document TOOL_INSTALL whitelist + new endpoints.

**Not modified:** `scheduler.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `layer0.py` (no new rule), `frontend/` (D-09's concern), existing D-01..D-07 levers.

**Runtime artifacts potentially touched:**
- `knowledge/autonomic/pending_approvals.jsonl` — id field on new entries.
- `knowledge/autonomic/ENABLED` — toggled by kill-switch endpoint.
- `models/llama_cpp/*.gguf` — created on successful `llama_cpp_pull`.

---

## 5. Testing strategy

### 5.1 Lever tests — `tests/autonomic/test_tool_install.py` (8 tests)

- metadata (`name=FIRE_TOOL_INSTALL`, `category=BODY`, `safety=YELLOW`, `executor=python`).
- `preconditions` True for valid `pip_install` / `ollama_pull` / `llama_cpp_pull` with correct params.
- `preconditions` False for unknown `command`.
- `preconditions` False for `llama_cpp_pull` with non-HTTPS URL or non-`.gguf` basename.
- `run` with `command="pip_install"`: monkey-patch `subprocess.run` to return rc=0 + stdout; verify outcome captures rc and tails.
- `run` with `command="ollama_pull"` when binary missing: monkey-patch `subprocess.run` to raise `FileNotFoundError`; verify SKIPPED + `reason="ollama_not_installed"`.
- `run` with `command="llama_cpp_pull"`: monkey-patch `httpx.stream`; verify file written to tmp dest + outcome has `dest_path` + file size.
- `run` with unknown command: FAILURE with `reason="unknown_command:..."`.

### 5.2 SafetyGate tests — extend `tests/autonomic/test_safety.py`

- `_queue` writes an `id` field; same call writes different ids.
- `list_pending` returns entries with ids.
- `remove_pending(id)` deletes only the matching entry; returns True.
- `remove_pending("nonexistent")` returns False; file unchanged.

### 5.3 Executor tests — extend `tests/autonomic/test_executor.py`

- `execute(..., bypass_safety=True)` on a yellow lever runs the lever immediately, writes `LeverReport`, emits event — does NOT hit `SafetyGate`.
- `execute(..., bypass_safety=False)` (existing) still queues yellow to pending — regression guard.

### 5.4 API tests — `tests/autonomic/test_api.py` (13 tests via TestClient)

Tests use FastAPI `TestClient(app)` after configuring env paths to a `tmp_path`. Covers:
- `GET /status` — fields present.
- `GET /ticks` returns recent tick_log entries in newest-first order.
- `GET /ticks?limit=5` respects limit.
- `GET /levers/FIRE_SERVER_HEALTH` returns only that lever's reports.
- `GET /levers/BOGUS` → 404.
- `POST /pending` body `{"lever": "FIRE_TOOL_INSTALL", "params": {...}}` → 200 with id.
- `POST /pending` body `{"lever": "FIRE_SERVER_HEALTH", ...}` → 400 (green lever not allowed via approval flow).
- `POST /pending` body `{"lever": "BOGUS", ...}` → 404.
- `GET /pending` after enqueue shows the entry with id.
- `POST /pending/{id}/approve` executes lever (mocked subprocess) and removes entry.
- `POST /pending/{id}/reject` removes entry, returns `rejected_id`.
- `GET /immune` returns seed signatures.
- `POST /kill-switch` body `{"enabled": false}` updates file; subsequent `GET /status` shows disabled.

### 5.5 Registry test — extend `tests/autonomic/test_registry.py`

- `register_default_autonomic_levers` registers 12 autonomic (was 11) — add `FIRE_TOOL_INSTALL` assertion.

**Test count target:** ~25 new. Combined autonomic suite post-D-08: ~254.

---

## 6. Open questions (not blocking)

1. **`LeverCategory.BODY` usage** — parent spec section 3 puts TOOL_INSTALL under "Тело и среда" which maps to `LeverCategory.BODY`. `BODY` is defined in `types.py` but currently unused. D-08 is the first user. No action — just note it.
2. **`pip_install` and sys.executable** — the test environment has `.venv/Scripts/python.exe`. The lever uses `sys.executable` which in a uvicorn process resolves to the same. In production the same rule applies. Correct and portable.
3. **Retention policy for pending_approvals.jsonl** — if a user never approves/rejects, entries accumulate. Not addressed in D-08 — acceptable at current scale. Future enhancement: a cleanup lever that expires entries older than N days (candidate for a post-19 micro-project).
4. **Body size limits on `POST /pending`** — FastAPI default limits. Not customised. Acceptable for `{lever, params}` small payloads.

---

## 7. What comes after D-08

**D-09** — Frontend: `AutonomicPanel.tsx` + `StatusBar` indicator. Consumes:
- `GET /status` — header: kill switch, scheduler running, 19 levers.
- `GET /ticks` — scrollable tick stream.
- `GET /levers/{name}` — per-lever detail drawer.
- `GET /pending` + `POST approve/reject` — pending approvals card (the main yellow-flow surface).
- `GET /immune` — immune signatures board.
- `POST /kill-switch` — toggle button.

D-09 is pure React/TypeScript + tests via `vitest` (manual visual review for UI). No backend changes. With D-08 + D-09 complete, Model X is at 19/19 levers with first-class observability — the full vision from [docs/superpowers/specs/2026-04-16-model-x-autonomic-design.md](./2026-04-16-model-x-autonomic-design.md) is delivered.
