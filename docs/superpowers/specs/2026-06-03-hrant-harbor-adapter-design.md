# Hrant — Harbor Terminal-Bench Adapter (callback-override) design

**Date:** 2026-06-03
**Status:** Spec (awaiting plan)
**Owner:** Hrant agent + repo maintainer

## Problem

The existing `hrant_agent.py` adapter under Harbor (`/.venv/.../harbor/agents/installed/hrant_agent.py`) is architecturally incomplete and silently fails Terminal-Bench:

1. **Speaker is GUEST.** Adapter posts `speaker: "benchmark_harness"` to `/api/chat`. Anything not under `webui:*` / approved Telegram is `guest`; guest is chat-only with no tools. So the agent enters every trial as a text-only chatter, never touches the task environment.
2. **`terminal_exec` is HOST-scoped.** Even if the agent had tools, `terminal_exec` runs `subprocess` on the Hrant host machine, not inside Harbor's per-task docker container. Tasks are isolated in containers; commands run on host see the wrong filesystem.

Result of the 2026-06-02 attempt: `asyncio.exceptions.CancelledError` (harbor per-trial timeout) with no measurable work done. The 2026-06-03 fall-back to `--agent hermes` produced a real bench number (0/2 mean, 1 NonZeroAgentExitCodeError) but measures **hermes**, not Hrant.

## Goal

Run Terminal-Bench with `--agent hrant` and get a **real Hrant score**: trials complete, reward values are computed, the bench summary table actually contains Hrant's performance. No prompt-level workarounds — Hrant's normal tool loop drives the trial, with the only change being that `terminal_exec` operates inside Harbor's task container instead of on the host.

## Non-goals

- Concurrent multi-trial parallelism (`--n-concurrent` stays at 1 for this iteration).
- Token-budget / cost optimisation specific to bench (Hrant's normal budgeting applies).
- Container-side install of Hrant CLI (the "embed Hrant in container" alternative was explicitly rejected — costs ~2-3min install per task and loses host-side knowledge dir).
- Migrating Hrant's other tools (`read_file`, `save_to_workspace`, `search_knowledge`, …) into the container. These are cognitive tools that legitimately operate against Hrant's own state and codebase; only `terminal_exec` needs container scope.
- Streaming responses to Harbor. Harbor wants one final answer per trial; we return synchronous.

## Architecture (Callback-override)

```
┌─────────────────────┐                      ┌─────────────────────┐
│  Harbor process     │                      │  hrant.service      │
│  (harbor venv)      │                      │  (FastAPI :3333)    │
│                     │                      │                     │
│  HrantAgent adapter │  POST /api/exec-     │                     │
│   ┌──────────────┐  │  ─protocol           │  Agent.run() with   │
│   │ aiohttp      │  │ ───────────────────► │  ContextVar set:    │
│   │ server       │  │   {task,             │  REMOTE_EXEC=cb_url │
│   │ :<eph-port>  │  │    callback_url,     │   ┌──────────────┐  │
│   │              │  │    session_id}       │   │ tool loop    │  │
│   │ POST /exec ──┼──┼──────────────────────┼──◄│ (skills, KG, │  │
│   │  ↓ runs in   │  │  Hrant's terminal_   │   │ identity,    │  │
│   │  environ-    │  │  exec is ContextVar  │   │ supervisor)  │  │
│   │  ment.exec() │  │  -overridden →       │   │              │  │
│   │              │  │  HTTP POST back to   │   │  terminal_   │  │
│   └──────────────┘  │  callback_url        │   │  exec  ──────┼──┼──► HTTP POST to
│                     │                      │   │  (override)  │  │   callback_url
└─────────────────────┘                      │   └──────────────┘  │
        │                                    └─────────────────────┘
        ▼
  environment.exec()
  → docker exec in
    task container
```

**Boundaries:**

- Both processes (Harbor adapter / Hrant gateway) live on the same host. All inter-process traffic is localhost.
- The task container is reachable ONLY through Harbor's own `environment.exec()` primitive (`docker exec ...` under the hood). Hrant never sees the container directly.
- Hrant runs its full normal loop — skills, knowledge graph, identity, supervisor, drift marker, self-correction — only `terminal_exec` is redirected. Other tools (`read_file`, `save_to_workspace`, `web_search`, `search_knowledge`, `start_background_job`) still operate on host. That is correct: they are cognitive tools (read codebase, search memory, queue follow-up work), not task-environment tools.

## Components

### Backend

#### `backend/tools/terminal_exec.py` — ContextVar override

Add a new module-level ContextVar:

```python
_REMOTE_EXEC_CALLBACK: ContextVar[str | None] = ContextVar(
    "hrant_terminal_exec_remote_callback", default=None,
)

def set_remote_exec_callback(url: str | None) -> Token: ...
def reset_remote_exec_callback(token: Token) -> None: ...
```

In `run_terminal(command, *, timeout_seconds, cwd)`, after the catastrophic-denylist validation but before `subprocess.run`:

- If `_REMOTE_EXEC_CALLBACK.get()` is non-empty: POST to that URL with JSON `{command, cwd, timeout_sec}` using `requests` (sync, since `run_terminal` itself is sync). Expect 200 with JSON `{stdout, stderr, return_code}`. Wrap into a `TerminalResult` exactly as the local path would.
- On HTTP error / timeout / connection refused: return `TerminalResult(ok=False, exit_code=-1, error=f"remote-exec callback failed: {e}")`. Caller sees a failed `terminal_exec` the same as a local subprocess failure — agent can decide to retry/pivot.
- Output cap and truncation still apply to the remote-returned stdout/stderr (same `MAX_OUTPUT_BYTES`, same `_truncate` helper).

The catastrophic denylist (`rm -rf /`, `dd of=/dev/sd*`, `curl|sh`, fork bomb, `mkfs /dev`, `kill 1`) **still runs** before the remote dispatch. We don't want the LLM to execute a destructive command inside a task container either (containers can have host volume mounts).

#### `backend/api/exec_protocol.py` (NEW) — endpoint

`POST /api/exec-protocol`:

- Body: `{task: str, callback_url: str, session_id: str = ""}`
- `callback_url` MUST start with `http://127.0.0.1:` — reject otherwise (`HTTPException(400, "callback_url must be loopback")`). We are localhost-only; an external callback URL would let a misconfigured client steer Hrant's terminal_exec output to a third party.
- Set `_REMOTE_EXEC_CALLBACK` to `callback_url` via the helper (token under `try`/`finally`).
- Call `Agent().run(task, speaker_id="webui:bench-harness", channel="webui", session_key=session_id or uuid4().hex)`.
- `webui:bench-harness` is treated as owner by the existing role system (`webui:*` is always owner — established session convention).
- Returns `{ok: True, answer: str, turn_id: str, token_usage: dict}` on success. On agent error: `{ok: False, error: str}` with 500. Harbor sees this through the adapter and writes whichever string ends up as the trial output.

Register the router in `backend/main.py` next to other API routers.

#### `backend/main.py` — wire the router

Add `from .api import exec_protocol as exec_protocol_api` to the import block and `app.include_router(exec_protocol_api.router)` in the registration block. Same shape as existing API routers (`background_jobs_api`, etc.).

### Adapter

#### Source-of-truth file in repo: `harbor_adapter/hrant_agent.py` (NEW)

We keep the canonical adapter in this repo (NOT only in the harbor venv) so changes are version-controlled. A small deploy hook copies it into `~/.hrant/data/workspace/terminal_bench_2_1/.venv/lib/python3.12/site-packages/harbor/agents/installed/hrant_agent.py` whenever it changes — implementation lives in the plan, not this spec.

Adapter structure:

```python
class HrantAgent(BaseInstalledAgent):
    SUPPORTS_ATIF = False
    @staticmethod
    def name() -> str: return "hrant"

    async def install(self, environment):
        # GET http://localhost:3333/api/version (or /api/health) to verify
        # gateway is reachable. Nothing to install inside the container —
        # the agent lives entirely on host.

    @with_prompt_template
    async def run(self, instruction, environment, context):
        port = pick_ephemeral_port()
        app = web.Application()
        app.router.add_post("/exec", make_exec_handler(environment))
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port); await site.start()
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    "http://localhost:3333/api/exec-protocol",
                    json={
                        "task": instruction,
                        "callback_url": f"http://127.0.0.1:{port}/exec",
                        "session_id": str(uuid.uuid4()),
                    },
                    timeout=aiohttp.ClientTimeout(total=900),
                ) as r:
                    payload = await r.json()
            answer = payload.get("answer") or json.dumps(payload)
            context.paths.output_file.write_text(answer, encoding="utf-8")
        finally:
            await runner.cleanup()
```

`make_exec_handler(environment)` returns an aiohttp handler that:
- parses `{command, cwd, timeout_sec}` from the request body
- `await environment.exec(command, cwd=cwd, timeout_sec=timeout_sec)`
- returns JSON `{stdout, stderr, return_code}`
- on any exception → 500 with `{error: str}` (which Hrant's `run_terminal` will surface as a failed `TerminalResult`)

## Data flow (one trial)

1. Harbor calls `HrantAgent.run(instruction, environment, context)`.
2. Adapter starts aiohttp on ephemeral port (e.g. `:42891`).
3. Adapter POSTs `http://localhost:3333/api/exec-protocol` with `{task=instruction, callback_url="http://127.0.0.1:42891/exec", session_id=…}`.
4. Endpoint sets ContextVar, calls `Agent.run`.
5. Agent runs its normal unified loop. When `terminal_exec` fires, `run_terminal` sees the ContextVar and POSTs to the callback URL with `{command, cwd, timeout_sec}`.
6. Adapter handler receives, `await environment.exec(...)`, returns `{stdout, stderr, return_code}`.
7. `run_terminal` wraps the response as a `TerminalResult`. Agent continues the loop.
8. Agent emits final answer. Endpoint returns 200 `{ok, answer, turn_id, token_usage}`.
9. Adapter writes `answer` into `context.paths.output_file`.
10. Adapter shuts down aiohttp server in `finally`.
11. Harbor reads `output_file` and scores the trial.

## Harness contract (configured defaults)

- **auth:** `/api/exec-protocol` is owner-mode by default. The gateway already binds to `127.0.0.1:3333`, so non-localhost reach is impossible without an additional reverse proxy. The endpoint validates `callback_url` starts with `http://127.0.0.1:` — the only authentication boundary needed.
- **speaker:** `webui:bench-harness` — `webui:*` already maps to owner; consistent with existing convention.
- **concurrency:** `harbor run --n-concurrent 1`. Hrant's tool registry holds module-level state (skills, in-progress jobs) that has not been audited for parallel `Agent.run` safety. Serial trials are correct; parallelism is a follow-up project.
- **timeout:** `harbor run --agent-timeout-multiplier 10`. Harbor's default per-trial agent timeout is tight; Hrant turns can legitimately take 5-15 minutes (cold tool-loop iterations with skill loads and KG searches).
- **inner exec timeout:** 300 seconds per `terminal_exec` callback. Each Hrant `terminal_exec` call hands an explicit `timeout_seconds` (defaults to 30, capped at 120 in the existing tool) — the adapter forwards that value to `environment.exec(timeout_sec=...)`. 300s is the harness cap if the agent ever asks for more.

## Failure modes

| Scenario | Behaviour |
|---|---|
| `callback_url` not loopback | endpoint returns 400, harbor adapter raises, harbor scores 0 |
| Adapter HTTP server dies mid-trial | next `terminal_exec` returns `ok=False, error="connection refused"` → agent sees failed tool → can retry / pivot / give up honestly |
| Hrant agent crashes inside `Agent.run` | endpoint returns 500 with error string → adapter writes the error into `output_file` → harbor scores 0 |
| Hrant turn exceeds harbor agent-timeout | harbor SIGTERMs adapter process group → `finally` cleanup runs → next time the adapter starts fresh |
| `environment.exec` raises | adapter handler returns 500 → `run_terminal` returns `ok=False` → agent treats as failed `terminal_exec` |
| Hrant gateway down at trial start | adapter's `install()` GET fails → harbor raises before any trial starts |
| Catastrophic-denylist hit (`rm -rf /`) | refused by Hrant's existing pre-flight; `run_terminal` returns `ok=False`, refusal reason in `error` |

## Testing strategy

- **Unit tests (run on dev box):**
  - `terminal_exec` ContextVar override: when set, no subprocess fires and the HTTP POST shape matches contract. When unset, normal subprocess path runs.
  - Catastrophic denylist still rejects in remote mode (denylist runs BEFORE the dispatch decision).
  - `/api/exec-protocol` endpoint: rejects non-loopback `callback_url`, accepts loopback, returns `{ok, answer, ...}` shape on a stubbed agent.
- **Integration test (run on dev box):**
  - End-to-end with a fake harbor environment: stand up the adapter's aiohttp server, set up a no-op fake `environment.exec` that echoes input, run a tiny task ("write the literal word DONE"). Assert the final `output_file` contains a non-empty answer and at least one fake exec call happened.
- **Real bench (run on prod after deploy):**
  - 1-task smoke (`harbor run --dataset terminal-bench --n-tasks 1 --agent hrant --agent-timeout-multiplier 10 --n-concurrent 1`). Manual eyeball on the result.json.
  - 2-task bench. Compare against the 2026-06-02 hermes baseline (mean 0.000, 1 exception).

## Out of scope (for this iteration; track separately)

- Parallel trials (`--n-concurrent > 1`) → audit module-level state in tool registry first.
- File delivery between container and host (MEDIA: convention adapter): Terminal-Bench tasks rarely need that.
- Trajectory recording (`Trajectory`, `Step`, `Observation` models in harbor) — adapter returns plain text for now; rich trajectory is a follow-up.
- Adapter unit-test scaffolding inside the harbor venv (it's a third-party install path; we test the protocol via Hrant-side unit tests + the integration test).

## Open follow-ups (after first green bench)

- Deploy hook: source-of-truth adapter file lives in `harbor_adapter/hrant_agent.py` in this repo; install or `hrant update` should `cp` it into the harbor venv. Spec'd as a follow-up because it requires `hrant update` script changes.
- Hrant adapter could also implement `Trajectory`-style step output so harbor's reporting shows real Hrant tool calls — useful but not required for a first real number.
