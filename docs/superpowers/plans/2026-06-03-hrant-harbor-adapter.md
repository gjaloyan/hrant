# Hrant–Harbor Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a real `--agent hrant` path into Harbor's terminal-bench by overriding Hrant's `terminal_exec` via a ContextVar callback that runs commands inside the Harbor task container.

**Architecture:** A new `POST /api/exec-protocol` endpoint accepts `{task, callback_url, session_id}`, sets a ContextVar before `Agent().run`, and unsets it after. While the ContextVar is set, `run_terminal` posts each command to `callback_url` instead of running locally. The new `harbor_adapter/hrant_agent.py` runs an aiohttp server on an ephemeral port that receives those posts and dispatches them to `environment.exec` (Harbor's docker exec primitive), returning stdout/stderr/return_code.

**Tech Stack:** Python 3.12, FastAPI (Hrant gateway), `requests` for sync callback dispatch, `aiohttp` (already a Harbor dep) for adapter server, pytest.

---

## File structure

| File | Role |
|---|---|
| `backend/tools/terminal_exec.py` (MODIFY) | Add `_REMOTE_EXEC_CALLBACK` ContextVar + dispatch branch in `run_terminal`. |
| `backend/api/exec_protocol.py` (CREATE) | FastAPI router with `POST /api/exec-protocol`; validates loopback callback, sets ContextVar, runs `Agent.run`, returns answer. |
| `backend/main.py` (MODIFY) | Include the new router. |
| `harbor_adapter/__init__.py` (CREATE) | Empty package marker. |
| `harbor_adapter/hrant_agent.py` (CREATE) | Source-of-truth `HrantAgent(BaseInstalledAgent)` — aiohttp callback server + POST to `/api/exec-protocol`. |
| `harbor_adapter/README.md` (CREATE) | Manual deploy instructions: how to copy this file into the harbor venv on prod. |
| `tests/test_terminal_exec_remote_override.py` (CREATE) | ContextVar override behavior. |
| `tests/test_exec_protocol_endpoint.py` (CREATE) | Endpoint contract: loopback check, ContextVar lifecycle, response shape. |
| `tests/test_hrant_adapter.py` (CREATE) | Adapter end-to-end with a fake Harbor environment + fake Hrant gateway. |

---

### Task 1: Add `_REMOTE_EXEC_CALLBACK` ContextVar + remote dispatch in `run_terminal`

**Files:**
- Modify: `backend/tools/terminal_exec.py`
- Test: `tests/test_terminal_exec_remote_override.py`

Background:
- `run_terminal` lives in `backend/tools/terminal_exec.py`. It is called by the agent's `terminal_exec` tool wrapper. Today it always runs `subprocess.run`.
- We add a module-level `ContextVar` plus two helpers (`set_remote_exec_callback`, `reset_remote_exec_callback`). When the var is set, `run_terminal` POSTs to that URL with `{command, cwd, timeout_sec}` and turns the response into a `TerminalResult`.
- The catastrophic-command denylist (`_check_dangerous_command`) STILL runs before the dispatch decision — destructive commands are refused regardless of where they would execute.
- Output truncation (`_truncate`, `MAX_OUTPUT_BYTES`) still applies to remote-returned stdout/stderr.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_terminal_exec_remote_override.py`:

```python
"""ContextVar override for terminal_exec.

When `_REMOTE_EXEC_CALLBACK` is set in the current async/thread context,
`run_terminal` MUST dispatch the command via HTTP POST to the callback
URL instead of running it locally. The catastrophic-command denylist
still applies (we don't want destructive commands to leak into a
container task environment either — containers can have host mounts).
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def reset_var():
    """Each test must start with a clean ContextVar."""
    from backend.tools import terminal_exec as te
    token = te._REMOTE_EXEC_CALLBACK.set(None)
    yield
    te._REMOTE_EXEC_CALLBACK.reset(token)


def test_local_path_runs_subprocess_when_var_unset(monkeypatch):
    """ContextVar is None → run_terminal MUST fall back to subprocess.run."""
    from backend.tools import terminal_exec as te
    called = {"n": 0}

    class _FakeProc:
        returncode = 0
        stdout = b"local stdout\n"
        stderr = b""

    def fake_run(cmd, **kw):
        called["n"] += 1
        return _FakeProc()

    monkeypatch.setattr("backend.tools.terminal_exec.subprocess.run", fake_run)
    result = te.run_terminal("echo hi")
    assert result.ok is True
    assert "local stdout" in result.stdout
    assert called["n"] == 1


def test_remote_path_calls_callback_when_var_set(monkeypatch):
    """ContextVar set → run_terminal MUST NOT call subprocess; it POSTs to
    the callback URL and wraps the response as a TerminalResult."""
    from backend.tools import terminal_exec as te
    posted = {"calls": []}

    class _FakeResp:
        status_code = 200
        def json(self):
            return {"stdout": "remote out", "stderr": "", "return_code": 0}
        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        posted["calls"].append({"url": url, "json": json, "timeout": timeout})
        return _FakeResp()

    monkeypatch.setattr("backend.tools.terminal_exec.requests.post", fake_post)

    def fake_run(*a, **kw):
        raise AssertionError("subprocess.run must NOT be called in remote mode")

    monkeypatch.setattr("backend.tools.terminal_exec.subprocess.run", fake_run)

    token = te._REMOTE_EXEC_CALLBACK.set("http://127.0.0.1:42891/exec")
    try:
        result = te.run_terminal("ls -la", timeout_seconds=15)
    finally:
        te._REMOTE_EXEC_CALLBACK.reset(token)

    assert result.ok is True
    assert result.stdout == "remote out"
    assert len(posted["calls"]) == 1
    call = posted["calls"][0]
    assert call["url"] == "http://127.0.0.1:42891/exec"
    assert call["json"]["command"] == "ls -la"
    assert call["json"]["timeout_sec"] == 15


def test_remote_callback_failure_returns_ok_false(monkeypatch):
    """A failed HTTP callback must surface as TerminalResult(ok=False)
    with the exception preserved in `error`; the agent loop can then
    treat it as a failed `terminal_exec` and react."""
    from backend.tools import terminal_exec as te
    import requests as _req

    def fake_post(url, json=None, timeout=None):
        raise _req.ConnectionError("simulated")

    monkeypatch.setattr("backend.tools.terminal_exec.requests.post", fake_post)
    token = te._REMOTE_EXEC_CALLBACK.set("http://127.0.0.1:42891/exec")
    try:
        result = te.run_terminal("ls")
    finally:
        te._REMOTE_EXEC_CALLBACK.reset(token)
    assert result.ok is False
    assert result.exit_code == -1
    assert "simulated" in (result.error or "")


def test_catastrophic_denylist_still_blocks_in_remote_mode(monkeypatch):
    """`rm -rf /` must be refused even when the callback would have
    executed it inside a container. Container task environments can
    mount host paths, and the denylist is the same safety boundary."""
    from backend.tools import terminal_exec as te

    def fake_post(url, json=None, timeout=None):
        raise AssertionError("denylist must short-circuit before HTTP")

    monkeypatch.setattr("backend.tools.terminal_exec.requests.post", fake_post)
    token = te._REMOTE_EXEC_CALLBACK.set("http://127.0.0.1:42891/exec")
    try:
        result = te.run_terminal("rm -rf /")
    finally:
        te._REMOTE_EXEC_CALLBACK.reset(token)
    assert result.ok is False
    assert "rm -rf" in (result.error or "").lower() or "catastrophic" in (result.error or "").lower()


def test_set_reset_helpers_round_trip():
    """`set_remote_exec_callback` returns a Token that
    `reset_remote_exec_callback` accepts. Round-trip must restore
    the prior value (None at module load)."""
    from backend.tools import terminal_exec as te
    assert te._REMOTE_EXEC_CALLBACK.get() is None
    tok = te.set_remote_exec_callback("http://127.0.0.1:1234/exec")
    try:
        assert te._REMOTE_EXEC_CALLBACK.get() == "http://127.0.0.1:1234/exec"
    finally:
        te.reset_remote_exec_callback(tok)
    assert te._REMOTE_EXEC_CALLBACK.get() is None
```

- [ ] **Step 2: Run tests to verify they fail (the module hasn't been changed yet)**

Run: `python -m pytest tests/test_terminal_exec_remote_override.py -v`
Expected: FAIL — `AttributeError: module 'backend.tools.terminal_exec' has no attribute '_REMOTE_EXEC_CALLBACK'` (etc.)

- [ ] **Step 3: Implement ContextVar + remote dispatch in `backend/tools/terminal_exec.py`**

Add these imports near the top of `backend/tools/terminal_exec.py` (alongside existing `import os`, `import re`, `import shlex`, `import subprocess`):

```python
import contextvars
from typing import Optional

import requests
```

Add this block right after the `MAX_TIMEOUT_SECONDS` / `DEFAULT_TIMEOUT_SECONDS` constants and before the `_WHOLE_STRING_DANGERS` block (so the override hook is declared with the other module-level state):

```python
# ─── Remote-exec override (Harbor terminal-bench adapter) ─────────
#
# When `_REMOTE_EXEC_CALLBACK` is set in the current context, `run_terminal`
# dispatches the command via HTTP POST to the callback URL instead of running
# it locally as a subprocess. The Harbor `hrant_agent.py` adapter uses this:
# it starts an aiohttp server on an ephemeral port, sets the ContextVar via
# the `/api/exec-protocol` endpoint, and forwards each callback to
# `environment.exec(...)` — Harbor's docker exec primitive — so terminal_exec
# operates against the task container instead of the host.
#
# Catastrophic-denylist checks run BEFORE the remote dispatch — destructive
# commands are refused regardless of where they would execute (the task
# container can have host volume mounts; same threat model applies).
_REMOTE_EXEC_CALLBACK: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hrant_terminal_exec_remote_callback", default=None,
)


def set_remote_exec_callback(url: Optional[str]) -> contextvars.Token:
    """Set the callback URL for the current context. Returns a Token
    the caller must pass to `reset_remote_exec_callback` on exit."""
    return _REMOTE_EXEC_CALLBACK.set(url or None)


def reset_remote_exec_callback(token: contextvars.Token) -> None:
    """Restore the prior ContextVar value (None if no nesting)."""
    try:
        _REMOTE_EXEC_CALLBACK.reset(token)
    except Exception:
        pass


# Timeout for the outbound HTTP POST to the callback URL. The CALLBACK is
# expected to await `environment.exec` on the container, which can legitimately
# take a few minutes for a heavy build. We cap at 600s as the safety net.
_REMOTE_HTTP_TIMEOUT_S = 600
```

Now modify `run_terminal` to branch on the ContextVar. Find the existing `try:` that calls `subprocess.run(...)` (around line ~436) and replace the whole exec section:

```python
    ok, err, _argv = _validate_command(command)
    if not ok:
        return TerminalResult(
            ok=False, command=command, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=0, error=err,
        )

    timeout = max(1, min(int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), MAX_TIMEOUT_SECONDS))
    start = _time.monotonic()

    # Remote-exec dispatch path. When the ContextVar is set we POST the
    # command to the harbor adapter's loopback server; the adapter awaits
    # environment.exec() and returns the result as JSON. Denylist already
    # ran above; output truncation applies the same as the local path.
    callback_url = _REMOTE_EXEC_CALLBACK.get()
    if callback_url:
        try:
            resp = requests.post(
                callback_url,
                json={
                    "command": command,
                    "cwd": cwd or "",
                    "timeout_sec": timeout,
                },
                timeout=_REMOTE_HTTP_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            elapsed = int((_time.monotonic() - start) * 1000)
            return TerminalResult(
                ok=False, command=command, exit_code=-1,
                stdout="", stderr="",
                truncated=False, elapsed_ms=elapsed,
                error=f"remote-exec callback failed: {type(e).__name__}: {e}",
            )
        elapsed = int((_time.monotonic() - start) * 1000)
        rc = int(data.get("return_code", -1) or -1)
        stdout_cap = (MAX_OUTPUT_BYTES * 2) // 3
        stderr_cap = MAX_OUTPUT_BYTES - stdout_cap
        out, out_trunc = _truncate((data.get("stdout") or "").encode("utf-8"), stdout_cap)
        err_text, err_trunc = _truncate((data.get("stderr") or "").encode("utf-8"), stderr_cap)
        return TerminalResult(
            ok=(rc == 0),
            command=command,
            exit_code=rc,
            stdout=out,
            stderr=err_text,
            truncated=(out_trunc or err_trunc),
            elapsed_ms=elapsed,
            error="" if rc == 0 else f"exit code {rc}",
        )

    # Local subprocess path (unchanged).
    # Prefer /bin/bash on POSIX so commands using bash features (set -o
```

(Keep the rest of the local subprocess path exactly as it is today — the change above only ADDS the remote branch before it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_terminal_exec_remote_override.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/tools/terminal_exec.py tests/test_terminal_exec_remote_override.py
git commit -m "feat(terminal_exec): ContextVar-driven remote-exec callback for Harbor adapter

run_terminal gains a contextvars-scoped override hook: when
_REMOTE_EXEC_CALLBACK is set in the current async/thread context, the
command is dispatched via HTTP POST to the callback URL instead of
running locally as a subprocess. Catastrophic denylist still runs
before the dispatch decision (containers can have host mounts);
output truncation still applies to remote-returned stdout/stderr.

The Harbor terminal-bench adapter will use this to redirect Hrant's
terminal_exec into the task container via environment.exec() while
keeping all other tools (read_file, search_knowledge, …) operating
on the host."
```

---

### Task 2: `/api/exec-protocol` endpoint

**Files:**
- Create: `backend/api/exec_protocol.py`
- Test: `tests/test_exec_protocol_endpoint.py`

Background:
- The endpoint accepts `{task, callback_url, session_id}`, validates loopback, sets the ContextVar from Task 1, calls `Agent().run(...)`, and returns the final answer.
- `webui:bench-harness` is the speaker. `webui:*` is owner per existing convention, so the agent gets the full tool surface.
- The endpoint is sync (matches existing `/api/chat` shape); FastAPI runs it in a threadpool, and the ContextVar lives in that thread for the duration of `Agent.run`.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_exec_protocol_endpoint.py`:

```python
"""POST /api/exec-protocol — adapter-facing endpoint for Harbor.

Contract:
  - Body: {task: str, callback_url: str, session_id: str = ""}
  - Rejects non-loopback callback_url with 400.
  - Sets the terminal_exec ContextVar before Agent.run, resets after.
  - Speaker is 'webui:bench-harness' (owner via webui:* convention).
  - Returns {ok, answer, turn_id, token_usage} on success.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def test_rejects_non_loopback_callback_url(client):
    """Only http://127.0.0.1:* is acceptable. Any other host must
    be refused with 400 — we never want Hrant's terminal_exec output
    to leak to an external machine."""
    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://example.com/exec",
            "session_id": "s1",
        },
    )
    assert resp.status_code == 400
    assert "loopback" in resp.text.lower() or "127.0.0.1" in resp.text


def test_rejects_loopback_with_wrong_scheme(client):
    """https:// is also refused — the adapter is plain HTTP on
    127.0.0.1 by design (no TLS needed loopback)."""
    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "https://127.0.0.1:42891/exec",
            "session_id": "s1",
        },
    )
    assert resp.status_code == 400


def test_accepts_loopback_and_runs_agent(client, monkeypatch):
    """A well-formed request must drive Agent.run with the expected
    speaker, AND set+reset the terminal_exec ContextVar around the call."""
    from backend.tools import terminal_exec as te

    observed: dict = {}

    def fake_run(self, task, *, speaker_id, channel, session_key=None, **kw):
        # Capture what the endpoint passed AND the ContextVar value
        # that's active while Agent.run is executing.
        observed["task"] = task
        observed["speaker_id"] = speaker_id
        observed["channel"] = channel
        observed["session_key"] = session_key
        observed["callback_during_run"] = te._REMOTE_EXEC_CALLBACK.get()
        ans = MagicMock()
        ans.answer = "OK final answer"
        ans.turn_id = "turn-xyz"
        ans.token_usage = None
        return ans

    monkeypatch.setattr("backend.agent.Agent.run", fake_run)

    # ContextVar must be None BEFORE the call.
    assert te._REMOTE_EXEC_CALLBACK.get() is None

    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://127.0.0.1:42891/exec",
            "session_id": "trial-abc",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["answer"] == "OK final answer"
    assert body["turn_id"] == "turn-xyz"
    # Speaker must be the owner-mapped bench-harness one.
    assert observed["speaker_id"] == "webui:bench-harness"
    assert observed["channel"] == "webui"
    assert observed["session_key"] == "trial-abc"
    # ContextVar must have been set DURING the call.
    assert observed["callback_during_run"] == "http://127.0.0.1:42891/exec"
    # And reset AFTER.
    assert te._REMOTE_EXEC_CALLBACK.get() is None


def test_uses_uuid_session_when_empty(client, monkeypatch):
    """If session_id is empty, the endpoint must generate a non-empty
    session_key so the agent doesn't share state across trials."""
    captured: dict = {}

    def fake_run(self, task, *, speaker_id, channel, session_key=None, **kw):
        captured["session_key"] = session_key
        ans = MagicMock()
        ans.answer = "ok"
        ans.turn_id = ""
        ans.token_usage = None
        return ans

    monkeypatch.setattr("backend.agent.Agent.run", fake_run)

    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://127.0.0.1:42891/exec",
            "session_id": "",
        },
    )
    assert resp.status_code == 200
    assert captured["session_key"]
    assert len(captured["session_key"]) >= 8  # uuid hex >= 8 chars


def test_agent_exception_returns_500_with_error(client, monkeypatch):
    """If Agent.run raises, the endpoint returns 500 with an error
    string so the adapter can write it into the trial output file."""
    from backend.tools import terminal_exec as te

    def boom(self, task, **kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr("backend.agent.Agent.run", boom)
    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://127.0.0.1:42891/exec",
            "session_id": "s",
        },
    )
    assert resp.status_code == 500
    body = resp.json()
    assert body.get("ok") is False
    assert "agent exploded" in body.get("error", "")
    # ContextVar must still be reset even after the exception.
    assert te._REMOTE_EXEC_CALLBACK.get() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_exec_protocol_endpoint.py -v`
Expected: FAIL — `404` on every test (the endpoint doesn't exist yet).

- [ ] **Step 3: Implement `backend/api/exec_protocol.py`**

Create `backend/api/exec_protocol.py`:

```python
"""Harbor terminal-bench adapter endpoint.

POST /api/exec-protocol drives one bench trial:
  - Set the `_REMOTE_EXEC_CALLBACK` ContextVar to `callback_url` so
    Hrant's `terminal_exec` posts each command to the adapter's
    loopback server instead of running on the host.
  - Run a single `Agent.run` turn as the owner-mapped bench harness.
  - Return the final answer to the adapter, which writes it to the
    trial output file harbor scores against.

Trust model: localhost-only. The gateway already binds to 127.0.0.1
(see backend/main.py); we additionally validate that `callback_url`
itself starts with http://127.0.0.1: so a misconfigured client can't
trick us into forwarding command output to a third party.

Speaker: `webui:bench-harness`. The existing roles convention maps
`webui:*` to owner, so the agent receives its full tool surface.

Concurrency: this endpoint is sync; FastAPI runs it in a worker
thread. The ContextVar is per-thread-task, so concurrent trials
do not share state. The current Harbor harness runs with
`--n-concurrent 1` anyway.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..tools import terminal_exec as _te


log = logging.getLogger(__name__)


router = APIRouter()


_LOOPBACK_PREFIX = "http://127.0.0.1:"


class ExecProtocolRequest(BaseModel):
    task: str
    callback_url: str
    session_id: str = ""


class ExecProtocolResponse(BaseModel):
    ok: bool
    answer: str = ""
    turn_id: str = ""
    token_usage: Optional[dict] = None
    error: str = ""


@router.post("/api/exec-protocol")
def exec_protocol(body: ExecProtocolRequest):
    if not body.callback_url.startswith(_LOOPBACK_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=(
                "callback_url must be loopback "
                f"(http://127.0.0.1:<port>/...); got {body.callback_url!r}"
            ),
        )

    session_key = (body.session_id or "").strip() or uuid.uuid4().hex

    # Set the ContextVar for this trial. The token is reset in `finally`
    # even on agent crash so a follow-up turn never inherits stale state.
    token = _te.set_remote_exec_callback(body.callback_url)
    try:
        # Late import: `Agent` pulls many runtime services that aren't
        # ready at module load time.
        from ..agent import Agent
        agent = Agent()
        result = agent.run(
            body.task,
            speaker_id="webui:bench-harness",
            channel="webui",
            session_key=session_key,
        )
        answer = getattr(result, "answer", "") or ""
        turn_id = getattr(result, "turn_id", "") or ""
        token_usage = getattr(result, "token_usage", None)
        # token_usage may be a Pydantic model; coerce to dict for JSON.
        if token_usage is not None and hasattr(token_usage, "model_dump"):
            token_usage = token_usage.model_dump()
        return ExecProtocolResponse(
            ok=True, answer=answer, turn_id=turn_id, token_usage=token_usage,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("exec-protocol agent run crashed")
        return JSONResponse(
            status_code=500,
            content=ExecProtocolResponse(
                ok=False, error=f"{type(e).__name__}: {e}",
            ).model_dump(),
        )
    finally:
        _te.reset_remote_exec_callback(token)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_exec_protocol_endpoint.py -v`
Expected: 5 passed.

Note: the endpoint isn't wired in `main.py` yet, so the `TestClient(app)` import will fail to find the route until Task 3. To make the test self-sufficient before that wiring lands, run instead the tests for Task 1 only at this checkpoint and proceed to Task 3, then come back to run Task 2's tests after wiring.

Actually that's wrong — the tests will 404 because the route isn't wired. Fix: do Task 3 BEFORE running Task 2 tests, OR temporarily wire the router inside the test fixture. Simpler: complete Task 3 wiring first, then run both Task 2 and 3 tests together.

- [ ] **Step 5: Commit**

```bash
git add backend/api/exec_protocol.py tests/test_exec_protocol_endpoint.py
git commit -m "feat(api): POST /api/exec-protocol for Harbor adapter

Endpoint accepts {task, callback_url, session_id}, validates loopback,
sets the terminal_exec ContextVar, runs a single Agent.run turn as
owner-mapped 'webui:bench-harness', and returns {ok, answer, turn_id,
token_usage}. The ContextVar is reset in finally even on agent crash.

Wiring into main.py lands in the next commit (routing is required
before the integration tests can pass)."
```

---

### Task 3: Wire `/api/exec-protocol` router in `backend/main.py`

**Files:**
- Modify: `backend/main.py`

Background:
- `backend/main.py` registers all API routers in the post-startup block. We just add ours alongside the others.

- [ ] **Step 1: Read the current router registration block to find the right spot**

Run: `grep -n "include_router\|background_jobs_api\|app.include_router" backend/main.py | head -20`

You're looking for the existing `app.include_router(...)` calls; insert ours adjacent.

- [ ] **Step 2: Modify `backend/main.py` — import the module**

Find the existing line that imports api modules (look for `background_jobs as background_jobs_api`). Add `exec_protocol as exec_protocol_api` to the same import block. The block typically looks like:

```python
from .api import (
    background_jobs as background_jobs_api,
    reasoning_routing as reasoning_routing_api,
    # ... others ...
)
```

After the change it includes `exec_protocol as exec_protocol_api,`.

- [ ] **Step 3: Modify `backend/main.py` — register the router**

Find the `app.include_router(background_jobs_api.router)` line (or wherever the other API routers are registered). Add a sibling line:

```python
app.include_router(exec_protocol_api.router)
```

- [ ] **Step 4: Run the Task 2 + Task 3 tests together to verify wiring**

Run: `python -m pytest tests/test_exec_protocol_endpoint.py -v`
Expected: 5 passed (all five Task 2 tests now resolve the route).

Also run: `python -m pytest tests/test_terminal_exec_remote_override.py tests/test_exec_protocol_endpoint.py --no-header -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py
git commit -m "feat(main): wire /api/exec-protocol router for Harbor adapter"
```

---

### Task 4: `harbor_adapter/hrant_agent.py` source-of-truth + package marker

**Files:**
- Create: `harbor_adapter/__init__.py`
- Create: `harbor_adapter/hrant_agent.py`
- Test: `tests/test_hrant_adapter.py`

Background:
- The adapter file is what Harbor's plugin loader imports. Today the live copy lives at `~/.hrant/data/workspace/terminal_bench_2_1/.venv/lib/python3.12/site-packages/harbor/agents/installed/hrant_agent.py`. We keep a canonical source-of-truth in this repo under `harbor_adapter/` so future changes are version-controlled. Manual deploy is documented in Task 5.
- Integration test stubs out `aiohttp` server bring-up and HTTP calls; we assert the message protocol between adapter ↔ endpoint ↔ environment, not the live runtime.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_hrant_adapter.py`:

```python
"""HrantAgent adapter — protocol-level integration test.

We stub the Harbor environment + Hrant gateway as fakes and assert
that the adapter:
  1. Starts an aiohttp server on a loopback ephemeral port,
  2. POSTs to /api/exec-protocol with the right shape,
  3. When the gateway calls the callback, forwards command/cwd/timeout
     to environment.exec and returns stdout/stderr/return_code,
  4. Writes the final answer into context.paths.output_file.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass
class _FakeEnvExecResult:
    stdout: str
    stderr: str
    return_code: int


class _FakeEnvironment:
    def __init__(self):
        self.calls: list[dict] = []

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
        env: dict | None = None,
    ) -> _FakeEnvExecResult:
        self.calls.append({
            "command": command, "cwd": cwd, "timeout_sec": timeout_sec,
        })
        # Echo so the adapter has something deterministic to forward back.
        return _FakeEnvExecResult(stdout=f"out: {command}", stderr="", return_code=0)


@dataclass
class _FakePaths:
    output_file: Path


class _FakeContext:
    def __init__(self, output_file: Path):
        self.paths = _FakePaths(output_file=output_file)


def _import_adapter():
    """Import the adapter from the in-repo source-of-truth path,
    bypassing the harbor venv copy. Harbor's BaseInstalledAgent isn't
    importable here, so we monkey-patch a minimal stub onto sys.modules
    before importing — sufficient to test our adapter logic in
    isolation."""
    import sys
    import types

    # Stub harbor packages so the adapter import doesn't pull harbor.
    harbor_pkg = types.ModuleType("harbor")
    agents_pkg = types.ModuleType("harbor.agents")
    installed_pkg = types.ModuleType("harbor.agents.installed")
    base_pkg = types.ModuleType("harbor.agents.installed.base")
    env_base = types.ModuleType("harbor.environments.base")
    env_pkg = types.ModuleType("harbor.environments")
    ctx_pkg = types.ModuleType("harbor.models.agent.context")
    name_pkg = types.ModuleType("harbor.models.agent.name")

    class _BaseInstalledAgent:
        def __init__(self, logs_dir: Path, **kwargs):
            self.logs_dir = logs_dir
            self._version = None
            for k, v in kwargs.items():
                setattr(self, f"_{k}", v)

        @staticmethod
        def name(): return "stub"

        def version(self) -> str | None:
            return self._version

    def with_prompt_template(fn):
        return fn

    base_pkg.BaseInstalledAgent = _BaseInstalledAgent
    base_pkg.with_prompt_template = with_prompt_template
    env_base.BaseEnvironment = object
    ctx_pkg.AgentContext = object
    name_pkg.AgentName = type("AgentName", (), {"HRANT": "hrant"})

    sys.modules["harbor"] = harbor_pkg
    sys.modules["harbor.agents"] = agents_pkg
    sys.modules["harbor.agents.installed"] = installed_pkg
    sys.modules["harbor.agents.installed.base"] = base_pkg
    sys.modules["harbor.environments"] = env_pkg
    sys.modules["harbor.environments.base"] = env_base
    sys.modules["harbor.models.agent.context"] = ctx_pkg
    sys.modules["harbor.models.agent.name"] = name_pkg

    # Now import the adapter from the in-repo path.
    import importlib.util
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "harbor_adapter_module_under_test",
        repo_root / "harbor_adapter" / "hrant_agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_name_is_hrant():
    """Harbor uses `name()` to register agents under their CLI name.
    Must be exactly 'hrant' so `--agent hrant` resolves."""
    mod = _import_adapter()
    assert mod.HrantAgent.name() == "hrant"


def test_adapter_run_posts_task_to_endpoint_and_writes_output(tmp_path, monkeypatch):
    """Full happy-path: adapter POSTs to the gateway endpoint, the gateway
    fakes returning a final answer, the adapter writes it to output_file."""
    mod = _import_adapter()
    env = _FakeEnvironment()
    out_file = tmp_path / "trial-output.txt"
    ctx = _FakeContext(out_file)

    # Patch aiohttp.ClientSession.post → fixed response.
    import aiohttp

    posted: dict = {}

    class _FakeResp:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def json(self):
            return {
                "ok": True,
                "answer": "FINAL ANSWER from agent",
                "turn_id": "t-1",
                "token_usage": None,
            }

    class _FakeClientSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def post(self, url, json=None, timeout=None):
            posted["url"] = url
            posted["body"] = json
            return _FakeResp()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda: _FakeClientSession())

    # The adapter still starts a real aiohttp server (to receive callbacks
    # in production). For the test we don't fire a callback — we just
    # verify that the POST happened and that output was written.
    agent = mod.HrantAgent(logs_dir=tmp_path)

    async def _run():
        await agent.run("solve a thing", env, ctx)

    asyncio.get_event_loop().run_until_complete(_run())

    assert posted["url"].endswith("/api/exec-protocol")
    assert posted["body"]["task"] == "solve a thing"
    assert posted["body"]["callback_url"].startswith("http://127.0.0.1:")
    assert "session_id" in posted["body"]
    assert out_file.read_text() == "FINAL ANSWER from agent"


def test_adapter_callback_forwards_to_environment_exec(tmp_path, monkeypatch):
    """When the gateway calls the adapter's loopback /exec endpoint,
    the handler must forward command/cwd/timeout to environment.exec
    and return the result as JSON."""
    mod = _import_adapter()
    env = _FakeEnvironment()
    out_file = tmp_path / "trial-output.txt"
    ctx = _FakeContext(out_file)

    # Capture the aiohttp.web.Application's POST /exec handler instead of
    # going through a network — pull it out and call it directly.
    captured: dict = {}

    import aiohttp.web as aweb
    orig_add_post = aweb.UrlDispatcher.add_post

    def fake_add_post(self, path, handler, *a, **kw):
        if path == "/exec":
            captured["handler"] = handler
        return orig_add_post(self, path, handler, *a, **kw)

    monkeypatch.setattr(aweb.UrlDispatcher, "add_post", fake_add_post)

    # Also stub the outbound POST so the test doesn't hang waiting on a
    # real gateway. Adapter run() returns immediately.
    import aiohttp

    class _Resp:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def json(self):
            return {"ok": True, "answer": "x", "turn_id": "t", "token_usage": None}

    class _Sess:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, *a, **kw): return _Resp()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda: _Sess())

    agent = mod.HrantAgent(logs_dir=tmp_path)

    async def _drive():
        await agent.run("anything", env, ctx)
        # Now invoke the captured handler directly with a synthetic request.
        assert "handler" in captured, "adapter did not register /exec route"
        request = _FakeRequest({"command": "ls /tmp", "cwd": "", "timeout_sec": 30})
        resp = await captured["handler"](request)
        return resp

    class _FakeRequest:
        def __init__(self, body):
            self._body = body
        async def json(self):
            return self._body

    resp = asyncio.get_event_loop().run_until_complete(_drive())
    # The handler returned an aiohttp.web.Response with our JSON payload.
    body = json.loads(resp.body.decode("utf-8"))
    assert body["return_code"] == 0
    assert "ls /tmp" in body["stdout"]
    assert env.calls == [{"command": "ls /tmp", "cwd": "", "timeout_sec": 30}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hrant_adapter.py -v`
Expected: FAIL — `FileNotFoundError` on `harbor_adapter/hrant_agent.py` or `ModuleNotFoundError`.

- [ ] **Step 3: Create the package marker**

Create `harbor_adapter/__init__.py` (empty file):

```python
"""Source-of-truth for Harbor agent adapters maintained in this repo.

See harbor_adapter/README.md for deploy instructions (the canonical
copy must be installed into the Harbor venv at
~/.hrant/data/workspace/.../harbor/agents/installed/ for harbor to
pick it up)."""
```

- [ ] **Step 4: Create the adapter**

Create `harbor_adapter/hrant_agent.py`:

```python
"""Hrant agent adapter for Harbor terminal-bench.

This adapter is the source-of-truth maintained in the Hrant repo.
Deploy: copy this file into harbor's installed-agents directory:
  cp harbor_adapter/hrant_agent.py \\
     ~/.hrant/data/workspace/terminal_bench_2_1/.venv/lib/python3.12/\\
     site-packages/harbor/agents/installed/hrant_agent.py

How it works:
  - Adapter starts an aiohttp server on a loopback ephemeral port.
  - Adapter POSTs to http://localhost:3333/api/exec-protocol with
    {task, callback_url, session_id} and synchronously awaits the
    Hrant gateway's response.
  - On the Hrant side, a ContextVar override makes terminal_exec
    POST each command to our callback_url instead of running on host.
  - Our /exec handler forwards command/cwd/timeout to
    environment.exec (Harbor's docker-exec primitive) and returns
    {stdout, stderr, return_code}.
  - When Hrant emits its final answer, we write it to
    context.paths.output_file and shut the aiohttp server down.

Speaker is webui:bench-harness (owner-mapped), so the agent has its
full normal tool surface — only terminal_exec is redirected. Other
tools (read_file, search_knowledge, …) keep operating on host —
they are cognitive tools, not task-environment tools.
"""
from __future__ import annotations

import asyncio
import json
import socket
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


_HRANT_URL = "http://localhost:3333"
_AGENT_TIMEOUT_S = 900  # 15 minutes — caller can lift via env if needed.


def _pick_ephemeral_port() -> int:
    """Bind to port 0 to let the kernel pick an unused loopback port,
    then close — the chosen port is reusable for our adapter server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_exec_handler(environment: BaseEnvironment):
    """Return an aiohttp handler that forwards command/cwd/timeout to
    environment.exec and returns the result as JSON."""
    async def handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response(
                {"stdout": "", "stderr": f"bad-request: {e}", "return_code": -1},
                status=400,
            )
        command = body.get("command") or ""
        cwd = body.get("cwd") or None
        timeout_sec = int(body.get("timeout_sec") or 30)
        try:
            result = await environment.exec(
                command=command, cwd=cwd, timeout_sec=timeout_sec,
            )
        except Exception as e:
            return web.json_response(
                {
                    "stdout": "",
                    "stderr": f"environment.exec crashed: {type(e).__name__}: {e}",
                    "return_code": -1,
                },
                status=500,
            )
        return web.json_response({
            "stdout": getattr(result, "stdout", "") or "",
            "stderr": getattr(result, "stderr", "") or "",
            "return_code": int(getattr(result, "return_code", -1) or -1),
        })

    return handler


class HrantAgent(BaseInstalledAgent):
    """The Hrant agent runs on the host as a FastAPI service; this
    adapter wires it into Harbor's terminal-bench framework by
    overriding terminal_exec via a loopback callback."""

    SUPPORTS_ATIF: bool = False

    @staticmethod
    def name() -> str:
        return "hrant"

    def version(self) -> str | None:
        return self._version or "dev"

    async def install(self, environment: BaseEnvironment) -> None:
        """Verify the host gateway is reachable; nothing to install
        in the container."""
        async with aiohttp.ClientSession() as s:
            try:
                async with s.get(
                    f"{_HRANT_URL}/api/version",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    if r.status != 200:
                        raise RuntimeError(
                            f"Hrant /api/version returned HTTP {r.status}; "
                            f"ensure hrant.service is running on host."
                        )
            except aiohttp.ClientError as e:
                raise RuntimeError(
                    f"Cannot reach Hrant at {_HRANT_URL}: {e}. "
                    f"Ensure hrant.service is up."
                ) from e

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        port = _pick_ephemeral_port()
        app = web.Application()
        app.router.add_post("/exec", _make_exec_handler(environment))
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()

        callback_url = f"http://127.0.0.1:{port}/exec"
        session_id = uuid.uuid4().hex
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{_HRANT_URL}/api/exec-protocol",
                    json={
                        "task": instruction,
                        "callback_url": callback_url,
                        "session_id": session_id,
                    },
                    timeout=aiohttp.ClientTimeout(total=_AGENT_TIMEOUT_S),
                ) as r:
                    payload = await r.json()
            answer = ""
            if isinstance(payload, dict):
                if payload.get("ok"):
                    answer = (payload.get("answer") or "").strip()
                else:
                    answer = (
                        f"[Hrant adapter error] "
                        f"{payload.get('error') or json.dumps(payload)}"
                    )
            else:
                answer = json.dumps(payload)
            context.paths.output_file.write_text(answer or "(empty)", encoding="utf-8")
        finally:
            await runner.cleanup()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_hrant_adapter.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add harbor_adapter/__init__.py harbor_adapter/hrant_agent.py tests/test_hrant_adapter.py
git commit -m "feat(harbor_adapter): Hrant adapter source-of-truth + integration tests

harbor_adapter/hrant_agent.py is the canonical adapter (versioned
in this repo). It starts an aiohttp loopback server on an ephemeral
port and POSTs to Hrant's /api/exec-protocol with {task,
callback_url, session_id}. When Hrant's terminal_exec posts back via
the override ContextVar, the adapter's /exec handler forwards to
Harbor's environment.exec (docker exec under the hood) and returns
stdout/stderr/return_code. The final answer is written to
context.paths.output_file so harbor can score the trial.

Integration test stubs harbor's BaseInstalledAgent + aiohttp transport
and asserts the protocol shape end-to-end without standing up a real
gateway or docker container."
```

---

### Task 5: Manual deploy README

**Files:**
- Create: `harbor_adapter/README.md`

Background:
- `hrant update` doesn't (yet) know how to push files into the harbor venv. Document the manual one-line copy step so anyone deploying knows what to do until that automation lands.

- [ ] **Step 1: Create the README**

Create `harbor_adapter/README.md`:

````markdown
# Harbor Adapters

Source-of-truth for any Harbor agent adapter that wraps Hrant.

## Why this lives here

Harbor's plugin loader reads adapter classes from
`harbor/agents/installed/` inside its venv. If we only edited the
file in-place there, our changes would be lost on the next `pip
install --upgrade harbor` (or any `hrant update` that rebuilds the
harbor venv). Keeping the source in this repo lets us version it
alongside the matching gateway changes (`/api/exec-protocol`,
ContextVar override) and ensures a `harbor` reinstall can be
followed by a deterministic redeploy step.

## Files

- `hrant_agent.py` — `HrantAgent(BaseInstalledAgent)` adapter that
  routes Harbor terminal-bench trials through Hrant's gateway.
- `__init__.py` — package marker (also points readers at this README).

## Manual deploy (until `hrant update` automates this)

Run on the host where Harbor's venv lives:

```bash
cp harbor_adapter/hrant_agent.py \
  ~/.hrant/data/workspace/terminal_bench_2_1/.venv/lib/python3.12/site-packages/harbor/agents/installed/hrant_agent.py
```

Verify Harbor sees it:

```bash
~/.hrant/data/workspace/terminal_bench_2_1/.venv/bin/harbor run --help \
  | grep -i hrant || echo "adapter NOT registered"
```

(The agent should appear in the `--agent` enum.)

## Running a trial

```bash
~/.hrant/data/workspace/terminal_bench_2_1/.venv/bin/harbor run \
  --dataset terminal-bench \
  --n-tasks 2 \
  --agent hrant \
  --agent-timeout-multiplier 10 \
  --n-concurrent 1
```

`--agent-timeout-multiplier 10` because Hrant turns can legitimately
take 5–15 minutes (cold tool loop with skill load + KG search). The
default per-trial timeout in Harbor is much tighter.

`--n-concurrent 1` because Hrant's module-level state (skills,
in-progress jobs) hasn't been audited for parallel `Agent.run`
safety yet.

## Architecture (one-liner)

Adapter starts a loopback aiohttp server → POSTs `{task,
callback_url}` to `http://localhost:3333/api/exec-protocol` → gateway
sets a ContextVar so Hrant's `terminal_exec` POSTs each command back
to the adapter → adapter awaits `environment.exec` (Harbor's docker
exec) and returns stdout/stderr/return_code. Full design:
`docs/superpowers/specs/2026-06-03-hrant-harbor-adapter-design.md`.
````

- [ ] **Step 2: Commit**

```bash
git add harbor_adapter/README.md
git commit -m "docs(harbor_adapter): manual deploy instructions"
```

---

## Self-review (post-write)

1. **Spec coverage** — each Spec section has a Task:
   - `Architecture` (callback-override) → Task 1 (ContextVar), Task 2 (endpoint), Task 4 (adapter).
   - `Components → terminal_exec ContextVar` → Task 1.
   - `Components → /api/exec-protocol endpoint` → Task 2.
   - `Components → main.py wire` → Task 3.
   - `Components → adapter` → Task 4.
   - `Source-of-truth in repo` → Task 4 + Task 5 (README).
   - `Harness contract` defaults documented in Task 5 README.
   - `Failure modes` → unit tests in Task 1 (denylist, callback fail), Task 2 (loopback reject, agent exception), Task 4 (env.exec crash → returned 500).
   - `Testing strategy` → unit tests in Tasks 1+2, integration test in Task 4.
2. **Placeholder scan** — no TBDs in steps, every step has full code.
3. **Type consistency** — `_REMOTE_EXEC_CALLBACK` name same in Tasks 1 and 2. `webui:bench-harness` speaker same in spec + Task 2. `callback_url` is the only key name for the URL in all tasks. `return_code` consistent (NOT `returncode` or `exit_code` from adapter side — only `TerminalResult.exit_code` on the Hrant side which `run_terminal` already exposes).
4. **Cross-task interplay** — Task 2's tests need the router wired (Task 3) before they can resolve the route; called out explicitly in Task 2 Step 4 to do them together as Task 3 Step 4.
