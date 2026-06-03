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
