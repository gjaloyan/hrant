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
    ctx = _FakeContext(tmp_path / "unused.txt")

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
        def post(self, url, json=None, timeout=None, headers=None):
            posted["url"] = url
            posted["body"] = json
            posted["headers"] = headers or {}
            return _FakeResp()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda: _FakeClientSession())
    monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "secret-for-harbor")

    agent = mod.HrantAgent(logs_dir=tmp_path)

    async def _run():
        await agent.run("solve a thing", env, ctx)

    asyncio.new_event_loop().run_until_complete(_run())

    assert posted["url"].endswith("/api/exec-protocol")
    assert posted["body"]["task"] == "solve a thing"
    assert posted["body"]["callback_url"].startswith("http://127.0.0.1:")
    assert "session_id" in posted["body"]
    # The endpoint requires a bearer since the 2026-09-05 audit; the
    # adapter has to carry it or every Harbor run 401s.
    assert posted["headers"]["Authorization"] == "Bearer secret-for-harbor"
    assert (tmp_path / "agent_output.txt").read_text() == "FINAL ANSWER from agent"


def test_adapter_callback_forwards_to_environment_exec(tmp_path, monkeypatch):
    """When the gateway calls the adapter's loopback /exec endpoint,
    the handler must forward command/cwd/timeout to environment.exec
    and return the result as JSON."""
    mod = _import_adapter()
    env = _FakeEnvironment()
    ctx = _FakeContext(tmp_path / "unused.txt")

    captured: dict = {}

    import aiohttp.web as aweb
    orig_add_post = aweb.UrlDispatcher.add_post

    def fake_add_post(self, path, handler, *a, **kw):
        if path == "/exec":
            captured["handler"] = handler
        return orig_add_post(self, path, handler, *a, **kw)

    monkeypatch.setattr(aweb.UrlDispatcher, "add_post", fake_add_post)

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

    class _FakeRequest:
        def __init__(self, body):
            self._body = body
        async def json(self):
            return self._body

    async def _drive():
        await agent.run("anything", env, ctx)
        assert "handler" in captured, "adapter did not register /exec route"
        request = _FakeRequest({"command": "ls /tmp", "cwd": "", "timeout_sec": 30})
        resp = await captured["handler"](request)
        return resp

    resp = asyncio.new_event_loop().run_until_complete(_drive())
    body = json.loads(resp.body.decode("utf-8"))
    assert body["return_code"] == 0
    assert "ls /tmp" in body["stdout"]
    assert env.calls == [{"command": "ls /tmp", "cwd": "", "timeout_sec": 30}]


def test_token_falls_back_to_the_repo_env_file(tmp_path, monkeypatch):
    """A bench run started from a plain shell does not inherit systemd's
    EnvironmentFile. The adapter reads the same `.env` the gateway does,
    or it 401s against a correctly configured server."""
    mod = _import_adapter()
    monkeypatch.delenv("HRANT_EXEC_PROTOCOL_TOKEN", raising=False)

    import pathlib
    env_file = pathlib.Path(mod.__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        import pytest
        pytest.skip("refusing to overwrite a real .env")
    env_file.write_text(
        "# comment\nOTHER=1\nHRANT_EXEC_PROTOCOL_TOKEN='from-env-file'\n",
        encoding="utf-8",
    )
    try:
        assert mod._auth_headers() == {"Authorization": "Bearer from-env-file"}
        # The environment still wins when it is set.
        monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "from-environ")
        assert mod._auth_headers() == {"Authorization": "Bearer from-environ"}
    finally:
        env_file.unlink()


def test_no_token_anywhere_sends_no_header(monkeypatch):
    mod = _import_adapter()
    monkeypatch.delenv("HRANT_EXEC_PROTOCOL_TOKEN", raising=False)
    monkeypatch.setattr(mod, "_token_from_env_file", lambda: "")
    assert mod._auth_headers() == {}
