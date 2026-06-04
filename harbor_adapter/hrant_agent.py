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
  - When Hrant emits its final answer, we shut the aiohttp server
    down. Scoring is done by Harbor's per-task verifier against
    container state — the adapter does not produce a scoreable
    output file. The final answer is persisted to
    `self.logs_dir/agent_output.txt` for post-hoc debugging only.

Speaker is webui:bench-harness (owner-mapped), so the agent has its
full normal tool surface — only terminal_exec is redirected. Other
tools (read_file, search_knowledge, ...) keep operating on host —
they are cognitive tools, not task-environment tools.
"""
from __future__ import annotations

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
_AGENT_TIMEOUT_S = 1800  # 30 minutes — caller can lift via env if needed.
# Bumped from 900s after the 2026-06-04 hardened-bench run showed a
# handful of trials legitimately exceeding 15 minutes once Hrant
# started running the real /tests/ suite mid-turn (each pytest run
# adds 30-90s).


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
        # Pass cwd through verbatim — environment.exec decides how to
        # interpret empty strings vs None. Coercing "" -> None loses
        # the caller's explicit signal.
        cwd = body.get("cwd")
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
        # Use explicit None checks — return_code=0 is success and must
        # not collapse to -1 via the `or` short-circuit on falsy zero.
        rc = getattr(result, "return_code", None)
        return web.json_response({
            "stdout": getattr(result, "stdout", "") or "",
            "stderr": getattr(result, "stderr", "") or "",
            "return_code": int(rc) if rc is not None else -1,
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
        in the container.

        Retries with exponential backoff. The gateway can be momentarily
        unresponsive between trials while it finishes post-processing
        the previous trial's turn (conversation log save, skill
        reflection, supervisor handoff). A 10s timeout on the first
        attempt + two 30s/60s retries gives a 100s total budget before
        we give up — versus the prior single 10s attempt that
        intermittently failed during the 20-task bench."""
        import asyncio as _asyncio
        last_err: Exception | None = None
        for attempt_idx, timeout_s in enumerate((10, 30, 60), start=1):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"{_HRANT_URL}/api/version",
                        timeout=aiohttp.ClientTimeout(total=timeout_s),
                    ) as r:
                        if r.status != 200:
                            raise RuntimeError(
                                f"Hrant /api/version returned HTTP {r.status}; "
                                f"ensure hrant.service is running on host."
                            )
                        return
            except (aiohttp.ClientError, _asyncio.TimeoutError) as e:
                last_err = e
                if attempt_idx < 3:
                    await _asyncio.sleep(2 * attempt_idx)
                continue
        raise RuntimeError(
            f"Cannot reach Hrant at {_HRANT_URL} after 3 attempts "
            f"(timeouts 10s/30s/60s). Last error: "
            f"{type(last_err).__name__}: {last_err}. "
            f"Ensure hrant.service is up and not blocked by a long "
            f"post-turn cleanup."
        ) from last_err

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
            # Harbor scores the trial by running the task's verifier
            # against container state; the adapter's job is to complete
            # `run` without raising. We persist the final answer to
            # `self.logs_dir/agent_output.txt` for post-hoc debugging
            # only — Harbor never reads it.
            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                (self.logs_dir / "agent_output.txt").write_text(
                    answer or "(empty)", encoding="utf-8",
                )
            except Exception:
                pass
        finally:
            await runner.cleanup()
