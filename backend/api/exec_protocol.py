"""Harbor terminal-bench adapter endpoint.

POST /api/exec-protocol drives one bench trial:
  - Set the `_REMOTE_EXEC_CALLBACK` ContextVar to `callback_url` so
    Hrant's `terminal_exec` posts each command to the adapter's
    loopback server instead of running on the host.
  - Run a single `Agent.run` turn as the owner-mapped bench harness.
  - Return the final answer to the adapter, which writes it to the
    trial output file harbor scores against.

Trust model, rewritten 2026-09-06 after an external audit. It used to
read "localhost-only", resting on the gateway binding to 127.0.0.1 —
a property of deployment, not of this endpoint. There was NO
authentication: any caller who could reach it got `Agent.run` as
`webui:bench-harness`, which `roles._IMPLICIT_OWNERS` maps to OWNER —
terminal_exec with no sandbox, self-modification, sending as the owner
on Telegram. A defect waiting for the day the gateway is published,
which is exactly when nobody re-reads this file.

And the callback guard was `startswith("http://127.0.0.1:")`, a string
prefix: `http://127.0.0.1@evil.example/` passes it and resolves to
evil.example.

So now: the endpoint does not exist unless HRANT_EXEC_PROTOCOL_TOKEN is
set, and the caller must present it as `Authorization: Bearer <token>`.
The callback host is PARSED and must be loopback. The Harbor adapter is
a real workflow, so this authenticates rather than removes it.

Speaker: `webui:bench-harness`, still owner-mapped — the harness needs
the full tool surface. The gate is at the door now, not in the role.

Concurrency: this endpoint is sync; FastAPI runs it in a worker
thread. The ContextVar is per-thread-task, so concurrent trials
do not share state. The current Harbor harness runs with
`--n-concurrent 1` anyway.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import secrets
import uuid
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..tools import terminal_exec as _te


log = logging.getLogger(__name__)


router = APIRouter()


TOKEN_ENV = "HRANT_EXEC_PROTOCOL_TOKEN"

_LOOPBACK_HOSTS = frozenset({"localhost"})


def _configured_token() -> str:
    return (os.environ.get(TOKEN_ENV) or "").strip()


def _is_loopback_callback(url: str) -> bool:
    """Does this URL actually point at this machine?

    Parsed, not prefix-matched. `http://127.0.0.1@evil.example/` starts
    with the old prefix and resolves to evil.example.
    """
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return False
    if (parsed.scheme or "").lower() != "http":
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _run_agent(*, task: str, session_key: str):
    """Behind a name so tests can stand in for a whole agent turn."""
    from ..agent import Agent
    return Agent().run(
        task, speaker_id="webui:bench-harness", channel="webui",
        session_key=session_key,
    )


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
def exec_protocol(body: ExecProtocolRequest,
                  authorization: str = Header(default="")):
    token = _configured_token()
    if not token:
        # 404, not 403: an endpoint nobody enabled should not announce
        # that it is there.
        raise HTTPException(status_code=404, detail="Not Found")
    presented = ""
    if authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    if not presented or not secrets.compare_digest(presented, token):
        raise HTTPException(
            status_code=401,
            detail=f"exec-protocol requires a valid {TOKEN_ENV} bearer token",
        )
    if not _is_loopback_callback(body.callback_url):
        raise HTTPException(
            status_code=400,
            detail=(
                "callback_url must resolve to loopback "
                f"(http://127.0.0.1:<port>/...); got {body.callback_url!r}"
            ),
        )

    session_key = (body.session_id or "").strip() or uuid.uuid4().hex

    token = _te.set_remote_exec_callback(body.callback_url)
    try:
        result = _run_agent(task=body.task, session_key=session_key)
        answer = getattr(result, "answer", "") or ""
        turn_id = getattr(result, "turn_id", "") or ""
        token_usage = getattr(result, "token_usage", None)
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
