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

    token = _te.set_remote_exec_callback(body.callback_url)
    try:
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
