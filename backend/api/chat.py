"""Chat SSE endpoint."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from ..agent import Agent
from ..conversation import CONVERSATION
from ..llm import TOKENS
from ..models import ChatRequest
from ..project_mode import PROJECTS
from ..sessions import SESSIONS

router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest):
    queue: asyncio.Queue = asyncio.Queue()

    def progress(event: str, msg: str, tool_call=None) -> None:
        # Round B: live tool-call streaming. When the agent's
        # progress() carries a structured ToolCallDetail (every
        # `event == "tool"` / `"tool_error"` step has one), serialize
        # it into the SSE payload so the WebUI can append a
        # ToolCallCard to the in-progress message in real time. For
        # text-only events (think, solve, verify, micro_ack, …) the
        # tool field is omitted, keeping the payload tiny.
        evt: dict = {"type": "progress", "event": event, "message": msg}
        if tool_call is not None:
            try:
                evt["tool_call"] = tool_call.model_dump()
            except Exception:
                evt["tool_call"] = None
        queue.put_nowait(evt)

    agent = Agent(progress=progress)

    async def runner():
        try:
            # Round C: caller can pick which channel context the
            # message belongs to. Default "webui" preserves prior
            # behaviour. "telegram" means the user is composing in
            # the WebUI to participate in the TG conversation
            # context; conversation memory + turn record are tagged
            # accordingly so a later TG turn picks up the thread.
            target_channel = (req.channel or "webui").strip().lower()
            if target_channel not in ("webui", "telegram"):
                target_channel = "webui"
            res = await asyncio.to_thread(
                lambda: agent.run(
                    req.message,
                    req.project or PROJECTS.current,
                    req.attachments or None,
                    channel=target_channel,
                ),
            )
            turn = {
                "ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": req.message,
                "answer": res.answer or "",
                "intent": "chat" if res.is_chat else "task",
                "is_chat": bool(res.is_chat),
                "confidence": res.verification.confidence if res.verification else 0,
                "topics": res.used_topics or [],
                # Round A: stamp the session entry with the on-disk
                # turn artefact id (P1) + the channel that produced
                # it. The frontend uses turn_id for lazy-loading
                # tool cards on history restore + channel for the
                # upcoming WebUI dropdown filter.
                "turn_id": getattr(res, "turn_id", "") or "",
                "channel": target_channel,
            }
            SESSIONS.add_turn(turn)
            if res.thinking_trace:
                TOKENS.save_request_trace(
                    question=req.message,
                    trace=[s.model_dump() for s in res.thinking_trace],
                    usage=res.token_usage.model_dump() if res.token_usage else {},
                )
            queue.put_nowait({"type": "answer", "data": res.model_dump()})
        except Exception as e:
            queue.put_nowait({"type": "error", "message": str(e)})
        finally:
            queue.put_nowait(None)

    async def stream() -> AsyncIterator[dict]:
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield {"data": json.dumps(item, ensure_ascii=False)}
        finally:
            await task

    return EventSourceResponse(stream())


# --- Round A: lazy-load turn artefacts + per-channel conversation -------


def _safe_turn_id(turn_id: str) -> str:
    """Defend against path traversal when looking up `<turn_id>.json`.
    Turn ids are timestamp + uuid hex (`20260507_120000_abc12345`); a
    request for `../../etc/passwd` mustn't escape `workspace/turns/`."""
    if not turn_id or "/" in turn_id or "\\" in turn_id or ".." in turn_id:
        raise HTTPException(status_code=400, detail="invalid turn_id")
    # Strip any `.json` extension the caller appended; we control it.
    return turn_id.rsplit(".json", 1)[0]


@router.get("/api/turns/{turn_id}")
async def get_turn(turn_id: str):
    """Return the full TurnWorkspace artefact for a single turn.

    The chat history endpoint returns lightweight rows (user message,
    short answer, turn_id pointer); the WebUI calls THIS endpoint when
    the user expands a message to see its tool calls / claims /
    evidence / verification / token breakdown. Lazy load keeps the
    chat history payload small while still letting any old turn
    surface its full record on demand.
    """
    safe = _safe_turn_id(turn_id)
    from ..workspace import get_workspace
    target = get_workspace().root / "turns" / f"{safe}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="turn not found")
    try:
        body = target.read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"read failed: {e}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"corrupted turn: {e}")
    return data


@router.get("/api/conversation")
async def get_conversation(
    channel: Optional[str] = Query(default=None),
    n: int = Query(default=50, ge=1, le=500),
):
    """Recent conversation turns, optionally filtered by `channel`.

    Round A: WebUI loads its own history on mount via this endpoint
    so a page refresh restores the chat (including turn_id pointers
    so per-message tool cards can be expanded). `channel=telegram`
    surfaces the Telegram conversation in the WebUI dropdown that
    Round C will add.

    Default channel is unset (returns ALL turns) so older clients
    that don't pass the param keep working — but those clients see
    cross-channel history which can be confusing. WebUI v2 always
    passes `channel=webui`.
    """
    turns = CONVERSATION.recent_full(n=n, channel=channel)
    return {"channel": channel, "turns": turns, "count": len(turns)}
