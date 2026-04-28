"""Chat SSE endpoint."""
from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from ..agent import Agent
from ..llm import TOKENS
from ..models import ChatRequest
from ..project_mode import PROJECTS
from ..sessions import SESSIONS

router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest):
    queue: asyncio.Queue = asyncio.Queue()

    def progress(event: str, msg: str) -> None:
        queue.put_nowait({"type": "progress", "event": event, "message": msg})

    agent = Agent(progress=progress)

    async def runner():
        try:
            res = await asyncio.to_thread(agent.run, req.message, req.project or PROJECTS.current)
            turn = {
                "ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": req.message,
                "answer": res.answer or "",
                "intent": "chat" if res.is_chat else "task",
                "is_chat": bool(res.is_chat),
                "confidence": res.verification.confidence if res.verification else 0,
                "topics": res.used_topics or [],
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
