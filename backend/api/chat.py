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
from ..job_runner import run_tracked
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
            # Caller picks which channel context the message belongs
            # to (default "webui"). "telegram" means the user is
            # composing in the WebUI to participate in a TG
            # conversation; the answer is also forwarded out the bot.
            target_channel = (req.channel or "webui").strip().lower()
            if target_channel not in ("webui", "telegram"):
                target_channel = "webui"
            # Phase 10: speaker_id is the primary partition key for
            # sessions + conversation memory + per-speaker user
            # profile. Default for WebUI: 'webui:default'. The UI
            # may pass a richer id (e.g. 'webui:gor') if a future
            # multi-user mode lands.
            from ..sessions import normalize_speaker
            target_speaker = normalize_speaker(req.speaker_id or f"{target_channel}:default")
            # Job tracking — every turn gets a durable record. If
            # this server dies mid-run, the boot recovery hook will
            # mark the job `interrupted` so the user can retry it
            # from the WebUI Jobs tab or `hrant jobs retry <id>`.
            res, job_id = await asyncio.to_thread(
                lambda: run_tracked(
                    agent,
                    req.message,
                    req.project or PROJECTS.current,
                    req.attachments or None,
                    channel=target_channel,
                    speaker_id=target_speaker,
                ),
            )
            # Round F-pre: include cheap summary fields directly in
            # the session row so the WebUI badges (token usage, tool
            # count, LLM count) survive a page refresh without
            # waiting for the lazy /api/turns/<id> fetch. Heavy data
            # (full thinking_trace, claims, evidence) still comes
            # via lazy load — these are just the ~5 small numbers a
            # restored chat needs to show counts and a token bar.
            tu = res.token_usage
            n_tools = sum(
                1 for s in (res.thinking_trace or [])
                if s.tool_call and (s.event == "tool" or s.event == "tool_error")
            )
            n_llm = len(res.llm_calls or [])
            turn = {
                "ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": req.message,
                "answer": res.answer or "",
                "intent": "chat" if res.is_chat else "task",
                "is_chat": bool(res.is_chat),
                "confidence": res.verification.confidence if res.verification else 0,
                "topics": res.used_topics or [],
                "turn_id": getattr(res, "turn_id", "") or "",
                "channel": target_channel,
                "speaker_id": target_speaker,
                "token_usage": tu.model_dump() if tu else None,
                "n_tool_calls": n_tools,
                "n_llm_calls": n_llm,
                # Link the session entry to its durable job record so the
                # WebUI can deep-link Conversation → Jobs.
                "job_id": job_id,
            }
            SESSIONS.add_turn(turn, speaker_id=target_speaker)
            if res.thinking_trace:
                TOKENS.save_request_trace(
                    question=req.message,
                    trace=[s.model_dump() for s in res.thinking_trace],
                    usage=res.token_usage.model_dump() if res.token_usage else {},
                )
            # Round E: TG forward. When the WebUI composed this turn
            # AS-IF it came from Telegram (channel=telegram), drop
            # the answer into the TG bot's most-recent chat too so
            # the user's TG thread doesn't go silent. Best-effort —
            # forwarding failure leaves the WebUI answer intact.
            if target_channel == "telegram":
                try:
                    from ..channels import CHANNELS
                    forwarded = CHANNELS.send_to_first_telegram(res.answer or "")
                    if not forwarded:
                        # Surface the no-bot-running case so the user
                        # in WebUI knows why TG stayed quiet. It
                        # rides on the SSE stream as a synthetic
                        # progress message; the answer event still
                        # arrives normally below.
                        queue.put_nowait({
                            "type": "progress",
                            "event": "tg_forward",
                            "message": (
                                "TG forward skipped: no Telegram bot is "
                                "currently running or it has no chat to "
                                "reply to yet. Send a message to the bot "
                                "from Telegram first."
                            ),
                        })
                except Exception as _e:
                    queue.put_nowait({
                        "type": "progress",
                        "event": "tg_forward",
                        "message": f"TG forward error: {_e}",
                    })
            # Attach job_id so the WebUI can deep-link to the Jobs
            # tab from the answer message (small string, no payload
            # bloat).
            answer_payload = res.model_dump()
            answer_payload["job_id"] = job_id
            queue.put_nowait({"type": "answer", "data": answer_payload})
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
    speaker_id: Optional[str] = Query(default=None),
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
    turns = CONVERSATION.recent_full(n=n, channel=channel, speaker_id=speaker_id)
    return {
        "channel": channel,
        "speaker_id": speaker_id,
        "turns": turns,
        "count": len(turns),
    }
