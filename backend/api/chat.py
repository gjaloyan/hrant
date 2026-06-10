"""Chat SSE endpoint."""
from __future__ import annotations
import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse


class AnswerQuestionRequest(BaseModel):
    """Audit follow-up (M-1): strict request shape for
    /api/chat/answer-question. The old `req: dict` was loosely
    typed — a malformed payload silently fell through to "404
    question_id" or worse. Pydantic gives us field-level errors
    that surface as 422 with a concrete message."""
    question_id: str = Field(..., min_length=1, max_length=64)
    choice: str = Field(default="", max_length=512)
    text: str = Field(default="", max_length=4000)

log = logging.getLogger(__name__)

from ..agent import Agent
from ..conversation import CONVERSATION
from ..job_runner import run_tracked
from ..llm import TOKENS
from ..models import ChatRequest
from ..project_mode import PROJECTS
from ._auth import require_owner_for_writes
from ._rate_limit import check_chat_rate

router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    # Owner-only chat from the WebUI. Telegram + other channels go
    # through their own handlers in channels.py and don't hit this
    # endpoint. Without this gate, a `--gateway`-bound install lets
    # anyone reachable consume the owner's LLM quota.
    require_owner_for_writes(action="chatting with the agent")
    # Audit #9: per-IP rate-limit. Even with the role gate above
    # in place, a leaked session or future bug could let unauth
    # traffic through; the limiter is defence-in-depth.
    check_chat_rate(request)
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
            # Audit-fix #2: SESSIONS.add_turn used to live HERE, but
            # the Telegram channel and CLI never duplicated it, so
            # 99% of production turns were invisible in the Sessions
            # panel. The write now happens inside `run_unified`
            # itself — one source of truth for every caller. This
            # endpoint no longer needs to assemble the turn record.
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
            # log.exception so the FULL traceback lands in journalctl.
            # Pre-fix, this handler emitted `{type: error, message:
            # str(e)}` only — when something raised a slice (a real
            # production bug we're tracking) the SSE event carried
            # `slice(None, 200, None)` to the WebUI and there was no
            # traceback to diagnose from.
            log.exception(
                "chat SSE handler failed for speaker=%s message=%r",
                target_speaker, (req.message or "")[:100],
            )
            queue.put_nowait({
                "type": "error",
                "message": f"{type(e).__name__}: {str(e) or 'no detail'}",
            })
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


# --- AskUserQuestion follow-up: answer a pending question ---------------


@router.post("/api/chat/answer-question")
async def answer_question(req: AnswerQuestionRequest, request: Request):
    """Submit a user's answer to a `ask_user`-issued question.

    Body shape:
      { "question_id": "q-...", "choice": "opt-0", "text": "..." }

    `choice` is the option `id` the user picked (for single-select)
    or a comma-separated list of ids (for multi-select). `text` is
    an optional free-text fallback (e.g. when the user typed "Other"
    via Telegram). Behaviour:

      1. Marks the question answered in the store (idempotent — a
         second click after the answer landed is a no-op return).
      2. Synthesizes a user-visible message describing the choice
         (e.g. "I picked: chocolate") and fires a fresh agent turn
         via the normal `/api/chat` pipeline so conversation memory
         + tool loop continue from where the previous turn ended.

    Returns the new turn's AgentAnswer JSON (same shape as `/api/chat`
    minus the SSE streaming — the WebUI calls this from a button
    click handler, not a chat-input form, so we keep the response
    flat)."""
    require_owner_for_writes(action="answering a pending question")
    check_chat_rate(request)
    from ..tools import ask_user as _aq
    qid = req.question_id.strip()
    choice = req.choice.strip()
    text = req.text.strip()
    if not qid:
        raise HTTPException(status_code=400, detail="question_id required")
    q_before = _aq.STORE.get(qid)
    if q_before is None:
        raise HTTPException(status_code=404, detail="question not found")
    if q_before.answered:
        # Idempotent return — the question already has an answer.
        # Surface the prior choice so the UI can update without
        # firing a second turn.
        return {
            "ok": True,
            "already_answered": True,
            "question_id": qid,
            "answer_choice": q_before.answer_choice,
            "answer_text": q_before.answer_text,
        }
    q = _aq.STORE.mark_answered(qid, choice=choice, text=text)
    if q is None:
        raise HTTPException(status_code=500, detail="failed to persist answer")
    # Compose a user-visible "I picked …" message that the agent's
    # next turn sees as the new user input. Prefer the option label
    # over the raw id so the conversation log reads naturally.
    label_for_choice = ""
    for opt in q.options:
        if opt.get("id") == choice:
            label_for_choice = opt.get("label") or ""
            break
    if not label_for_choice and "," in choice:
        # multi-select — assemble labels in order
        ids = [c.strip() for c in choice.split(",") if c.strip()]
        labels = []
        for oid in ids:
            for opt in q.options:
                if opt.get("id") == oid:
                    labels.append(opt.get("label") or oid)
                    break
        label_for_choice = ", ".join(labels)
    if text:
        user_message = f"My answer (free text): {text}"
    elif label_for_choice:
        user_message = f"My choice: {label_for_choice}"
    else:
        user_message = f"My choice: {choice or '(none)'}"
    # Run the new turn synchronously and return its result. Same
    # pattern as /api/chat but without SSE streaming — the answer
    # is delivered as a flat JSON response so the UI can render it
    # inline below the question card.
    from ..sessions import normalize_speaker
    speaker = normalize_speaker(q.asker_speaker_id or "webui:default")
    agent = Agent()
    res, job_id = await asyncio.to_thread(
        lambda: run_tracked(
            agent,
            user_message,
            PROJECTS.current,
            None,
            channel="webui" if q.channel == "webui" else "telegram",
            speaker_id=speaker,
        ),
    )
    out = res.model_dump()
    out["job_id"] = job_id
    out["answered_question_id"] = qid
    out["answer_choice"] = choice
    out["answer_text"] = text
    return out


@router.get("/api/chat/open-questions")
async def open_questions():
    """List un-answered questions so the WebUI can restore them on
    page reload."""
    from ..tools import ask_user as _aq
    items = _aq.STORE.list_open(limit=50)
    return {"questions": [q.to_dict() for q in items]}


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
