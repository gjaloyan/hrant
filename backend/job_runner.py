"""Thin wrapper around `Agent.run` that produces a durable Job
record. Every entry point (WebUI chat, Telegram handler, voice
intake) should call `run_tracked` instead of `agent.run` directly.

Why a wrapper instead of pushing tracking into `Agent.run`:
  - Agent.run already has a long signature + state machine; adding
    a hidden Job dependency makes unit tests harder.
  - run_tracked is the natural place to translate AgentAnswer back
    into a Job update (extract tool-call trace, persist response).
  - Phase B (failover) sits between this wrapper and Agent.run —
    keeps Agent itself provider-agnostic.

The wrapper does NOT swallow exceptions — the caller still gets
the exception so the WebUI / Telegram / SSE layer can surface it
to the user. The job's `failed` state is just our audit log; the
foreground caller decides how to present the failure.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import jobs
from .agent import Agent, AgentAnswer

log = logging.getLogger(__name__)


def _extract_tool_calls(answer: AgentAnswer) -> list[dict]:
    """Flatten the tool-call trace from an AgentAnswer into a list
    of compact dicts suitable for storage. Each entry:
        {name, args_summary, ok, error?, elapsed_ms?}

    We deliberately don't store full tool args — they can be huge
    (file contents, search results). The summary is enough for the
    WebUI's collapsed view; the full trace stays in TOKENS / the
    per-turn payload."""
    out: list[dict] = []
    for step in (answer.thinking_trace or []):
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        # `step.event` is one of 'tool' (success) | 'tool_error' (fail).
        # Anything else without a tool_call is filtered above.
        if step.event not in ("tool", "tool_error"):
            continue
        try:
            data = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
        except Exception:
            data = {}
        out.append({
            "name": data.get("name") or data.get("tool") or "?",
            "args_summary": (data.get("args_summary") or data.get("args") or "")[:200],
            "ok": step.event == "tool",
            "error": data.get("error") if step.event == "tool_error" else None,
            "elapsed_ms": data.get("elapsed_ms"),
        })
    return out


def run_tracked(
    agent: Agent,
    task: str,
    project: Optional[str] = None,
    attachments: Optional[list[str]] = None,
    *,
    channel: str = "webui",
    speaker_id: str = "webui:default",
    reply_to: Optional[dict] = None,
) -> tuple[AgentAnswer, str]:
    """Run a single agent turn with a Job record.

    Returns (answer, job_id). The Job is left in `completed` state
    on success, `failed` state on exception. The caller can attach
    the `job_id` to its response payload so the WebUI can link the
    SSE-streamed turn to the persistent job record.

    Exceptions propagate — the caller still has to handle the
    foreground error (return 5xx to client, send "I had a problem"
    to Telegram, etc.). The job record is just the persistent audit
    trail next to that path.
    """
    job = jobs.JOBS.create(
        prompt=task,
        channel=channel,
        speaker_id=speaker_id,
        reply_to=reply_to or {},
    )
    jobs.JOBS.mark_running(job.id)
    try:
        answer = agent.run(
            task,
            project,
            attachments,
            channel=channel,
            speaker_id=speaker_id,
        )
    except Exception as e:
        # Anything that escapes Agent.run is a real failure — LLM
        # provider error, tool crash, internal bug. Persist the
        # reason so the user can see it in the Jobs tab and retry.
        jobs.JOBS.mark_failed(job.id, error=f"{type(e).__name__}: {e}")
        log.exception("job %s failed", job.id)
        raise

    # Successful run — persist response + tool trace.
    jobs.JOBS.mark_completed(
        job.id,
        response=answer.answer or "",
        tool_calls=_extract_tool_calls(answer),
    )
    return answer, job.id
