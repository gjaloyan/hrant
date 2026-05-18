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

import json
import logging
from typing import Optional

from . import jobs
from .agent import Agent, AgentAnswer

log = logging.getLogger(__name__)


_ARGS_SUMMARY_CAP = 1500


def _summarize_tool_args(raw: object) -> str:
    """Render whatever the `args` field holds as a short string for
    storage. `ToolCallDetail.args` is typed as `dict`, so the natural
    repr is JSON. Pre-fix, the code did
        `(data.get("args_summary") or data.get("args") or "")[:200]`
    and crashed with `KeyError: slice(None, 200, None)` when args
    was a non-empty dict — `dict[:200]` does dict lookup, not string
    slicing. Caught in production via /api/chat + Telegram on every
    tool-using turn (web_search, fetch_url, read_file).

    Cap is 1500 bytes — enough to capture a full terminal_exec
    command, a typical fetch_url, or the head of a long run_python
    payload. The earlier 200-byte cap left every interesting
    debugging case as the literal empty string in jobs.json, which
    made the post-Phase-3 audit unable to reconstruct what the
    agent actually tried during a failed video-processing turn."""
    if raw is None or raw == "" or raw == {}:
        return ""
    if isinstance(raw, str):
        return raw[:_ARGS_SUMMARY_CAP]
    if isinstance(raw, (dict, list)):
        try:
            return json.dumps(raw, ensure_ascii=False)[:_ARGS_SUMMARY_CAP]
        except Exception:
            return str(raw)[:_ARGS_SUMMARY_CAP]
    return str(raw)[:_ARGS_SUMMARY_CAP]


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
            "args_summary": _summarize_tool_args(
                data.get("args_summary") or data.get("args")
            ),
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
    session_key: Optional[str] = None,
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
    # Phase B: make the active job id visible to the failover layer
    # via ContextVar so every LLM call inside `agent.run` can append
    # its (provider, model, ok, error) attempt onto the same Job.
    from . import failover as _fo
    token = _fo.set_current_job_id(job.id)
    # Track whether a terminal status (completed / failed) was
    # written. If we leave the function without writing one, the
    # `finally` block stamps `failed` — pre-fix, a crash AFTER
    # `mark_running` but BEFORE `mark_completed` (e.g. the slice
    # KeyError in `_extract_tool_calls`) left the job stuck in
    # `running` forever. The new flag closes that gap.
    terminal_written = False
    answer: Optional[AgentAnswer] = None
    try:
        try:
            answer = agent.run(
                task,
                project,
                attachments,
                channel=channel,
                speaker_id=speaker_id,
                session_key=session_key,
                job_id=job.id,
            )
        except Exception as e:
            # Anything that escapes Agent.run is a real failure — LLM
            # provider error, tool crash, internal bug. Persist the
            # reason so the user can see it in the Jobs tab and retry.
            jobs.JOBS.mark_failed(job.id, error=f"{type(e).__name__}: {e}")
            terminal_written = True
            log.exception("job %s failed", job.id)
            raise

        # Successful run — persist response + tool trace. The
        # `_extract_tool_calls` call is the historical crash site
        # (slice KeyError, fixed in 66c506d3) so we still wrap the
        # whole block: any future bug here marks failed instead of
        # leaving the job orphaned in `running`.
        try:
            tool_calls = _extract_tool_calls(answer)
        except Exception as e:
            jobs.JOBS.mark_failed(
                job.id,
                error=f"post-run tool-trace serialisation failed: "
                      f"{type(e).__name__}: {e}",
            )
            terminal_written = True
            log.exception(
                "job %s failed post-run during tool-trace extraction",
                job.id,
            )
            raise
        try:
            jobs.JOBS.mark_completed(
                job.id,
                response=answer.answer or "",
                tool_calls=tool_calls,
            )
            terminal_written = True
        except Exception as e:
            jobs.JOBS.mark_failed(
                job.id,
                error=f"mark_completed failed: {type(e).__name__}: {e}",
            )
            terminal_written = True
            log.exception(
                "job %s failed at mark_completed step", job.id,
            )
            raise
        return answer, job.id
    finally:
        # Always clear the ContextVar — leaking it would mean the
        # next request's LLM calls record onto the previous Job.
        _fo.reset_current_job_id(token)
        # Belt-and-braces: if execution somehow exits this function
        # without a terminal status written (e.g. KeyboardInterrupt,
        # SystemExit, an unwound future raise that we didn't model),
        # stamp the job as failed so the Jobs tab doesn't show a
        # phantom "running" row forever.
        if not terminal_written:
            try:
                jobs.JOBS.mark_failed(
                    job.id,
                    error="job exited without a terminal status "
                          "(no exception, no completion)",
                )
            except Exception:
                # Last-resort logging only — don't shadow the original
                # exit cause with a bookkeeping error.
                log.exception(
                    "job %s: also failed to write fallback `failed` status",
                    job.id,
                )
