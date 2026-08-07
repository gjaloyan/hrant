"""Dispatch a subtask to a role-specific subagent.

The dispatch is a single LLM tool-loop run with:
  - the role's `system_prompt`,
  - only the role's allowed tools exposed (everything else is
    invisible to the role's LLM),
  - role-specific `max_iterations` cap.

This is deliberately NOT a full `Agent.run` invocation. Skipping the
full pipeline (intent classifier, think, verify, learn, memory
extraction) keeps subagent cost predictable and the parent context
clean. The parent sees ONE tool-call result — the child's final
answer — not the child's intermediate trace.

Recursive delegation is blocked: the `delegate` tool is intentionally
NOT in any role's `tools` list, so the LLM running a subagent can't
spawn its own subagent. Depth is hard-capped at 1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..llm import LLMError, TaskType, router
from ..roles import current_speaker, is_owner
from .roles import available_role_names, get_role
from .store import SUBAGENT_STORE


# Hard cap shared by the dispatcher + the recursive-delegation guard.
# Production hermes uses the same default (configurable up to 3).
MAX_DEPTH: int = 1


@dataclass
class SubagentResult:
    """All a parent (or its caller) needs to know about a subagent run."""
    ok: bool
    role: str
    task: str
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str = ""
    iterations: int = 0
    # Tool-name → call-count summary so the parent's audit trail can
    # report "researcher used web_search(2) + fetch_url(3)" without
    # carrying the full result bodies.
    tool_summary: dict[str, int] = field(default_factory=dict)


def available_roles() -> dict[str, str]:
    """Map role name → one-line description. Used by the `delegate`
    tool description so the parent LLM knows what's available."""
    out: dict[str, str] = {}
    for name in available_role_names():
        role = get_role(name)
        if role is not None:
            out[name] = role.description
    return out


def _tool_schemas_for_role(allowed: tuple[str, ...]) -> list[dict]:
    """Build the JSON schema list `call_with_tools` expects, filtered
    to the role's allowed names. Schemas are pulled from the global
    registry — we don't redefine them per role so the role's view of
    a tool always matches the parent's."""
    from ..tool_registry import get_registry as _get_registry

    full = _get_registry()
    schemas: list[dict] = []
    for name in allowed:
        spec = full.tools.get(name)
        if spec is None:
            continue
        # spec.to_anthropic_schema() / spec.schema_dict() varies by
        # ToolRegistry implementation; we just need {name,
        # description, input_schema} for the Anthropic-style call.
        schemas.append({
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        })
    return schemas


def _make_tool_executor(allowed: tuple[str, ...]):
    """Build an execute_tool(name, args) → (result_text, is_error)
    callback that ONLY runs tools in the role's allowlist. A request
    for a name outside the allowlist returns a structured error
    string + is_error=True — the LLM sees that the tool isn't
    available and (hopefully) stops trying."""
    from ..tool_registry import get_registry as _get_registry

    allowed_set = set(allowed)
    full = _get_registry()

    def _execute(name: str, args: dict) -> tuple[str, bool]:
        if name not in allowed_set:
            import json as _j
            return _j.dumps({
                "error": (
                    f"tool {name!r} is not available to this subagent role; "
                    f"allowed: {sorted(allowed_set)}"
                ),
            }, ensure_ascii=False), True
        result, is_error = full.execute(name, args)
        # Subagents used to call full.execute directly, bypassing the parent
        # turn's interception point entirely — so a builder subagent that
        # edited a config and never restarted the service produced an EMPTY
        # parent contract, and the parent reported success (2026-08-06). The
        # contract is a ContextVar, so a subagent running in this thread
        # shares the parent's; the obligation it raises is the parent's to
        # discharge.
        try:
            from ..unified_agent import _BUILD_WRITE_TOOLS
            if not is_error and name in _BUILD_WRITE_TOOLS:
                from ..turn_contract import note_mutation
                marker = note_mutation()
                if marker:
                    result = f"{marker}{result}"
        except Exception:
            pass
        return result, is_error

    return _execute


def _current_parent_job_id() -> str:
    """Best-effort lookup of the parent agent's Job.id. Used so the
    persisted SubagentSession can be linked back to the row in the
    WebUI's Jobs tab. Returns '' when no parent job is in scope
    (CLI / test / direct API call)."""
    try:
        from .. import failover as _fo
        sid = _fo.get_current_job_id()
        return sid or ""
    except Exception:
        return ""


def run_subagent(
    role_name: str,
    task: str,
    *,
    depth: int = 0,
    require_owner: bool = True,
    speaker_id: str | None = None,
    on_progress: Callable[[str, str], None] | None = None,
    on_session: Callable[[str], None] | None = None,
) -> SubagentResult:
    """Run one subagent turn for the given role.

    Parameters:
      role_name: registry key (researcher / coder / reviewer).
      task:      free-form task the role should answer.
      depth:     0 = parent calling; >=1 = nested. Caller passes the
                 parent's depth; we refuse if depth >= MAX_DEPTH.
      require_owner: when True, the dispatch refuses unless the
                 current speaker (via the speaker ContextVar) is the
                 OWNER. Subagents have direct access to tools that
                 cost money / hit external services — non-owner
                 callers shouldn't be able to spawn them.
      speaker_id: optional override of the speaker check (used by
                 internal callers that have already authorised).
      on_progress: optional `(event, message)` callback so the
                 parent's progress stream can show "🔀 delegating to
                 researcher" / "🔀 researcher: 3 tools used".

    Returns: SubagentResult — never raises. All failure modes land
    in `ok=False` with an `error` field the parent can surface.
    """
    start = time.monotonic()
    role = get_role(role_name)
    if role is None:
        return SubagentResult(
            ok=False, role=role_name, task=task, answer="",
            elapsed_ms=int((time.monotonic() - start) * 1000),
            error=(
                f"unknown role {role_name!r}; available: "
                f"{', '.join(available_role_names())}"
            ),
        )

    if depth >= MAX_DEPTH:
        return SubagentResult(
            ok=False, role=role.name, task=task, answer="",
            elapsed_ms=int((time.monotonic() - start) * 1000),
            error=(
                f"delegation depth {depth} >= MAX_DEPTH {MAX_DEPTH}; "
                "nested delegation is disabled"
            ),
        )

    if require_owner:
        sid = speaker_id if speaker_id is not None else current_speaker()
        if not is_owner(sid):
            return SubagentResult(
                ok=False, role=role.name, task=task, answer="",
                elapsed_ms=int((time.monotonic() - start) * 1000),
                error="permission denied — delegate is owner-only",
            )

    if not task or not task.strip():
        return SubagentResult(
            ok=False, role=role.name, task=task, answer="",
            elapsed_ms=int((time.monotonic() - start) * 1000),
            error="empty task",
        )

    if on_progress is not None:
        try:
            on_progress("delegate", f"→ {role.name}: {task[:80]}")
        except Exception:
            pass

    # Persist a session BEFORE the LLM call so the WebUI's "Active"
    # panel can show it live. `parent_job_id` links the session back
    # to the row in the Jobs tab (when called from inside an agent
    # turn — empty for direct API / test callers).
    parent_speaker = speaker_id if speaker_id is not None else (current_speaker() or "")
    session = SUBAGENT_STORE.create(
        role=role.name,
        task=task,
        parent_job_id=_current_parent_job_id(),
        parent_speaker=parent_speaker,
    )
    if on_session is not None:
        # Hand the session id to the caller immediately — background
        # delegation returns this as a ticket while the run continues.
        try:
            on_session(session.id)
        except Exception:
            pass

    tool_schemas = _tool_schemas_for_role(role.tools)
    execute_tool = _make_tool_executor(role.tools)

    # Tool-call accounting so the parent can show a tight summary
    # without ferrying the whole tool trace.
    tool_summary: dict[str, int] = {}
    tool_records: list[dict] = []

    def _on_tool_call(name: str, args: dict, result: str, is_error: bool) -> None:
        tool_summary[name] = tool_summary.get(name, 0) + 1
        # Cap the per-call result preview so a giant web_search blob
        # doesn't bloat the SubagentResult payload.
        preview = (result or "")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        tool_records.append({
            "name": name,
            "args": args or {},
            "result_preview": preview,
            "is_error": bool(is_error),
        })

    task_type_name = (role.task_type or "complex_solving").lower()
    try:
        task_type = getattr(TaskType, task_type_name.upper())
    except AttributeError:
        task_type = TaskType.COMPLEX_SOLVING

    try:
        rt = router()
        answer = rt.call_with_tools(
            task_type,
            role.system_prompt,
            task,
            tools=tool_schemas,
            execute_tool=execute_tool,
            max_iterations=role.max_iterations,
            on_tool_call=_on_tool_call,
        )
    except LLMError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        SUBAGENT_STORE.finalize(
            session.id, status="failed",
            error=f"LLM error: {e}",
            iterations=sum(tool_summary.values()),
            tool_summary=tool_summary,
            tool_calls=tool_records,
            elapsed_ms=elapsed,
        )
        return SubagentResult(
            ok=False, role=role.name, task=task, answer="",
            tool_calls=tool_records,
            tool_summary=tool_summary,
            iterations=sum(tool_summary.values()),
            elapsed_ms=int((time.monotonic() - start) * 1000),
            error=f"LLM error: {e}",
        )
    except Exception as e:  # pragma: no cover — defensive
        elapsed = int((time.monotonic() - start) * 1000)
        SUBAGENT_STORE.finalize(
            session.id, status="failed",
            error=f"{type(e).__name__}: {e}",
            iterations=sum(tool_summary.values()),
            tool_summary=tool_summary,
            tool_calls=tool_records,
            elapsed_ms=elapsed,
        )
        return SubagentResult(
            ok=False, role=role.name, task=task, answer="",
            tool_calls=tool_records,
            tool_summary=tool_summary,
            iterations=sum(tool_summary.values()),
            elapsed_ms=elapsed,
            error=f"unexpected error: {type(e).__name__}: {e}",
        )

    answer_text = (answer or "").strip()
    if on_progress is not None:
        try:
            summary_str = ", ".join(f"{n}({c})" for n, c in tool_summary.items()) or "no tools"
            on_progress(
                "delegate_done",
                f"← {role.name}: {len(answer_text)} chars, {summary_str}",
            )
        except Exception:
            pass

    elapsed = int((time.monotonic() - start) * 1000)
    final_status = "completed" if answer_text else "failed"
    final_error = "" if answer_text else "empty answer from subagent"
    SUBAGENT_STORE.finalize(
        session.id, status=final_status,
        answer=answer_text,
        error=final_error,
        iterations=sum(tool_summary.values()),
        tool_summary=tool_summary,
        tool_calls=tool_records,
        elapsed_ms=elapsed,
    )
    return SubagentResult(
        ok=bool(answer_text),
        role=role.name,
        task=task,
        answer=answer_text,
        tool_calls=tool_records,
        tool_summary=tool_summary,
        iterations=sum(tool_summary.values()),
        elapsed_ms=elapsed,
        error=final_error,
    )
