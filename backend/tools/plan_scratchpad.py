"""Per-turn plan scratchpad — plan-execute-verify for interactive turns.

AGI-roadmap item (2026-06-11). The supervisor pattern gives background
jobs an LLM that plans, retries, and seals chains — interactive turns
had nothing equivalent: on a 5-step task a small model wanders,
forgets steps, and declares victory early. The post-hoc endpoint
check catches non-delivery of the FINAL action but can't see that
step 3 of 5 was silently dropped.

The mechanism is deliberately lightweight — a visible checklist, not
a controller:

  set_plan(steps)            — LLM declares the checklist up front.
  update_plan(step, status)  — marks steps done/skipped as it works.

Both tools echo the FULL current checklist in their result, so the
plan re-enters the conversation context on every iteration that
touches it — a persistent working memory the model can't lose track
of (tool results survive in history; system prompts don't change
mid-turn).

Enforcement is the same post-hoc pattern as the other turn guards:
`_decide_self_correction` refuses to accept a final answer while the
turn's plan still has pending steps — one corrective re-prompt that
lists exactly what's unfinished (finish them, mark them skipped with
a reason, or honestly escalate).

State is a ContextVar reset at run_unified entry — same lifecycle as
the duplicate-call cache and the endpoint judgment cache.
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar

log = logging.getLogger(__name__)


_MAX_STEPS = 12
_STATUSES = ("pending", "done", "skipped")

# Per-turn plan state: {"steps": [{"text": str, "status": str,
# "note": str}]} or None when no plan was declared this turn.
_plan_state: ContextVar["dict | None"] = ContextVar(
    "_plan_scratchpad_state", default=None,
)


def reset_plan() -> None:
    """Called at run_unified entry — each turn starts plan-less."""
    _plan_state.set(None)


def current_plan() -> "dict | None":
    return _plan_state.get()


def unfinished_steps() -> list[tuple[int, str]]:
    """1-based (index, text) of steps still pending. Empty when no
    plan exists or everything is done/skipped."""
    state = _plan_state.get()
    if not state:
        return []
    out: list[tuple[int, str]] = []
    for i, step in enumerate(state.get("steps") or [], start=1):
        if step.get("status") == "pending":
            out.append((i, step.get("text") or ""))
    return out


def _render(state: dict) -> str:
    marks = {"pending": "[ ]", "done": "[x]", "skipped": "[-]"}
    lines = ["PLAN:"]
    for i, step in enumerate(state.get("steps") or [], start=1):
        mark = marks.get(step.get("status") or "pending", "[ ]")
        note = step.get("note") or ""
        suffix = f"  ({note})" if note else ""
        lines.append(f"{mark} {i}. {step.get('text') or ''}{suffix}")
    pending = len([s for s in state.get("steps") or []
                   if s.get("status") == "pending"])
    if pending:
        lines.append(f"{pending} step(s) still pending.")
    else:
        lines.append("All steps complete.")
    return "\n".join(lines)


def set_plan_handler(steps) -> str:
    """Declare (or replace) this turn's checklist."""
    if isinstance(steps, str):
        # Tolerate a JSON-encoded array — some models stringify.
        try:
            steps = json.loads(steps)
        except Exception:
            steps = [s.strip() for s in steps.split("\n") if s.strip()]
    if not isinstance(steps, list):
        return json.dumps({
            "ok": False,
            "error": "steps must be an array of short step descriptions",
        }, ensure_ascii=False)
    cleaned = [str(s).strip() for s in steps if str(s).strip()]
    if not cleaned:
        return json.dumps({
            "ok": False, "error": "plan needs at least one step",
        }, ensure_ascii=False)
    if len(cleaned) > _MAX_STEPS:
        cleaned = cleaned[:_MAX_STEPS]
    state = {
        "steps": [{"text": s, "status": "pending", "note": ""}
                  for s in cleaned],
    }
    _plan_state.set(state)
    return _render(state)


def update_plan_handler(step: int, status: str, note: str = "") -> str:
    """Mark one step done/skipped (or back to pending). `step` is the
    1-based index from the rendered checklist."""
    state = _plan_state.get()
    if not state:
        return json.dumps({
            "ok": False,
            "error": "no plan set this turn — call set_plan first",
        }, ensure_ascii=False)
    steps = state.get("steps") or []
    try:
        idx = int(step)
    except (TypeError, ValueError):
        return json.dumps({
            "ok": False, "error": f"step must be an integer, got {step!r}",
        }, ensure_ascii=False)
    if not (1 <= idx <= len(steps)):
        return json.dumps({
            "ok": False,
            "error": f"step {idx} out of range (plan has {len(steps)})",
        }, ensure_ascii=False)
    status_norm = (status or "").strip().lower()
    if status_norm not in _STATUSES:
        return json.dumps({
            "ok": False,
            "error": f"status must be one of {list(_STATUSES)}",
        }, ensure_ascii=False)
    if status_norm == "skipped" and not (note or "").strip():
        return json.dumps({
            "ok": False,
            "error": "skipping a step requires a `note` explaining why",
        }, ensure_ascii=False)
    steps[idx - 1]["status"] = status_norm
    steps[idx - 1]["note"] = (note or "").strip()
    _plan_state.set(state)
    return _render(state)
