"""Shared auth helpers for REST handlers.

Mirrors the pattern already in `backend/api/skills.py:_require_owner_for_writes`
but lifted into one place so the Phase 14+/15+/16+ endpoints don't
each re-implement the role check. Picking up a consistent gate
helps when the agent's WebUI binds to 0.0.0.0 via `hrant gateway
start --gateway` — without this, anyone on the LAN could fire a
consolidation (cost-amplification), wipe jobs, or rewrite the
failover chain.

Semantics match the existing skills check:
  - Within a real request handled via the WebUI: the speaker is
    derived from a ContextVar (Phase 11). If the speaker is set
    and isn't an owner, the gate raises 403.
  - In contexts with no speaker context (CLI, tests, autonomic
    ticks), the gate is a no-op — we trust those callers.

For pure read endpoints (status/list/get) the gate isn't needed —
those are safe to expose. Apply this helper to mutations:
POST/PUT/DELETE that change state, run LLM calls, or move money.
"""
from __future__ import annotations

from fastapi import HTTPException

from ..roles import current_speaker, is_owner


def require_owner_for_writes(*, action: str = "this action") -> None:
    """Raise 403 if the current request's speaker isn't an owner.

    `action` is interleaved into the error message so the WebUI /
    CLI can surface what the user was trying to do (e.g. "trigger
    consolidation requires owner role")."""
    sp = current_speaker()
    if sp is None:
        # No speaker context — CLI, tests, autonomic. Trust them.
        return
    if not is_owner(sp):
        raise HTTPException(
            status_code=403,
            detail=f"{action} requires owner role",
        )
