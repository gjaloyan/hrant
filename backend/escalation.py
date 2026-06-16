"""Explicit turn escalation levels — one source of truth for how much
verification weight a turn carries.

See docs/superpowers/specs/2026-06-16-escalation-levels-design.md.
Pure logic: no I/O, no LLM call. The level is derived from the fast-path
decision plus the tool classes that actually ran.
"""
from __future__ import annotations

from enum import IntEnum


class Level(IntEnum):
    L0_CHAT = 0     # direct answer, no tools
    L1_ACTION = 1   # tool turn, pure state-mutation — nothing to fact-verify
    L2_TASK = 2     # tool turn that produced assertable information


# Tools whose ONLY effect is a state mutation / confirmation. A turn that ran
# exclusively these has no assertable factual claims, so the claim verifier
# has nothing to check. Deliberately NOT endpoint_check._EXECUTE_TOOLS, which
# includes information-producing executors (sandbox_exec / agent_browser /
# delegate) and omits terminal_exec. save_to_workspace / save_knowledge are
# intentionally absent — their CONTENT can be wrong, so they still get verified.
_PURE_ACTION_TOOLS: frozenset[str] = frozenset({
    "save_user_fact", "set_setting", "schedule_message",
    "start_background_job", "define_task_endpoint", "complete_supervisor",
    "kick_supervisor", "grant_telegram_access", "revoke_telegram_access",
    "approve_pairing", "propose_skill", "propose_self_modification",
    "ask_user",
})


def tool_names_from_trace(trace) -> list[str]:
    """Names of the tools that actually ran in a turn's thinking trace.
    Mirrors the endpoint-check extraction: `tool_call` is a ToolCallDetail
    pydantic model on ThinkingStep (attribute access) with a dict fallback
    for older trace formats. Only completed steps (event 'tool'/'tool_error')
    count — a 'tool_starting' step has no result yet."""
    names: list[str] = []
    for step in (trace or []):
        if getattr(step, "event", "") not in ("tool", "tool_error"):
            continue
        tc = getattr(step, "tool_call", None)
        if tc is None:
            continue
        name = getattr(tc, "name", None)
        if name is None and isinstance(tc, dict):
            name = tc.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def decide_level(*, was_fast_chat: bool, tool_names: list[str]) -> Level:
    """Pick a turn's level from what actually happened.
    - was_fast_chat            -> L0 (the chat lane answered directly).
    - non-empty trace, EVERY tool pure-action -> L1 (skip the verifier).
    - anything else            -> L2 (an information tool ran, or a no-tool
      reasoned answer escalated off the fast path -> verify it)."""
    if was_fast_chat:
        return Level.L0_CHAT
    if tool_names and all(n in _PURE_ACTION_TOOLS for n in tool_names):
        return Level.L1_ACTION
    return Level.L2_TASK


def should_run_verifier(level: Level) -> bool:
    """The claim verifier runs only at L2 — L0/L1 have nothing to ground."""
    return level >= Level.L2_TASK


def should_verify(tool_outputs, trace) -> bool:
    """THE unified verifier gate, as one testable function the production
    code calls. Run the claim verifier only when there is grounding material
    (`tool_outputs`) AND the turn is L2 (an information tool ran)."""
    if not tool_outputs:
        return False
    level = decide_level(
        was_fast_chat=False, tool_names=tool_names_from_trace(trace),
    )
    return should_run_verifier(level)
