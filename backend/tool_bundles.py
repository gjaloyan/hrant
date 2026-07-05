"""Tool schema diet — bundle definitions + per-turn loaded-bundle state.

Phase 2 of the cost audit cycle (2026-05-23): the per-iteration tool
schema dropped from ~38 KB to ~16 KB by splitting 12 heavy/niche
tools into 4 named bundles. The LLM unlocks a bundle mid-turn via
`load_tool_bundle(name)`; the next iteration's schema rebuild
picks it up. Bundle state resets at turn end (ContextVar default).

Spec: docs/superpowers/specs/2026-05-23-tool-schema-diet-design.md
"""
from __future__ import annotations

import contextvars
from typing import Final


TOOL_BUNDLES: Final[dict[str, list[str]]] = {
    # The former `bench` bundle was dissolved 2026-05-27:
    # start_background_job / define_task_endpoint /
    # complete_supervisor moved to BASE_TOOLS so supervisor turns
    # and any long-running launch work without a bundle dance.
    # See the audit follow-up commit.
    "admin": [
        "set_setting",
        "grant_telegram_access",
        "revoke_telegram_access",
        "approve_pairing",
        "list_telegram_access",
        "list_pending_pairings",
    ],
    "self": [
        "propose_skill",
        "propose_self_modification",
    ],
    "media": [
        "agent_browser",
        "sandbox_exec",
    ],
}


BUNDLE_DESCRIPTIONS: Final[dict[str, str]] = {
    "admin": (
        "Mutate agent configuration (set_setting), manage Telegram "
        "user access (grant / revoke / list / approve_pairing / "
        "list_pending_pairings)."
    ),
    "self": (
        "Write a new reusable skill (propose_skill), propose "
        "structural code changes to the agent itself "
        "(propose_self_modification)."
    ),
    "media": (
        "Drive a headless Chromium for JS-rendered / login-walled "
        "pages (agent_browser — for plain HTML, the always-on "
        "fetch_url is cheaper); run untrusted binaries under "
        "bubblewrap/firejail/unshare isolation (sandbox_exec)."
    ),
}


BASE_TOOLS: Final[frozenset[str]] = frozenset({
    # File I/O
    "read_file", "save_to_workspace",
    # Execution
    "terminal_exec", "run_python",
    # Search / navigation + knowledge education (read + deliberate write)
    "locate_symbol", "search_knowledge", "save_knowledge",
    "list_skills", "load_skill",
    # Web (basic — agent_browser is in `media` bundle)
    "fetch_url", "web_search",
    # Multimodal
    "analyze_image",
    # Interaction + reminders. schedule_message moved to BASE 2026-06-18:
    # it was gated behind the `admin` bundle, so reminder turns couldn't reach
    # it and the model hand-rolled scheduling via raw python/terminal (slow,
    # fragile, sometimes a useless outbox file). A self-reminder is a core,
    # common, owner-gated action — it belongs always-on.
    "ask_user", "save_user_fact", "schedule_message",
    # Project tracker (the living-projects feature). Added to BASE 2026-06-19:
    # these were registered as builtins but never placed in BASE or any
    # bundle, so the per-turn schema filter (BASE | loaded-bundles) dropped
    # them — the model never saw them. Asked to "заведи трекер" it spent
    # ~120s exploring the codebase and hand-rolled a dead markdown note in
    # workspace/notes/ instead of using the real store (tracker.json +
    # check-ins + due-date scheduling + WebUI board). Same gating-induced
    # hand-roll class as schedule_message. Managing a project is a core,
    # conversational, owner-gated action — it belongs always-on.
    "create_tracker", "list_trackers", "get_tracker",
    "add_step", "update_step",
    # Critical-thinking framing — reachable always so the agent can frame a
    # big task into its real components and confirm scope before building
    # (2026-06-23). Gating it would re-create the "model can't reach the tool,
    # so it hand-rolls" trap.
    "frame_problem",
    # Verification-as-action (2026-06-25): render what you built the way the
    # user's browser would, BEFORE claiming done. Pairs with the hard
    # verify-before-done gate in unified_agent.
    "verify_web",
    # Delegation — moved to BASE 2026-06-23 so the agent USES subagents often
    # (research / build a component / review) to keep its own context clean,
    # instead of doing everything in one window. Gating it behind `self` meant
    # the model rarely loaded it and rarely delegated.
    "delegate",
    # Plan scratchpad (2026-06-11) — multi-step checklist the model
    # declares up front and ticks off; self-correction refuses a
    # final answer while steps are still pending.
    "set_plan", "update_plan",
    # Jobs — full read+write so supervisor turns + any long-running
    # launch work without a `load_tool_bundle("bench")` dance.
    # Moved here 2026-05-27 (the former `bench` bundle is gone).
    "list_background_jobs", "get_background_job",
    "start_background_job", "define_task_endpoint",
    "complete_supervisor", "kick_supervisor",
    # Self-surface acknowledgement (audit 2026-05-28). Always-on so
    # the agent can mark UNRESOLVED AGENT-SIDE FAILURES from the
    # system-prompt block as explained on any turn.
    "acknowledge_provider_issue",
    # Meta — the LLM's discovery hook for bundles
    "load_tool_bundle",
})


def expand_loaded(bundles: set[str]) -> set[str]:
    """Return the union of all tool names across the requested bundles.

    Unknown bundle names are silently dropped — the `load_tool_bundle`
    handler is the validation point; this helper is purely a schema
    assembler used by the per-iteration tool-list builder.
    """
    out: set[str] = set()
    for name in bundles:
        members = TOOL_BUNDLES.get(name)
        if members:
            out.update(members)
    return out


# Per-turn loaded-bundles state — initialised at run_unified entry,
# mutated by the `load_tool_bundle` handler, read by the per-iteration
# schema rebuild. Default `frozenset()` keeps test callers (no
# run_unified frame) honest about the "nothing loaded" baseline.
_loaded_bundles: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "hrant_loaded_tool_bundles", default=frozenset(),
)


def get_loaded_bundles() -> set[str]:
    """Snapshot of the current turn's loaded bundles. Returns a fresh
    set so the caller can't mutate the stored ContextVar value."""
    return set(_loaded_bundles.get())


def set_loaded_bundles(bundles: set[str]) -> None:
    """Replace the current turn's loaded set. Stored as a frozenset
    so accidental aliasing across turns can't bleed state."""
    _loaded_bundles.set(frozenset(bundles))
