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
    "bench": [
        "start_background_job",
        "define_task_endpoint",
        "complete_supervisor",
    ],
    "admin": [
        "set_setting",
        "grant_telegram_access",
        "revoke_telegram_access",
        "approve_pairing",
        "list_telegram_access",
        "list_pending_pairings",
        "schedule_message",
    ],
    "self": [
        "propose_skill",
        "propose_self_modification",
        "delegate",
    ],
    "media": [
        "agent_browser",
        "sandbox_exec",
    ],
}


BUNDLE_DESCRIPTIONS: Final[dict[str, str]] = {
    "bench": (
        "Launch long-running benchmarks / background jobs "
        "(start_background_job), define their success criteria + "
        "prerequisites (define_task_endpoint), finalise supervisor "
        "turns (complete_supervisor)."
    ),
    "admin": (
        "Mutate agent configuration (set_setting), manage Telegram "
        "user access (grant / revoke / list / approve_pairing / "
        "list_pending_pairings), schedule outbound messages "
        "(schedule_message)."
    ),
    "self": (
        "Write a new reusable skill (propose_skill), propose "
        "structural code changes to the agent itself "
        "(propose_self_modification), delegate a focused subtask "
        "to a specialised subagent (delegate)."
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
    # Search / navigation
    "locate_symbol", "search_knowledge", "list_skills", "load_skill",
    # Web (basic — agent_browser is in `media` bundle)
    "fetch_url", "web_search",
    # Multimodal
    "analyze_image",
    # Interaction
    "ask_user", "save_user_fact",
    # Jobs (read-only — write side is in `bench` bundle)
    "list_background_jobs", "get_background_job",
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
