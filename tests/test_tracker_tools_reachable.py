"""Tracker tools must be reachable in the per-turn schema, not orphaned.

Root cause of "the agent writes a markdown note instead of a real tracker":
`create_tracker` / `list_trackers` / `get_tracker` / `add_step` / `update_step`
were registered as builtins but placed in neither BASE_TOOLS nor any bundle.
The per-turn schema filter is `BASE | expand_loaded(loaded_bundles)`, so the
model never saw them. Asked to manage a project it explored the codebase for
~120s and hand-rolled a dead markdown note in workspace/notes/ — bypassing the
real store (tracker.json + due-date check-ins + WebUI board).

This test pins the whole reachability contract: every registered builtin must
be reachable from SOME surface (BASE or a bundle), so a newly-registered tool
that is never wired up fails CI instead of silently hand-rolling in prod.
"""
from __future__ import annotations

from backend.tool_bundles import BASE_TOOLS, TOOL_BUNDLES

TRACKER_TOOLS = {
    "create_tracker", "list_trackers", "get_tracker", "add_step", "update_step",
}


def test_tracker_tools_are_base():
    assert TRACKER_TOOLS <= set(BASE_TOOLS)


def test_no_registered_builtin_is_orphaned():
    """No registered builtin may be unreachable (absent from BASE and every
    bundle). Guards against the tracker-style 'registered but never wired'
    regression for any future tool."""
    import backend.builtin_tools as bt
    bt.register_builtin_tools()
    from backend.tool_registry import REGISTRY

    registered = set(REGISTRY.names())
    reachable = set(BASE_TOOLS)
    for members in TOOL_BUNDLES.values():
        reachable |= set(members)

    orphans = registered - reachable
    assert not orphans, (
        f"registered builtins unreachable from BASE or any bundle: "
        f"{sorted(orphans)} — add them to BASE_TOOLS or a bundle"
    )
