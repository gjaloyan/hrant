"""A write-capable `builder` subagent role + `delegate` reachable in BASE, so
the agent can actually DELEGATE BUILD WORK (not just read-only research/review)
and reach the tool without loading a bundle."""
from __future__ import annotations

from backend.subagents.roles import ROLE_REGISTRY
from backend.tool_bundles import BASE_TOOLS, TOOL_BUNDLES


def test_builder_role_exists_and_can_build():
    assert "builder" in ROLE_REGISTRY
    tools = set(ROLE_REGISTRY["builder"].tools)
    # NOT read-only — it must have write/run tools
    assert {"save_to_workspace", "terminal_exec", "run_python"} <= tools


def test_read_only_roles_stay_read_only():
    # researcher/reviewer must NOT gain build tools by accident
    for ro in ("researcher", "reviewer"):
        assert "terminal_exec" not in set(ROLE_REGISTRY[ro].tools)


def test_delegate_is_base_not_gated():
    assert "delegate" in BASE_TOOLS
    assert "delegate" not in TOOL_BUNDLES.get("self", [])
