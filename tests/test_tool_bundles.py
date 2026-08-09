"""Tests for the tool-schema diet (Phase 2)."""
from __future__ import annotations

import pytest


def test_bundles_constant_shape():
    """V2 (2026-05-27): the former `bench` bundle was dissolved.
    `start_background_job` / `define_task_endpoint` /
    `complete_supervisor` moved to BASE_TOOLS so supervisor turns
    work without a bundle dance."""
    from backend.tool_bundles import TOOL_BUNDLES
    # `self` was dissolved into BASE on 2026-08-09: measured over 49 prod turn
    # artifacts, load_tool_bundle("media") was called TEN times to reach
    # propose_self_modification FOUR times. Self-improvement is the agent's
    # core loop and should not pay a discovery round-trip.
    assert set(TOOL_BUNDLES.keys()) == {"admin", "media"}
    from backend.tool_bundles import BASE_TOOLS
    assert {"propose_skill", "propose_self_modification"} <= set(BASE_TOOLS)
    for name, members in TOOL_BUNDLES.items():
        assert isinstance(members, list), f"{name} value must be list"
        assert all(isinstance(m, str) for m in members)
        assert len(members) >= 1, f"{name} bundle must have at least one tool"


def test_bundle_descriptions_present_for_every_bundle():
    from backend.tool_bundles import TOOL_BUNDLES, BUNDLE_DESCRIPTIONS
    assert set(BUNDLE_DESCRIPTIONS.keys()) == set(TOOL_BUNDLES.keys())
    for desc in BUNDLE_DESCRIPTIONS.values():
        assert isinstance(desc, str) and len(desc) > 20


def test_base_tools_constant_shape():
    """V2: jobs-control tools (start_background_job /
    define_task_endpoint / complete_supervisor) are now in
    BASE_TOOLS so any turn — including supervisor — can use them
    without `load_tool_bundle` first."""
    from backend.tool_bundles import BASE_TOOLS
    assert isinstance(BASE_TOOLS, frozenset)
    assert len(BASE_TOOLS) >= 20
    assert "load_tool_bundle" in BASE_TOOLS, (
        "the meta-tool must be in base — otherwise LLM can't unlock bundles"
    )
    for tool in (
        "read_file", "terminal_exec", "ask_user", "load_skill",
        "list_skills", "search_knowledge", "fetch_url", "web_search",
        "save_to_workspace", "save_user_fact", "list_background_jobs",
        "get_background_job", "analyze_image", "run_python",
        "locate_symbol",
        # V2: jobs control promoted from `bench` bundle to base.
        "start_background_job", "define_task_endpoint",
        "complete_supervisor",
    ):
        assert tool in BASE_TOOLS, f"{tool} missing from BASE_TOOLS"


def test_base_and_bundles_are_disjoint():
    from backend.tool_bundles import BASE_TOOLS, TOOL_BUNDLES
    all_bundled = set().union(*(set(v) for v in TOOL_BUNDLES.values()))
    overlap = BASE_TOOLS & all_bundled
    assert overlap == set(), f"tool(s) in both base and bundle: {overlap}"


def test_bundles_have_no_internal_duplicates():
    from backend.tool_bundles import TOOL_BUNDLES
    seen: dict[str, str] = {}
    for bundle, tools in TOOL_BUNDLES.items():
        for t in tools:
            assert t not in seen, (
                f"{t!r} is in both {seen[t]!r} and {bundle!r}"
            )
            seen[t] = bundle


def test_expand_loaded_empty():
    from backend.tool_bundles import expand_loaded
    assert expand_loaded(set()) == set()


def test_expand_loaded_single_bundle():
    from backend.tool_bundles import expand_loaded, TOOL_BUNDLES
    assert expand_loaded({"admin"}) == set(TOOL_BUNDLES["admin"])


def test_expand_loaded_multiple_bundles_union():
    from backend.tool_bundles import expand_loaded, TOOL_BUNDLES
    out = expand_loaded({"admin", "media"})
    assert out == set(TOOL_BUNDLES["admin"]) | set(TOOL_BUNDLES["media"])


def test_expand_loaded_unknown_bundle_ignored():
    from backend.tool_bundles import expand_loaded
    assert expand_loaded({"not_a_bundle"}) == set()


def test_contextvar_default_empty():
    from backend.tool_bundles import get_loaded_bundles
    assert get_loaded_bundles() == set()


def test_contextvar_set_and_get_roundtrip():
    from backend.tool_bundles import set_loaded_bundles, get_loaded_bundles
    set_loaded_bundles({"admin"})
    assert get_loaded_bundles() == {"admin"}
    set_loaded_bundles(set())


def test_contextvar_returns_copy_not_reference():
    from backend.tool_bundles import set_loaded_bundles, get_loaded_bundles
    set_loaded_bundles({"admin"})
    snapshot = get_loaded_bundles()
    snapshot.add("admin")
    assert get_loaded_bundles() == {"admin"}
    set_loaded_bundles(set())


def test_handler_loads_valid_bundle():
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import (
        get_loaded_bundles, set_loaded_bundles, TOOL_BUNDLES,
    )
    import json
    set_loaded_bundles(set())
    try:
        raw = _load_tool_bundle_handler(name="admin")
        body = json.loads(raw)
        assert body["ok"] is True
        assert body["name"] == "admin"
        assert set(body["added"]) == set(TOOL_BUNDLES["admin"])
        assert "next iteration" in body["note"].lower()
        assert get_loaded_bundles() == {"admin"}
    finally:
        set_loaded_bundles(set())


def test_handler_idempotent_on_repeat():
    """Loading a bundle twice in the same turn is a no-op success —
    not an error. Empty `added` signals 'already in your toolbox'."""
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import set_loaded_bundles
    import json
    set_loaded_bundles(set())
    try:
        _load_tool_bundle_handler(name="admin")
        raw = _load_tool_bundle_handler(name="admin")
        body = json.loads(raw)
        assert body["ok"] is True
        assert body["added"] == []
        assert "already loaded" in body["note"].lower()
    finally:
        set_loaded_bundles(set())


def test_handler_rejects_unknown_bundle():
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import (
        get_loaded_bundles, set_loaded_bundles, TOOL_BUNDLES,
    )
    import json
    set_loaded_bundles(set())
    try:
        raw = _load_tool_bundle_handler(name="not_a_real_bundle")
        body = json.loads(raw)
        assert body["ok"] is False
        assert "unknown" in body["error"].lower()
        assert set(body["available"]) == set(TOOL_BUNDLES.keys())
        # State must NOT have been mutated.
        assert get_loaded_bundles() == set()
    finally:
        set_loaded_bundles(set())


def test_handler_loads_multiple_independent_bundles():
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import set_loaded_bundles, get_loaded_bundles
    set_loaded_bundles(set())
    try:
        _load_tool_bundle_handler(name="admin")
        _load_tool_bundle_handler(name="media")
        assert get_loaded_bundles() == {"admin", "media"}
    finally:
        set_loaded_bundles(set())


def test_load_tool_bundle_registered_in_global_registry():
    """The handler must be wired into the global registry so the LLM
    actually sees it in the schema."""
    from backend.tool_registry import get_registry
    names = {t["name"] for t in get_registry().to_anthropic_list()}
    assert "load_tool_bundle" in names
