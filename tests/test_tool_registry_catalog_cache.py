"""Tool registry catalog cache (audit 2026-06-10 I2).

The unified loop calls `to_anthropic_list` once per LLM iteration with
the same `filter_names` set. Without cache, that's N iterations of the
tools dict (one per loop iteration × ~10 per turn). With cache, the
second-and-onward calls return the prior result in O(1) — invalidated
only on register / unregister.
"""
from __future__ import annotations

from backend.tool_registry import ToolRegistry, Tool


def _make_tool(name: str = "t1") -> Tool:
    return Tool(
        name=name,
        description="desc",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **kw: "ok",
    )


def test_to_anthropic_list_returns_cached_object_on_repeat_call():
    """Same call shape returns the SAME list object (identity check)."""
    r = ToolRegistry()
    r.register(_make_tool("a"))
    r.register(_make_tool("b"))

    first = r.to_anthropic_list()
    second = r.to_anthropic_list()
    assert first is second, (
        "cached call must return the same list object — not just an "
        "equal one — to prove no rebuild happened"
    )


def test_to_anthropic_list_caches_per_filter_set():
    """Two distinct filter sets get two distinct cached entries; each
    is consistent across repeats."""
    r = ToolRegistry()
    r.register(_make_tool("a"))
    r.register(_make_tool("b"))
    r.register(_make_tool("c"))

    abc1 = r.to_anthropic_list({"a", "b", "c"})
    abc2 = r.to_anthropic_list({"c", "a", "b"})  # different ordering
    ab1 = r.to_anthropic_list({"a", "b"})
    ab2 = r.to_anthropic_list({"a", "b"})

    # Same set (regardless of ordering) returns the same cached list.
    assert abc1 is abc2
    assert ab1 is ab2
    # Distinct filter sets must NOT collide.
    assert abc1 is not ab1
    assert len(abc1) == 3
    assert len(ab1) == 2


def test_register_invalidates_cache():
    """Adding a tool drops the cache so the next call sees it."""
    r = ToolRegistry()
    r.register(_make_tool("a"))
    first = r.to_anthropic_list()
    assert len(first) == 1

    # Add a new tool.
    r.register(_make_tool("b"))

    second = r.to_anthropic_list()
    assert second is not first, "register() must invalidate the cache"
    assert len(second) == 2
    names = {t["name"] for t in second}
    assert names == {"a", "b"}


def test_unregister_invalidates_cache():
    """Removing a tool drops the cache so the next call sees the
    smaller catalog."""
    r = ToolRegistry()
    r.register(_make_tool("a"))
    r.register(_make_tool("b"))
    r.to_anthropic_list()  # populate cache

    r.unregister("a")
    after = r.to_anthropic_list()
    assert len(after) == 1
    assert after[0]["name"] == "b"


def test_unregister_missing_name_does_not_invalidate():
    """Unregistering a name that wasn't there is a no-op — must NOT
    invalidate the cache (no state change happened)."""
    r = ToolRegistry()
    r.register(_make_tool("a"))
    first = r.to_anthropic_list()

    r.unregister("not-there")
    second = r.to_anthropic_list()

    assert first is second, (
        "no-op unregister must not invalidate cache"
    )


def test_cache_survives_independent_filter_call():
    """A call with filter_names doesn't trample the cache slot for
    the all-tools (filter=None) call."""
    r = ToolRegistry()
    r.register(_make_tool("a"))
    r.register(_make_tool("b"))

    all_first = r.to_anthropic_list()
    _ = r.to_anthropic_list({"a"})  # separate cache key
    all_second = r.to_anthropic_list()

    assert all_first is all_second
