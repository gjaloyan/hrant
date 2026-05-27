"""Per-turn duplicate-call guard.

When the LLM calls the same tool with the same arguments twice
within one turn, the second call short-circuits with a synthesized
result that includes the previous output + a "use the cached result"
hint. Catches the failure mode from prod logs 2026-05-26 (17×
similar `terminal_exec` calls, 2× `load_skill`, 2× `load_tool_bundle`).
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_dedup_state():
    """Each test starts with a clean per-turn cache."""
    from backend import tool_registry as tr
    tr.reset_per_turn_call_cache()
    yield
    tr.reset_per_turn_call_cache()


def _make_test_registry(name="echo"):
    from backend.tool_registry import ToolRegistry
    reg = ToolRegistry()
    counter = {"n": 0}

    def _handler(**kwargs):
        counter["n"] += 1
        return {"ok": True, "n": counter["n"], "args": kwargs}

    reg.register_func(
        name=name,
        description="echo",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        handler=_handler,
    )
    return reg, counter


def test_first_call_executes_normally():
    reg, counter = _make_test_registry()
    text, is_error = reg.execute("echo", {"x": "hello"})
    assert is_error is False
    assert counter["n"] == 1
    assert "hello" in text


def test_duplicate_call_with_same_args_short_circuits():
    """The second call with the same (name, args) should NOT
    re-run the handler. Counter stays at 1."""
    reg, counter = _make_test_registry()
    reg.execute("echo", {"x": "hello"})
    text2, _ = reg.execute("echo", {"x": "hello"})
    assert counter["n"] == 1, "handler ran a second time — dedup failed"
    # The synthesized result must reference the duplicate condition.
    assert "DUPLICATE" in text2 or "already called" in text2.lower()


def test_duplicate_call_preserves_previous_output_in_hint():
    """The cached result must be embedded so the LLM can still use
    the data without re-calling."""
    reg, _ = _make_test_registry()
    text1, _ = reg.execute("echo", {"x": "hello"})
    text2, _ = reg.execute("echo", {"x": "hello"})
    # Original payload is reachable in the duplicate-warned text.
    assert "hello" in text2
    assert text1 in text2 or text1.strip() in text2


def test_different_args_do_not_trigger_dedup():
    reg, counter = _make_test_registry()
    reg.execute("echo", {"x": "hello"})
    reg.execute("echo", {"x": "world"})
    assert counter["n"] == 2


def test_different_tools_do_not_collide():
    """The cache key is (name, args), not just args."""
    from backend.tool_registry import ToolRegistry
    reg = ToolRegistry()
    n_a = {"v": 0}
    n_b = {"v": 0}

    def _a(**k): n_a["v"] += 1; return "A"
    def _b(**k): n_b["v"] += 1; return "B"

    reg.register_func(name="a", description="", input_schema={
        "type": "object", "properties": {}}, handler=_a)
    reg.register_func(name="b", description="", input_schema={
        "type": "object", "properties": {}}, handler=_b)
    reg.execute("a", {})
    reg.execute("b", {})
    assert n_a["v"] == 1 and n_b["v"] == 1


def test_args_canonicalised_so_key_order_does_not_matter():
    """`{"a":1,"b":2}` and `{"b":2,"a":1}` are the same call."""
    reg, counter = _make_test_registry()
    reg.execute("echo", {"a": 1, "b": 2})
    reg.execute("echo", {"b": 2, "a": 1})
    assert counter["n"] == 1


def test_reset_per_turn_cache_allows_re_execution():
    """The reset at turn entry must let the same call run again
    on the next turn."""
    from backend import tool_registry as tr
    reg, counter = _make_test_registry()
    reg.execute("echo", {"x": "hello"})
    assert counter["n"] == 1
    tr.reset_per_turn_call_cache()
    reg.execute("echo", {"x": "hello"})
    assert counter["n"] == 2, "after reset, the same call should run again"


def test_duplicate_warning_carries_error_flag_of_original():
    """If the first call errored, the cached repeat must also
    surface is_error=True so the loop sees the failure consistently."""
    from backend.tool_registry import ToolRegistry
    reg = ToolRegistry()
    counter = {"n": 0}

    def _failing_handler(**kwargs):
        counter["n"] += 1
        return "[fetch error: kaboom]"

    reg.register_func(
        name="flaky", description="",
        input_schema={"type": "object", "properties": {}},
        handler=_failing_handler,
    )
    _, is_err_1 = reg.execute("flaky", {})
    _, is_err_2 = reg.execute("flaky", {})
    assert is_err_1 is True
    assert is_err_2 is True
    assert counter["n"] == 1  # still deduped


def test_dedup_handles_None_arguments():
    """Some handlers accept zero kwargs and the LLM may pass null."""
    reg, counter = _make_test_registry()
    reg.execute("echo", None)  # type: ignore[arg-type]
    reg.execute("echo", None)  # type: ignore[arg-type]
    reg.execute("echo", {})  # canonically same as None → no-args
    assert counter["n"] == 1
