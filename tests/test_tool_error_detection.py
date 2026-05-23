"""Tests for the 2026-05-23 tool error instrumentation (audit Important #6).

Pre-fix: only tool handlers that RAISED were tagged is_error=True.
Most production tools catch internally and return error-shaped
strings, slipping through with is_error=False — that's why the
audit saw "0/416 reported errors" across the last 20 turns.

Post-fix: `_looks_like_error(name, text)` runs after every successful
handler return; if the result looks like a known error shape, the
tuple flips is_error=True so the dev panel + supervisor + LogBus all
see the failure consistently."""
from __future__ import annotations

import pytest


@pytest.fixture
def registry():
    """Fresh ToolRegistry for each test — avoids pollution from
    other tool-registration-time tests."""
    from backend.tool_registry import ToolRegistry
    return ToolRegistry()


# ─── _looks_like_error: bracket-prefix patterns ───────────────────


def test_detects_fetch_error_bracket():
    from backend.tool_registry import _looks_like_error
    assert _looks_like_error("fetch_url", "[fetch error: timeout]")
    assert _looks_like_error("fetch_url", "[fetch refused: in 127.0.0.0/8]")


def test_detects_bad_arguments_bracket():
    from backend.tool_registry import _looks_like_error
    assert _looks_like_error("anything", "[bad arguments for X: missing 'path']")


def test_detects_no_results_bracket():
    from backend.tool_registry import _looks_like_error
    assert _looks_like_error("web_search", "[no results]")


def test_detects_generic_error_bracket():
    """A handler-returned `[error: ...]` string must be flagged.
    Note: `[X runtime error]` shape is what `execute()` emits when a
    handler RAISES — that path already sets is_error=True directly,
    so it never round-trips through this heuristic."""
    from backend.tool_registry import _looks_like_error
    assert _looks_like_error("X", "[error: file not found]")


def test_detects_permission_denied():
    from backend.tool_registry import _looks_like_error
    assert _looks_like_error("terminal", "[permission denied — owner-only]")


# ─── _looks_like_error: JSON {"ok": false} ────────────────────────


def test_detects_json_ok_false():
    from backend.tool_registry import _looks_like_error
    assert _looks_like_error("propose_skill", '{"ok": false, "error": "duplicate"}')
    # No space — common in compact JSON.
    assert _looks_like_error("agent_browser", '{"ok":false,"error":"timeout"}')


def test_does_not_detect_json_ok_true():
    from backend.tool_registry import _looks_like_error
    assert not _looks_like_error("ask_user", '{"ok": true, "question_id": "q-x"}')


# ─── _looks_like_error: subprocess non-zero returncode ────────────


def test_detects_nonzero_returncode():
    """run_python / terminal_exec wrap subprocess output as JSON; a
    non-zero returncode field means the wrapped command failed."""
    from backend.tool_registry import _looks_like_error
    assert _looks_like_error(
        "terminal_exec",
        '{"stdout": "", "stderr": "command not found", "returncode": 127}',
    )
    assert _looks_like_error(
        "run_python",
        '{"stdout": "", "stderr": "Traceback...", "returncode": 1}',
    )


def test_does_not_detect_zero_returncode():
    from backend.tool_registry import _looks_like_error
    assert not _looks_like_error(
        "terminal_exec",
        '{"stdout": "hello\\n", "stderr": "", "returncode": 0}',
    )


def test_does_not_detect_negative_returncode_word_in_prose():
    """A prose result that just happens to contain the substring
    'returncode' (e.g. a help text from --help) must not trip the
    detector — the digit-match regex requires colon + integer."""
    from backend.tool_registry import _looks_like_error
    assert not _looks_like_error(
        "terminal_exec",
        "the returncode varies. exit 0 means success.",
    )


# ─── Conservative on edge cases ───────────────────────────────────


def test_empty_or_non_string_not_error():
    """Empty result is NOT an error (some tools legitimately produce
    nothing). Caller can layer 'empty after N attempts' logic on top."""
    from backend.tool_registry import _looks_like_error
    assert not _looks_like_error("x", "")
    assert not _looks_like_error("x", None)  # type: ignore[arg-type]


def test_does_not_detect_prose_starting_with_bracket():
    """A normal answer that happens to start with `[Note: ...]` or
    similar prose bracket must not be flagged — only the explicit
    error prefixes count."""
    from backend.tool_registry import _looks_like_error
    assert not _looks_like_error("x", "[Note: this is fine]")
    assert not _looks_like_error("x", "[OK] done")


# ─── End-to-end: execute() now reports is_error for these shapes ──


def test_execute_flags_handler_returning_error_string(registry):
    """Handler doesn't raise — returns an `[fetch error: ...]`
    string. Pre-fix: is_error=False. Post-fix: is_error=True."""
    def _handler():
        return "[fetch error: connection refused]"
    registry.register_func(
        name="my_fetch", description="", input_schema={"type": "object"},
        handler=_handler,
    )
    text, is_error = registry.execute("my_fetch", {})
    assert "fetch error" in text
    assert is_error is True


def test_execute_flags_handler_returning_ok_false_json(registry):
    def _handler():
        return {"ok": False, "error": "duplicate skill"}
    registry.register_func(
        name="my_tool", description="", input_schema={"type": "object"},
        handler=_handler,
    )
    text, is_error = registry.execute("my_tool", {})
    assert '"ok": false' in text
    assert is_error is True


def test_execute_flags_handler_returning_nonzero_returncode(registry):
    def _handler():
        return {"stdout": "", "stderr": "boom", "returncode": 2}
    registry.register_func(
        name="my_exec", description="", input_schema={"type": "object"},
        handler=_handler,
    )
    text, is_error = registry.execute("my_exec", {})
    assert is_error is True


def test_execute_keeps_success_for_zero_returncode(registry):
    """A successful subprocess wrapped as JSON must stay is_error=False."""
    def _handler():
        return {"stdout": "hi\n", "stderr": "", "returncode": 0}
    registry.register_func(
        name="my_exec_ok", description="", input_schema={"type": "object"},
        handler=_handler,
    )
    text, is_error = registry.execute("my_exec_ok", {})
    assert is_error is False


def test_execute_keeps_raising_path(registry):
    """Original raising-handler path still flips is_error=True (this
    is the safety net the audit fix layers on top of, not replaces)."""
    def _handler():
        raise ValueError("nope")
    registry.register_func(
        name="my_throw", description="", input_schema={"type": "object"},
        handler=_handler,
    )
    text, is_error = registry.execute("my_throw", {})
    assert "runtime error" in text
    assert "ValueError" in text
    assert is_error is True
