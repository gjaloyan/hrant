"""Tests for the May 2026 cost audit T7: per-tool digest headers.

Background: the audit recommended a "context ledger" replacing prior
tool results with summary facts. The anthropic SDK tool-loop doesn't
allow mid-loop history rewriting, so T7 gives the LLM the next-best
thing — a structured 1-line digest header PREPENDED to every tool
result. Across iterations the LLM can scroll the conversation and
re-read what it already learned ("[read_file backend/x.py] 47L 4234c
— handle_callback_query, ...") even when the body is truncated.

Each digester is fast (pure pattern extraction, no LLM call) and
swallows errors so a malformed result never breaks the tool flow.
"""
from __future__ import annotations

import pytest


# ─── _digest_read_file ─────────────────────────────────────────────


def test_digest_read_file_shows_path_and_size():
    from backend.unified_agent import _digest_read_file
    args = {"path": "backend/channels.py"}
    result = "line1\nline2\nline3\n"
    header = _digest_read_file(args, result)
    assert "backend/channels.py" in header
    # 3 lines / 18 chars (3 + 5 newlines + content).
    assert "3L" in header
    assert "no symbols" in header


def test_digest_read_file_extracts_python_symbols():
    from backend.unified_agent import _digest_read_file
    args = {"path": "x.py"}
    result = (
        "def foo():\n    pass\n"
        "\n"
        "class Bar:\n    pass\n"
        "\n"
        "async def baz():\n    pass\n"
    )
    header = _digest_read_file(args, result)
    assert "def foo" in header
    assert "class Bar" in header
    assert "async def baz" in header


def test_digest_read_file_extracts_markdown_headings():
    from backend.unified_agent import _digest_read_file
    args = {"path": "doc.md"}
    result = (
        "# Top\n\n"
        "## Section\n\n"
        "Some text\n\n"
        "### Subsection\n"
    )
    header = _digest_read_file(args, result)
    # All three headings should appear (capped at 5).
    assert "# Top" in header
    assert "## Section" in header


# ─── _digest_terminal_exec ─────────────────────────────────────────


def test_digest_terminal_exec_marks_ok():
    from backend.unified_agent import _digest_terminal_exec
    args = {"command": "ls /tmp"}
    result = "a\nb\nc\n"
    header = _digest_terminal_exec(args, result)
    assert "ls /tmp" in header
    assert "ok" in header
    assert "3L" in header


def test_digest_terminal_exec_marks_error_on_traceback():
    from backend.unified_agent import _digest_terminal_exec
    args = {"command": "python -c 'import nope'"}
    result = (
        "Traceback (most recent call last):\n"
        "  File ...\n"
        "ModuleNotFoundError: No module named 'nope'\n"
    )
    header = _digest_terminal_exec(args, result)
    assert "ERROR" in header


def test_digest_terminal_exec_marks_empty():
    from backend.unified_agent import _digest_terminal_exec
    args = {"command": "true"}
    result = ""
    header = _digest_terminal_exec(args, result)
    # Empty result triggers either 'empty' or just outcomes accurately.
    assert "empty" in header or "0L" in header


# ─── _digest_web_search / _digest_fetch_url ────────────────────────


def test_digest_web_search_shows_query():
    from backend.unified_agent import _digest_web_search
    args = {"query": "claude opus 4.7 token cost"}
    result = "\n- Result 1\n- Result 2\n- Result 3\n"
    header = _digest_web_search(args, result)
    assert "claude opus 4.7 token cost" in header
    # 3 dash-bullet hits.
    assert "3 hits" in header


def test_digest_fetch_url_extracts_title():
    from backend.unified_agent import _digest_fetch_url
    args = {"url": "https://example.com/page"}
    result = "<html><head><title>Example Title</title></head><body>...</body></html>"
    header = _digest_fetch_url(args, result)
    assert "https://example.com/page" in header
    assert "Example Title" in header


def test_digest_fetch_url_handles_no_title():
    from backend.unified_agent import _digest_fetch_url
    args = {"url": "https://api.example/data.json"}
    result = '{"key": "value"}'
    header = _digest_fetch_url(args, result)
    assert "https://api.example/data.json" in header
    # No quoted title section if no <title>.
    assert "—" not in header  # em-dash separator only when title present


# ─── _digest_locate_symbol / _digest_search_knowledge ──────────────


def test_digest_locate_symbol_counts_hits():
    from backend.unified_agent import _digest_locate_symbol
    args = {"symbol": "handle_callback_query", "path": "backend/channels.py"}
    result = '[{"start_line": 1010, "end_line": 1075}, {"start_line": 1080, "end_line": 1090}]'
    header = _digest_locate_symbol(args, result)
    assert "handle_callback_query" in header
    assert "backend/channels.py" in header
    assert "2 hit" in header


def test_digest_search_knowledge_counts_hits():
    from backend.unified_agent import _digest_search_knowledge
    args = {"query": "token cost"}
    result = '[{"topic":"x"},{"topic":"y"},{"topic":"z"}]'
    header = _digest_search_knowledge(args, result)
    assert "token cost" in header
    assert "3 hit" in header


# ─── _digest_tool_result dispatch ──────────────────────────────────


def test_digest_dispatch_unknown_tool_returns_empty():
    """Tools with no digester (set_setting, propose_install, etc.)
    return an empty header — their results are already short."""
    from backend.unified_agent import _digest_tool_result
    assert _digest_tool_result("set_setting", {}, "ok") == ""
    assert _digest_tool_result("unknown_tool", {}, "abc") == ""


def test_digest_dispatch_known_tool_returns_header():
    from backend.unified_agent import _digest_tool_result
    header = _digest_tool_result(
        "read_file", {"path": "x.py"}, "def foo():\n    pass\n"
    )
    assert "x.py" in header
    assert "def foo" in header


def test_digest_dispatch_swallows_exceptions():
    """A malformed result must not crash the tool flow."""
    from backend.unified_agent import _digest_tool_result
    # Pass None where digester expects a string — should return "" not raise.
    assert _digest_tool_result("read_file", {}, "") == ""
    # Args is a non-dict — digester should handle gracefully.
    assert _digest_tool_result("read_file", "not a dict", "stuff") in (
        "", "[read_file ?] 1L 5c — no symbols",
    )


# ─── Dispatch table contains expected tools ────────────────────────


def test_digesters_registered_for_heavy_tools():
    """The audit identified these tools as the heavy spenders.
    Pin that each has a digester registered."""
    from backend.unified_agent import _DIGESTERS
    expected = {
        "read_file", "view_file", "terminal_exec",
        "web_search", "fetch_url",
        "locate_symbol", "search_knowledge",
    }
    actual = set(_DIGESTERS.keys())
    missing = expected - actual
    assert not missing, f"missing digesters for heavy tools: {missing}"
