"""Chat fast-path bare-tool-call guard (2026-06-13).

Found in Gor's real history: a chat turn was answered with the
literal text `web_search(query="...")` instead of running it — the
model emitted a bare parenthesised function call as its answer, and
the fast-path's `<tool_call` XML guard didn't catch that form. The
user got a non-answer and had to poke "Hrant?". The guard now also
recognises a bare call to a REAL registered tool.
"""
from __future__ import annotations

from backend.unified_agent import _looks_like_tool_call_dump


def test_bare_known_tool_call_detected():
    assert _looks_like_tool_call_dump(
        'web_search(query="global cryptocurrency market cap today")'
    ) is True
    assert _looks_like_tool_call_dump("read_file(path='/etc/hosts')") is True
    assert _looks_like_tool_call_dump("  terminal_exec(command='ls')") is True


def test_ordinary_prose_not_misread():
    # Legit text that happens to contain function-call-looking bits.
    assert _looks_like_tool_call_dump(
        "You can use print(x) to debug this."
    ) is False
    assert _looks_like_tool_call_dump(
        "The market cap is about $2.17T according to CoinGecko."
    ) is False
    # A parenthesised call to something that is NOT a registered tool.
    assert _looks_like_tool_call_dump("frobnicate(42)") is False
    assert _looks_like_tool_call_dump("") is False


def test_guard_only_matches_at_head():
    # A real answer that mentions a tool mid-sentence must not trip it.
    assert _looks_like_tool_call_dump(
        "I would run web_search(...) but I already know the answer."
    ) is False


def test_tool_code_block_dump_detected():
    """Gemini/code-style: <tool_code>print(web_search(...))</tool_code>
    leaked as a chat answer (found gathering Q2 data 2026-06-15)."""
    assert _looks_like_tool_call_dump(
        '<tool_code>\nprint(web_search(query="SSD price 2026"))\n</tool_code>'
    ) is True
    assert _looks_like_tool_call_dump(
        'print(read_file(path="/etc/hosts"))'
    ) is True
    assert _looks_like_tool_call_dump(
        '```tool_code\nweb_search(query="x")\n```'
    ) is True


def test_print_of_nontool_not_misread():
    assert _looks_like_tool_call_dump("print(x + 1)  # debug") is False
    assert _looks_like_tool_call_dump("print('hello world')") is False
