"""Tests for backend/tg_format.py — Hermes-style Telegram message
rendering.

The owner asked for two things on the Telegram integration:

  1. Hermes-style aesthetic: clear answer formatting, visible
     tool-calling/status blocks, professional look. This module
     handles the formatting.
  2. Push notifications — handled in channels.py by deleting the
     placeholder + sending a fresh reply (orthogonal to this file).

What this test file pins:
  - markdown → Telegram HTML conversion (bold/italic/code/link/strike)
  - safety: <script> escapes, javascript: links are NOT auto-anchored
  - footer formatters produce HTML with emoji + emphasis tags
  - render_answer_with_footer composes a clean message with divider
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ─── markdown → HTML conversion ────────────────────────────────────


def test_md_plain_text_is_escaped():
    """Plain text with no markdown — must be HTML-escaped so user
    content can't break the parser or inject markup."""
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html("hello <world> & friends")
    assert out == "hello &lt;world&gt; &amp; friends"


def test_md_bold_converts():
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html("This is **bold** text.")
    assert out == "This is <b>bold</b> text."


def test_md_italic_converts_only_when_paired():
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html("This is *italic* text.")
    assert out == "This is <i>italic</i> text."


def test_md_strike_converts():
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html("This is ~~struck~~ out.")
    assert out == "This is <s>struck</s> out."


def test_md_inline_code_converts():
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html("Use `read_file(path)` here.")
    assert out == "Use <code>read_file(path)</code> here."


def test_md_fenced_code_block_converts():
    from backend.tg_format import markdown_to_telegram_html
    src = "Here:\n\n```python\nprint('hi')\n```\nDone."
    out = markdown_to_telegram_html(src)
    assert '<pre><code class="language-python">' in out
    assert "</code></pre>" in out
    # Content inside the block must NOT be re-converted (no <b>, etc.)
    assert "print('hi')" in out


def test_md_fenced_code_block_no_language():
    from backend.tg_format import markdown_to_telegram_html
    src = "```\nplain code\n```"
    out = markdown_to_telegram_html(src)
    assert "<pre>plain code\n</pre>" in out
    # No <code class=...> when no language given.
    assert 'class="language-' not in out


def test_md_link_safe_https_converts_to_anchor():
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html("See [docs](https://example.com/x).")
    assert '<a href="https://example.com/x">docs</a>' in out


def test_md_link_unsafe_javascript_left_as_text():
    """javascript: / data: / file: URLs must NOT become anchor tags —
    they could execute arbitrary code in the recipient's client."""
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html(
        "Bad: [click](javascript:alert(1)) and [also](data:foo)."
    )
    assert "<a href=" not in out
    assert "javascript:alert" in out  # left visible, just escaped


def test_md_xss_attempt_is_escaped():
    """User content with HTML must be escaped, not interpreted."""
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html(
        "User: <script>alert('x')</script> said hi"
    )
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_md_header_becomes_bold():
    """Markdown headers (#, ##, ...) collapse to bold — Telegram has
    no native header tag."""
    from backend.tg_format import markdown_to_telegram_html
    out = markdown_to_telegram_html("# Title\nbody")
    assert "<b>Title</b>" in out


def test_md_empty_returns_empty():
    from backend.tg_format import markdown_to_telegram_html
    assert markdown_to_telegram_html("") == ""
    assert markdown_to_telegram_html(None) == ""  # type: ignore


# ─── format_trace_footer ───────────────────────────────────────────


def test_trace_footer_empty_for_empty_trace():
    from backend.tg_format import format_trace_footer
    assert format_trace_footer([]) == ""
    assert format_trace_footer(None) == ""  # type: ignore


def test_trace_footer_shows_distinct_stages():
    from backend.tg_format import format_trace_footer
    trace = [
        SimpleNamespace(event="chat", ts=0.1, tool_call=None),
        SimpleNamespace(event="solve", ts=0.5, tool_call=None),
        SimpleNamespace(event="verify", ts=0.8, tool_call=None),
    ]
    out = format_trace_footer(trace, total_time_s=0.8)
    assert "chat → solve → verify" in out
    assert "🧠" in out
    assert "(3 steps · 0.8s)" in out


def test_trace_footer_collapses_repeated_stages():
    """Multiple events with the same name should only appear once
    in the stage chain — otherwise the chain looks noisy."""
    from backend.tg_format import format_trace_footer
    trace = [
        SimpleNamespace(event="found", ts=0.1, tool_call=None),
        SimpleNamespace(event="found", ts=0.2, tool_call=None),
        SimpleNamespace(event="found", ts=0.3, tool_call=None),
        SimpleNamespace(event="solve", ts=0.5, tool_call=None),
    ]
    out = format_trace_footer(trace, total_time_s=0.5)
    # Only one "found" appears in the chain.
    assert out.count("found") == 1


def test_trace_footer_counts_tools():
    from backend.tg_format import format_trace_footer
    trace = [
        SimpleNamespace(event="tool", ts=0.1, tool_call=SimpleNamespace(name="read_file")),
        SimpleNamespace(event="tool", ts=0.2, tool_call=SimpleNamespace(name="read_file")),
        SimpleNamespace(event="tool", ts=0.3, tool_call=SimpleNamespace(name="web_search")),
        SimpleNamespace(event="solve", ts=0.5, tool_call=None),
    ]
    out = format_trace_footer(trace, total_time_s=0.5)
    assert "🔧" in out
    assert "<code>read_file</code>×2" in out
    assert "<code>web_search</code>×1" in out


def test_trace_footer_skips_tool_events_in_stage_chain():
    """Tool events go to the 🔧 line, not the 🧠 stage line."""
    from backend.tg_format import format_trace_footer
    trace = [
        SimpleNamespace(event="chat", ts=0.1, tool_call=None),
        SimpleNamespace(event="tool", ts=0.2, tool_call=SimpleNamespace(name="x")),
        SimpleNamespace(event="solve", ts=0.5, tool_call=None),
    ]
    out = format_trace_footer(trace, total_time_s=0.5)
    # "tool" must not appear in the chain itself.
    assert "tool →" not in out
    assert "→ tool" not in out
    # chat → solve must.
    assert "chat → solve" in out


# ─── format_stats_block ────────────────────────────────────────────


def test_stats_block_empty_when_no_usage():
    from backend.tg_format import format_stats_block
    assert format_stats_block(None) == ""
    assert format_stats_block(SimpleNamespace(total_tokens=0)) == ""


def test_stats_block_basic_one_line():
    from backend.tg_format import format_stats_block
    tu = SimpleNamespace(
        total_tokens=1234, input_tokens=1000, output_tokens=234,
        cache_read_tokens=0, cache_creation_tokens=0,
        llm_calls=1, by_stage={},
    )
    out = format_stats_block(tu)
    # Numbers with thousands separator.
    assert "1,234" in out
    assert "1,000" in out
    assert "234" in out
    # Single LLM call -> "1 call" not "1 calls".
    assert "1 call" in out
    assert "1 calls" not in out
    # No per-stage breakdown for small turns.
    assert "📊" not in out


def test_stats_block_shows_cache_when_nonzero():
    from backend.tg_format import format_stats_block
    tu = SimpleNamespace(
        total_tokens=5000, input_tokens=4800, output_tokens=200,
        cache_read_tokens=2000, cache_creation_tokens=500,
        llm_calls=3, by_stage={},
    )
    out = format_stats_block(tu)
    assert "cache" in out
    assert "r 2,000" in out
    assert "w 500" in out


def test_stats_block_shows_per_stage_for_large_turn():
    """When input ≥ 5k tokens and there are multiple stages, the
    per-stage breakdown helps the user see where the bill went."""
    from backend.tg_format import format_stats_block
    tu = SimpleNamespace(
        total_tokens=10_000, input_tokens=9_500, output_tokens=500,
        cache_read_tokens=0, cache_creation_tokens=0,
        llm_calls=2,
        by_stage={
            "solve": {"input_tokens": 7500},
            "verify": {"input_tokens": 1500},
            "memory": {"input_tokens": 500},
        },
    )
    out = format_stats_block(tu)
    assert "📊" in out
    assert "solve" in out
    assert "7,500" in out


# ─── render_answer_with_footer ─────────────────────────────────────


def test_render_just_body_when_no_footer():
    from backend.tg_format import render_answer_with_footer
    out = render_answer_with_footer(
        answer_html="<b>hi</b>",
        trace_footer="",
        stats_block="",
    )
    # No divider when there's no footer.
    assert "─" not in out
    assert out == "<b>hi</b>"


def test_render_with_divider_when_footer_present():
    from backend.tg_format import render_answer_with_footer
    out = render_answer_with_footer(
        answer_html="<b>hi</b>",
        trace_footer="🧠 trace line",
        stats_block="🔢 stats",
    )
    assert "<b>hi</b>" in out
    # Divider em-rule appears once.
    assert out.count("─────────") == 1
    # Footer appears after body.
    assert out.index("<b>hi</b>") < out.index("🧠 trace line")
    assert out.index("🧠 trace line") < out.index("🔢 stats")


def test_render_with_only_stats_no_trace():
    """Fast-chat path has stats but no trace footer."""
    from backend.tg_format import render_answer_with_footer
    out = render_answer_with_footer(
        answer_html="Quick answer",
        trace_footer="",
        stats_block="🔢 8,616 tok",
    )
    assert "Quick answer" in out
    assert "🔢 8,616 tok" in out
    # Divider still appears because there's at least one footer part.
    assert "─" in out
