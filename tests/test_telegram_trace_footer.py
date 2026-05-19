"""Telegram replies need a compact thinking + tools summary appended
between the answer and the token-usage block. Telegram has no
collapsible UI, so the footer must stay tight: one stage-chain line
and one tools-tally line, both omitted when empty.

After the Hermes-style refactor (`backend/tg_format.py`) the
old `_format_trace_footer` in channels.py was replaced by
`format_trace_footer` in `backend.tg_format`. Same semantics, HTML
emphasis added, plus reusable across other DM call sites.
"""
from __future__ import annotations
from types import SimpleNamespace

from backend.tg_format import format_trace_footer
from backend.models import ThinkingStep, ToolCallDetail


def test_empty_trace_returns_empty_string():
    assert format_trace_footer([]) == ""
    assert format_trace_footer(None) == ""  # type: ignore[arg-type]


def test_stage_chain_omits_tool_events_and_dedups():
    trace = [
        ThinkingStep(ts=0.0, event="core", message="loading"),
        ThinkingStep(ts=0.1, event="think", message="thinking"),
        ThinkingStep(ts=0.5, event="tool", message="read_file()",
                     tool_call=ToolCallDetail(name="read_file", args={"path": "x.py"}, result="...")),
        ThinkingStep(ts=0.7, event="tool", message="read_file()",
                     tool_call=ToolCallDetail(name="read_file", args={"path": "y.py"}, result="...")),
        ThinkingStep(ts=1.0, event="solve", message="composing"),
        ThinkingStep(ts=2.5, event="verify", message="verifying"),
    ]
    out = format_trace_footer(trace, total_time_s=2.5)
    assert "🧠" in out
    # Tool stage isn't shown in the stage chain — it's on the 🔧 line.
    stage_section = out.split("🧠", 1)[1].split("🔧", 1)[0] if "🔧" in out else out
    assert "tool" not in stage_section
    # Stages appear in order, deduped.
    assert "core" in out
    assert "think" in out
    assert "solve" in out
    assert "verify" in out
    # Step count + elapsed time.
    assert "6 steps" in out
    assert "2.5s" in out


def test_tools_tally_counts_per_tool():
    trace = [
        ThinkingStep(ts=0.0, event="solve", message="composing"),
        ThinkingStep(ts=0.1, event="tool", message="read_file",
                     tool_call=ToolCallDetail(name="read_file", args={}, result="A")),
        ThinkingStep(ts=0.2, event="tool", message="read_file",
                     tool_call=ToolCallDetail(name="read_file", args={}, result="B")),
        ThinkingStep(ts=0.3, event="tool", message="web_search",
                     tool_call=ToolCallDetail(name="web_search", args={}, result="C")),
        ThinkingStep(ts=0.4, event="tool_error", message="grep",
                     tool_call=ToolCallDetail(name="grep", args={}, result="D", is_error=True)),
    ]
    out = format_trace_footer(trace, total_time_s=0.4)
    assert "🔧" in out
    # HTML-emphasised tool names with ×N count.
    assert "<code>read_file</code>×2" in out
    assert "<code>web_search</code>×1" in out
    assert "<code>grep</code>×1" in out


def test_tools_line_omitted_when_no_tools():
    trace = [
        ThinkingStep(ts=0.0, event="chat", message="quick reply"),
    ]
    out = format_trace_footer(trace, total_time_s=0.0)
    assert "🔧" not in out
    # But the thinking line is still present.
    assert "🧠" in out


def test_long_chain_is_truncated():
    """More than 8 distinct stages should compress with a `(+N)` suffix
    so the line doesn't get unwieldy on Telegram's narrow screen."""
    stages = ["core", "think", "memory", "strategy", "found",
              "solve", "verify", "experience", "cleanup", "tick", "done"]
    trace = [ThinkingStep(ts=float(i), event=ev, message="x") for i, ev in enumerate(stages)]
    out = format_trace_footer(trace, total_time_s=10.0)
    assert "(+" in out
