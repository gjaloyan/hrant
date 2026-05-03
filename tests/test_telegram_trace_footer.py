"""Telegram replies need a compact thinking + tools summary appended
between the answer and the token-usage block. Telegram has no
collapsible UI, so the footer must stay tight: one stage-chain line
and one tools-tally line, both omitted when empty."""
from __future__ import annotations
from types import SimpleNamespace

from backend.channels import _format_trace_footer
from backend.models import ThinkingStep, ToolCallDetail


def _result(trace: list[ThinkingStep]) -> SimpleNamespace:
    """Minimal stand-in for AgentAnswer — only the field the formatter reads."""
    return SimpleNamespace(thinking_trace=trace)


def test_empty_trace_returns_empty_string():
    assert _format_trace_footer(_result([])) == ""
    # Missing attribute too — chat fast-path may produce no trace.
    assert _format_trace_footer(SimpleNamespace()) == ""


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
    out = _format_trace_footer(_result(trace))
    assert "🧠 Thinking:" in out
    # Tool stage isn't shown in the chain
    assert "tool" not in out.split("Thinking:", 1)[1].split("Tools:")[0]
    # Stages appear in order, deduped
    chain = out.split("Thinking:", 1)[1].split("(", 1)[0]
    assert "core" in chain and "think" in chain and "solve" in chain and "verify" in chain
    # Step count + elapsed time tail
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
    out = _format_trace_footer(_result(trace))
    assert "🔧 Tools:" in out
    assert "read_file(2)" in out
    assert "web_search(1)" in out
    assert "grep(1)" in out


def test_tools_line_omitted_when_no_tools():
    trace = [
        ThinkingStep(ts=0.0, event="chat", message="quick reply"),
    ]
    out = _format_trace_footer(_result(trace))
    assert "🔧 Tools:" not in out
    # But the thinking line is still present
    assert "🧠 Thinking:" in out


def test_long_chain_is_truncated():
    """More than 8 distinct stages should compress with a `(+N)` suffix
    so the line doesn't get unwieldy on Telegram's narrow screen."""
    stages = ["core", "think", "memory", "strategy", "found",
              "solve", "verify", "experience", "cleanup", "tick", "done"]
    trace = [ThinkingStep(ts=float(i), event=ev, message="x") for i, ev in enumerate(stages)]
    out = _format_trace_footer(_result(trace))
    assert "(+" in out
