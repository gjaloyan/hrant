"""Round 4 / P1: the self-analysis guard must require an actual
read_file / view_file in the trace. Non-empty tool_context from
calc / web_search / run_python doesn't prove the agent looked at
its own source — the agent could call calc("2+2"), get "4" back,
and still hallucinate about its own architecture.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import Agent, SOURCE_READ_TOOLS
from backend.models import (
    ThinkingResult,
    ThinkingStep,
    ToolCallDetail,
    VerificationResult,
)


def test_source_read_tools_set_membership():
    assert "read_file" in SOURCE_READ_TOOLS
    assert "view_file" in SOURCE_READ_TOOLS
    # Things that must NOT count as source reads.
    assert "calc" not in SOURCE_READ_TOOLS
    assert "web_search" not in SOURCE_READ_TOOLS
    assert "run_python" not in SOURCE_READ_TOOLS


def _trace_step_with_tool(name: str) -> ThinkingStep:
    return ThinkingStep(
        ts=0.5, event="tool", message=f"{name}()",
        tool_call=ToolCallDetail(name=name, args={}, result="ok"),
    )


def test_guard_fires_when_only_calc_was_called():
    """Solver answered self-analysis turn but only called calc — must
    trigger the source-read guard. We probe the same `any(...)`
    expression the guard uses against a synthetic trace."""
    trace = [
        ThinkingStep(ts=0.0, event="solve", message="composing"),
        _trace_step_with_tool("calc"),
        _trace_step_with_tool("web_search"),
    ]
    source_files_read = any(
        step.tool_call is not None and step.tool_call.name in SOURCE_READ_TOOLS
        for step in trace
    )
    assert source_files_read is False


def test_guard_passes_when_read_file_was_called():
    trace = [
        _trace_step_with_tool("calc"),
        _trace_step_with_tool("read_file"),
    ]
    source_files_read = any(
        step.tool_call is not None and step.tool_call.name in SOURCE_READ_TOOLS
        for step in trace
    )
    assert source_files_read is True


def test_view_file_also_counts():
    """view_file is the line-range cousin from Round 3 — it MUST
    qualify as a source read or the guard would fire spuriously."""
    trace = [_trace_step_with_tool("view_file")]
    source_files_read = any(
        step.tool_call is not None and step.tool_call.name in SOURCE_READ_TOOLS
        for step in trace
    )
    assert source_files_read is True


def test_steps_without_tool_call_are_ignored():
    """Plain progress events (no tool_call payload) must not be
    interpreted as tool invocations."""
    trace = [
        ThinkingStep(ts=0.0, event="think", message="thinking"),
        ThinkingStep(ts=0.1, event="strategy", message="planned"),
        ThinkingStep(ts=0.2, event="solve", message="composing"),
    ]
    source_files_read = any(
        step.tool_call is not None and step.tool_call.name in SOURCE_READ_TOOLS
        for step in trace
    )
    assert source_files_read is False
