"""Endpoint-aware verifier confidence — deterministic fast-path tests.

Only tests that exercise the deterministic short-circuits (execute-class
tool present, or MEDIA: delivery in answer) are kept here. The LLM-judge
path (_llm_endpoint_met) is not unit-testable without network access.
"""
from __future__ import annotations


def test_endpoint_met_when_execute_tool_in_trace():
    """Execute-class tool in trace -> endpoint met without LLM call."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="run terminal-bench on 5 tasks",
        answer="started job j-7a3c",
        tool_names=["define_task_endpoint", "start_background_job"],
    ) is True


def test_endpoint_met_when_media_line_in_answer():
    """MEDIA: delivery line in answer -> endpoint met without LLM call."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="make me a video without logo",
        answer="done\n\nMEDIA:/home/x/.hrant/data/workspace/outbox/clip.mp4",
        tool_names=["read_file", "preprocess_video", "run_python"],
    ) is True


def test_cap_confidence_passthrough_when_execute_tool():
    """Execute-class tool -> confidence passes through cap unchanged."""
    from backend.endpoint_check import cap_confidence_for_endpoint
    assert cap_confidence_for_endpoint(
        task="run terminal-bench",
        answer="started job",
        tool_names=["start_background_job"],
        confidence=75,
    ) == 75


def test_cap_confidence_passthrough_when_media_delivery():
    """MEDIA: line -> confidence passes through cap unchanged."""
    from backend.endpoint_check import cap_confidence_for_endpoint
    assert cap_confidence_for_endpoint(
        task="make a video",
        answer="done\nMEDIA:/tmp/out.mp4",
        tool_names=[],
        confidence=80,
    ) == 80
