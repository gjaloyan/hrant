"""Endpoint-aware verifier confidence — deterministic fast-path tests.

One deterministic short-circuit remains: an execute-class tool in the trace.
The `"MEDIA:" in answer` short-circuit was deleted on 2026-08-06 — see
test_completion_gate.py for why — so the tests that pinned it now assert the
opposite: an attached file is evidence for the judge, never a verdict.
"""
from __future__ import annotations

import pytest


def test_endpoint_met_when_execute_tool_in_trace():
    """Execute-class tool in trace -> endpoint met without LLM call."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="run terminal-bench on 5 tasks",
        answer="started job j-7a3c",
        tool_names=["define_task_endpoint", "start_background_job"],
    ) is True


def test_media_line_is_judged_not_trusted(monkeypatch):
    """A MEDIA: line no longer short-circuits: the judge is consulted, and it
    is handed code-produced evidence rather than the assistant's prose."""
    import backend.endpoint_check as ec
    seen = {}

    def _fake_llm(task, answer, evidence=""):
        seen["evidence"] = evidence
        return True

    monkeypatch.setattr(ec, "_llm_endpoint_met", _fake_llm)
    assert ec.endpoint_met(
        task="make me a video without logo",
        answer="done\n\nMEDIA:/home/x/.hrant/data/workspace/outbox/clip.mp4",
        tool_names=["read_file", "preprocess_video", "run_python"],
    ) is True
    assert "evidence" in seen, "the judge was skipped — short-circuit is back"
    assert "read_file, preprocess_video, run_python" in seen["evidence"]


def test_cap_confidence_passthrough_when_execute_tool():
    """Execute-class tool -> confidence passes through cap unchanged."""
    from backend.endpoint_check import cap_confidence_for_endpoint
    assert cap_confidence_for_endpoint(
        task="run terminal-bench",
        answer="started job",
        tool_names=["start_background_job"],
        confidence=75,
    ) == 75


def test_cap_confidence_when_media_delivery_defers_to_the_judge(monkeypatch):
    """The cap no longer passes through on the strength of a MEDIA: substring.
    A judge that rules "not delivered" must be able to clip the confidence of
    an answer that attached a file — that is the whole 2026-08-06 fix."""
    import backend.endpoint_check as ec
    monkeypatch.setattr(ec, "_llm_endpoint_met",
                        lambda task, answer, evidence="": False)
    capped = ec.cap_confidence_for_endpoint(
        task="calibrate the search engines",
        answer="done\nMEDIA:/tmp/scratch_matrix.txt",
        tool_names=[],
        confidence=80,
    )
    assert capped < 80

    monkeypatch.setattr(ec, "_llm_endpoint_met",
                        lambda task, answer, evidence="": True)
    assert ec.cap_confidence_for_endpoint(
        task="make a video", answer="done\nMEDIA:/tmp/out.mp4",
        tool_names=[], confidence=80,
    ) == 80
