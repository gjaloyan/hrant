"""P0 — claim/evidence layer (Phase A).

Phase A is purely additive: new dataclasses on AgentAnswer, populated
from existing data (verifier buckets + thinking trace tool calls).
The solver and verifier are not yet emitting these natively — that's
Phase B. These tests pin the data contract so Phase B has something
stable to integrate against.
"""
from __future__ import annotations

import pytest

from backend.claims import (
    _format_tool_ref,
    build_claims_and_evidence,
)
from backend.models import (
    Claim,
    EvidenceItem,
    ThinkingStep,
    ToolCallDetail,
    VerificationResult,
)


# --- model surface --------------------------------------------------------


def test_claim_has_required_fields():
    c = Claim(id="c_001", text="user lives in Yerevan")
    assert c.id == "c_001"
    assert c.text == "user lives in Yerevan"
    assert c.status == "unverified"
    assert c.evidence_ids == []
    assert c.risk == "low"


def test_evidence_item_has_required_fields():
    e = EvidenceItem(id="ev_001")
    assert e.id == "ev_001"
    assert e.source_type == "model"
    assert e.confidence == 1.0


def test_agent_answer_carries_claims_and_evidence_optional():
    """Backward compat: an AgentAnswer built without claims/evidence
    must default both to empty lists, not raise."""
    from backend.models import AgentAnswer
    a = AgentAnswer(
        answer="x",
        verification=VerificationResult(confidence=50),
    )
    assert a.claims == []
    assert a.evidence == []


# --- build_claims_and_evidence: claims side --------------------------------


def test_builds_one_claim_per_verified_string():
    vr = VerificationResult(
        confidence=80,
        verified_claims=["alpha", "beta"],
        unverified_claims=[],
        contradictions=[],
    )
    claims, _ = build_claims_and_evidence(vr, [], user_message="")
    assert len(claims) == 2
    assert all(c.status == "verified" for c in claims)
    assert all(c.risk == "low" for c in claims)
    assert {c.text for c in claims} == {"alpha", "beta"}


def test_unverified_and_contradicted_get_higher_risk():
    vr = VerificationResult(
        confidence=50,
        verified_claims=[],
        unverified_claims=["maybe"],
        contradictions=["definitely wrong"],
    )
    claims, _ = build_claims_and_evidence(vr, [])
    statuses = {c.status: c.risk for c in claims}
    assert statuses["unverified"] == "medium"
    assert statuses["contradicted"] == "high"


def test_claim_ids_are_deterministic_and_unique():
    """Same verification → same ids → UI can diff across re-renders."""
    vr = VerificationResult(
        confidence=80,
        verified_claims=["a", "b"],
        unverified_claims=["c"],
        contradictions=["d"],
    )
    claims_1, _ = build_claims_and_evidence(vr, [])
    claims_2, _ = build_claims_and_evidence(vr, [])
    ids_1 = [c.id for c in claims_1]
    ids_2 = [c.id for c in claims_2]
    assert ids_1 == ids_2
    assert len(set(ids_1)) == len(ids_1)  # unique


def test_empty_claim_strings_are_skipped():
    vr = VerificationResult(
        confidence=80,
        verified_claims=["", "  ", "real"],
    )
    claims, _ = build_claims_and_evidence(vr, [])
    assert len(claims) == 1
    assert claims[0].text == "real"


def test_long_claim_text_is_capped():
    long_text = "x" * 5000
    vr = VerificationResult(
        confidence=80,
        verified_claims=[long_text],
    )
    claims, _ = build_claims_and_evidence(vr, [])
    assert len(claims[0].text) <= 800


# --- build_claims_and_evidence: evidence side -----------------------------


def _trace_step(name: str, args: dict, result: str = "ok",
                is_error: bool = False) -> ThinkingStep:
    return ThinkingStep(
        ts=0.0, event="tool", message=f"{name}()",
        tool_call=ToolCallDetail(
            name=name, args=args, result=result, is_error=is_error,
        ),
    )


def test_evidence_built_from_tool_calls_in_trace():
    trace = [
        _trace_step("read_file", {"path": "x.py"}, result="def x(): pass"),
        _trace_step("calc", {"expression": "2+2"}, result="4"),
    ]
    _, evidence = build_claims_and_evidence(
        VerificationResult(confidence=50), trace,
    )
    assert len(evidence) == 2
    types = {e.source_type for e in evidence}
    assert types == {"tool"}


def test_evidence_ref_uses_compact_format_for_read_file():
    trace = [_trace_step("read_file", {
        "path": "backend/llm.py", "start_line": 1026, "end_line": 1180,
    })]
    _, evidence = build_claims_and_evidence(VerificationResult(confidence=50), trace)
    assert evidence[0].source_ref == "read_file:backend/llm.py:1026-1180"


def test_evidence_ref_for_locate_symbol():
    ref = _format_tool_ref("locate_symbol", {
        "name": "Agent.run", "path": "backend/agent.py",
    })
    assert ref == "locate_symbol:Agent.run@backend/agent.py"


def test_evidence_ref_for_calc_uses_expression():
    ref = _format_tool_ref("calc", {"expression": "2+2*3"})
    assert ref == "calc:2+2*3"


def test_evidence_ref_for_web_search_uses_query():
    ref = _format_tool_ref("web_search", {"query": "latest python release"})
    assert ref.startswith("web_search:")
    assert "latest python release" in ref


def test_evidence_ref_falls_back_to_bare_name():
    """Unknown tool with no recognised args → just the name."""
    assert _format_tool_ref("totally_made_up_tool", {"foo": "bar"}) == "totally_made_up_tool"


def test_failed_tool_call_gets_zero_confidence():
    trace = [_trace_step("read_file", {"path": "x.py"},
                         result="[error]", is_error=True)]
    _, evidence = build_claims_and_evidence(VerificationResult(confidence=50), trace)
    assert evidence[0].confidence == 0.0


def test_user_message_becomes_evidence_item():
    _, evidence = build_claims_and_evidence(
        VerificationResult(confidence=50), [],
        user_message="my brother's name is Tigran",
    )
    user_evs = [e for e in evidence if e.source_type == "user"]
    assert len(user_evs) == 1
    assert user_evs[0].source_ref == "user_turn"
    assert "Tigran" in user_evs[0].quote


def test_empty_user_message_is_not_added_as_evidence():
    _, evidence = build_claims_and_evidence(
        VerificationResult(confidence=50), [], user_message="",
    )
    assert all(e.source_type != "user" for e in evidence)


def test_long_tool_quote_is_capped():
    big = "z" * 5000
    trace = [_trace_step("read_file", {"path": "x.py"}, result=big)]
    _, evidence = build_claims_and_evidence(VerificationResult(confidence=50), trace)
    assert len(evidence[0].quote) <= 801  # 800 + ellipsis char


def test_steps_without_tool_call_are_skipped():
    """Non-tool steps in the trace shouldn't produce evidence."""
    trace = [
        ThinkingStep(ts=0.0, event="think", message="thinking", tool_call=None),
        ThinkingStep(ts=0.1, event="solve", message="solving", tool_call=None),
        _trace_step("calc", {"expression": "1+1"}, result="2"),
    ]
    _, evidence = build_claims_and_evidence(VerificationResult(confidence=50), trace)
    # Only the tool step contributes.
    assert len(evidence) == 1


# --- Agent.run integration ------------------------------------------------


def test_agent_run_populates_claims_and_evidence(tmp_kb):
    """End-to-end: when the agent runs a real (mocked) task turn, the
    AgentAnswer must carry claims + evidence built from the
    verification result and the trace."""
    from unittest.mock import patch

    from backend.agent import Agent
    from backend.finetune import store as finetune_store
    from backend.llm import TaskType

    # Re-use the FakeRouter shape from test_agent.py.
    import json

    class FakeRouter:
        def __init__(self):
            self.calls = []

        def call(self, task_type, system, user, **kw):
            if task_type == TaskType.COMPLEX_SOLVING:
                return "RS-485 is a differential bus."
            if task_type == TaskType.QUICK_ANSWER:
                return ""
            return ""

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, task_type, system, user, **kw):
            self.calls.append(task_type)
            if task_type == TaskType.TASK_ANALYSIS:
                return {
                    "required_topics": ["RS-485"],
                    "plan": ["answer"],
                    "confidence": 50,
                }
            if task_type == TaskType.VERIFICATION:
                return {
                    "verified_claims": ["differential bus"],
                    "unverified_claims": ["minor"],
                    "contradictions": [],
                    "notes_used": [],
                }
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "task", "reason": "test"}
            return {}

    tmp_kb.save_note(
        topic="RS-485",
        body="## What\nDifferential bus.",
        category="profession",
        keywords=["rs485"],
        source="test",
    )

    fake = FakeRouter()
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        res = agent.run("tell me about RS-485")

    # Claims built from verification buckets.
    statuses = {c.status for c in res.claims}
    assert "verified" in statuses
    assert any(c.text == "differential bus" for c in res.claims)
    # User message captured as one evidence item.
    user_evs = [e for e in res.evidence if e.source_type == "user"]
    assert len(user_evs) == 1
    assert "RS-485" in user_evs[0].quote
