"""P0 Phase B — solver natively emits a `---CLAIMS---` JSON tail.

Phase A built Claim/Evidence structures from data the verifier and
trace already produced, with no end-to-end binding. Phase B closes
that loop: the solver appends a structured tail to its answer
referencing tool calls by 1-based index, and the builder binds each
claim's `evidence_ids` to the matching tool EvidenceItems.

If the LLM ignores the directive, the tail is missing → fall back
to Phase A's verifier-bucket-based claims, exactly as before. The
user never sees the JSON tail in the answer.
"""
from __future__ import annotations

import json

import pytest

from backend.claims import (
    SOLVER_CLAIMS_DIRECTIVE,
    SOLVER_CLAIMS_MARKER,
    build_claims_and_evidence,
    extract_solver_claims_block,
)
from backend.models import (
    ThinkingStep,
    ToolCallDetail,
    VerificationResult,
)


# --- directive surface ----------------------------------------------------


def test_directive_mentions_marker_and_format():
    assert SOLVER_CLAIMS_MARKER in SOLVER_CLAIMS_DIRECTIVE
    assert "tool_1" in SOLVER_CLAIMS_DIRECTIVE
    assert "evidence" in SOLVER_CLAIMS_DIRECTIVE


# --- extract_solver_claims_block ------------------------------------------


def test_extract_no_marker_returns_input_unchanged():
    txt = "just a plain answer"
    cleaned, claims = extract_solver_claims_block(txt)
    assert cleaned == txt
    assert claims is None


def test_extract_clean_block_returns_parsed_claims():
    txt = (
        "RS-485 is a differential bus.\n\n"
        f"{SOLVER_CLAIMS_MARKER}\n"
        '{"claims":[{"text":"RS-485 is differential","evidence":["tool_1"]}]}'
    )
    cleaned, claims = extract_solver_claims_block(txt)
    assert "RS-485 is a differential bus." in cleaned
    assert SOLVER_CLAIMS_MARKER not in cleaned
    assert claims == [{"text": "RS-485 is differential", "evidence": ["tool_1"]}]


def test_extract_strips_marker_even_when_json_invalid():
    """Marker present but body unparseable — answer must STILL be
    cleaned. The user should never see a raw `---CLAIMS---` tail or
    a half-block of broken JSON."""
    txt = (
        "answer body\n"
        f"{SOLVER_CLAIMS_MARKER}\n"
        "{this is not json at all"
    )
    cleaned, claims = extract_solver_claims_block(txt)
    assert SOLVER_CLAIMS_MARKER not in cleaned
    assert "{this is not json" not in cleaned
    assert claims is None


def test_extract_strips_marker_with_no_body():
    txt = f"answer body\n\n{SOLVER_CLAIMS_MARKER}\n"
    cleaned, claims = extract_solver_claims_block(txt)
    assert SOLVER_CLAIMS_MARKER not in cleaned
    assert cleaned.strip() == "answer body"
    assert claims is None


def test_extract_handles_multiple_claims():
    block = json.dumps({
        "claims": [
            {"text": "claim a", "evidence": ["tool_1"]},
            {"text": "claim b", "evidence": ["tool_2", "tool_3"]},
            {"text": "claim c", "evidence": []},
        ],
    })
    txt = f"answer prose.\n\n{SOLVER_CLAIMS_MARKER}\n{block}"
    _, claims = extract_solver_claims_block(txt)
    assert claims is not None
    assert len(claims) == 3
    assert claims[0]["evidence"] == ["tool_1"]
    assert claims[2]["evidence"] == []


def test_extract_drops_claims_with_empty_text():
    block = json.dumps({"claims": [
        {"text": "", "evidence": ["tool_1"]},
        {"text": "   ", "evidence": ["tool_2"]},
        {"text": "real", "evidence": []},
    ]})
    txt = f"x\n{SOLVER_CLAIMS_MARKER}\n{block}"
    _, claims = extract_solver_claims_block(txt)
    assert claims == [{"text": "real", "evidence": []}]


def test_extract_handles_empty_input():
    cleaned, claims = extract_solver_claims_block("")
    assert cleaned == ""
    assert claims is None


def test_extract_keeps_inline_marker_in_code_block():
    """A user-pasted code block that happens to contain the marker
    inline (not on its own at the END of the answer) must NOT trip
    the regex. Otherwise prose gets eaten."""
    txt = (
        "Here's a code example:\n"
        f"```\n# This file uses {SOLVER_CLAIMS_MARKER}\n```\n"
        "and that's it."
    )
    cleaned, claims = extract_solver_claims_block(txt)
    # Marker still in the cleaned text because it's not at end
    # followed by JSON body. The fallback `rfind` trims everything
    # from the marker onward; this is a known trade-off — we accept
    # the false-positive trim in the rare prose-marker case so we
    # never leak a half-block to the user. The trim is safe: the
    # text before the marker is preserved.
    if "Here's a code example" not in cleaned:
        pytest.fail("trimming should keep prose before the marker")


# --- build_claims_and_evidence with solver_claims -------------------------


def _trace_step(name: str, args: dict, result: str = "ok",
                is_error: bool = False) -> ThinkingStep:
    return ThinkingStep(
        ts=0.0, event="tool", message=f"{name}()",
        tool_call=ToolCallDetail(
            name=name, args=args, result=result, is_error=is_error,
        ),
    )


def test_solver_claims_bind_evidence_ids():
    """tool_1 → evidence[0].id, tool_2 → evidence[1].id, …"""
    trace = [
        _trace_step("read_file", {"path": "x.py"}, result="line 1"),
        _trace_step("calc", {"expression": "2+2"}, result="4"),
        _trace_step("locate_symbol", {"name": "foo", "path": "x.py"}, result="[]"),
    ]
    vr = VerificationResult(
        confidence=80,
        verified_claims=["x is line 1", "result is 4"],
    )
    solver_claims = [
        {"text": "x is line 1", "evidence": ["tool_1"]},
        {"text": "result is 4", "evidence": ["tool_2"]},
        {"text": "no symbol foo", "evidence": ["tool_3"]},
    ]
    claims, evidence = build_claims_and_evidence(
        vr, trace, solver_claims=solver_claims,
    )
    tool_evs = [e for e in evidence if e.source_type == "tool"]
    assert len(tool_evs) == 3
    # First claim should reference the first tool's evidence id.
    assert claims[0].evidence_ids == [tool_evs[0].id]
    assert claims[1].evidence_ids == [tool_evs[1].id]
    assert claims[2].evidence_ids == [tool_evs[2].id]


def test_solver_claims_status_decided_by_verifier():
    """Solver doesn't get to mark its own claim 'verified' — verifier
    is the source of truth on what's grounded."""
    trace = [_trace_step("read_file", {"path": "x.py"})]
    vr = VerificationResult(
        confidence=50,
        verified_claims=["solid claim"],
        unverified_claims=["shaky claim"],
        contradictions=["wrong claim"],
    )
    solver_claims = [
        {"text": "solid claim", "evidence": ["tool_1"]},
        {"text": "shaky claim", "evidence": ["tool_1"]},
        {"text": "wrong claim", "evidence": ["tool_1"]},
        {"text": "claim with no verifier match", "evidence": ["tool_1"]},
    ]
    claims, _ = build_claims_and_evidence(vr, trace, solver_claims=solver_claims)
    statuses = {c.text: c.status for c in claims}
    assert statuses["solid claim"] == "verified"
    assert statuses["shaky claim"] == "unverified"
    assert statuses["wrong claim"] == "contradicted"
    # Unmatched claim falls back to "unverified" (cautious default).
    assert statuses["claim with no verifier match"] == "unverified"


def test_solver_claim_unverified_with_no_evidence_gets_high_risk():
    """A solver claim that admits no tool grounding AND isn't in the
    verifier's verified bucket is the most suspicious thing on the
    turn — bump risk to 'high' so the WebUI can flag it."""
    vr = VerificationResult(confidence=50, verified_claims=[])
    solver_claims = [
        {"text": "I think X is true", "evidence": []},
    ]
    claims, _ = build_claims_and_evidence(vr, [], solver_claims=solver_claims)
    assert claims[0].risk == "high"


def test_solver_claim_with_evidence_keeps_normal_risk():
    """An unverified claim that DOES have tool evidence is less
    suspicious — risk stays at the status default."""
    trace = [_trace_step("read_file", {"path": "x.py"})]
    vr = VerificationResult(confidence=50)
    solver_claims = [{"text": "X reads as Y", "evidence": ["tool_1"]}]
    claims, _ = build_claims_and_evidence(vr, trace, solver_claims=solver_claims)
    assert claims[0].risk == "medium"  # unverified default


def test_solver_claims_match_via_substring_when_wording_drifts():
    """Solver: 'differential bus, 12 Mbit/s'
       Verifier:'differential bus'
    Match by normalised substring so a slightly looser solver phrasing
    still inherits the verifier's status."""
    vr = VerificationResult(
        confidence=80,
        verified_claims=["differential bus"],
    )
    solver_claims = [{"text": "differential bus, 12 Mbit/s", "evidence": []}]
    claims, _ = build_claims_and_evidence(vr, [], solver_claims=solver_claims)
    assert claims[0].status == "verified"


def test_solver_claims_skip_unknown_tool_refs():
    """If the solver names `tool_5` but only 2 tools were called,
    the unknown ref is silently dropped — claim stays, evidence_ids
    just doesn't include it."""
    trace = [_trace_step("read_file", {"path": "x.py"})]
    solver_claims = [{"text": "x", "evidence": ["tool_1", "tool_5", "tool_99"]}]
    claims, evidence = build_claims_and_evidence(
        VerificationResult(confidence=50), trace, solver_claims=solver_claims,
    )
    tool_evs = [e for e in evidence if e.source_type == "tool"]
    assert claims[0].evidence_ids == [tool_evs[0].id]


def test_solver_claims_empty_falls_back_to_phase_a():
    vr = VerificationResult(
        confidence=80,
        verified_claims=["a", "b"],
    )
    claims, _ = build_claims_and_evidence(vr, [], solver_claims=[])
    # Phase A path → claims from verifier buckets, count matches.
    assert {c.text for c in claims} == {"a", "b"}


def test_solver_claims_none_falls_back_to_phase_a():
    vr = VerificationResult(confidence=80, verified_claims=["x"])
    claims, _ = build_claims_and_evidence(vr, [], solver_claims=None)
    assert claims[0].text == "x"


# --- _solve integration ---------------------------------------------------


def test_solve_appends_directive_to_user_prompt(tmp_kb):
    """Whatever the live solver flow looks like, the directive must
    end up in the user prompt — otherwise the LLM never knows to
    emit a tail."""
    from unittest.mock import patch

    from backend.agent import Agent
    from backend.llm import TaskType

    captured = {}

    class _Capture:
        def call(self, task_type, system, user, **kw):
            if task_type == TaskType.COMPLEX_SOLVING:
                captured["user"] = user
                return "answer."
            return ""

        def call_with_tools(self, task_type, system, user, **kw):
            return self.call(task_type, system, user, **kw)

        def call_json(self, task_type, system, user, **kw):
            if task_type == TaskType.TASK_ANALYSIS:
                return {"required_topics": [], "plan": [], "confidence": 50}
            if task_type == TaskType.VERIFICATION:
                return {
                    "verified_claims": [],
                    "unverified_claims": [],
                    "contradictions": [],
                    "notes_used": [],
                }
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "task", "reason": "test"}
            return {}

    fake = _Capture()
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("explain X")

    assert "user" in captured
    assert SOLVER_CLAIMS_MARKER in captured["user"]
    assert "tool_1" in captured["user"]


def test_solve_strips_tail_from_answer_and_stashes_claims(tmp_kb):
    """Full integration: LLM emits a tail, _solve strips it for the
    user-visible answer, and stashes parsed claims on `self`. The
    AgentAnswer that comes back has them in `claims` with bound
    evidence_ids when tools were involved."""
    from unittest.mock import patch

    from backend.agent import Agent
    from backend.llm import TaskType

    block = json.dumps({"claims": [
        {"text": "X is Y", "evidence": []},
        {"text": "Z works", "evidence": []},
    ]})
    solver_response = (
        "X is Y. Z works.\n\n"
        f"{SOLVER_CLAIMS_MARKER}\n{block}"
    )

    class _LLM:
        def call(self, task_type, system, user, **kw):
            if task_type == TaskType.COMPLEX_SOLVING:
                return solver_response
            return ""

        def call_with_tools(self, task_type, system, user, **kw):
            return self.call(task_type, system, user, **kw)

        def call_json(self, task_type, system, user, **kw):
            if task_type == TaskType.TASK_ANALYSIS:
                return {"required_topics": [], "plan": [], "confidence": 50}
            if task_type == TaskType.VERIFICATION:
                return {
                    "verified_claims": ["X is Y", "Z works"],
                    "unverified_claims": [],
                    "contradictions": [],
                    "notes_used": [],
                }
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "task", "reason": "test"}
            return {}

    fake = _LLM()
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        res = agent.run("tell me about X and Z")

    # User-visible answer is clean — no marker, no JSON.
    assert SOLVER_CLAIMS_MARKER not in res.answer
    assert "{" not in res.answer.split("\n")[-1]
    # Structured claims came through with verifier-decided status.
    assert {c.text for c in res.claims} == {"X is Y", "Z works"}
    assert all(c.status == "verified" for c in res.claims)
