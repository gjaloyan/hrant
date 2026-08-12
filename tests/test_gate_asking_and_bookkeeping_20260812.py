"""Asking, and filing a note, are not doing.

Two shapes the owner hit repeatedly on 2026-08-12, both passing the
completion gate as delivered.

1. He wrote "continue". The agent replied:

       ❓ Продолжать в подтверждённом узком scope: smoke-test Graf-J локально?

   `ask_user` was in the delivery set, on the reasoning that "asking IS the
   terminal act of the turn". So a question auto-satisfied the gate without
   the answer ever being read: no work, no risk, full marks. The gate was
   paying the agent to ask instead of act — three times in one conversation.

2. He answered "Да, продолжай". The agent then ran three tools —
   get_background_job, read_file, save_knowledge — wrote a note about the
   previous result and said "дальше надо применять Graf-J к реальной CAPTCHA".
   Bookkeeping about the task, plus a description of the next step. The gate
   passed it: real tools ran, something real was written.

Note the budget was NOT the constraint here. After the iteration ceiling went
to 500 this turn used three calls. It stopped because stopping scored.
"""
import pytest

from backend.endpoint_check import (
    _DELIVERY_TOOLS, _ENDPOINT_JUDGE_SYSTEM, _INSTRUMENT_TOOLS,
    _turn_evidence, endpoint_met,
)


def test_asking_no_longer_auto_passes():
    assert "ask_user" not in _DELIVERY_TOOLS
    assert "ask_user" in _INSTRUMENT_TOOLS


def test_a_question_turn_reaches_the_judge(monkeypatch):
    import backend.endpoint_check as ec
    seen = {}

    def _judge(task, answer, evidence=""):
        seen["answer"] = answer
        return False

    monkeypatch.setattr(ec, "_llm_endpoint_met", _judge)
    out = endpoint_met(task="continue", answer="❓ Shall I proceed?",
                       tool_names=["ask_user"])
    assert out is False
    assert seen["answer"] == "❓ Shall I proceed?"


def test_the_judge_separates_a_real_question_from_a_permission_request():
    p = _ENDPOINT_JUDGE_SYSTEM
    assert "shall I proceed?" in p
    assert "credential" in p
    assert "the user has already answered it" in p


def test_the_judge_knows_bookkeeping_is_not_the_work():
    p = _ENDPOINT_JUDGE_SYSTEM
    assert "BOOKKEEPING IS NOT THE WORK" in p
    assert "Saving a note" in p
    assert "a report about it" in p


# ── the evidence block now carries results, not just names ──────────

def test_evidence_includes_what_the_tools_returned():
    """Names alone cannot settle a concrete claim. Asked to rule on
    "recognised '6wuf'" against a list reading `agent_browser`, the judge has
    nothing to confirm and says not-delivered — a false NOT DONE on a turn
    that did the work, which is worse than the miss the gate exists for."""
    ev = _turn_evidence(["agent_browser"], "answer",
                        [("agent_browser", '{"success":true,"text":"6wuf"}')])
    assert "agent_browser -> " in ev
    assert "6wuf" in ev


def test_evidence_is_bounded():
    results = [("terminal_exec", "x" * 5000) for _ in range(20)]
    ev = _turn_evidence(["terminal_exec"] * 20, "a", results)
    assert len(ev) < 2000


def test_evidence_keeps_the_tail_not_the_head():
    """A turn works and then reports; its final claim is backed by the last
    calls, not the first."""
    results = [("t", f"result-{i}") for i in range(10)]
    ev = _turn_evidence(["t"] * 10, "a", results)
    assert "result-9" in ev
    assert "result-0" not in ev


def test_evidence_survives_missing_results():
    assert _turn_evidence(["x"], "a", None)
    assert _turn_evidence(["x"], "a", [("x", "")])


def test_the_gate_threads_results_from_the_trace():
    """Guard the wiring: the evidence upgrade is worthless if the caller
    never passes them."""
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua._decide_self_correction)
    assert "_turn_tool_results(trace)" in src


def test_turn_tool_results_reads_a_trace_list():
    from backend.unified_agent import _turn_tool_results

    class _TC:
        def __init__(self, n, r):
            self.name, self.result = n, r

    class _S:
        def __init__(self, tc):
            self.tool_call = tc

    got = _turn_tool_results([_S(_TC("agent_browser", "ok"))])
    assert got == [("agent_browser", "ok")]
    assert _turn_tool_results(None) == []
