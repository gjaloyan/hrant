"""The two ends of the turn must judge the same evidence.

From the 2026-09-07 re-audit, which is the one that hurt: the agent
fixed `shipping.py` and passed 22 independent checks, and ran a real
sandbox command to exit 0 — and BOTH turns came back
`endpoint_met=false`, confidence clipped from 100 to 30, with a
contradiction line saying "without execute-class tool call or MEDIA:
delivery" on a turn that had executed a shell.

Cause, in three parts:

  * the final check hand-rolled its own name collector, which counted
    every trace step carrying a `tool_call` — and the trace emits
    `tool_starting` AND `tool` per call, so 9 calls arrived as 18. The
    fix for exactly that lives in `_turn_tool_names`; this site never
    got it.
  * it passed no `tool_results`, so a judge told to rule from the
    evidence was handed nothing but names. `_turn_tool_results` exists
    because of that failure and says so in its own docstring.
  * the differing name list changed the cache key, so the early
    verdict was not reused and the judge ran a second time on strictly
    worse input.
"""
from __future__ import annotations

import inspect

import pytest

import backend.unified_agent as ua
from backend import endpoint_check as ec


def test_the_final_check_uses_the_shared_collector():
    src = inspect.getsource(ua.run_unified) if hasattr(ua, "run_unified") else ""
    if not src:
        src = inspect.getsource(ua)
    assert "_trace_tool_names = _turn_tool_names(agent)" in src
    assert "_trace_tool_results = _turn_tool_results(agent._trace or [])" in src


def test_the_final_check_passes_the_results():
    src = inspect.getsource(ua)
    i = src.find("_was_met = endpoint_met(")
    assert i > 0
    call = src[i:i + 320]
    assert "tool_results=_trace_tool_results" in call, call


def test_the_cap_is_asked_the_same_question():
    src = inspect.getsource(ua)
    i = src.find("_capped = cap_confidence_for_endpoint(")
    assert i > 0
    assert "tool_results=_trace_tool_results" in src[i:i + 320]


def test_starting_events_are_not_counted_as_calls():
    """A 9-call turn must not present as 18 names."""
    class _TC:
        def __init__(self, name):
            self.name = name

    class _Step:
        def __init__(self, event, name):
            self.event = event
            self.tool_call = _TC(name)

    class _Agent:
        _trace = [
            _Step("tool_starting", "terminal_exec"),
            _Step("tool", "terminal_exec"),
            _Step("tool_starting", "sandbox_exec"),
            _Step("tool", "sandbox_exec"),
        ]

    assert ua._turn_tool_names(_Agent()) == ["terminal_exec", "sandbox_exec"]


def test_the_cache_distinguishes_calls_with_and_without_evidence():
    """Same names, different evidence, must not share a verdict — the
    first caller to arrive used to decide for the second."""
    bare = ec._cache_key("endpoint_met", "t", "a", ["terminal_exec"], None)
    rich = ec._cache_key("endpoint_met", "t", "a", ["terminal_exec"],
                         [("terminal_exec", "exit 0, tests 22/22")])
    other = ec._cache_key("endpoint_met", "t", "a", ["terminal_exec"],
                          [("terminal_exec", "exit 1, failed")])
    assert bare != rich and rich != other


def test_evidence_reaches_the_judge(monkeypatch):
    """Through `endpoint_met`, not the key helper: the results must
    arrive in the block the judge reads."""
    seen = {}

    def fake_judge(task, answer, evidence=""):
        seen["evidence"] = evidence
        return True

    monkeypatch.setattr(ec, "_llm_endpoint_met", fake_judge)
    ec.endpoint_met(
        task="fix the function", answer="done",
        tool_names=["terminal_exec"],
        tool_results=[("terminal_exec", "22 passed")],
    )
    assert "22 passed" in seen["evidence"], seen


# ── the training queue ───────────────────────────────────────────────

@pytest.mark.parametrize("status", ["failed", "not_checked", "partial"])
def test_an_unverified_turn_does_not_become_training_data(status, monkeypatch):
    """The full cycle keeps confidence at 85 when nothing checked the
    answer and carries the truth in `check_status`. The collector read
    only the number, so those turns went in stamped `verified=True` —
    the model fine-tuned on its own unexamined output."""
    import backend.finetune as ft
    from backend.models import VerificationResult

    added = []
    monkeypatch.setattr(ft, "store", lambda: type("S", (), {
        "confidence_threshold": 85,
        "add": lambda self, **kw: added.append(kw),
    })())

    vr = VerificationResult(confidence=85, endpoint_met=True,
                            notes_used=["note"], check_status=status)
    out = ft.collect_from_turn(
        task="a real question about the world",
        answer="an answer long enough to pass the length gate " * 2,
        vr=vr, tool_names=["web_search"], is_chat=False,
        supervisor_mode=False, project=None,
    )
    assert out is None, f"{status} was accepted"
    assert added == []


def test_a_verified_turn_still_gets_collected(monkeypatch):
    import backend.finetune as ft
    from backend.models import VerificationResult

    added = []
    monkeypatch.setattr(ft, "store", lambda: type("S", (), {
        "confidence_threshold": 85,
        "add": lambda self, **kw: (added.append(kw), "pair")[1],
    })())

    vr = VerificationResult(confidence=90, endpoint_met=True,
                            notes_used=["note"], check_status="verified")
    out = ft.collect_from_turn(
        task="a real question about the world",
        answer="an answer long enough to pass the length gate " * 2,
        vr=vr, tool_names=["web_search"], is_chat=False,
        supervisor_mode=False, project=None,
    )
    assert out is not None and added


def test_a_turn_from_before_the_field_is_not_punished(monkeypatch):
    """None means "recorded before this existed", not "unchecked"."""
    import backend.finetune as ft
    from backend.models import VerificationResult

    added = []
    monkeypatch.setattr(ft, "store", lambda: type("S", (), {
        "confidence_threshold": 85,
        "add": lambda self, **kw: (added.append(kw), "pair")[1],
    })())
    vr = VerificationResult(confidence=90, endpoint_met=True,
                            notes_used=["note"])
    assert vr.check_status is None
    assert ft.collect_from_turn(
        task="a real question about the world",
        answer="an answer long enough to pass the length gate " * 2,
        vr=vr, tool_names=["web_search"], is_chat=False,
        supervisor_mode=False, project=None,
    ) is not None
