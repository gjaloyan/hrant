"""The correction round must not delete the turn's own notes.

Measured 2026-08-10 on the owner's DataLex task, from the trace rather than
from reasoning. `call_with_tools` takes `(system, user, tools, execute_tool)`
and starts a fresh message list — it has no history parameter. The corrective
re-prompt passed only `task + corrective`, so the agent entered round two
knowing nothing it had just spent fifty tool calls learning.

The trace shows the consequence exactly: it had found `#search_form` and
enumerated its controls, then opened the site's home page and started again.
Twice — once per correction round. The corrective it was handed even says
"keep going with it THIS TURN", which was impossible to obey.

That is not a forgetful model. We deleted its notes and asked why it started
over.
"""
import pytest

from backend.unified_agent import (
    _FINDINGS_MAX_CALLS, _FINDINGS_TOTAL_CAP, _turn_findings,
)


class _TC:
    def __init__(self, name, args, result, is_error=False):
        self.name, self.args, self.result, self.is_error = (
            name, args, result, is_error)


class _Step:
    def __init__(self, tc):
        self.tool_call = tc


class _Agent:
    def __init__(self, steps):
        self._trace = steps


def _agent(*calls):
    return _Agent([_Step(_TC(*c)) for c in calls])


# ── the findings reach the next round ───────────────────────────────

def test_what_the_turn_learned_is_carried_forward():
    a = _agent(
        ("agent_browser", {"command": "open https://x.test"}, '{"ok":true}'),
        ("agent_browser", {"command": "snapshot"}, "e31 button Search"),
    )
    out = _turn_findings(a, previous_answer="I did not finish.")
    assert "do not start over" in out
    assert "snapshot" in out and "e31 button Search" in out
    assert "I did not finish." in out


def test_errors_are_marked_so_they_are_not_retried_blindly():
    a = _agent(("agent_browser", {"command": "click @e23"},
                "Element not found", True))
    out = _turn_findings(a)
    assert "[ERROR]" in out
    assert "Element not found" in out


def test_calls_are_shown_in_the_order_they_happened():
    a = _agent(
        ("agent_browser", {"command": "first"}, "a"),
        ("agent_browser", {"command": "second"}, "b"),
        ("agent_browser", {"command": "third"}, "c"),
    )
    out = _turn_findings(a)
    assert out.index("first") < out.index("second") < out.index("third")


def test_the_tail_survives_the_budget_not_the_head():
    """A fifty-call turn cannot fit. What matters is where it GOT to, so the
    most recent calls are the ones kept."""
    calls = [("agent_browser", {"command": f"step{i}"}, f"result{i}")
             for i in range(60)]
    out = _turn_findings(_agent(*calls))
    assert "step59" in out
    assert "step0" not in out


def test_the_digest_is_budgeted():
    calls = [("agent_browser", {"command": f"c{i}"}, "x" * 5000)
             for i in range(60)]
    out = _turn_findings(_agent(*calls))
    assert len(out) < _FINDINGS_TOTAL_CAP * 2
    assert out.count("[ok]") <= _FINDINGS_MAX_CALLS


# ── it stays quiet when there is nothing to say ─────────────────────

def test_a_turn_with_no_tools_and_no_answer_adds_nothing():
    assert _turn_findings(_Agent([])) == ""


def test_a_previous_answer_alone_is_still_worth_carrying():
    out = _turn_findings(_Agent([]), previous_answer="Here is what I found.")
    assert "Here is what I found." in out


def test_a_missing_trace_does_not_raise():
    class _NoTrace:
        pass
    assert _turn_findings(_NoTrace()) == ""


def test_dict_shaped_trace_entries_are_understood():
    """Some producers put plain dicts on the trace rather than the model."""
    class _S:
        tool_call = {"name": "fetch_url", "args": {"url": "u"},
                     "result": "body", "is_error": False}
    out = _turn_findings(_Agent([_S()]))
    assert "fetch_url" in out and "body" in out


# ── the corrective prompt actually includes it ──────────────────────

def test_run_unified_actually_uses_the_findings():
    """The correction loop lives inside a 900-line function that no unit test
    can call, so this is checked structurally — but on the USE of the value,
    not on the call that produces it.

    The first version asserted only that `_turn_findings(...)` appears in the
    source. That stayed true when the result was computed and dropped on the
    floor, so it passed against a mutation reproducing the original bug
    exactly. A test that survives its own bug is not a test."""
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua.run_unified)
    assert "_findings = _turn_findings(agent, previous_answer=answer)" in src
    assert 'f"{_findings}\\n\\n" if _findings else ""' in src, (
        "the computed findings must be interpolated into the user message")
