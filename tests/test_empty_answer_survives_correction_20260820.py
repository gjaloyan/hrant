"""A correction round must never hand the user an empty message.

Measured on the owner's own Telegram turn, 2026-08-19: 104 tool calls,
113 LLM calls, 1,050,255 input tokens, $5.80 — and what arrived was the
bare gate footer, "NOT DONE - this turn changed state and never verified
it", with no content at all.

The empty-answer guard was already in place and it fired. That was the
problem. Its placeholder ("I produced no answer for this turn...") is by
construction a non-delivery, so a completion gate fired on it, the
correction round re-prompted, the re-prompt came back blank, and `answer`
was overwritten with "". Nothing looked again after the loop.

The guard created the condition that erased the guard.

A note on what these tests do and do not prove. The correction loop lives
inside `run_unified`, which is far too large to drive from a test without
mocking most of the turn. So `_run_correction_loop` below MIRRORS the
loop's assignment shape rather than executing it: those cases pin the
intended semantics, and they will not catch the shipped loop drifting away
from them. The two tests at the bottom read the real source and are the
ones that fail when the fix is removed — verified by deleting it. If you
change the loop, change the mirror with it, or the mirror becomes a
comfortable lie.
"""
import pytest

import backend.unified_agent as ua


class _Agent:
    """Minimal stand-in: the loop only reads progress + _trace off it."""
    def __init__(self):
        self._trace = []
        self.events = []

    def progress(self, *a, **kw):
        self.events.append(a)


@pytest.fixture
def _always_correct(monkeypatch):
    """Force one correction round, the way a real gate would."""
    monkeypatch.setattr(
        ua, "_decide_self_correction",
        lambda **kw: ("forced", "do it properly"))


def _run_correction_loop(monkeypatch, agent, answer, reanswers):
    """Drive the same loop shape the turn uses, with a scripted model.

    Kept deliberately close to the source: the defect was in how the loop
    assigns `answer`, so the test has to exercise assignment, not a
    reimplementation of the decision logic.
    """
    calls = iter(reanswers)

    class _Router:
        def call_with_tools(self, *a, **kw):
            return next(calls)

    import backend.llm as _llm
    monkeypatch.setattr(_llm, "router", lambda: _Router())
    monkeypatch.setattr(ua, "_turn_tool_names", lambda a: ["terminal_exec"])
    monkeypatch.setattr(ua, "_turn_findings", lambda *a, **kw: "")
    monkeypatch.setattr(ua, "_rewrite_xml_tool_call_dump", lambda ans, a: ans)

    last_tag = ""
    for _round in range(2):
        tag, corrective = ua._decide_self_correction(
            task="t", answer=answer, turn_tools=["terminal_exec"],
            trace=agent._trace, speaker_id="", job_id="")
        if not corrective:
            last_tag = ""
            break
        last_tag = tag
        previous = answer
        import backend.llm as _llm
        answer = _llm.router().call_with_tools()
        answer = ua._rewrite_xml_tool_call_dump(answer, agent)
        if not (answer or "").strip():
            answer = previous
            last_tag = tag
            break
    if not (answer or "").strip():
        answer = ("I produced no answer for this turn. That is a failure, "
                  "not a result: I ran tools and then returned nothing.")
    return answer, last_tag


def test_a_blank_re_answer_does_not_erase_the_previous_one(
        monkeypatch, _always_correct):
    """The measured failure, reduced: the turn holds a placeholder, the
    correction comes back empty, and the user must not get nothing."""
    answer, _ = _run_correction_loop(
        monkeypatch, _Agent(),
        answer="I produced no answer for this turn.",
        reanswers=[""])
    assert answer.strip(), "the user received an empty message"
    assert "no answer" in answer


def test_whitespace_counts_as_blank(monkeypatch, _always_correct):
    """A model that returns "\\n\\n" is not answering either."""
    answer, _ = _run_correction_loop(
        monkeypatch, _Agent(), answer="something real",
        reanswers=["   \n\t  "])
    assert answer == "something real"


def test_a_real_re_answer_still_replaces_the_draft(
        monkeypatch, _always_correct):
    """The fix must not freeze the answer — correction has to still work.

    Two rounds are scripted because the forced corrective fires on both;
    the point is that a non-blank round always wins over what preceded it.
    """
    answer, _ = _run_correction_loop(
        monkeypatch, _Agent(), answer="first draft",
        reanswers=["round one", "the corrected answer"])
    assert answer == "the corrected answer"


def test_a_blank_round_stops_the_loop(monkeypatch, _always_correct):
    """Re-prompting a model that just returned nothing spends money to get
    nothing twice. The second scripted answer must go unused."""
    answer, _ = _run_correction_loop(
        monkeypatch, _Agent(), answer="draft",
        reanswers=["", "would have been round two"])
    assert answer == "draft"


def test_the_backstop_fills_a_blank_that_reaches_the_end(monkeypatch):
    """Belt and braces: whatever else empties the answer, the user still
    gets a sentence saying so rather than a bare gate footer."""
    monkeypatch.setattr(ua, "_decide_self_correction",
                        lambda **kw: ("", ""))
    answer, _ = _run_correction_loop(
        monkeypatch, _Agent(), answer="", reanswers=[])
    assert "That is a failure, not a result" in answer


# ── the shipped source, not just this harness ───────────────────────

def test_the_loop_keeps_the_previous_answer_on_a_blank_round():
    import inspect
    src = inspect.getsource(ua)
    assert "_previous_answer" in src, (
        "the correction loop has no way to fall back to what it had")
    assert "answer = _previous_answer" in src


def test_the_backstop_runs_after_the_loop():
    """Placement is the whole point — the original guard ran BEFORE the
    loop, which is why the loop could undo it."""
    import inspect
    src = inspect.getsource(ua)
    guard = "I produced no answer for this turn."
    assert src.count(guard) >= 2, (
        "there must be a check after the correction loop, not only before it")
