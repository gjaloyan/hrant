"""The judge must see the call that DELIVERED, not just the last five.

Measured, 2026-08-17. A turn solved a CAPTCHA, opened the record and pulled
~12 KB of case data out of the browser. It then spent its final five calls
corroborating: a search provider answered with an anti-bot challenge, a
fetcher returned a script skeleton, and it re-read a note. `_turn_tool_results`
kept the last six and `_turn_evidence` cropped that to four, so every result
the judge saw was a failure — of a DIFFERENT channel than the one that had
worked. It ruled not-delivered, the correction round told the agent it had
produced nothing usable, and the agent retracted data it was holding.

The user's summary of the class: "сделал и отрёкся".

Two defects, either sufficient alone:
  1. the evidence window was a pure tail, and the delivering call is not
     reliably last — a turn succeeds and then tidies up;
  2. nothing told the judge that a later failure of one tool says nothing
     about what another tool already retrieved.
"""
import pytest

from backend.endpoint_check import (
    _ENDPOINT_JUDGE_SYSTEM, _EVIDENCE_MAX_RESULTS, _turn_evidence,
)
from backend.unified_agent import _turn_tool_results


class _TC:
    def __init__(self, name, result):
        self.name = name
        self.result = result


class _Step:
    def __init__(self, name, result):
        self.tool_call = _TC(name, result)


CASE_CARD = "Դատական Գործ N: ՍնԴ/0038/04/22 " + ("x" * 4000)


def _the_measured_turn():
    """Browser retrieves the record, then ELEVEN calls tidy up.

    The count is from the live trace, and it is the whole point: the record
    landed at call 147 of 158. Anything that only looks at the last handful
    cannot see it. A fixture that leaves the record inside the tail would pass
    against code that does nothing — verified by deleting the selection and
    watching this file stay green.
    """
    return (
        [_Step("agent_browser", "clicked") for _ in range(20)]
        + [_Step("agent_browser", CASE_CARD)]
        + [_Step("agent_browser", "{dialog fragment}") for _ in range(6)]
        + [_Step("waive_proof", "ok"),
           _Step("web_search", "every provider failed: anti-bot challenge"),
           _Step("fetch_url", "<html><script>app.js</script></html>"),
           _Step("fetch_url", "<html><script>app.js</script></html>"),
           _Step("read_file", "notes about the case")]
    )


# ── defect 1: the window was a tail ─────────────────────────────────

def test_the_delivering_call_survives_the_tidy_up():
    out = _turn_tool_results(_the_measured_turn())
    joined = " ".join(r for _, r in out)
    assert "ՍնԴ/0038/04/22" in joined, (
        "the retrieved record must reach the judge even when five calls "
        "followed it")


def test_the_recent_calls_are_still_there():
    """Recency did not stop mattering — the fix adds, it does not replace."""
    out = _turn_tool_results(_the_measured_turn())
    names = [n for n, _ in out]
    assert "read_file" in names and "fetch_url" in names


def test_results_stay_in_call_order():
    """Out-of-order evidence would let the judge read the tidy-up as having
    happened before the retrieval."""
    steps = [_Step("a", "1"), _Step("b", "2" * 5000), _Step("c", "3"),
             _Step("d", "4"), _Step("e", "5"), _Step("f", "6"),
             _Step("g", "7")]
    out = _turn_tool_results(steps)
    names = [n for n, _ in out]
    assert names == sorted(names, key=lambda n: "abcdefg".index(n))


def test_no_call_is_reported_twice():
    """A big result inside the tail must not also arrive as a 'richest'."""
    steps = [_Step("small", "x"), _Step("big", "y" * 5000)]
    out = _turn_tool_results(steps)
    assert len(out) == len(steps)


def test_the_evidence_block_does_not_re_crop_what_was_selected():
    """The caller decides what the judge needs; cropping it again one function
    later is how the record was assembled and then discarded."""
    results = _turn_tool_results(_the_measured_turn())
    assert len(results) <= _EVIDENCE_MAX_RESULTS, (
        "the selection must fit the block, or the fix is undone downstream")
    ev = _turn_evidence(["agent_browser"], "", results)
    assert "ՍնԴ/0038/04/22" in ev


def test_the_evidence_block_says_the_last_call_is_not_the_delivery():
    ev = _turn_evidence(["agent_browser"], "", [("agent_browser", "data")])
    low = ev.lower()
    assert "largest" in low
    assert "last call is not necessarily" in low


def test_an_empty_turn_still_renders():
    ev = _turn_evidence([], "", [])
    assert "(none)" in ev


# ── defect 2: the epistemic rule ────────────────────────────────────

def test_the_judge_is_told_a_later_failure_does_not_unmake_a_result():
    p = _ENDPOINT_JUDGE_SYSTEM
    assert "DOES NOT EXPIRE BECAUSE A LATER CALL FAILED" in p


def test_the_judge_is_told_which_tool_the_failure_describes():
    """The live turn blamed the case page for an anti-bot challenge that a
    search engine had served."""
    p = _ENDPOINT_JUDGE_SYSTEM.lower()
    assert "anti-bot" in p
    assert "script skeleton" in p


def test_withholding_an_obtained_result_is_itself_the_failure():
    p = _ENDPOINT_JUDGE_SYSTEM.lower()
    assert "retracts it" in p
    assert "holding the result" in p


def test_the_rule_did_not_cost_the_honesty_protection():
    """Telling the judge to reject a retraction must not teach the agent to
    overclaim — the older rule has to survive alongside it."""
    p = _ENDPOINT_JUDGE_SYSTEM.lower()
    assert "honesty is never itself a failure" in p
    assert "not to punish the confession" in p


@pytest.mark.parametrize("tail,rich", [(0, 3), (6, 0), (1, 1)])
def test_the_window_is_configurable_without_crashing(tail, rich):
    out = _turn_tool_results(_the_measured_turn(), limit=tail, rich=rich)
    assert isinstance(out, list)
    assert len(out) <= tail + rich
