"""Let the cheap lane check its own claims instead of paying for a full turn.

Making the lane hand over on any unchecked claim was correct and
expensive: measured 2026-09-04, a factual question went from ~40k tokens
(lane, no tools) to ~193k-320k (full agent with tools), 5-8x.

The lane's defect was never that it was the wrong lane. It was that it
had no tools at all. And by the time the claims are known there is
nothing left to guess: the judge has already named what needs settling,
so the search is targeted rather than composed blind.

Draft -> name the unchecked claims -> search for those -> redraft with
what came back. Two lane calls and at most two searches, with no tool
schemas in the prompt, instead of a whole full-agent turn.
"""
from unittest.mock import patch

from backend import unified_agent as ua


class _Agent:
    def progress(self, *a, **k):
        pass


def test_an_answer_with_nothing_to_check_is_served_untouched():
    with patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck([], "checked")):
        out = ua._ground_fast_answer(task="привет", answer="Привет, Гор!",
                                     agent=_Agent(), speaker_id="webui:default",
                                     snapshot="", convo="")
    assert out == "Привет, Гор!"


def test_claims_are_searched_and_the_answer_redrafted():
    searched = []

    def _search(query, max_results=5):
        searched.append(query)
        return '[{"title": "Grain moisture meter", "snippet": "влагомер зерна"}]'

    with patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["прибор называется гигрометр"], "checked")), \
         patch.object(ua, "_web_search_for_lane", _search), \
         patch.object(ua, "_try_chat_path", return_value="Это влагомер зерна.") as redraft:
        out = ua._ground_fast_answer(
            task="как называется прибор", answer="Это гигрометр.",
            agent=_Agent(), speaker_id="webui:default", snapshot="", convo="")

    assert out == "Это влагомер зерна."
    assert searched == ["прибор называется гигрометр"]
    # The evidence has to reach the redraft, or the second call is just
    # the first one again.
    passed = redraft.call_args.kwargs["snapshot"]
    assert "влагомер зерна" in passed


def test_at_most_two_claims_are_searched():
    """A cap, or one verbose answer becomes eight searches."""
    searched = []

    with patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["a", "b", "c", "d"], "checked")), \
         patch.object(ua, "_web_search_for_lane",
                      lambda q, max_results=5: searched.append(q) or "[]"), \
         patch.object(ua, "_try_chat_path", return_value="x"):
        ua._ground_fast_answer(task="q", answer="a", agent=_Agent(),
                               speaker_id="s", snapshot="", convo="")
    assert len(searched) == 2


def test_a_search_that_returns_nothing_hands_over():
    """No evidence means the lane cannot do better than it already did,
    and must not serve the unchecked draft either."""
    with patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["a claim"], "checked")), \
         patch.object(ua, "_web_search_for_lane", lambda q, max_results=5: ""), \
         patch.object(ua, "_try_chat_path", return_value="should not be used"):
        agent = _Agent()
        out = ua._ground_fast_answer(task="q", answer="draft", agent=agent,
                                     speaker_id="s", snapshot="", convo="")
    assert out is None
    assert "a claim" in getattr(agent, "_escalated_because", "")


def test_a_redraft_that_escalates_hands_over():
    """`_try_chat_path` returns None when the lane asks for tools. That
    answer is not servable."""
    with patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["a claim"], "checked")), \
         patch.object(ua, "_web_search_for_lane",
                      lambda q, max_results=5: '[{"snippet": "something"}]'), \
         patch.object(ua, "_try_chat_path", return_value=None):
        out = ua._ground_fast_answer(task="q", answer="draft", agent=_Agent(),
                                     speaker_id="s", snapshot="", convo="")
    assert out is None


def test_a_failure_anywhere_hands_over_rather_than_serving_the_draft():
    """Fail CLOSED here, unlike the judge: the draft is known to contain
    unchecked claims, so serving it because the search broke would be the
    old behaviour with extra steps."""
    def _boom(*a, **k):
        raise RuntimeError("search down")

    with patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["a claim"], "checked")), \
         patch.object(ua, "_web_search_for_lane", _boom):
        out = ua._ground_fast_answer(task="q", answer="draft", agent=_Agent(),
                                     speaker_id="s", snapshot="", convo="")
    assert out is None
