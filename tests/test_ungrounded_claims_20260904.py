"""Deciding to search AFTER the draft, not before.

Deciding "should I search?" before answering asks the model what it does
not know, which is the thing it is worst at. Measured 2026-09-04: the
same question three times gave escalate, escalate, answer-directly.

After the draft the question is different and answerable: "this sentence
states a checkable fact and nothing this turn checked it." The turn that
prompted this asserted, from weights, what a stuck-glass-stopper
extractor is called and how to use it, with zero tools.

`should_verify` cannot help: it returns False when there are no tool
outputs, so the turns that most need checking are exactly the ones that
skip verification. This runs where the existing self-correction loop
already decides to re-prompt, so a claim found here is re-answered with
evidence instead of being reported to the user as a confidence number.
"""
from unittest.mock import patch

from backend import unified_agent as ua


def test_a_checkable_assertion_with_no_tools_is_corrected():
    with patch("backend.endpoint_check.unbacked_action_claim", return_value=""), \
         patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["Экстрактор притёртых пробок — винтовой съёмник"], "checked")):
        tag, corrective = ua._decide_self_correction(
            task="есть ли приспособление для пробки",
            answer="Да, экстрактор притёртых пробок — винтовой съёмник.",
            turn_tools=[])
    assert tag
    assert "ungrounded" in tag
    low = corrective.lower()
    assert "web_search" in low or "search_knowledge" in low
    assert "винтовой съёмник" in corrective


def test_nothing_checkable_means_no_correction():
    with patch("backend.endpoint_check.unbacked_action_claim", return_value=""), \
         patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck([], "checked")):
        tag, corrective = ua._decide_self_correction(
            task="привет", answer="Привет, Гор!", turn_tools=[])
    assert (tag, corrective) == ("", "")


def test_an_action_claim_still_wins_the_branch():
    """(A) is the older, sharper failure -- claiming a save that never
    happened. It must not be shadowed by the new check."""
    with patch("backend.endpoint_check.unbacked_action_claim",
               return_value="I saved it"), \
         patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["something"], "checked")) as ungrounded:
        tag, corrective = ua._decide_self_correction(
            task="запомни это", answer="Запомнил.", turn_tools=[])
    assert "unbacked claim" in tag
    ungrounded.assert_not_called()


def test_the_corrective_permits_an_honest_i_do_not_know():
    """Forcing a search on every assertion would trade one confident
    wrong answer for another. Saying so plainly has to stay allowed."""
    with patch("backend.endpoint_check.unbacked_action_claim", return_value=""), \
         patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["X is Y"], "checked")):
        _, corrective = ua._decide_self_correction(
            task="q", answer="X is Y.", turn_tools=[])
    low = corrective.lower()
    assert "cannot" in low or "could not" in low or "say so" in low


def test_the_judge_fails_open(monkeypatch):
    """A judge that raises must not block the turn -- an unchecked answer
    is better than no answer."""
    def _boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(ua, "_claims_judge_call", _boom, raising=False)
    got = ua._ungrounded_factual_claims("q", "a")
    assert got.claims == []
    # ...and says it failed rather than passing for a clean bill of
    # health (2026-09-05 audit, finding 3).
    assert got.status == "failed"
