"""The critic was structurally paid to delete the answer.

`revise_and_pick` kept the revision whenever it scored higher — by any
amount. The score measures groundedness: how many claims have evidence
behind them. Deleting a claim raises it mechanically, so the
highest-scoring revision of any answer is the empty one.

Measured on the owner's turns, 2026-08-21:

  085424  content-confidence 37, 17 unverified -> revision kept at 42
  083551  content-confidence 47,  8 unverified -> revision kept at 78
  084008  content-confidence 44,  5 unverified -> rejected (blind retraction)

The middle one is a real correction and should stand. The first bought
five points by replacing an answer about the differences between two
legal codes with a list of things it would not claim — the owner asked
what changed and received an inventory of doubts, recorded as an
improvement. His summary: "i dont recive interested me information".

So a revision now earns its place: a large gain may shrink the answer,
a small one must keep the substance.
"""
import pytest

from backend.answer_critic import (
    MIN_RETENTION, SUBSTANTIAL_GAIN, blind_retraction, revision_wins,
)


def _win(old, new, old_len, new_len):
    return revision_wins(old_score=old, new_score=new,
                         old_text="x" * old_len, new_text="y" * new_len)


# ── the measured cases ──────────────────────────────────────────────

def test_the_law_turn_revision_is_now_rejected():
    """37 -> 42 while halving the answer. Five points is not worth the
    content the owner asked for."""
    won, why = _win(37, 42, 1800, 900)
    assert won is False
    assert "cutting" in why


def test_the_genuine_correction_still_wins():
    """47 -> 78 is a real fix and may shorten — that is what a correction
    looks like. Rejecting this would trade one failure for its opposite."""
    won, why = _win(47, 78, 1000, 400)
    assert won is True
    assert "substantial" in why


def test_grounding_without_shrinking_wins_on_a_small_gain():
    """The healthy shape: same substance, better supported."""
    won, why = _win(60, 66, 1000, 950)
    assert won is True
    assert "kept" in why


# ── the rule itself ─────────────────────────────────────────────────

def test_a_revision_that_scores_no_better_never_wins():
    assert _win(50, 50, 1000, 1000)[0] is False
    assert _win(50, 40, 1000, 1000)[0] is False


def test_an_empty_revision_cannot_win_on_a_small_gain():
    """The degenerate case the old rule rewarded: assert nothing, score
    high."""
    assert _win(60, 70, 2000, 5)[0] is False


def test_an_empty_revision_is_still_refused_at_the_boundary():
    """Even a large gain should not be bought with an answer that says
    nothing — but that is the blind-retraction guard's job, so here we
    only pin that this rule does not pretend to cover it."""
    won, _ = _win(40, 90, 2000, 5)
    assert won is True, (
        "a 50-point gain is allowed through here; emptiness is caught by "
        "blind_retraction, and this test exists so the division of labour "
        "is deliberate rather than assumed")


@pytest.mark.parametrize("gain", [SUBSTANTIAL_GAIN, SUBSTANTIAL_GAIN + 5])
def test_a_substantial_gain_may_shorten(gain):
    assert _win(40, 40 + gain, 2000, 300)[0] is True


@pytest.mark.parametrize("gain", [1, SUBSTANTIAL_GAIN - 1])
def test_a_small_gain_may_not_shorten_much(gain):
    assert _win(40, 40 + gain, 2000, 400)[0] is False


def test_the_retention_bar_is_where_it_says_it_is():
    """Just above and just below, so the constant is honest."""
    keep = int(2000 * MIN_RETENTION) + 10
    cut = int(2000 * MIN_RETENTION) - 10
    assert _win(40, 45, 2000, keep)[0] is True
    assert _win(40, 45, 2000, cut)[0] is False


def test_an_empty_original_does_not_divide_by_zero():
    won, _ = revision_wins(old_score=0, new_score=50, old_text="",
                           new_text="something")
    assert won is True


def test_the_reason_explains_itself_to_the_log():
    """The rejection is invisible to the user; the log line is the only
    place anyone learns why an answer was kept."""
    _, why = _win(37, 42, 1800, 900)
    assert "score rises when claims are deleted" in why


# ── the two guards stay distinct ────────────────────────────────────

def test_blind_retraction_still_covers_the_zero_check_case():
    """This rule is about the price of a gain; that one is about
    retracting without having looked. Neither replaces the other."""
    assert blind_retraction("I cannot confirm the file was written",
                            verify_calls=0) is True
    assert blind_retraction("I cannot confirm the file was written",
                            verify_calls=2) is False


def test_the_caller_uses_the_rule():
    import inspect
    from backend.answer_critic import revise_and_pick
    src = inspect.getsource(revise_and_pick)
    assert "revision_wins(" in src
    assert "_content_score(new_vr) > _content_score(vr)" not in src, (
        "the any-improvement-wins rule is what made deletion profitable")


# ── the owner decides, not the machine ──────────────────────────────

def test_rewriting_is_off_by_default():
    """His instruction: "we need to more free agent(model). i can say
    agent what good and what bed. not verifier(hard code)."

    The agent's answer now reaches him as written. Verification still
    runs and still reports — what stops is the silent replacement.
    """
    from backend.answer_critic import rewriting_enabled
    assert rewriting_enabled() is False


def test_the_gate_refuses_before_looking_at_anything_else(monkeypatch):
    """No confidence bar, no claim count, no contradiction can start a
    rewrite while the flag is off."""
    from backend.answer_critic import should_critique
    from backend.models import VerificationResult
    vr = VerificationResult(confidence=1, unverified_claims=["a", "b"],
                            contradictions=["c"])
    fire, why = should_critique(vr, answer="an answer with real content")
    assert fire is False
    assert why == "critic-rewrite-disabled"


def test_turning_it_back_on_restores_the_old_behaviour(monkeypatch):
    """The flag is a decision, not a deletion — he can reverse it, and
    `revision_wins` guards it when he does."""
    import backend.answer_critic as ac
    monkeypatch.setattr(ac, "rewriting_enabled", lambda: True)
    from backend.models import VerificationResult
    vr = VerificationResult(confidence=10, unverified_claims=["a"],
                            contradictions=[])
    fire, why = ac.should_critique(vr, answer="an answer")
    assert fire is True
    assert "critic-rewrite-disabled" not in why


def test_the_flag_is_read_live_not_captured_at_import():
    """A switch that needs a redeploy is not a decision he can make."""
    import inspect
    from backend.answer_critic import rewriting_enabled
    assert "_cfg(" in inspect.getsource(rewriting_enabled)


def test_verification_itself_is_untouched():
    """Measuring stays on: the confidence number and the unverified list
    are what let him see when the agent is on thin ice."""
    from backend.config import CONFIG
    assert CONFIG.verification["enabled"] is True
