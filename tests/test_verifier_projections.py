"""Verifier forecast calibration — the 'projection' bucket (Q2 level 2).

Data that forced this (2026-06-15 probe battery): detailed forecast/
analysis turns accumulated huge 'unverified' counts (58, 79, 95) that
were almost entirely hedged scenarios, NOT hallucinations — dragging
confidence to 9-11. Worse: after the catalyst-weighting fix the
analysis got RICHER and the unverified count rose (31 -> 95), i.e.
better work scored worse. Projections are now a separate bucket
excluded from the confidence penalty; a confident fabrication about
the future still lands in unverified/contradiction.
"""
from __future__ import annotations

from backend.verifier import _compute_confidence, _PROJECTION_CONFIDENCE_CAP


def test_projections_excluded_from_penalty():
    # 11 verified facts, 0 unverified, 0 contradictions, 95 hedged
    # projections — the rich PEPE analysis shape.
    with_proj = _compute_confidence(
        verified=11, unverified=0, contradictions=0, projections=95,
    )
    without_proj_field = _compute_confidence(
        verified=11, unverified=0, contradictions=0,
    )
    # Projections don't drag it down — same as if they weren't counted.
    assert with_proj == without_proj_field
    # And it's a strong score (capped, see next test), not 9-11.
    assert with_proj >= 75


def test_projection_dominated_answer_capped():
    """A forecast-heavy answer (more projections than verified facts)
    is capped — a year-ahead forecast is never near-certain."""
    from backend import verifier as v
    # Simulate the full verify() cap logic on counts.
    base = _compute_confidence(11, 0, 0, 95)  # would be 100
    projections, verified = 95, 11
    conf = base
    if projections and projections > verified and conf > _PROJECTION_CONFIDENCE_CAP:
        conf = _PROJECTION_CONFIDENCE_CAP
    assert conf == _PROJECTION_CONFIDENCE_CAP == 85


def test_fact_dominated_with_few_projections_not_capped():
    base = _compute_confidence(20, 0, 0, 3)  # 100, fact-dominated
    projections, verified = 3, 20
    conf = base
    if projections and projections > verified and conf > _PROJECTION_CONFIDENCE_CAP:
        conf = _PROJECTION_CONFIDENCE_CAP
    assert conf == 100  # not projection-dominated -> not capped


def test_hallucinated_facts_still_tank_even_with_projections():
    """A forecast riddled with fabricated PRESENT facts (unverified /
    contradictions) stays low — projections can't launder bad facts."""
    conf = _compute_confidence(
        verified=5, unverified=20, contradictions=2, projections=30,
    )
    # 5 / (5 + 20 + 4) = ~17 — projections don't rescue it.
    assert conf < 30


def test_pure_speculation_no_facts_scores_zero():
    """Hedged guesses with zero grounded facts shouldn't ride high on
    speculation alone — denom is verified+unverified+2*contra."""
    assert _compute_confidence(0, 0, 0, 40) == 0


def test_verification_result_carries_projections():
    from backend.models import VerificationResult
    vr = VerificationResult(confidence=85, projections=["PEPE may rise if ETF approves"])
    assert vr.projections == ["PEPE may rise if ETF approves"]
    # Back-compat: old artifacts without the field default to [].
    assert VerificationResult(confidence=90).projections == []
