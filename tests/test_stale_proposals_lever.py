"""FIRE_STALE_PROPOSALS — auto-reject self-mod proposals that have
been pending for too long without human review.

Audit 2026-05-27 prod state: 10 proposals all `status="pending"`,
ages 10-13 days. The autonomic system kept generating them; the
review side was idle. Stale pending entries pollute the registry
and obscure fresh suggestions.

This lever auto-rejects proposals older than STALE_DAYS with a
clear `review_note` so the WebUI can render "stale, not seen".
"""
from __future__ import annotations

from datetime import datetime, timedelta


def _make_proposal(*, age_days: int, status: str = "pending", **overrides):
    from backend.self_modifier import Proposal
    created = (datetime.now() - timedelta(days=age_days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return Proposal(
        id=overrides.pop("id", f"p_{age_days}d"),
        module=overrides.pop("module", "backend/verifier.py"),
        title=overrides.pop("title", "stub"),
        description=overrides.pop("description", "stub"),
        status=status,
        created=created,
        **overrides,
    )


def test_lever_registered_in_autonomic_defaults():
    from backend.autonomic.levers import (
        register_default_autonomic_levers, clear_registry, list_levers,
    )
    clear_registry()
    try:
        register_default_autonomic_levers()
        assert "FIRE_STALE_PROPOSALS" in list_levers()
    finally:
        clear_registry()


def test_lever_skip_when_no_pending(monkeypatch):
    """No pending proposals at all → nothing to archive."""
    from backend.autonomic.levers.stale_proposals import (
        FIRE_STALE_PROPOSALS,
    )
    from backend.autonomic.types import LeverStatus
    from backend import self_modifier as sm

    monkeypatch.setattr(sm.SELF_MODIFIER, "_proposals", [])
    lever = FIRE_STALE_PROPOSALS()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "no_pending" in report.reason


def test_lever_skip_when_pending_but_fresh(monkeypatch):
    """Pending proposals exist but all are within the freshness
    window — no action."""
    from backend.autonomic.levers.stale_proposals import (
        FIRE_STALE_PROPOSALS,
    )
    from backend.autonomic.types import LeverStatus
    from backend import self_modifier as sm

    fresh = [_make_proposal(age_days=2), _make_proposal(age_days=5)]
    monkeypatch.setattr(sm.SELF_MODIFIER, "_proposals", fresh)
    saved = {"n": 0}
    monkeypatch.setattr(sm.SELF_MODIFIER, "_save",
                        lambda: saved.update({"n": saved["n"] + 1}))

    lever = FIRE_STALE_PROPOSALS()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "no_stale" in report.reason
    assert saved["n"] == 0


def test_lever_rejects_proposals_older_than_threshold(monkeypatch):
    """Proposals older than STALE_DAYS get auto-rejected with a
    clear review_note. Younger ones stay pending."""
    from backend.autonomic.levers.stale_proposals import (
        FIRE_STALE_PROPOSALS, STALE_DAYS,
    )
    from backend.autonomic.types import LeverStatus
    from backend import self_modifier as sm

    stale = _make_proposal(age_days=STALE_DAYS + 1, id="stale_1")
    fresh = _make_proposal(age_days=2, id="fresh_1")
    proposals = [stale, fresh]
    monkeypatch.setattr(sm.SELF_MODIFIER, "_proposals", proposals)
    saved = {"n": 0}
    monkeypatch.setattr(sm.SELF_MODIFIER, "_save",
                        lambda: saved.update({"n": saved["n"] + 1}))

    lever = FIRE_STALE_PROPOSALS()
    report = lever.run({}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["rejected"] == 1
    assert report.outcome["kept"] == 1
    assert stale.status == "rejected"
    assert fresh.status == "pending"
    assert "stale" in stale.review_note.lower()
    assert saved["n"] >= 1


def test_lever_only_touches_pending(monkeypatch):
    """Already-approved / already-rejected / applied proposals are
    untouched regardless of age — the lever is for clearing pending
    cruft, not rewriting decisions."""
    from backend.autonomic.levers.stale_proposals import (
        FIRE_STALE_PROPOSALS, STALE_DAYS,
    )
    from backend import self_modifier as sm

    old_approved = _make_proposal(
        age_days=STALE_DAYS + 10, id="ok", status="approved"
    )
    old_rejected = _make_proposal(
        age_days=STALE_DAYS + 10, id="rej", status="rejected"
    )
    old_applied = _make_proposal(
        age_days=STALE_DAYS + 10, id="app", status="applied"
    )
    monkeypatch.setattr(
        sm.SELF_MODIFIER, "_proposals",
        [old_approved, old_rejected, old_applied],
    )
    monkeypatch.setattr(sm.SELF_MODIFIER, "_save", lambda: None)

    lever = FIRE_STALE_PROPOSALS()
    report = lever.run({}, {})
    assert old_approved.status == "approved"
    assert old_rejected.status == "rejected"
    assert old_applied.status == "applied"


def test_lever_custom_threshold_via_params(monkeypatch):
    """`params["max_age_days"]` overrides the default threshold —
    operator can run a one-shot purge with a different cutoff."""
    from backend.autonomic.levers.stale_proposals import (
        FIRE_STALE_PROPOSALS,
    )
    from backend import self_modifier as sm

    p3d = _make_proposal(age_days=3, id="p3")
    p7d = _make_proposal(age_days=7, id="p7")
    monkeypatch.setattr(sm.SELF_MODIFIER, "_proposals", [p3d, p7d])
    monkeypatch.setattr(sm.SELF_MODIFIER, "_save", lambda: None)

    lever = FIRE_STALE_PROPOSALS()
    report = lever.run({"max_age_days": 5}, {})
    assert p3d.status == "pending"     # 3 days < 5 → kept
    assert p7d.status == "rejected"    # 7 days > 5 → archived


def test_lever_handles_unparseable_created_date(monkeypatch):
    """If a proposal's `created` field is malformed, fall back to
    treating it as NOT stale (conservative). Never raise."""
    from backend.autonomic.levers.stale_proposals import (
        FIRE_STALE_PROPOSALS,
    )
    from backend.autonomic.types import LeverStatus
    from backend import self_modifier as sm
    from backend.self_modifier import Proposal

    weird = Proposal(
        id="weird", module="x", title="x", description="x",
        status="pending", created="not-a-date",
    )
    monkeypatch.setattr(sm.SELF_MODIFIER, "_proposals", [weird])
    monkeypatch.setattr(sm.SELF_MODIFIER, "_save", lambda: None)

    lever = FIRE_STALE_PROPOSALS()
    report = lever.run({}, {})
    # Either skipped (treated as fresh) or success-with-zero-rejected.
    assert report.status in (LeverStatus.SKIPPED, LeverStatus.SUCCESS)
    assert weird.status == "pending"
