"""Levers that can never fire must be visible, not silent.

2026-08-09 dead-code audit, measured on prod against the real lever log
(20 distinct levers seen firing) and the Layer0 rule table: of 29 lever
modules on disk, EIGHT have no rule naming them and therefore cannot be
selected at all. Two are `self_heal` and `service_repair` — and there are
knowledge modules on disk describing what they do, so the agent believes it
can repair itself. If the box breaks, nothing runs.

This is the same class as the $5 budget cap that could not cap: a capability
that is registered, documented and inert. The guard does not fix the levers —
wiring a rule changes autonomous behaviour and is the owner's call — it makes
their absence impossible to miss.
"""
from __future__ import annotations

from backend.autonomic.startup import unreachable_levers


def test_a_lever_with_no_rule_is_reported():
    out = unreachable_levers()
    assert isinstance(out, list)
    # These were measured as ruleless on 2026-08-09. If one disappears from
    # this list it means somebody wired it — good — and the assertion below
    # should be updated deliberately rather than by accident.
    assert "self_heal" in out
    # service_repair was WIRED on 2026-08-10 after its five defects were
    # fixed, so it is deliberately no longer here.
    assert "service_repair" not in out
    assert "tool_install" in out


def test_a_lever_that_fires_in_production_is_not_reported():
    """FIRE_SCHEDULED_MESSAGES has fired 21026 times on prod; it must never
    appear in this list, or the warning becomes noise nobody reads."""
    out = unreachable_levers()
    for alive in ("scheduled_messages", "error_triage", "goal_drive",
                  "integrity_heartbeat"):
        assert alive not in out, f"{alive} fires in production"


def test_the_check_never_raises(monkeypatch):
    """It runs inside scheduler startup — it must not be able to stop it."""
    import backend.autonomic.startup as st
    monkeypatch.setattr(st.os, "listdir", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    assert unreachable_levers() == []
