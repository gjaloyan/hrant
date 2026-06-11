"""Nightly prune wiring (audit 2026-06-11).

Bundle B (2026-06-10) shipped prune() helpers deliberately without an
auto-fire site. The nightly consolidation is now that site — same
"sleep" window, best-effort semantics per store.
"""
from __future__ import annotations

import pytest


def test_prune_stores_calls_all_three(monkeypatch):
    from backend.consolidation import scheduler as sched

    called = {"bg": 0, "sched": 0, "ev": 0}

    class _BgStore:
        def prune(self, *a, **kw):
            called["bg"] += 1
            return 3

    import backend.tools.background_jobs as _bg_mod
    monkeypatch.setattr(_bg_mod, "STORE", _BgStore())

    import backend.scheduled_messages as _sched_mod
    def _sched_prune(*a, **kw):
        called["sched"] += 1
        return 2
    monkeypatch.setattr(_sched_mod, "prune", _sched_prune)

    import backend.evaluator as _ev_mod
    class _Ev:
        def prune(self, *a, **kw):
            called["ev"] += 1
            return 1
    monkeypatch.setattr(_ev_mod, "EVALUATOR", _Ev())

    sched._prune_stores()

    assert called == {"bg": 1, "sched": 1, "ev": 1}


def test_one_failing_prune_does_not_block_others(monkeypatch):
    from backend.consolidation import scheduler as sched

    called = {"sched": 0, "ev": 0}

    class _BgStore:
        def prune(self, *a, **kw):
            raise OSError("disk hostile")

    import backend.tools.background_jobs as _bg_mod
    monkeypatch.setattr(_bg_mod, "STORE", _BgStore())

    import backend.scheduled_messages as _sched_mod
    def _sched_prune(*a, **kw):
        called["sched"] += 1
        return 0
    monkeypatch.setattr(_sched_mod, "prune", _sched_prune)

    import backend.evaluator as _ev_mod
    class _Ev:
        def prune(self, *a, **kw):
            called["ev"] += 1
            return 0
    monkeypatch.setattr(_ev_mod, "EVALUATOR", _Ev())

    # Must not raise despite the first store failing.
    sched._prune_stores()

    assert called == {"sched": 1, "ev": 1}
