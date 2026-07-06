"""Proposal hygiene (2026-07-06 audit: 385 pending, 288 for one module,
near-duplicates minted the same minute — the review gate was clogged and the
self-improvement loop dead):

- duplicate pending (module, title) must not be re-added;
- pending per module is capped;
- FIRE_STALE_PROPOSALS actually has a layer0 rule (it was registered but
  ruleless — never fired once).
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def sm(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config
    importlib.reload(config)
    import backend.self_modifier as sm_mod
    importlib.reload(sm_mod)
    return sm_mod


def test_propose_skips_duplicate_pending(sm):
    a = sm.propose(description="Refactor router chain", files=["backend/llm.py"])
    assert a is not None
    b = sm.propose(description="Refactor router chain", files=["backend/llm.py"])
    assert b is None                                  # duplicate rejected
    pend = [p for p in sm.SELF_MODIFIER._proposals if p.status == "pending"]
    assert len(pend) == 1


def test_pending_helpers_and_cap_value(sm):
    m = sm.SELF_MODIFIER
    for i in range(3):
        sm.propose(description=f"fix {i}", files=["backend/x.py"])
    assert len(m._pending_for_module("backend/x.py")) == 3
    assert m._is_dupe_pending("backend/x.py", "fix 1") is True
    assert m._is_dupe_pending("backend/x.py", "unseen") is False
    assert m.PENDING_PER_MODULE_CAP == 15


def test_stale_proposals_has_layer0_rule():
    from backend.autonomic.layer0 import default_rules
    levers = [r.lever for r in default_rules()]
    assert "FIRE_STALE_PROPOSALS" in levers
