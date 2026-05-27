"""Tests for Phase 3 — self-modification inline-keyboard approval.

Pinned behaviour:
  - self_modifier.propose() creates a Proposal and fires the
    on_proposal_created callback registry.
  - register_on_proposal_created is idempotent.
  - prop:diff:<id>   returns a followup_text containing the diff (or
                     a placeholder when no diff has been generated).
  - prop:apply:<id>  approves + applies (when there's a diff to apply).
  - prop:reject:<id> marks status=rejected.
  - Only owners may press the buttons.
  - CallbackResult.followup_text round-trips through the dispatcher.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def isolated_self_mod(tmp_path, monkeypatch):
    """Redirect proposals.json to tmp_path and reset the in-memory
    SELF_MODIFIER + callback registry between tests."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    from backend import self_modifier
    self_modifier.SELF_MODIFIER._proposals = []
    self_modifier.SELF_MODIFIER.path = tmp_path / "proposals.json"
    saved_subs = list(self_modifier._ON_PROPOSAL_CREATED)
    self_modifier._ON_PROPOSAL_CREATED.clear()
    yield self_modifier
    self_modifier._ON_PROPOSAL_CREATED.clear()
    self_modifier._ON_PROPOSAL_CREATED.extend(saved_subs)


# ─── propose() + on_proposal_created hook ────────────────────────────


def test_propose_creates_proposal_and_fires_callback(isolated_self_mod):
    sm = isolated_self_mod
    fired: list = []
    sm.register_on_proposal_created(lambda p: fired.append(p))

    proposal = sm.propose(
        description="Improve cache TTL handling",
        files=["backend/cache.py"],
        rationale="Audit found stale entries",
        requester="webui:default",
    )
    assert proposal is not None
    assert proposal.status == "pending"
    assert proposal.module == "backend/cache.py"
    assert proposal.description == "Improve cache TTL handling"
    assert proposal.reasoning == "Audit found stale entries"
    # Persisted
    assert sm.SELF_MODIFIER._proposals[-1].id == proposal.id
    # Callback fired exactly once
    assert len(fired) == 1
    assert fired[0].id == proposal.id


def test_register_on_proposal_created_is_idempotent(isolated_self_mod):
    sm = isolated_self_mod
    calls: list = []

    def cb(p):
        calls.append(p)

    sm.register_on_proposal_created(cb)
    sm.register_on_proposal_created(cb)
    sm.register_on_proposal_created(cb)
    sm.propose(description="x", rationale="y", requester="webui:default")
    assert len(calls) == 1


def test_propose_with_no_files_still_works(isolated_self_mod):
    """A pure description (no module hint) still creates a pending
    proposal — the agent or autonomic loop can fill in the diff
    later."""
    sm = isolated_self_mod
    proposal = sm.propose(description="generic improvement")
    assert proposal is not None
    assert proposal.module == ""
    assert proposal.title == "generic improvement"


def test_propose_callback_failure_does_not_break_proposer(isolated_self_mod):
    """A subscriber raising must NOT propagate to the propose() caller
    — proposal-created events are best-effort."""
    sm = isolated_self_mod

    def bad(p):
        raise RuntimeError("subscriber kaboom")

    sm.register_on_proposal_created(bad)
    proposal = sm.propose(description="test", requester="webui:default")
    assert proposal is not None
    assert proposal.status == "pending"


# ─── Callback bridge prop:diff / prop:apply / prop:reject ────────────


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    return tmp_path


def test_prop_diff_returns_followup_text(isolated_state, isolated_self_mod):
    """prop:diff:<id> for a proposal carrying an old/new diff returns
    a followup_text with a <pre> block."""
    from backend import tg_interactive, self_modifier
    from backend.roles import set_role
    set_role("telegram:111", "owner", label="Owner")

    proposal = self_modifier.propose(description="t")
    # Patch in a real diff so _render_diff_for_telegram has content.
    p = self_modifier.SELF_MODIFIER._proposals[-1]
    p.module = "backend/cache.py"
    p.old_code = "x = 1\ny = 2\n"
    p.new_code = "x = 10\ny = 20\n"
    self_modifier.SELF_MODIFIER._save()

    res = tg_interactive.dispatch_callback(
        f"prop:diff:{proposal.id}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert res.followup_text is not None
    assert "<pre>" in res.followup_text
    assert "x = 10" in res.followup_text


def test_prop_diff_without_diff_shows_description_placeholder(isolated_state, isolated_self_mod):
    """A pending proposal with no old_code/new_code yet still returns
    a placeholder so the owner sees SOMETHING in the followup."""
    from backend import tg_interactive, self_modifier
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    proposal = self_modifier.propose(description="placeholder test", rationale="why")
    res = tg_interactive.dispatch_callback(
        f"prop:diff:{proposal.id}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "no diff yet" in (res.followup_text or "")
    assert "placeholder test" in (res.followup_text or "")


def test_prop_reject_marks_proposal_rejected(isolated_state, isolated_self_mod):
    from backend import tg_interactive, self_modifier
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    proposal = self_modifier.propose(description="will be rejected")
    res = tg_interactive.dispatch_callback(
        f"prop:reject:{proposal.id}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "Rejected" in (res.edited_text or "")
    # Persisted
    p_after = self_modifier.SELF_MODIFIER.get(proposal.id) if hasattr(self_modifier.SELF_MODIFIER, "get") else next(
        x for x in self_modifier.SELF_MODIFIER._proposals if x.id == proposal.id
    )
    assert p_after.status == "rejected"


def test_prop_callbacks_refuse_non_owner_clicker(isolated_state, isolated_self_mod):
    from backend import tg_interactive, self_modifier
    from backend.roles import set_role
    set_role("telegram:222", "trusted")
    proposal = self_modifier.propose(description="blocked test")
    res = tg_interactive.dispatch_callback(
        f"prop:reject:{proposal.id}",
        ctx={"clicker_speaker_id": "telegram:222"},
    )
    assert res.ok is False
    assert "owner" in (res.toast or "").lower()
    p_after = next(x for x in self_modifier.SELF_MODIFIER._proposals if x.id == proposal.id)
    assert p_after.status == "pending"  # unchanged


def test_prop_diff_for_missing_proposal_returns_error(isolated_state, isolated_self_mod):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    res = tg_interactive.dispatch_callback(
        "prop:diff:DEADBEEF",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is False
    assert "not found" in (res.toast or "").lower()


def test_prop_apply_fails_cleanly_when_no_diff(isolated_state, isolated_self_mod):
    """A pending proposal without old_code/new_code can't be applied.

    Audit 2026-05-28 added an EARLY refusal at the Telegram callback
    layer (before approve() runs) so the toast is concrete: "proposal
    has no diff yet — ask the agent to regenerate". Previously the
    refusal happened deeper in apply() after approve() had already
    flipped the status; the new path keeps the proposal at pending
    until a real diff exists."""
    from backend import tg_interactive, self_modifier
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    proposal = self_modifier.propose(description="no diff yet")
    res = tg_interactive.dispatch_callback(
        f"prop:apply:{proposal.id}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is False
    msg = (res.toast or "") + " " + (res.edited_text or "")
    assert (
        "no diff" in msg.lower()
        or "Apply failed" in msg
        or "failed" in msg.lower()
    )
    # Proposal stays pending — the early refusal must NOT have
    # flipped status to approved (which would let a later apply
    # attempt by any caller succeed before the diff is fixed).
    p_after = next(
        x for x in self_modifier.SELF_MODIFIER._proposals
        if x.id == proposal.id
    )
    assert p_after.status == "pending"


# ─── CallbackResult.followup_text shape ──────────────────────────────


def test_callback_result_has_followup_text_field():
    from backend.tg_interactive import CallbackResult
    cr = CallbackResult(ok=True, followup_text="hello")
    assert cr.followup_text == "hello"
    # Default is None
    cr2 = CallbackResult(ok=True)
    assert cr2.followup_text is None
