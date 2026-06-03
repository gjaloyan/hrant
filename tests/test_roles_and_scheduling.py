"""Tests for Phase 11 — roles, relationships, scheduled messages.

The contract being pinned:
  - role_of() returns 'owner' for webui:default ALWAYS (back-stop
    against locking yourself out via a bad edit of roles.json).
  - is_owner / is_at_least respect role rank.
  - set_role mutates correctly + persists.
  - permissions_block is non-empty and differentiates per role.
  - run_python (code_executor) refuses non-owner via direct param.
  - SelfModifier.apply refuses non-owner via the ContextVar.
  - schedule_message tool: guest refused, trusted can only schedule
    to owner, owner can schedule to anyone, alias resolution works.
  - Telegram chat_id capture is idempotent.
  - FIRE_SCHEDULED_MESSAGES dispatcher marks delivered rows.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    """Redirect CONFIG.knowledge['base_dir'] to a tmp_path so each
    test gets a fresh roles.json / relationships.json / scheduled
    ledger without touching the user's real data."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    return tmp_path


# --- roles -----------------------------------------------------------


def test_webui_default_is_always_owner(isolated_kb):
    from backend.roles import is_owner, role_of
    assert role_of("webui:default") == "owner"
    assert is_owner("webui:default") is True


def test_role_of_unknown_speaker_defaults_to_guest(isolated_kb):
    from backend.roles import role_of
    assert role_of("telegram:9999") == "guest"


def test_set_role_persists_to_disk(isolated_kb):
    from backend.roles import role_of, set_role
    set_role("telegram:222", "trusted", label="Wife")
    assert role_of("telegram:222") == "trusted"
    # Round-trip through a fresh _load().
    from backend.roles import list_roles
    state = list_roles()
    assert state["speakers"]["telegram:222"]["role"] == "trusted"
    assert state["speakers"]["telegram:222"]["label"] == "Wife"


def test_set_owner_adds_to_owner_list(isolated_kb):
    from backend.roles import is_owner, list_roles, set_role
    set_role("telegram:111", "owner", label="Gor TG")
    assert is_owner("telegram:111")
    assert "telegram:111" in list_roles()["owner_speaker_ids"]


def test_demoting_webui_default_is_blocked(isolated_kb):
    """The local user can't accidentally lock themselves out of
    their own box even by issuing a 'guest' demotion against
    webui:default."""
    from backend.roles import is_owner, set_role
    set_role("webui:default", "guest")
    assert is_owner("webui:default")  # still owner!


def test_is_at_least_role_rank(isolated_kb):
    from backend.roles import is_at_least, set_role
    set_role("telegram:222", "trusted")
    set_role("telegram:333", "guest")
    set_role("telegram:111", "owner")
    # owner >= anything
    assert is_at_least("telegram:111", "guest")
    assert is_at_least("telegram:111", "owner")
    # trusted >= guest, trusted < owner
    assert is_at_least("telegram:222", "guest")
    assert is_at_least("telegram:222", "trusted")
    assert not is_at_least("telegram:222", "owner")
    # guest only >= guest
    assert is_at_least("telegram:333", "guest")
    assert not is_at_least("telegram:333", "trusted")


def test_permissions_block_differs_per_role(isolated_kb):
    from backend.roles import permissions_block, set_role
    set_role("telegram:222", "trusted")
    set_role("telegram:333", "guest")
    owner_block = permissions_block("webui:default")
    trusted_block = permissions_block("telegram:222")
    guest_block = permissions_block("telegram:333")
    assert "OWNER" in owner_block
    assert "TRUSTED" in trusted_block
    assert "GUEST" in guest_block
    # Trusted explicitly allows cross-speaker-to-owner; guest doesn't.
    assert "owner" in trusted_block.lower()
    assert "refuse" in guest_block.lower()


# --- code_executor gate ---------------------------------------------


def test_run_python_refuses_non_owner(isolated_kb):
    from backend.tools.code_executor import run_python
    res = run_python("print('hi')", speaker_id="telegram:999")
    assert res.returncode == -2
    assert "refused" in res.stderr.lower()


def test_run_python_owner_passes_through(isolated_kb):
    """Owner gate should NOT prevent execution. We mock the actual
    subprocess so the test doesn't spawn python."""
    from backend.tools import code_executor as ce
    # subprocess.run patched to return a fake CompletedProcess.
    fake = MagicMock(returncode=0, stdout="ok\n", stderr="")
    with patch("subprocess.run", return_value=fake):
        res = ce.run_python("print('hi')", speaker_id="webui:default")
    assert res.returncode == 0


# --- self_modifier gate ---------------------------------------------


def test_self_modifier_apply_refuses_non_owner_via_context(isolated_kb):
    """The ContextVar gate is the LAST line of defence — even if the
    LLM is talked into requesting a self-mod, the apply path refuses
    when the current speaker isn't owner."""
    from backend import roles as _roles
    from backend.self_modifier import SelfModifier

    sm = SelfModifier()
    # Stuff a proposal as if approved.
    from backend.self_modifier import Proposal
    p = Proposal(
        id="x1",
        module="backend/foo.py",
        title="t",
        description="d",
        old_code="a\n",
        new_code="b\n",
        impact="performance",
        risk="low",
        reasoning="r",
        status="approved",
    )
    sm._proposals = [p]
    token = _roles.set_current_speaker("telegram:999")  # non-owner
    try:
        result = sm.apply("x1")
    finally:
        _roles.reset_current_speaker(token)
    assert result["ok"] is False
    assert "owner" in (result.get("message") or "").lower()


# --- contacts -------------------------------------------------------


def test_remember_telegram_user_idempotent(isolated_kb):
    from backend.contacts import load_chat_ids, remember_telegram_user
    remember_telegram_user(111, 99999, username="gor", label="Gor")
    remember_telegram_user(111, 99999, username="gor")  # second call
    state = load_chat_ids()
    assert state["111"]["chat_id"] == 99999
    # Label preserved (don't overwrite a user-set label silently).
    assert state["111"]["label"] == "Gor"


def test_chat_id_for_speaker_returns_int(isolated_kb):
    from backend.contacts import chat_id_for_speaker, remember_telegram_user
    remember_telegram_user(222, 55555)
    assert chat_id_for_speaker("telegram:222") == 55555
    assert chat_id_for_speaker("telegram:doesnotexist") is None
    # Non-Telegram speakers return None — no other channels delivered yet.
    assert chat_id_for_speaker("webui:default") is None


def test_resolve_alias_to_speaker(isolated_kb):
    from backend.contacts import resolve, save_relationships
    save_relationships({"wife": "telegram:222"})
    assert resolve("wife") == "telegram:222"
    # Already-qualified speaker_id passes through.
    assert resolve("telegram:333") == "telegram:333"
    # Unknown alias → None.
    assert resolve("nobody") is None


# --- scheduled messages -------------------------------------------


def test_schedule_creates_pending_row(isolated_kb):
    from backend.scheduled_messages import list_pending, schedule
    row = schedule(
        target_speaker="telegram:222",
        text="call me",
        due_at="2026-05-14T07:00:00Z",
        requested_by="webui:default",
    )
    assert row["status"] == "pending"
    assert row["target_speaker"] == "telegram:222"
    assert list_pending() == [row]


def test_due_now_returns_only_past_due_rows(isolated_kb):
    from backend.scheduled_messages import due_now, schedule
    schedule(
        target_speaker="telegram:222", text="past",
        due_at="2000-01-01T00:00:00Z", requested_by="webui:default",
    )
    schedule(
        target_speaker="telegram:222", text="future",
        due_at="2999-01-01T00:00:00Z", requested_by="webui:default",
    )
    due = due_now()
    assert len(due) == 1
    assert due[0]["text"] == "past"


def test_cancel_pending_message(isolated_kb):
    from backend.scheduled_messages import cancel, list_pending, schedule
    row = schedule(
        target_speaker="telegram:222", text="x",
        due_at="2999-01-01T00:00:00Z", requested_by="webui:default",
    )
    assert cancel(row["id"]) is True
    assert list_pending() == []


def test_deliver_marks_failed_when_no_chat_id(isolated_kb):
    """If we've never received a message from the target Telegram
    user, we don't have their chat_id and delivery must fail
    gracefully (mark as failed, don't crash the dispatcher)."""
    from backend.scheduled_messages import deliver, list_all, schedule
    row = schedule(
        target_speaker="telegram:777", text="hi",
        due_at="2000-01-01T00:00:00Z", requested_by="webui:default",
    )
    ok, err = deliver(row)
    assert ok is False
    assert "chat_id" in err
    rows = list_all()
    assert rows[0]["status"] == "failed"


# --- schedule_message tool gate ------------------------------------


def test_schedule_message_handler_guest_refused(isolated_kb):
    from backend import roles as _roles
    from backend.builtin_tools import _schedule_message_handler

    _roles.set_role("telegram:guest1", "guest")
    token = _roles.set_current_speaker("telegram:guest1")
    try:
        out = _schedule_message_handler(
            target="telegram:111", text="hi",
            due_at="2999-01-01T00:00:00Z",
        )
    finally:
        _roles.reset_current_speaker(token)
    body = json.loads(out)
    assert body["ok"] is False
    assert "refused" in body["error"].lower()


def test_schedule_message_trusted_can_only_target_owner(isolated_kb):
    from backend import roles as _roles
    from backend.builtin_tools import _schedule_message_handler

    _roles.set_role("telegram:wife", "trusted")
    _roles.set_role("telegram:other", "guest")
    token = _roles.set_current_speaker("telegram:wife")
    try:
        # Wife → guest: refused.
        out_bad = _schedule_message_handler(
            target="telegram:other", text="x",
            due_at="2999-01-01T00:00:00Z",
        )
        # Wife → owner (webui:default): allowed.
        out_ok = _schedule_message_handler(
            target="webui:default", text="x",
            due_at="2999-01-01T00:00:00Z",
        )
    finally:
        _roles.reset_current_speaker(token)
    bad = json.loads(out_bad)
    ok = json.loads(out_ok)
    assert bad["ok"] is False
    assert "owner" in bad["error"].lower()
    assert ok["ok"] is True


def test_schedule_message_owner_can_schedule_anywhere(isolated_kb):
    from backend import contacts, roles as _roles
    from backend.builtin_tools import _schedule_message_handler

    contacts.save_relationships({"wife": "telegram:222"})
    token = _roles.set_current_speaker("webui:default")
    try:
        out = _schedule_message_handler(
            target="wife",  # resolved via alias
            text="call gor",
            due_at="2999-01-01T00:00:00Z",
        )
    finally:
        _roles.reset_current_speaker(token)
    body = json.loads(out)
    assert body["ok"] is True
    assert body["target_speaker"] == "telegram:222"


def test_bench_harness_speaker_is_implicit_owner():
    """The Harbor terminal-bench endpoint always runs as
    `webui:bench-harness`. That speaker MUST resolve to owner so the
    in-trial agent can call execute-class tools (terminal_exec,
    start_background_job, set_setting, ...). The endpoint is
    loopback-only, so the trust boundary is the socket, not the
    file-based role list."""
    from backend.roles import role_of, is_owner
    assert role_of("webui:bench-harness") == "owner"
    assert is_owner("webui:bench-harness") is True


def test_bench_harness_owner_status_survives_missing_roles_file(tmp_path, monkeypatch):
    """Even with no roles.json on disk (fresh install, corrupt file),
    bench-harness must still resolve to owner — the implicit set is
    a process-level constant, not a file-loaded list."""
    from backend import roles
    # Point the loader at a non-existent dir so _load() returns the
    # defaults — bench-harness must still be owner.
    monkeypatch.setattr(
        "backend.roles._roles_path",
        lambda: tmp_path / "definitely-not-here" / "roles.json",
    )
    assert roles.role_of("webui:bench-harness") == "owner"
