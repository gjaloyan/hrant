"""A trusted user may set a reminder for themselves.

Measured, 2026-08-31. The owner granted his brother access with the words
"дай разрешение чтобы он мог установить напоминание", and the agent
confirmed: "Тиграну предоставлено доверенное разрешение". The brother then
asked four times, in Armenian and Russian, to be reminded to call his
dentist. Every attempt returned:

    refused: trusted users may only schedule messages to the owner

The grant did the thing it was named for and not the thing it was asked
for. `trusted` allowed scheduling only TO THE OWNER, so the one message a
trusted user actually wants — a reminder for himself — was the one shape
the role forbade.

The rule the gate exists for is untouched: a trusted user must not send
messages to THIRD parties. A reminder someone sets for themselves is not
outbound traffic to anyone else.
"""
import json

import pytest

from backend import builtin_tools as bt


OWNER = "telegram:848732236"
BROTHER = "telegram:1358056500"
STRANGER = "telegram:999000111"


@pytest.fixture
def _as(monkeypatch):
    """Run the handler as a given speaker with a given role."""
    def _setup(speaker, role):
        monkeypatch.setattr(bt, "_schedule_message_handler",
                            bt._schedule_message_handler)
        import backend.roles as roles
        import backend.contacts as contacts
        import backend.scheduled_messages as sm
        monkeypatch.setattr(roles, "current_speaker", lambda: speaker)
        monkeypatch.setattr(roles, "current_role", lambda: role)
        monkeypatch.setattr(roles, "is_owner", lambda s: s == OWNER)
        monkeypatch.setattr(contacts, "resolve", lambda t: t)
        monkeypatch.setattr(
            sm, "schedule",
            lambda **kw: {"id": "row1", "target_speaker": kw["target_speaker"],
                          "due_at": kw["due_at"], "repeat": kw.get("repeat", "")})
    return _setup


def _call(target, text="call the dentist", due="2026-09-01T06:30:00Z"):
    return json.loads(bt._schedule_message_handler(
        target=target, text=text, due_at=due))


# ── the measured failure ────────────────────────────────────────────

def test_a_trusted_user_can_remind_themselves(_as):
    """The brother's dentist reminder — refused four times."""
    _as(BROTHER, "trusted")
    out = _call(BROTHER)
    assert out["ok"] is True, out.get("error")
    assert out["target_speaker"] == BROTHER


def test_a_trusted_user_can_still_message_the_owner(_as):
    """The behaviour the gate was written for must survive."""
    _as(BROTHER, "trusted")
    assert _call(OWNER)["ok"] is True


def test_a_trusted_user_still_cannot_message_a_third_party(_as):
    """The actual risk: trusted access must not become a way to send
    messages to other people."""
    _as(BROTHER, "trusted")
    out = _call(STRANGER)
    assert out["ok"] is False
    assert "not to other people" in out["error"]


def test_the_refusal_says_what_is_allowed(_as):
    """The old text named only the owner, which is why four attempts in a
    row read as a flat no rather than a fixable one."""
    _as(BROTHER, "trusted")
    err = _call(STRANGER)["error"]
    assert "owner" in err and "themselves" in err


# ── the other roles are unchanged ───────────────────────────────────

def test_the_owner_may_still_schedule_to_anyone(_as):
    _as(OWNER, "owner")
    assert _call(STRANGER)["ok"] is True
    assert _call(BROTHER)["ok"] is True


def test_a_guest_is_still_refused_outright(_as):
    _as("telegram:5", "guest")
    out = _call("telegram:5")
    assert out["ok"] is False
    assert "trusted or owner" in out["error"]


# ── shapes that must not slip through ───────────────────────────────

def test_self_targeting_is_compared_after_normalisation(_as, monkeypatch):
    """`resolve()` and `current_speaker()` need not agree on spelling; a
    raw string compare would refuse a legitimate self-reminder."""
    import backend.contacts as contacts
    _as(BROTHER, "trusted")
    monkeypatch.setattr(contacts, "resolve", lambda t: f"  {BROTHER}  ")
    assert _call("me")["ok"] is True


def test_the_gate_reads_the_resolved_target_not_the_raw_alias(_as, monkeypatch):
    """An alias that resolves to a third party must still be refused."""
    import backend.contacts as contacts
    _as(BROTHER, "trusted")
    monkeypatch.setattr(contacts, "resolve", lambda t: STRANGER)
    assert _call("myself")["ok"] is False


# ── isolation: each person manages only their own ───────────────────
#
# The owner's rule, in his words: "notifications need to work isolated
# for each user."

@pytest.fixture
def _ledger(tmp_path, monkeypatch):
    """A ledger of this test's own — the real one belongs to a person."""
    import backend.scheduled_messages as sm
    monkeypatch.setattr(sm, "_path", lambda: tmp_path / "sched.jsonl")
    import backend.roles as roles
    monkeypatch.setattr(roles, "is_owner", lambda s: s == OWNER)
    return sm


def _row(sm, requester, target):
    return sm.schedule(target_speaker=target, text="t",
                       due_at="2026-09-01T06:30:00Z", requested_by=requester)


def test_a_user_may_cancel_their_own_reminder(_ledger):
    """The brother's card came with a Cancel button that refused him."""
    sm = _ledger
    row = _row(sm, BROTHER, BROTHER)
    assert sm.cancel(row["id"], by_speaker=BROTHER) is True


def test_a_user_may_not_cancel_someone_elses(_ledger):
    sm = _ledger
    row = _row(sm, OWNER, OWNER)
    assert sm.cancel(row["id"], by_speaker=BROTHER) is False
    assert [r for r in sm.list_pending() if r["id"] == row["id"]], (
        "the row must survive a refused cancel")


def test_the_owner_may_cancel_anything(_ledger):
    sm = _ledger
    row = _row(sm, BROTHER, BROTHER)
    assert sm.cancel(row["id"], by_speaker=OWNER) is True


def test_the_recipient_may_cancel_a_reminder_addressed_to_them(_ledger):
    """Being messaged is enough to stop being messaged."""
    sm = _ledger
    row = _row(sm, OWNER, BROTHER)
    assert sm.cancel(row["id"], by_speaker=BROTHER) is True


def test_an_internal_caller_with_no_person_behind_it_still_works(_ledger):
    """The re-arm path and migrations have no speaker; requiring one would
    break them for no safety gain."""
    sm = _ledger
    row = _row(sm, BROTHER, BROTHER)
    assert sm.cancel(row["id"]) is True


def test_an_anonymous_click_cannot_cancel(_ledger):
    """An empty speaker is not 'internal' — it is an unidentified click."""
    sm = _ledger
    row = _row(sm, BROTHER, BROTHER)
    assert sm.cancel(row["id"], by_speaker="") is False


def test_the_telegram_button_passes_the_clicker(_ledger):
    import inspect
    import backend.scheduled_messages as sm
    src = inspect.getsource(sm._register_sched_callback)
    assert "by_speaker=clicker_id" in src
    assert "only the owner can manage scheduled messages" not in src, (
        "that refusal is what met a trusted user on his own reminder")


def test_listing_can_be_scoped_to_one_person(_ledger):
    """Isolation is not only about writes — one user's pending list must
    not show another's."""
    sm = _ledger
    _row(sm, OWNER, OWNER)
    mine = _row(sm, BROTHER, BROTHER)
    got = sm.list_pending(requested_by=BROTHER)
    assert [r["id"] for r in got] == [mine["id"]]
