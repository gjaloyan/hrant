"""The backlog must speak up again, and say what is about to be lost.

A proposal is announced once, when it is created. If that message scrolls
past, nothing raises it again until FIRE_STALE_PROPOSALS auto-rejects it 14
days later. Prod on 2026-09-01: 25 pending, 30 rejected, 2 applied — the
approve-and-apply path works, it just was not being reached.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from backend import proposal_digest as pd

YEREVAN = ZoneInfo("Asia/Yerevan")
NOW = datetime(2026, 9, 1, 14, 0, 0)          # afternoon, well outside quiet hours


class P:
    def __init__(self, pid, days_old, status="pending", title="", module=""):
        self.id = pid
        self.status = status
        self.title = title or ("change " + pid)
        self.description = self.title
        self.module = module
        self.created = (NOW - timedelta(days=days_old)).strftime(
            "%Y-%m-%d %H:%M:%S")


def test_the_soonest_to_expire_comes_first():
    # Ordering by deadline is the point: the top of the list is what you
    # lose first. Ordering by age would bury a fresh urgent one.
    got = pd.pending_for_digest([P("a", 1), P("c", 13), P("b", 7)], NOW)
    assert [p.id for p in got] == ["c", "b", "a"]


def test_only_pending_ones_are_raised():
    got = pd.pending_for_digest(
        [P("a", 2), P("b", 2, status="applied"), P("c", 2, status="rejected")],
        NOW)
    assert [p.id for p in got] == ["a"]


def test_an_undated_proposal_is_not_treated_as_urgent():
    p = P("x", 1)
    p.created = "not a date"
    got = pd.pending_for_digest([p, P("y", 13)], NOW)
    assert [q.id for q in got] == ["y", "x"], "no deadline is not a deadline"


def test_the_deadline_is_in_the_message():
    # "3 waiting" is noise; "one expires tomorrow" is a reason to look.
    text = pd.render(pd.pending_for_digest([P("a", 13.2)], NOW), NOW)
    assert "expires tomorrow" in text or "expires today" in text


def test_the_message_says_what_approving_does():
    text = pd.render(pd.pending_for_digest([P("a", 1)], NOW), NOW)
    assert "test" in text.lower(), "the owner should know tests run on apply"


def test_the_overflow_is_counted_not_dumped():
    ps = pd.pending_for_digest([P(str(i), i + 1) for i in range(9)], NOW)
    text = pd.render(ps, NOW)
    assert "and 6 more" in text
    assert text.count("•") == pd.MAX_WITH_BUTTONS


def test_it_does_not_repeat_within_the_interval():
    recent = (NOW - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    assert pd.due_for_digest(recent, NOW) is False
    old = (NOW - timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")
    assert pd.due_for_digest(old, NOW) is True


def test_the_first_one_always_goes():
    assert pd.due_for_digest(None, NOW) is True
    assert pd.due_for_digest("", NOW) is True


def test_never_at_night():
    # 03:00 Yerevan. The agent invented this message; it does not get to
    # wake anyone with it.
    night = datetime(2026, 9, 1, 23, 0, tzinfo=timezone.utc)   # 03:00 +04
    assert pd.due_for_digest(None, night, tz=YEREVAN) is False
    day = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)     # 14:00 +04
    assert pd.due_for_digest(None, day, tz=YEREVAN) is True


def test_a_broken_timezone_does_not_silence_it():
    # Failing closed here would mean the backlog goes quiet forever on a
    # config typo — the exact failure this whole module exists to prevent.
    class Boom:
        def utcoffset(self, _):
            raise ValueError("bad zone")
    assert pd.due_for_digest(None, NOW, tz=Boom()) is True


# ── the lever ──────────────────────────────────────────────────────────

def test_the_lever_is_registered_and_has_a_rule():
    """Registered-but-unscheduled is the exact fault recorded in layer0 for
    stale_proposals: it existed for months, never fired, and 385 proposals
    piled up. A lever with no rule is a lever that does not run."""
    from backend.autonomic import layer0
    from backend.autonomic.levers import proposal_digest as lev

    assert lev.FIRE_PROPOSAL_DIGEST.name == "FIRE_PROPOSAL_DIGEST"
    rules = layer0.default_rules() if hasattr(layer0, "default_rules") else None
    if rules is None:
        pytest.skip("layer0 rule accessor not found")
    assert any(r.lever == "FIRE_PROPOSAL_DIGEST" for r in rules), (
        "the lever would never fire")


class _Bot:
    """Stands in for TelegramBot, which is what owns the digest.

    The fake CHANNELS below deliberately has NO `send_proposal_digest`: the
    method lives on TelegramBot, beside `_send_with_buttons`, and the first
    version of the lever called `CHANNELS.send_proposal_digest()` — an
    AttributeError swallowed into its own error branch. A stub that answers
    to everything would have hidden that.
    """

    calls = 0

    def __init__(self, result):
        self._running = True
        self._result = result

    def send_proposal_digest(self):
        _Bot.calls += 1
        return self._result


class _Chan:
    def __init__(self, bot):
        self._bots = {"tg": bot} if bot else {}


def _lever(tmp_path, monkeypatch, sent_result, bot=True):
    from backend.autonomic.levers import proposal_digest as lev
    import backend.channels as ch
    from backend import follow_up as fu

    # Pin the clock. These assert the lever's OWN gating — interval, stamp,
    # failure handling — and without this they inherit the wall clock, so
    # the whole file went red at midnight when quiet hours legitimately
    # suppressed the send. Quiet hours have their own tests above, with an
    # explicit time.
    # `due_for_digest` imports it from follow_up at call time.
    monkeypatch.setattr(fu, "in_quiet_hours", lambda *a, **k: False)
    _Bot.calls = 0
    monkeypatch.setattr(ch, "CHANNELS",
                        _Chan(_Bot(sent_result) if bot else None))
    monkeypatch.setattr(lev, "resolve_knowledge_path",
                        lambda p: tmp_path / "digest.json")
    return lev.FIRE_PROPOSAL_DIGEST(), _Bot


def test_the_digest_lives_on_the_bot_not_the_manager():
    """Pin the object that owns it. ChannelManager is what `CHANNELS` is;
    the send path (`_send_with_buttons`) is on TelegramBot."""
    from backend.channels import ChannelManager, TelegramBot

    assert hasattr(TelegramBot, "send_proposal_digest")
    assert hasattr(TelegramBot, "_send_with_buttons")
    assert not hasattr(ChannelManager, "send_proposal_digest")


def test_no_running_bot_is_a_skip_not_a_failure(tmp_path, monkeypatch):
    """No bot yet is ordinary — during startup, or outside the gateway. It
    must read as SKIPPED, not fill the lever log with FAILUREs."""
    from backend.autonomic.types import LeverStatus

    lever, _ = _lever(tmp_path, monkeypatch, {"sent": 1}, bot=False)
    rep = lever.run({}, {})
    assert rep.status == LeverStatus.SKIPPED, rep.reason
    assert "bot" in (rep.reason or "")
    assert not (tmp_path / "digest.json").exists()


def test_a_delivered_digest_starts_the_clock(tmp_path, monkeypatch):
    lever, chan = _lever(tmp_path, monkeypatch, {"sent": 1, "pending": 4})
    lever.run({}, {})
    lever.run({}, {})
    assert chan.calls == 1, "it announced the same backlog twice in a row"


def test_an_undelivered_digest_does_not(tmp_path, monkeypatch):
    """Stamping on a message that never arrived would mute the queue for a
    day for no reason — the failure this lever exists to prevent."""
    lever, chan = _lever(tmp_path, monkeypatch,
                         {"sent": 0, "pending": 0, "reason": "nothing_pending"})
    lever.run({}, {})
    lever.run({}, {})
    assert chan.calls == 2
    assert not (tmp_path / "digest.json").exists()


def test_a_send_failure_is_reported_not_swallowed(tmp_path, monkeypatch):
    from backend.autonomic.levers import proposal_digest as lev
    from backend import follow_up as fu
    import backend.channels as ch

    monkeypatch.setattr(fu, "in_quiet_hours", lambda *a, **k: False)

    class _Boom:
        _running = True

        def send_proposal_digest(self):
            raise RuntimeError("telegram down")

    monkeypatch.setattr(ch, "CHANNELS", _Chan(_Boom()))
    monkeypatch.setattr(lev, "resolve_knowledge_path",
                        lambda p: tmp_path / "digest.json")
    rep = lev.FIRE_PROPOSAL_DIGEST().run({}, {})
    assert "telegram down" in (rep.reason or "")
    assert not (tmp_path / "digest.json").exists()
