"""'Every day' had no tool behind it.

Measured, 2026-08-20. The owner asked the agent to follow a public
Telegram crypto channel and send him a daily market summary. The agent
fetched the channel successfully (`fetch_url https://t.me/s/COIN22T`
returned 2611 characters, so the capability was there), saved a note that
the user wants a daily digest — and then asked what time to send it.
Twice. Nothing was scheduled: `list_pending()` returned 0.

It was not stalling. `schedule_message` took a single `due_at` and had no
notion of recurrence, so "every day" was not expressible. Nothing in the
tool said so either, which is why the agent asked about the hour as
though the rest were already handled.

Recurrence is a closed set of three intervals rather than cron: the
caller is a language model, and every extra degree of freedom is another
way to schedule something nobody asked for.
"""
from datetime import datetime, timedelta, timezone

import pytest

from backend import scheduled_messages as sm


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── what counts as a repeat ─────────────────────────────────────────

def test_the_three_intervals_are_accepted():
    for word in ("daily", "weekly", "monthly"):
        assert sm.normalize_repeat(word) == word


def test_case_and_padding_do_not_matter():
    assert sm.normalize_repeat("  DAILY ") == "daily"


def test_an_unknown_interval_degrades_to_one_shot():
    """Safe direction: a message that fires once when it should have
    repeated is a disappointment the user reports. One that repeats when
    it should not is a bot that will not stop messaging them."""
    for word in ("hourly", "every 5 minutes", "cron", "yes", None, ""):
        assert sm.normalize_repeat(word) == ""


# ── when the next one lands ─────────────────────────────────────────

def test_a_one_shot_has_no_next_occurrence():
    assert sm.next_due(_iso(datetime.now(timezone.utc)), "") == ""


def test_the_next_daily_keeps_the_original_hour():
    """Counted from the DUE time, not from now — otherwise a tick that
    ran late would walk the 09:00 digest later every day."""
    base = datetime.now(timezone.utc).replace(
        hour=9, minute=0, second=0, microsecond=0) - timedelta(days=1)
    nxt = sm.next_due(_iso(base), "daily")
    assert nxt.endswith("T09:00:00Z")


def test_a_long_outage_does_not_queue_a_backlog():
    """If the box was off for a week the user gets the next digest, not
    seven of them."""
    base = datetime.now(timezone.utc) - timedelta(days=7)
    nxt = datetime.strptime(sm.next_due(_iso(base), "daily"),
                            "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert nxt > datetime.now(timezone.utc)
    assert nxt <= datetime.now(timezone.utc) + timedelta(days=1)


def test_weekly_and_monthly_step_by_their_interval():
    base = datetime.now(timezone.utc) - timedelta(days=1)
    wk = datetime.strptime(sm.next_due(_iso(base), "weekly"),
                           "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert timedelta(days=5) < wk - base < timedelta(days=8)


def test_a_malformed_due_at_does_not_raise():
    assert sm.next_due("not a timestamp", "daily") == ""


# ── re-arming after delivery ────────────────────────────────────────

def test_a_delivered_repeat_queues_the_next_one():
    row = {"id": "x", "target_speaker": "telegram:1", "text": "digest",
           "due_at": _iso(datetime.now(timezone.utc) - timedelta(days=1)),
           "requested_by": "telegram:1", "kind": "message",
           "repeat": "daily", "meta": {}}
    summary = {"sent": ["x"], "failed": []}
    sm._rearm(row, summary)
    assert summary.get("rearmed"), "the series stopped after one delivery"


def test_a_one_shot_is_not_re_armed():
    row = {"id": "x", "target_speaker": "telegram:1", "text": "once",
           "due_at": _iso(datetime.now(timezone.utc)), "requested_by": "t:1",
           "kind": "message", "repeat": "", "meta": {}}
    summary = {"sent": ["x"], "failed": []}
    sm._rearm(row, summary)
    assert "rearmed" not in summary


def test_re_arming_happens_only_after_a_successful_delivery():
    """A row that re-armed up front would keep firing while its
    deliveries fail, and the user would get nothing while the ledger
    fills with attempts."""
    import inspect
    lines = inspect.getsource(sm.deliver_due).splitlines()
    rearm = next(i for i, l in enumerate(lines) if "_rearm(row, summary)" in l)
    # The LAST success-append before _rearm; an earlier one belongs to the
    # check_in branch and is a different block entirely.
    sent = max(i for i, l in enumerate(lines)
               if 'summary["sent"].append(row["id"])' in l and i < rearm)
    # Same block as the success append, and before the else that handles
    # failure — indentation is the structure here, not character offsets.
    indent = lambda l: len(l) - len(l.lstrip())
    assert indent(lines[rearm]) == indent(lines[sent])
    following_else = next(i for i, l in enumerate(lines)
                          if l.strip() == "else:" and i > rearm)
    assert rearm < following_else


def test_re_arming_cannot_break_the_sweep():
    import inspect
    assert "except Exception" in inspect.getsource(sm._rearm)


# ── the tool the model reads ────────────────────────────────────────

def _tool():
    from backend import builtin_tools as bt
    from backend.tool_registry import get_registry
    bt.register_builtin_tools()
    return get_registry().tools["schedule_message"]


def test_the_tool_offers_repeat():
    props = _tool().input_schema["properties"]
    assert "repeat" in props
    assert set(props["repeat"]["enum"]) == {"", "daily", "weekly", "monthly"}


def test_the_description_claims_standing_requests():
    d = _tool().description
    assert "STANDING REQUESTS ARE THIS TOOL" in d
    assert "every day" in d


def test_the_description_forbids_asking_instead_of_setting_up():
    """The measured behaviour: asked what hour, scheduled nothing."""
    d = _tool().description.lower()
    assert "do not answer a recurring request by asking what time" in d
    assert "pick a sensible hour" in d


def test_the_reply_says_whether_it_actually_recurs():
    """A model that asked for daily and silently got a one-shot would tell
    the user their standing digest is running when it fires once."""
    import inspect
    from backend import builtin_tools as bt
    src = inspect.getsource(bt._schedule_message_handler)
    assert '"recurring"' in src
    assert '"repeat"' in src
