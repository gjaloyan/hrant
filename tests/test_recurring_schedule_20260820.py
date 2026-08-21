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


# ── the kind that actually does work ────────────────────────────────

def test_a_scheduled_task_runs_instead_of_being_mailed():
    """The two existing kinds could not express "do this every morning":
    `message` mails the instruction to the user verbatim, and `check_in`
    silently returns unless meta names a live tracker step. A digest
    scheduled as a check_in is marked delivered and re-armed while doing
    nothing at all, every day, in silence — caught before it shipped."""
    import inspect
    src = inspect.getsource(sm.deliver_due)
    assert 'row.get("kind") == "agent_task"' in src
    assert "run_agent_task(row)" in src


def test_a_scheduled_task_re_arms_like_any_other_row():
    import inspect
    lines = inspect.getsource(sm.deliver_due).splitlines()
    task = next(i for i, l in enumerate(lines) if "run_agent_task(row)" in l)
    rearm = next(i for i, l in enumerate(lines)
                 if "_rearm(row, summary)" in l and i > task)
    fail = next(i for i, l in enumerate(lines)
                if "mark_failed" in l and i > task)
    assert rearm < fail, "a failed run must stop the series, not continue it"


def test_a_task_row_without_a_target_or_text_raises():
    """Raising, not returning: the caller marks the row failed and the
    series stops visibly rather than repeating a no-op."""
    with pytest.raises(ValueError):
        sm.run_agent_task({"target_speaker": "", "text": "x"})
    with pytest.raises(ValueError):
        sm.run_agent_task({"target_speaker": "telegram:1", "text": "  "})


def test_the_check_in_kind_still_needs_a_tracker():
    """The silent-return is correct FOR check_in — it exists to skip a step
    that was completed before its date. The bug was using that kind for
    something else, not the behaviour itself."""
    import inspect
    from backend.tracker_checkin import run_check_in
    src = inspect.getsource(run_check_in)
    assert "tracker_id" in src and "return" in src


# ── what the owner is shown ─────────────────────────────────────────

def test_a_re_armed_row_records_where_it_came_from():
    """The Telegram preview needs to tell a fresh series from its own
    continuation; without it the owner gets the card every morning."""
    row = {"id": "orig", "target_speaker": "telegram:1", "text": "digest",
           "due_at": _iso(datetime.now(timezone.utc) - timedelta(days=1)),
           "requested_by": "telegram:1", "kind": "agent_task",
           "repeat": "daily", "meta": {}}
    summary = {"sent": ["orig"], "failed": []}
    sm._rearm(row, summary)
    new_id = summary["rearmed"][0]
    new_row = [r for r in sm.list_all() if r["id"] == new_id][0]
    assert new_row["meta"]["rearmed_from"] == "orig"


def test_the_preview_stays_quiet_for_a_re_armed_row():
    """Reported live: the card arrived every morning beside the digest it
    had just produced. The owner accepted the series once."""
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert 'get("rearmed_from")' in src
    assert "accepted the series" in src


def test_the_preview_does_not_quote_an_agent_task_body():
    """The body of an agent_task is an instruction to the agent. Quoting it
    showed the owner `channel_updates(channel="COIN22T")` and implied that
    was the text he would receive each morning."""
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert 'row.get("kind") == "agent_task"' in src
    assert "Recurring task set" in src
    assert "instruction to the agent" in src


# ── the answer has to actually reach the user ───────────────────────

def test_a_scheduled_task_delivers_its_answer(monkeypatch):
    """The bug the owner reported: "i dont recive review".

    The digest ran every morning, produced real text, and threw it away.
    `Agent.run` RETURNS an AgentAnswer — the Telegram send lives in the
    channel layer, after `run_tracked`. The first version assumed
    otherwise in a comment and was never checked against a real inbox.
    """
    sent = {}

    class _Answer:
        answer = "Here is your digest."

    class _Agent:
        def run(self, text, **kw):
            return _Answer()

    monkeypatch.setattr(sm, "Agent", _Agent, raising=False)
    import backend.agent as _ag
    monkeypatch.setattr(_ag, "Agent", _Agent)
    monkeypatch.setattr(sm, "send_to_speaker",
                        lambda t, b: (sent.update({"to": t, "body": b}), (True, ""))[1])
    sm.run_agent_task({"id": "x", "target_speaker": "telegram:1",
                       "text": "do the digest"})
    assert sent["to"] == "telegram:1"
    assert sent["body"] == "Here is your digest."


def test_a_task_that_produced_nothing_is_a_failure(monkeypatch):
    """Silently delivering an empty message would look like the bug it
    replaced. Raising stops the series where the owner can see it."""
    class _Empty:
        answer = "   "

    class _Agent:
        def run(self, text, **kw):
            return _Empty()

    import backend.agent as _ag
    monkeypatch.setattr(_ag, "Agent", _Agent)
    with pytest.raises(RuntimeError, match="no answer"):
        sm.run_agent_task({"id": "x", "target_speaker": "telegram:1",
                           "text": "t"})


def test_a_delivery_failure_is_raised_not_swallowed(monkeypatch):
    """Running and failing to deliver is indistinguishable from never
    running, from the owner's side."""
    class _Answer:
        answer = "text"

    class _Agent:
        def run(self, text, **kw):
            return _Answer()

    import backend.agent as _ag
    monkeypatch.setattr(_ag, "Agent", _Agent)
    monkeypatch.setattr(sm, "send_to_speaker",
                        lambda t, b: (False, "no chat_id"))
    with pytest.raises(RuntimeError, match="delivery failed"):
        sm.run_agent_task({"id": "x", "target_speaker": "telegram:1",
                           "text": "t"})


def test_the_transport_refuses_an_empty_body():
    assert sm.send_to_speaker("telegram:1", "")[0] is False
    assert sm.send_to_speaker("", "hi")[0] is False


def test_the_transport_reports_an_unknown_channel():
    ok, err = sm.send_to_speaker("carrierpigeon:1", "hi")
    assert ok is False and "unsupported channel" in err
