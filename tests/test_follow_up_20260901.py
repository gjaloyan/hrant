"""Follow-up must back off, must stop, and must not wake anyone at 3am."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from backend import follow_up as fu

YEREVAN = ZoneInfo("Asia/Yerevan")


def _utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_the_gap_grows():
    # A thing forgotten this morning is raised again today; a thing ignored
    # all week is not raised five times a day.
    now = _utc(2026, 9, 1, 9)          # 13:00 Yerevan, well inside waking hours
    gaps = []
    for n in range(len(fu.BACKOFF_HOURS)):
        nxt = datetime.strptime(fu.next_nudge_at(n, now, YEREVAN), fu.ISO)
        gaps.append((nxt.replace(tzinfo=timezone.utc) - now).total_seconds())
    assert gaps == sorted(gaps), gaps
    assert gaps[0] < gaps[-1], "the backoff is flat -- that is nagging"


def test_it_stops():
    # A list that nags forever gets muted, and a muted list tracks nothing.
    spent = len(fu.BACKOFF_HOURS)
    assert fu.exhausted(spent)
    assert fu.next_nudge_at(spent, _utc(2026, 9, 1, 9), YEREVAN) is None
    assert fu.remaining(spent) == 0


def test_a_nudge_landing_at_night_is_moved_to_morning():
    # 21:00 Yerevan + 8h = 05:00 -- the owner is asleep.
    now = _utc(2026, 9, 1, 17)                 # 21:00 Yerevan
    when = fu.next_nudge_at(2, now, YEREVAN)   # BACKOFF_HOURS[2] == 8
    local = datetime.strptime(when, fu.ISO).replace(
        tzinfo=timezone.utc).astimezone(YEREVAN)
    assert local.hour == fu.QUIET_END, local
    assert not fu.in_quiet_hours(
        datetime.strptime(when, fu.ISO).replace(tzinfo=timezone.utc), YEREVAN)


def test_late_evening_nudge_waits_for_the_next_morning():
    # 23:30 local must become 08:00 the NEXT day, not 08:00 the same one.
    now = _utc(2026, 9, 1, 19, 30)             # 23:30 Yerevan
    when = datetime.strptime(fu.next_nudge_at(0, now, YEREVAN), fu.ISO)
    local = when.replace(tzinfo=timezone.utc).astimezone(YEREVAN)
    assert (local.hour, local.day) == (fu.QUIET_END, 2), local


def test_a_daytime_nudge_is_left_exactly_where_it_falls():
    # Quiet hours must not drag every follow-up to 08:00; that would
    # collapse the backoff into one daily batch.
    now = _utc(2026, 9, 1, 6)                  # 10:00 Yerevan
    when = datetime.strptime(fu.next_nudge_at(0, now, YEREVAN), fu.ISO)
    assert when.replace(tzinfo=timezone.utc) == now + timedelta(hours=1)


@pytest.mark.parametrize("local_hour,quiet", [
    (0, True), (3, True), (7, True), (8, False), (15, False),
    (22, False), (23, True),
])
def test_the_night_boundary(local_hour, quiet):
    when = datetime(2026, 9, 1, local_hour, tzinfo=YEREVAN).astimezone(timezone.utc)
    assert fu.in_quiet_hours(when, YEREVAN) is quiet


def test_the_first_follow_up_uses_the_first_interval(tmp_path, monkeypatch):
    """The 1h gap must actually be reachable.

    Live run 2026-09-01 armed +3h, +8h, +24h, +48h -- BACKOFF_HOURS[0] was
    dead code, because arm_follow_up incremented the counter before looking
    the interval up. The 1h nudge is the valuable one: it catches the task
    the same day.
    """
    from datetime import datetime, timezone
    from backend import tracker as tk

    monkeypatch.setattr(tk, "data_dir", lambda: tmp_path)
    (tmp_path / "knowledge" / "projects").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tk.TrackerStore, "_schedule_check_in", lambda *a, **k: None)
    monkeypatch.setattr("backend.settings.user_timezone", lambda: "Asia/Yerevan")

    store = tk.TrackerStore()
    t = store.create(title="probe", requested_by="telegram:1",
                     steps=[{"title": "x", "due_at": "2026-09-01T09:00:00Z"}])
    sid = t["steps"][0]["id"]

    # Compare against next_nudge_at called with an EXPLICIT index at the
    # same instant, rather than against raw hour gaps. Raw gaps made this
    # test depend on the time of day: run it in the evening and the 3h and
    # 8h steps land in quiet hours, get pushed to 08:00, and read as 12h.
    # This form still catches the bug it was written for — an off-by-one
    # in which interval `arm_follow_up` picks — because the expectation is
    # pinned to the index.
    for i in range(len(fu.BACKOFF_HOURS)):
        before = datetime.now(timezone.utc)
        step = store.arm_follow_up(t["id"], sid)
        assert step is not None and step["next_check_at"], (
            "ran out of follow-ups at index %d" % i)
        want = datetime.strptime(fu.next_nudge_at(i, before), fu.ISO)
        got = datetime.strptime(step["next_check_at"], fu.ISO)
        assert abs((got - want).total_seconds()) <= 5, (
            "call %d used the wrong interval: got %s, expected %s"
            % (i + 1, got, want))
