"""When to nudge again, and when to stop.

A reminder used to fire exactly once. If the owner did not reply, the step
stayed "pending" forever and nothing ever asked again -- so the tracker
could not do the one thing a task list exists for: carry something until it
is actually finished. The owner's words: "agent can track work or todu
until over. with notifications to user."

Two things make follow-up useful rather than irritating:

  * It backs OFF. The gap grows 1h -> 3h -> 8h -> 24h -> 48h, so a thing
    forgotten this morning is raised again today, and a thing ignored all
    week is not raised five times a day.
  * It STOPS. After the last interval the step is parked as "stalled" and
    the agent asks once whether it still matters. A list that nags forever
    gets muted, and a muted list tracks nothing.

Quiet hours apply ONLY to these agent-chosen times. An explicit "wake me at
6:00" is the owner's decision and is delivered at 6:00; it is a nudge the
agent invented on its own that must not arrive at 03:00.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# Growing gaps: raised again soon at first, then left alone.
BACKOFF_HOURS: tuple[int, ...] = (1, 3, 8, 24, 48)

# Local hours during which the agent will not invent a notification.
QUIET_START = 23   # inclusive
QUIET_END = 8      # exclusive

ISO = "%Y-%m-%dT%H:%M:%SZ"


def _zone() -> ZoneInfo | timezone:
    from .settings import user_timezone
    try:
        return ZoneInfo(user_timezone())
    except Exception:
        return timezone.utc


def in_quiet_hours(when_utc: datetime, tz=None) -> bool:
    """Is this instant inside the owner's night?"""
    local = when_utc.astimezone(tz or _zone())
    return local.hour >= QUIET_START or local.hour < QUIET_END


def respect_quiet_hours(when_utc: datetime, tz=None) -> datetime:
    """Push an agent-chosen time out of the owner's night to 08:00 local."""
    tz = tz or _zone()
    local = when_utc.astimezone(tz)
    if local.hour >= QUIET_START:
        local = local + timedelta(days=1)
    elif local.hour >= QUIET_END:
        return when_utc
    local = local.replace(hour=QUIET_END, minute=0, second=0, microsecond=0)
    return local.astimezone(timezone.utc)


def exhausted(nudges: int) -> bool:
    """Has this step used up every follow-up it is allowed?"""
    return int(nudges or 0) >= len(BACKOFF_HOURS)


def next_nudge_at(nudges: int, now_utc: Optional[datetime] = None,
                  tz=None) -> Optional[str]:
    """UTC stamp for the next follow-up, or None when the step is spent.

    `nudges` is how many have already been sent, so the first call (0)
    schedules BACKOFF_HOURS[0] from now.
    """
    n = int(nudges or 0)
    if exhausted(n):
        return None
    now = now_utc or datetime.now(timezone.utc)
    raw = now + timedelta(hours=BACKOFF_HOURS[n])
    return respect_quiet_hours(raw, tz).strftime(ISO)


def remaining(nudges: int) -> int:
    """How many follow-ups are still owed. Shown to the agent so it can say
    'I will ask twice more' instead of nagging with no visible end."""
    return max(0, len(BACKOFF_HOURS) - int(nudges or 0))
