"""What is still waiting for the owner, and when it runs out of time.

A self-modification proposal is announced to the owner exactly once, at the
moment it is created (`_on_self_mod_proposal`). If that message scrolls past
-- and on a busy chat it does -- nothing ever raises it again. Fourteen days
later `FIRE_STALE_PROPOSALS` auto-rejects it and the work is gone.

Measured on prod 2026-09-01: 25 proposals pending, 30 already rejected, 2
applied. The approve-and-apply path works and runs tests; it simply was not
being reached, because nobody was reminded that anything was waiting.

Same fire-once-and-forget shape as the reminders fixed earlier that day, and
the same answer: raise it again, on a schedule, and say what happens if it
keeps being ignored. The deadline is the actionable part -- "3 waiting" is
noise, "one expires tomorrow" is a reason to look.

This module is deliberately transport-free: it decides WHETHER to send and
WHAT to say. `channels.py` owns chat ids, bots and buttons.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

# Kept equal to FIRE_STALE_PROPOSALS' own window. Imported rather than
# duplicated where possible; this constant is the fallback.
DEFAULT_STALE_DAYS = 14

# How often the owner hears about the same backlog at most.
MIN_INTERVAL_HOURS = 20

# Telegram inline keyboards get unusable long before this matters; three
# gives the urgent ones a button each and keeps the message readable.
MAX_WITH_BUTTONS = 3

_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> Optional[datetime]:
    for f in _FMTS:
        try:
            return datetime.strptime((ts or "").strip(), f)
        except ValueError:
            continue
    return None


def days_left(created: str, now: datetime, stale_days: int = DEFAULT_STALE_DAYS
              ) -> Optional[float]:
    """Days until this proposal is auto-rejected. None when unparseable."""
    dt = _parse(created)
    if dt is None:
        return None
    return round((dt + timedelta(days=stale_days) - now).total_seconds() / 86400, 1)


def pending_for_digest(proposals: Iterable[Any], now: Optional[datetime] = None,
                       stale_days: int = DEFAULT_STALE_DAYS) -> list[Any]:
    """Pending proposals, soonest to expire first.

    Ordering by deadline rather than by age is what makes the message
    useful: the top of the list is what you lose first.
    """
    now = now or datetime.now()
    out = [p for p in proposals if getattr(p, "status", "") == "pending"]

    def key(p: Any) -> tuple[int, float]:
        d = days_left(getattr(p, "created", "") or "", now, stale_days)
        # Unparseable dates sort last: no deadline is not urgent, and it
        # must not be presented as if it were.
        return (1, 0.0) if d is None else (0, d)

    out.sort(key=key)
    return out


def due_for_digest(last_sent: Optional[str], now: Optional[datetime] = None,
                   min_interval_hours: int = MIN_INTERVAL_HOURS,
                   tz: Any = None) -> bool:
    """Should a digest go out right now?

    Never at night. This is a message the agent invents on its own, and the
    same rule applies as to a self-scheduled reminder: an explicit request
    from the owner is delivered when asked, an idea of the agent's is not
    allowed to arrive at 03:00.
    """
    from .follow_up import in_quiet_hours

    now = now or datetime.now()
    aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    try:
        if in_quiet_hours(aware, tz):
            return False
    except Exception:
        pass  # a bad timezone must not silence the digest entirely

    if not last_sent:
        return True
    prev = _parse(last_sent)
    if prev is None:
        return True
    return (now - prev) >= timedelta(hours=min_interval_hours)


def render(proposals: list[Any], now: Optional[datetime] = None,
           stale_days: int = DEFAULT_STALE_DAYS) -> str:
    """The digest text. HTML, matching the rest of the Telegram surface."""
    from .tg_interactive import escape_html

    now = now or datetime.now()
    n = len(proposals)
    head = (
        f"🛠 <b>{n} change{'' if n == 1 else 's'} waiting for you</b>\n"
        f"<i>Approving one applies it and runs the tests.</i>\n"
    )

    lines = []
    for p in proposals[:MAX_WITH_BUTTONS]:
        title = escape_html(
            (getattr(p, "title", "") or getattr(p, "description", "")
             or getattr(p, "id", ""))[:90])
        module = escape_html(getattr(p, "module", "") or "")
        left = days_left(getattr(p, "created", "") or "", now, stale_days)
        if left is None:
            when = "no date on it"
        elif left <= 0:
            when = "⚠️ expires today"
        elif left < 1:
            when = "⚠️ expires today"
        elif left < 2:
            when = "⚠️ expires tomorrow"
        else:
            when = f"{int(left)} days left"
        where = f" · <code>{module}</code>" if module else ""
        lines.append(f"\n• <b>{title}</b>\n  {when}{where}")

    rest = n - len(proposals[:MAX_WITH_BUTTONS])
    tail = (
        f"\n\n<i>…and {rest} more. Open Settings → Self-Modifications to "
        f"see them all.</i>" if rest > 0 else ""
    )
    note = (
        f"\n\n<i>Anything untouched for {stale_days} days is dropped "
        f"automatically.</i>"
    )
    return head + "".join(lines) + tail + note
