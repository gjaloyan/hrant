"""Follow public Telegram channels by polling, and keep what they post.

The owner asked the agent to "subscribe to the channel and receive
updates". A bot cannot: Telegram only pushes channel posts to bots that
administer the channel, and this is somebody else's. Real subscription
needs a USER account over MTProto, which means handing the agent a phone
number and login code.

What a public channel does expose is `t.me/s/<name>`, a plain HTML page
of recent posts with stable per-post ids. Polling it and keeping what is
new gives the outcome that was actually wanted — updates collected,
nothing missed between digests — with no credentials at all. The word
"subscribe" is the only part that does not survive; the behaviour does.

Two things this module is careful about:

**Posts are kept, not just summarised.** The page shows only the last
~16 posts, so a channel busier than the digest interval would silently
lose the rest. Polling on a shorter cycle and appending to a ledger is
what makes "collect them" true rather than aspirational.

**Reviewed state is a watermark, not a delete.** The digest marks how
far it got; the posts stay. A failed or truncated digest can be re-run,
and the owner can always look at what the agent actually saw.

Polling is pure fetch-and-parse and costs no LLM call. The digest that
reads the ledger is a separate, scheduled agent turn.
"""
from __future__ import annotations

import html as _html
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import CONFIG

log = logging.getLogger(__name__)

_LOCK = threading.RLock()

# One page is ~16 posts. Anything busier than that between polls would be
# lost, so the poll interval matters more than the digest interval.
PAGE_URL = "https://t.me/s/{channel}"
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
FETCH_TIMEOUT = 30

_POST_RE = re.compile(
    r'data-post="(?P<chan>[^"/]+)/(?P<id>\d+)"(?P<rest>.*?)'
    r'(?=data-post="|\Z)', re.S)
_TEXT_RE = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(?P<body>.*?)</div>', re.S)
_TIME_RE = re.compile(r'<time[^>]+datetime="(?P<ts>[^"]+)"')


def _dir() -> Path:
    p = Path(CONFIG.knowledge["base_dir"]) / "channel_watch"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _posts_path(channel: str) -> Path:
    return _dir() / f"{channel.lower()}.jsonl"


def _state_path(channel: str) -> Path:
    return _dir() / f"{channel.lower()}.state.json"


def normalize_channel(value: str) -> str:
    """Accept a @name, a t.me link, or a bare name; return the bare name."""
    v = str(value or "").strip()
    # Scheme optional: owners paste "t.me/name" as often as a full URL, and
    # the bare-host form silently resolved to the channel "t.me".
    v = re.sub(r"^(https?://)?t\.me/(s/)?", "", v, flags=re.I)
    v = v.split("?")[0].split("/")[0]
    return v.lstrip("@").strip()


def strip_markup(fragment: str) -> str:
    """Post body HTML to plain text, keeping line structure."""
    t = re.sub(r"<br\s*/?>", "\n", fragment or "")
    t = re.sub(r"</?(p|div)[^>]*>", "\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\n{3,}", "\n\n", _html.unescape(t)).strip()


def parse_posts(page_html: str, channel: str) -> list:
    """Extract [{id, channel, ts, text, link}] from a t.me/s page.

    A post with no text block is still returned, with an empty `text`:
    photo- and video-only posts are real activity, and dropping them
    would make the ledger disagree with the channel about what happened.
    The digest can say it saw media it could not read — that is honest,
    where silence is not.
    """
    out = []
    for m in _POST_RE.finditer(page_html or ""):
        if m.group("chan").lower() != channel.lower():
            continue
        rest = m.group("rest")
        body = _TEXT_RE.search(rest)
        when = _TIME_RE.search(rest)
        pid = int(m.group("id"))
        out.append({
            "id": pid,
            "channel": channel,
            "ts": when.group("ts") if when else "",
            "text": strip_markup(body.group("body")) if body else "",
            "has_media": body is None,
            "link": f"https://t.me/{channel}/{pid}",
        })
    out.sort(key=lambda p: p["id"])
    return out


def _read_posts(channel: str) -> list:
    p = _posts_path(channel)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fetch_page(channel: str, before: Optional[int] = None) -> str:
    import httpx
    url = PAGE_URL.format(channel=channel)
    params = {"before": before} if before else None
    r = httpx.get(url, params=params, timeout=FETCH_TIMEOUT,
                  headers={"User-Agent": _UA}, follow_redirects=True)
    r.raise_for_status()
    return r.text


def poll(channel: str) -> dict:
    """Fetch the channel page and append posts not already stored.

    Returns {channel, fetched, new, latest_id}. Never raises — a poll
    runs on a timer, and one unreachable fetch must not stop the series.
    """
    channel = normalize_channel(channel)
    if not channel:
        return {"channel": "", "fetched": 0, "new": 0, "error": "empty channel"}
    try:
        page = fetch_page(channel)
    except Exception as e:
        log.warning("channel_watch poll %s failed: %s", channel, e)
        return {"channel": channel, "fetched": 0, "new": 0,
                "error": f"{type(e).__name__}: {e}"}

    found = parse_posts(page, channel)
    with _LOCK:
        known = {p["id"] for p in _read_posts(channel)}
        fresh = [p for p in found if p["id"] not in known]
        if fresh:
            with _posts_path(channel).open("a", encoding="utf-8") as f:
                for p in fresh:
                    p["seen_at"] = datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ")
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("channel_watch %s: %d on page, %d new",
             channel, len(found), len(fresh))
    return {"channel": channel, "fetched": len(found), "new": len(fresh),
            "latest_id": max((p["id"] for p in found), default=0)}


def watched() -> list:
    """Channels this deployment follows, from config. Empty by default."""
    raw = (CONFIG.knowledge.get("watch_channels")
           if isinstance(CONFIG.knowledge, dict) else None) or []
    if isinstance(raw, str):
        raw = [x for x in raw.replace(",", " ").split() if x]
    seen, out = set(), []
    for c in raw:
        c = normalize_channel(c)
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def poll_all() -> list:
    return [poll(c) for c in watched()]


# ── what the digest consumes ────────────────────────────────────────

def _state(channel: str) -> dict:
    p = _state_path(channel)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def unreviewed(channel: str, limit: int = 200) -> list:
    """Posts collected since the last digest, oldest first."""
    channel = normalize_channel(channel)
    mark = int(_state(channel).get("last_reviewed_id") or 0)
    rows = [p for p in _read_posts(channel) if int(p.get("id") or 0) > mark]
    rows.sort(key=lambda p: int(p.get("id") or 0))
    return rows[:limit]


def mark_reviewed(channel: str, upto_id: int) -> dict:
    """Move the watermark. Posts are kept: a digest that failed halfway
    can be re-run, and the owner can see what the agent actually had."""
    channel = normalize_channel(channel)
    with _LOCK:
        st = _state(channel)
        st["last_reviewed_id"] = max(int(upto_id or 0),
                                     int(st.get("last_reviewed_id") or 0))
        st["reviewed_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        _state_path(channel).write_text(
            json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    return st


def digest_input(channel: str, limit: int = 200) -> dict:
    """Everything the daily review needs, in one call."""
    channel = normalize_channel(channel)
    rows = unreviewed(channel, limit=limit)
    return {
        "channel": channel,
        "count": len(rows),
        "with_media_only": sum(1 for p in rows if p.get("has_media")),
        "latest_id": max((int(p["id"]) for p in rows), default=0),
        "posts": rows,
    }
