"""Tiny in-process rate limiter for the chat surface.

No external dependency: a deque of timestamps per IP, evicted as
they age past the window. At single-process scale (one uvicorn
worker is the norm for this agent) the memory cost is bounded by
the number of distinct IPs that hit /api/chat × the rolling
window — fine for personal use.

Why this exists: even with the owner-role gate on /api/chat, an
attacker who somehow gets a `webui:default` ContextVar (a future
session-cookie scheme that leaks, etc.) can still drive arbitrary
LLM cost. A cheap per-IP cap is defence-in-depth.

Defaults are intentionally generous — a power user firing one
question per second for 60 seconds straight stays under. Bots and
brute-force probes (~hundreds/min) blow through the cap and 429.

Config via env:
    HRANT_CHAT_RATE_PER_MIN   max chat calls per IP per 60s
    HRANT_CHAT_RATE_BURST     max simultaneous tokens
"""
from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock
from typing import Deque, Optional

from fastapi import HTTPException, Request


WINDOW_SECONDS = 60.0
PER_MINUTE = int(os.environ.get("HRANT_CHAT_RATE_PER_MIN", 60))
BURST = int(os.environ.get("HRANT_CHAT_RATE_BURST", 10))


# Per-IP timestamp ring. Single-process; if you ever run multiple
# uvicorn workers, replace with a Redis-backed limiter.
_buckets: dict[str, Deque[float]] = {}
_lock = Lock()


def _client_ip(request: Request) -> str:
    """Best-effort IP. Trusts `X-Forwarded-For` since the typical
    deployment puts the agent behind a Tailscale or local proxy;
    if you bind to 0.0.0.0 without one, `request.client.host` is
    what you want."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        # First value is the originating client.
        return fwd.split(",")[0].strip()
    return (request.client.host if request.client else "unknown") or "unknown"


def check_chat_rate(request: Request) -> None:
    """Raise 429 when the caller IP exceeds the configured rate.

    Sliding-window: drop timestamps older than WINDOW_SECONDS, then
    check current count + a short-term burst cap. The burst guard
    protects against a sudden hundred-requests-in-one-second spike
    that fits the minute budget but still amplifies cost.
    """
    ip = _client_ip(request)
    now = time.monotonic()
    with _lock:
        bucket = _buckets.setdefault(ip, deque())
        # Evict anything older than the window.
        while bucket and (now - bucket[0]) > WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"chat rate-limit: {PER_MINUTE} requests/minute per IP "
                    "exceeded — wait a minute and retry"
                ),
            )
        # Burst guard: requests in the last 5 seconds.
        recent = sum(1 for ts in bucket if (now - ts) <= 5.0)
        if recent >= BURST:
            raise HTTPException(
                status_code=429,
                detail=f"chat burst-limit: {BURST} requests/5s per IP exceeded",
            )
        bucket.append(now)


def reset_buckets() -> None:
    """Used by tests. Clears all per-IP state so a fresh test
    isn't carrying counts from the previous one."""
    with _lock:
        _buckets.clear()
