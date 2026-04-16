"""Simple in-process event bus for autonomic coordination.

Synchronous publish: all subscribers invoked immediately in the publish call.
Exceptions in one subscriber do not prevent others from running.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], None]


@dataclass
class _Subscription:
    topic: str
    handler: Handler


@dataclass
class EventBus:
    _subs: dict[int, _Subscription] = field(default_factory=dict)
    _next_id: int = 0

    def subscribe(self, topic: str, handler: Handler) -> int:
        token = self._next_id
        self._next_id += 1
        self._subs[token] = _Subscription(topic=topic, handler=handler)
        return token

    def unsubscribe(self, token: int) -> None:
        self._subs.pop(token, None)

    def publish(self, topic: str, event: dict[str, Any]) -> None:
        for sub in list(self._subs.values()):
            if sub.topic != topic:
                continue
            try:
                sub.handler(event)
            except Exception as exc:
                log.warning("EventBus subscriber for %r raised: %s", topic, exc)
