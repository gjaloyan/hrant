import pytest

from backend.autonomic.events import EventBus


def test_subscribe_and_publish():
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("tick", received.append)
    bus.publish("tick", {"n": 1})
    assert received == [{"n": 1}]


def test_multiple_subscribers():
    bus = EventBus()
    a: list[dict] = []
    b: list[dict] = []
    bus.subscribe("ev", a.append)
    bus.subscribe("ev", b.append)
    bus.publish("ev", {"k": "v"})
    assert a == [{"k": "v"}]
    assert b == [{"k": "v"}]


def test_no_subscribers_noop():
    bus = EventBus()
    bus.publish("nobody_listens", {})


def test_subscriber_exception_does_not_break_others():
    bus = EventBus()
    b_received: list[dict] = []

    def failing(event):
        raise RuntimeError("boom")

    bus.subscribe("ev", failing)
    bus.subscribe("ev", b_received.append)
    bus.publish("ev", {"ok": True})
    assert b_received == [{"ok": True}]


def test_unsubscribe():
    bus = EventBus()
    received: list[dict] = []
    token = bus.subscribe("ev", received.append)
    bus.publish("ev", {"x": 1})
    bus.unsubscribe(token)
    bus.publish("ev", {"x": 2})
    assert received == [{"x": 1}]
