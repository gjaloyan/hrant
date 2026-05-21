"""Tests for the unified LogBus that backs the WebUI Logs tab."""
from __future__ import annotations

import time

import pytest


def test_log_event_to_dict_shape():
    from backend.log_bus import LogEvent
    ev = LogEvent(
        ts=1234.5,
        level="info",
        source="python",
        logger="backend.agent",
        message="hello",
        meta={"k": "v"},
        request_id="t-abc",
    )
    d = ev.to_dict()
    assert d["ts"] == 1234.5
    assert d["level"] == "info"
    assert d["source"] == "python"
    assert d["logger"] == "backend.agent"
    assert d["message"] == "hello"
    assert d["meta"] == {"k": "v"}
    assert d["request_id"] == "t-abc"


def _mk_event(level="info", source="python", message="x", logger_name="t"):
    from backend.log_bus import LogEvent
    return LogEvent(
        ts=time.time(),
        level=level,
        source=source,
        logger=logger_name,
        message=message,
    )


def test_publish_appends_to_ring():
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=100)
    assert bus.tail() == []
    bus.publish(_mk_event(message="m1"))
    bus.publish(_mk_event(message="m2"))
    rows = bus.tail()
    assert len(rows) == 2
    assert rows[0]["message"] == "m1"
    assert rows[1]["message"] == "m2"


def test_ring_drops_oldest_when_full():
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=3)
    for i in range(5):
        bus.publish(_mk_event(message=f"m{i}"))
    rows = bus.tail()
    assert len(rows) == 3
    assert [r["message"] for r in rows] == ["m2", "m3", "m4"]


def test_tail_filters_by_level():
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=50)
    bus.publish(_mk_event(level="debug", message="d"))
    bus.publish(_mk_event(level="info", message="i"))
    bus.publish(_mk_event(level="warning", message="w"))
    bus.publish(_mk_event(level="error", message="e"))
    levels = {r["level"] for r in bus.tail(level="warning")}
    assert levels == {"warning"}
    multi = {r["level"] for r in bus.tail(level=["warning", "error"])}
    assert multi == {"warning", "error"}


def test_tail_filters_by_source():
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=50)
    bus.publish(_mk_event(source="python", message="p"))
    bus.publish(_mk_event(source="tool", message="t"))
    bus.publish(_mk_event(source="job", message="j"))
    sources = {r["source"] for r in bus.tail(source="tool")}
    assert sources == {"tool"}


def test_tail_filters_by_substring_search():
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=50)
    bus.publish(_mk_event(message="loading file foo.py"))
    bus.publish(_mk_event(message="loading file bar.py"))
    bus.publish(_mk_event(message="finished"))
    hits = bus.tail(search="foo")
    assert len(hits) == 1
    assert "foo" in hits[0]["message"]


def test_tail_search_is_case_insensitive():
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=50)
    bus.publish(_mk_event(message="Some ERROR happened"))
    hits = bus.tail(search="error")
    assert len(hits) == 1


def test_tail_limit_returns_newest():
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=50)
    for i in range(20):
        bus.publish(_mk_event(message=f"m{i}"))
    rows = bus.tail(limit=5)
    assert len(rows) == 5
    assert [r["message"] for r in rows] == ["m15", "m16", "m17", "m18", "m19"]


def test_tail_before_ts_returns_older():
    from backend.log_bus import LogBus
    from backend.log_bus import LogEvent
    bus = LogBus(maxlen=50)
    bus.publish(LogEvent(ts=100.0, level="info", source="python",
                         logger="t", message="old"))
    bus.publish(LogEvent(ts=200.0, level="info", source="python",
                         logger="t", message="mid"))
    bus.publish(LogEvent(ts=300.0, level="info", source="python",
                         logger="t", message="new"))
    rows = bus.tail(before_ts=250.0)
    assert {r["message"] for r in rows} == {"old", "mid"}


def test_publish_rejects_invalid_level_and_source():
    """A bad level/source must NOT poison the bus — it lands at a
    well-defined fallback so the UI dropdown stays clean."""
    from backend.log_bus import LogBus
    bus = LogBus(maxlen=10)
    bus.publish(_mk_event(level="totally_made_up", message="x"))
    bus.publish(_mk_event(source="not_a_source", message="y"))
    rows = bus.tail()
    assert all(r["level"] in ("debug", "info", "warning", "error", "critical")
               for r in rows)
    assert all(r["source"] in ("python", "tool", "job", "supervisor", "agent")
               for r in rows)
