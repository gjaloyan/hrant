"""Tests for the unified LogBus that backs the WebUI Logs tab."""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _isolate_bus(tmp_path, monkeypatch):
    """Reset the module-level singleton around every test so that
    publishers added by future tasks (LogBusHandler from Task 2,
    cross-cutting publishers from Task 8) don't leak events into
    unrelated test runs. Also point the JSONL writer at tmp_path so
    the test suite doesn't pollute ~/.hrant/data/logs/ with stray
    daily files."""
    from backend import log_bus as _lb
    monkeypatch.setattr(_lb, "_logs_dir", lambda: tmp_path)
    _lb.BUS.clear()
    yield
    _lb.BUS.clear()


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


def test_tool_call_publishes_event():
    from backend.log_bus import BUS, publish_tool_event
    publish_tool_event(
        name="read_file",
        args={"path": "x.py"},
        result_preview="abc",
        is_error=False,
        request_id="t-123",
    )
    rows = BUS.tail()
    assert len(rows) == 1
    assert rows[0]["source"] == "tool"
    assert rows[0]["logger"] == "read_file"
    assert rows[0]["request_id"] == "t-123"
    assert rows[0]["meta"].get("args") == {"path": "x.py"}
    assert rows[0]["level"] == "info"


def test_tool_error_publishes_error_event():
    from backend.log_bus import BUS, publish_tool_event
    publish_tool_event(
        name="terminal_exec",
        args={"command": "false"},
        result_preview="exit code 1",
        is_error=True,
    )
    rows = BUS.tail()
    assert rows[0]["level"] == "error"


def test_job_status_change_publishes():
    from backend.log_bus import BUS, publish_job_event
    publish_job_event(
        job_id="2b7d6ed82c76",
        new_status="completed",
        prev_status="running",
    )
    rows = BUS.tail()
    assert len(rows) == 1
    assert rows[0]["source"] == "job"
    assert rows[0]["meta"].get("job_id") == "2b7d6ed82c76"
    assert rows[0]["meta"].get("from") == "running"
    assert rows[0]["meta"].get("to") == "completed"


def test_agent_progress_publishes():
    from backend.log_bus import BUS, publish_agent_event
    publish_agent_event(event="think", message="planning", request_id="t-456")
    rows = BUS.tail()
    assert rows[0]["source"] == "agent"
    assert rows[0]["logger"] == "think"
    assert rows[0]["message"] == "planning"
    assert rows[0]["request_id"] == "t-456"


def test_supervisor_decision_publishes():
    from backend.log_bus import BUS, publish_supervisor_event
    publish_supervisor_event(
        job_id="abc",
        decision="done",
        message="all criteria met",
    )
    rows = BUS.tail()
    assert rows[0]["source"] == "supervisor"
    assert rows[0]["meta"].get("decision") == "done"


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
