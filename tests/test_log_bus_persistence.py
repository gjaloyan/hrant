"""Persistence: every publish also appends a JSONL line to a daily
rotating file under <data_dir>/logs/agent-YYYYMMDD.jsonl. The GC
sweep deletes files older than the retention window."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def isolated_logs_dir(tmp_path, monkeypatch):
    """Point the bus's file writer at a clean tmp dir."""
    from backend import log_bus as _lb
    monkeypatch.setattr(_lb, "_logs_dir", lambda: tmp_path)
    _lb.BUS.clear()
    yield tmp_path
    _lb.BUS.clear()


def test_publish_appends_jsonl_line(isolated_logs_dir):
    from backend.log_bus import BUS, LogEvent
    BUS.publish(LogEvent(
        ts=time.time(), level="info", source="python",
        logger="t", message="hello",
    ))
    today = datetime.now().strftime("%Y%m%d")
    f = isolated_logs_dir / f"agent-{today}.jsonl"
    assert f.exists()
    lines = f.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["message"] == "hello"
    assert parsed["source"] == "python"


def test_two_publishes_two_lines(isolated_logs_dir):
    from backend.log_bus import BUS, LogEvent
    for i in range(3):
        BUS.publish(LogEvent(
            ts=time.time(), level="info", source="python",
            logger="t", message=f"m{i}",
        ))
    today = datetime.now().strftime("%Y%m%d")
    f = isolated_logs_dir / f"agent-{today}.jsonl"
    assert len(f.read_text(encoding="utf-8").splitlines()) == 3


def test_gc_old_deletes_files_past_retention(isolated_logs_dir):
    """The GC keeps files dated within `days`; older daily files
    are unlinked. Files with non-matching names are left alone."""
    from backend.log_bus import gc_old
    now = datetime.now()
    for d in range(10):
        date = (now - timedelta(days=d)).strftime("%Y%m%d")
        (isolated_logs_dir / f"agent-{date}.jsonl").write_text("x\n")
    (isolated_logs_dir / "README").write_text("keep me")
    removed = gc_old(days=7)
    assert removed == 2  # the day-8 and day-9 files
    survivors = sorted(p.name for p in isolated_logs_dir.iterdir())
    assert "README" in survivors
    daily_survivors = [p for p in survivors if p.startswith("agent-")]
    assert len(daily_survivors) == 8


def test_writer_failure_does_not_break_bus(isolated_logs_dir, monkeypatch):
    """If the JSONL write fails (disk full, permission denied), the
    in-memory ring + subscribers must still work."""
    from backend.log_bus import BUS, LogEvent
    import backend.log_bus as _lb
    def _boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(_lb, "_write_jsonl_line", _boom)
    BUS.publish(LogEvent(
        ts=time.time(), level="info", source="python",
        logger="t", message="x",
    ))
    rows = BUS.tail()
    assert len(rows) == 1
