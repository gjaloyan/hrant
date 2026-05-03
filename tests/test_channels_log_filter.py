"""Regression: python-telegram-bot's Conflict-during-poll logs are
collapsed to a single throttled warning instead of a 30-line stack
trace per retry. Background: uvicorn --reload race spawns a fresh
poller before the old child finishes its in-flight getUpdates;
Telegram terminates the old session with `Conflict` and the lib
retries every few seconds, dumping the same trace each time.
"""
from __future__ import annotations
import logging
import time

from backend.channels import _ConflictNoiseFilter


def _record(msg: str, *, exc: Exception | None = None) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="telegram.ext.Updater",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=(type(exc), exc, exc.__traceback__) if exc else None,
    )
    return rec


def test_filter_passes_unrelated_records():
    f = _ConflictNoiseFilter()
    rec = _record("some other error")
    assert f.filter(rec) is True


def test_filter_collapses_conflict_message():
    f = _ConflictNoiseFilter()
    rec = _record("Exception happened while polling for updates.")
    # Simulate the lib's exc_info — message holds 'Conflict'.
    rec.exc_info = (type(Exception()), Exception("Conflict: terminated by other getUpdates request"), None)
    assert f.filter(rec) is True
    assert rec.exc_info is None
    assert rec.levelname == "WARNING"
    assert "preempted" in rec.msg.lower()


def test_filter_throttles_repeats():
    f = _ConflictNoiseFilter()
    f.THROTTLE_SECONDS = 9999.0  # never expire within test

    rec1 = _record("Conflict in poll loop")
    rec2 = _record("Conflict in poll loop")
    assert f.filter(rec1) is True   # first one passes
    assert f.filter(rec2) is False  # second is suppressed entirely


def test_filter_admits_after_throttle_window():
    f = _ConflictNoiseFilter()
    f.THROTTLE_SECONDS = 0.01

    rec1 = _record("Conflict in poll loop")
    assert f.filter(rec1) is True
    time.sleep(0.02)
    rec2 = _record("Conflict in poll loop")
    assert f.filter(rec2) is True
