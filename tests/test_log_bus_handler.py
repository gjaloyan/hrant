"""Bridging Python's stdlib logging into the LogBus so every existing
`log.info(...)` / `log.warning(...)` call in the codebase becomes a
LogEvent without per-call instrumentation."""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def clean_bus():
    from backend.log_bus import BUS
    BUS.clear()
    yield BUS
    BUS.clear()


def test_handler_emits_info_event(clean_bus):
    from backend.log_bus import LogBusHandler
    handler = LogBusHandler()
    logger = logging.getLogger("test.handler.info")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.info("hello %s", "world")
    finally:
        logger.removeHandler(handler)
    rows = clean_bus.tail()
    assert len(rows) == 1
    assert rows[0]["level"] == "info"
    assert rows[0]["source"] == "python"
    assert rows[0]["logger"] == "test.handler.info"
    assert rows[0]["message"] == "hello world"


def test_handler_maps_warning_and_error_levels(clean_bus):
    from backend.log_bus import LogBusHandler
    handler = LogBusHandler()
    logger = logging.getLogger("test.handler.levels")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.warning("w")
        logger.error("e")
        logger.critical("c")
        logger.debug("d")
    finally:
        logger.removeHandler(handler)
    levels = [r["level"] for r in clean_bus.tail()]
    assert levels == ["warning", "error", "critical", "debug"]


def test_handler_includes_exception_meta(clean_bus):
    """`log.exception(...)` (called inside an except: block) attaches
    the traceback. We surface it in the event's meta so the UI can
    show it in a collapsible row."""
    from backend.log_bus import LogBusHandler
    handler = LogBusHandler()
    logger = logging.getLogger("test.handler.exc")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.exception("oops")
    finally:
        logger.removeHandler(handler)
    rows = clean_bus.tail()
    assert len(rows) == 1
    assert rows[0]["level"] == "error"
    assert "boom" in (rows[0]["meta"].get("traceback") or "")


def test_handler_never_recurses_into_itself(clean_bus):
    """If the handler's own emit() logs a warning, the warning must
    NOT re-enter the bus (infinite loop). Pinned by using a private
    `_in_handle` flag."""
    from backend.log_bus import LogBusHandler

    class Recursive(LogBusHandler):
        def emit(self, record):
            super().emit(record)
            log_in_handler = logging.getLogger("test.handler.recurse")
            log_in_handler.warning("from emit")

    handler = Recursive()
    logger = logging.getLogger("test.handler.recurse")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.info("seed")
    finally:
        logger.removeHandler(handler)
    rows = clean_bus.tail()
    messages = [r["message"] for r in rows]
    # Deterministic: only the outer "seed" event reaches the bus. The
    # subclass's post-super `log.warning("from emit")` re-enters
    # handle() on the same handler while the outer call is still on
    # the stack, so the recursion guard suppresses it. A regression
    # that drops the guard balloons this list (the inner warning
    # would publish, re-trigger emit(), publish again, ...).
    assert messages == ["seed"], (
        f"unexpected event sequence: {messages}"
    )
