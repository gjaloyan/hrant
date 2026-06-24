"""Structural backstop: after several build-write actions without frame_problem,
the tool loop injects a one-time FRAME-CHECK nudge. Behavioural (reads the
agent's own tool stream), NOT keyword routing on the user's message.
"""
from __future__ import annotations

from backend.unified_agent import (
    _build_frame_marker, _should_block_build,
    _BUILD_FRAME_THRESHOLD, _BUILD_BLOCK_THRESHOLD,
)


def test_block_refuses_build_write_after_threshold_without_frame():
    st = {"writes": _BUILD_BLOCK_THRESHOLD, "framed": False}
    assert _should_block_build(st, "terminal_exec") is True
    assert _should_block_build(st, "save_to_workspace") is True
    # a read-only tool is never blocked
    assert _should_block_build(st, "read_file") is False


def test_block_allows_when_under_threshold():
    st = {"writes": _BUILD_BLOCK_THRESHOLD - 1, "framed": False}
    assert _should_block_build(st, "terminal_exec") is False


def test_block_allows_after_framing():
    st = {"writes": _BUILD_BLOCK_THRESHOLD + 5, "framed": True}
    assert _should_block_build(st, "terminal_exec") is False


def test_fires_once_after_threshold_build_writes():
    st = {}
    for _ in range(_BUILD_FRAME_THRESHOLD - 1):
        assert _build_frame_marker(st, "terminal_exec", False) == ""
    m = _build_frame_marker(st, "save_to_workspace", False)  # crosses threshold
    assert "FRAME-CHECK" in m
    # fires only once
    assert _build_frame_marker(st, "run_python", False) == ""


def test_silent_when_already_framed():
    st = {}
    _build_frame_marker(st, "frame_problem", False)  # framed this turn
    for _ in range(_BUILD_FRAME_THRESHOLD + 3):
        assert _build_frame_marker(st, "terminal_exec", False) == ""


def test_create_tracker_also_counts_as_framed():
    st = {}
    _build_frame_marker(st, "create_tracker", False)
    for _ in range(_BUILD_FRAME_THRESHOLD + 1):
        assert _build_frame_marker(st, "save_to_workspace", False) == ""


def test_errors_and_readonly_tools_do_not_count():
    st = {}
    for _ in range(_BUILD_FRAME_THRESHOLD + 2):
        _build_frame_marker(st, "terminal_exec", True)   # errored — ignored
        _build_frame_marker(st, "read_file", False)      # read-only — not a build write
    assert st.get("writes", 0) == 0
