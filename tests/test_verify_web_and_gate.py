"""Verification as a first-class action.

(1) `verify_web` tool: headless-render a URL and report what ACTUALLY renders
(HTTP status, DOM size, expected texts present, headings) — the check I kept
running by hand while the agent claimed "done" three times without looking.

(2) Verify-gate: in a turn that made build-writes, `update_step(status=done)`
is blocked until a verification ran (verify_web) — same structural, behavioral
pattern as the build-without-frame gate (no keywords, agent's own tool stream).
"""
from __future__ import annotations

import json

from backend.unified_agent import _should_block_done, _note_verify_tools
from backend.tool_bundles import BASE_TOOLS


# ─── the gate decision ────────────────────────────────────────────────

def test_done_blocked_after_builds_without_verification():
    st = {"writes": 5, "verified": False}
    assert _should_block_done(st, "update_step", {"status": "done"}) is True


def test_done_allowed_when_verified():
    st = {"writes": 5, "verified": False}
    _note_verify_tools(st, "verify_web")
    assert st["verified"] is True
    assert _should_block_done(st, "update_step", {"status": "done"}) is False


def test_done_allowed_without_builds():
    # reviewing/planning turn (no build-writes) may mark steps done freely
    st = {"writes": 0, "verified": False}
    assert _should_block_done(st, "update_step", {"status": "done"}) is False


def test_non_done_updates_pass():
    st = {"writes": 9, "verified": False}
    assert _should_block_done(st, "update_step", {"status": "active"}) is False
    assert _should_block_done(st, "read_file", {}) is False


# ─── the tool ─────────────────────────────────────────────────────────

def test_verify_web_registered_and_base():
    import backend.builtin_tools as bt
    bt.register_builtin_tools()
    from backend.tool_registry import REGISTRY
    assert "verify_web" in REGISTRY.names()
    assert "verify_web" in BASE_TOOLS


def test_verify_web_unreachable_is_honest():
    import backend.builtin_tools as bt
    out = json.loads(bt._verify_web_handler(
        url="http://127.0.0.1:59999/", expect_texts=["x"]))
    assert out["ok"] is False
    assert "unreachable" in (out.get("error") or "").lower()


def test_builder_subagent_can_verify_web():
    from backend.subagents.roles import ROLE_REGISTRY
    assert "verify_web" in ROLE_REGISTRY["builder"].tools
