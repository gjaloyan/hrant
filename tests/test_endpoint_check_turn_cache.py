"""endpoint_check turn cache (audit 2026-06-10 I3).

Without the cache a single turn could trip 2-3 LLM CLASSIFICATION
calls on identical (task, answer, tool_names) — _decide_self_correction,
the verifier-cap branch in run_unified, and cap_confidence_for_endpoint
internally all evaluate the same judgment. The cache makes the 2nd
and 3rd calls instant.
"""
from __future__ import annotations

import pytest


def test_endpoint_met_called_only_once_per_turn(monkeypatch):
    """Two calls to endpoint_met with the same args, inside the same
    turn cache window, hit the LLM exactly once."""
    from backend import endpoint_check as ec

    call_count = {"n": 0}

    def _fake_llm(task, answer, evidence=""):
        call_count["n"] += 1
        return False

    monkeypatch.setattr(ec, "_llm_endpoint_met", _fake_llm)

    token = ec.begin_turn_cache()
    try:
        r1 = ec.endpoint_met(
            task="email Anna", answer="I sent it.", tool_names=[],
        )
        r2 = ec.endpoint_met(
            task="email Anna", answer="I sent it.", tool_names=[],
        )
        r3 = ec.endpoint_met(
            task="email Anna", answer="I sent it.", tool_names=[],
        )
    finally:
        ec.reset_turn_cache(token)

    assert r1 == r2 == r3 is False
    assert call_count["n"] == 1, (
        f"expected 1 LLM trip, got {call_count['n']}"
    )


def test_endpoint_met_cache_key_includes_tool_names(monkeypatch):
    """Different tool_names = different key — must NOT collide."""
    from backend import endpoint_check as ec

    seen = []

    def _fake_llm(task, answer, evidence=""):
        seen.append((task, answer))
        return False

    monkeypatch.setattr(ec, "_llm_endpoint_met", _fake_llm)

    token = ec.begin_turn_cache()
    try:
        ec.endpoint_met(task="t", answer="a", tool_names=["read_file"])
        # Different tool list — but since read_file is read-only and the
        # answer has no MEDIA:, both calls fall through to the LLM.
        # The cache MUST treat ["read_file"] vs ["grep"] as different
        # keys, not collide.
        ec.endpoint_met(task="t", answer="a", tool_names=["grep"])
    finally:
        ec.reset_turn_cache(token)

    assert len(seen) == 2


def test_endpoint_met_no_cache_outside_turn_window(monkeypatch):
    """When no turn cache is open, every call hits the LLM. Behaviour
    must be transparent — turning the cache off must not break calls."""
    from backend import endpoint_check as ec

    call_count = {"n": 0}

    def _fake_llm(task, answer, evidence=""):
        call_count["n"] += 1
        return True

    monkeypatch.setattr(ec, "_llm_endpoint_met", _fake_llm)

    # No begin_turn_cache call.
    r1 = ec.endpoint_met(task="x", answer="y", tool_names=[])
    r2 = ec.endpoint_met(task="x", answer="y", tool_names=[])

    assert r1 is True and r2 is True
    assert call_count["n"] == 2


def test_execute_tool_short_circuits_before_llm(monkeypatch):
    """An execute-class tool in tool_names returns True without
    hitting the LLM at all. Cache MUST still record this so a
    follow-up call doesn't accidentally re-run the cheap branch
    (we want the result cached either way for consistency)."""
    from backend import endpoint_check as ec

    call_count = {"n": 0}

    def _fake_llm(task, answer, evidence=""):
        call_count["n"] += 1
        return False

    monkeypatch.setattr(ec, "_llm_endpoint_met", _fake_llm)

    token = ec.begin_turn_cache()
    try:
        r = ec.endpoint_met(
            task="t", answer="a", tool_names=["start_background_job"],
        )
    finally:
        ec.reset_turn_cache(token)

    assert r is True
    assert call_count["n"] == 0


def test_unbacked_action_claim_also_cached(monkeypatch):
    """Same dedupe story for unbacked_action_claim — it's the other
    classification call _decide_self_correction triggers."""
    from backend import endpoint_check as ec

    call_count = {"n": 0}

    def _fake_router():
        class R:
            @staticmethod
            def call_json(*a, **kw):
                call_count["n"] += 1
                return {"unbacked_claim": "I sent an email"}
        return R()

    monkeypatch.setattr("backend.endpoint_check.router", _fake_router,
                        raising=False)
    # router lives in llm; need to monkeypatch the actual symbol used.
    import backend.llm as _llm
    monkeypatch.setattr(_llm, "router", _fake_router)

    token = ec.begin_turn_cache()
    try:
        r1 = ec.unbacked_action_claim("send email", "I sent it.", [])
        r2 = ec.unbacked_action_claim("send email", "I sent it.", [])
    finally:
        ec.reset_turn_cache(token)

    assert r1 == r2 == "I sent an email"
    assert call_count["n"] == 1
