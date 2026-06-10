"""skill_reflection gates (audit 2026-06-10 I4).

Two new gates added on top of the existing ones:
  7. verifier_confidence < SKILL_REFLECTION_CONFIDENCE_FLOOR -> skip
  8. endpoint_was_met is False -> skip

These avoid spending an LLM call to canonize a flawed/research-only
turn into a skill that would carry the same flaw forward.
"""
from __future__ import annotations

import pytest


class _StubTraceStep:
    def __init__(self, name: str, event: str = "tool"):
        self.event = event
        self.tool_call = type("TC", (), {"name": name, "args": {}})()


class _StubAgent:
    def __init__(self, tool_names):
        self._trace = [_StubTraceStep(n) for n in tool_names]


@pytest.fixture
def stub_skills(monkeypatch):
    """Ensure skill_creator looks loaded so gate 6 doesn't fire first.
    monkeypatch.setattr restores the real SKILLS singleton when the
    test exits — without that, the next test in the same process
    inherits our stub and AttributeErrors on real method calls."""
    from backend import skills as sk

    class _S:
        def __init__(self, name="skill_creator", body="body"):
            self.name = name
            self.body = body
            self.enabled = True
            self.triggers = []
            self.tags = []
            self.source = "core"

    class _Store:
        def get(self, n):
            return _S(n) if n == "skill_creator" else None

        def list(self):
            return [_S()]

    monkeypatch.setattr(sk, "SKILLS", _Store())
    yield _Store()


def test_low_confidence_skips_reflection(stub_skills):
    """verifier_confidence below the floor -> reason includes
    'low-confidence' and should_run is False."""
    from backend import unified_agent as ua

    agent = _StubAgent(["read_file", "grep", "terminal_exec"])

    should, reason = ua._should_reflect_for_skill(
        agent, "ok answer",
        verifier_confidence=35,  # below the 50 floor
        endpoint_was_met=True,
    )
    assert should is False
    assert "low-confidence" in reason
    assert "35" in reason


def test_high_confidence_proceeds(stub_skills):
    """verifier_confidence above the floor passes gate 7."""
    from backend import unified_agent as ua

    agent = _StubAgent(["read_file", "grep", "terminal_exec"])

    should, reason = ua._should_reflect_for_skill(
        agent, "ok answer",
        verifier_confidence=85,
        endpoint_was_met=True,
    )
    assert should is True, f"expected ok, got reason={reason}"


def test_endpoint_not_met_skips_reflection(stub_skills):
    """endpoint_was_met=False -> pure-research turn, no skill needed."""
    from backend import unified_agent as ua

    agent = _StubAgent(["read_file", "grep", "search_knowledge"])

    should, reason = ua._should_reflect_for_skill(
        agent, "ok answer",
        verifier_confidence=85,
        endpoint_was_met=False,
    )
    assert should is False
    assert reason == "endpoint-not-met"


def test_endpoint_unknown_does_not_block(stub_skills):
    """endpoint_was_met=None (verifier path didn't run) does NOT
    veto reflection — falls through to the original gates."""
    from backend import unified_agent as ua

    agent = _StubAgent(["read_file", "grep", "terminal_exec"])

    should, reason = ua._should_reflect_for_skill(
        agent, "ok answer",
        verifier_confidence=None,
        endpoint_was_met=None,
    )
    assert should is True


def test_confidence_unknown_does_not_block(stub_skills):
    """verifier_confidence=None (verifier skipped, e.g. no tool_outputs)
    does NOT veto reflection — falls through to the original gates."""
    from backend import unified_agent as ua

    agent = _StubAgent(["read_file", "grep", "terminal_exec"])

    should, reason = ua._should_reflect_for_skill(
        agent, "ok answer",
        verifier_confidence=None,
        endpoint_was_met=True,
    )
    assert should is True


def test_floor_constant_value():
    """Sanity: the floor lines up with the daily-report cap (30 from
    cap_confidence_for_endpoint on a missed endpoint should always
    veto)."""
    from backend import unified_agent as ua

    assert ua.SKILL_REFLECTION_CONFIDENCE_FLOOR == 50
    # A missed-endpoint turn is capped at 30 — must be below the floor
    # so it always gates out via the confidence check too (belt and
    # suspenders with the explicit endpoint_was_met=False gate).
    from backend.endpoint_check import _MISSED_ENDPOINT_CAP
    assert _MISSED_ENDPOINT_CAP < ua.SKILL_REFLECTION_CONFIDENCE_FLOOR
