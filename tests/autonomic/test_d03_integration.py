import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.layer0 import Layer0Engine, default_rules
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
from backend.autonomic.safety import SafetyGate
from backend.autonomic.state import StateSnapshotBuilder
from backend.autonomic.tick import make_real_tick
from backend.autonomic.types import LeverReport


@pytest.fixture(autouse=True)
def _reg():
    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
    yield
    clear_registry()


def _fake_consolidation_response() -> dict:
    return {
        "session_summary": "Discussed project plans.",
        "user_profile_facts": [
            {"summary": "User works on autonomic agent", "confidence": 0.9, "category": "project"},
        ],
        "durable_facts": [
            {
                "summary": "agent uses FastAPI and pytest",
                "triples": [["agent", "uses", "fastapi"], ["agent", "uses", "pytest"]],
                "tags": ["tech"],
                "category": "technical",
                "confidence": 0.9,
            }
        ],
        "topic_threads": ["autonomic design"],
    }


def _build_tick(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    lever_log = tmp_path / "lever_log.jsonl"
    tick_log = tmp_path / "tick_log.jsonl"
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=lever_log,
    )
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )
    return tick, lever_log, tick_log


def test_three_scheduled_levers_fire_in_sequence(tmp_path: Path):
    """Three consecutive ticks should fire integrity, goal_propose,
    capability_scan in that order — the fall-through cooldown
    behavior in Layer0Engine.

    Audit T3.2 (2026-05-27): the old test asserted consolidation
    fired third. That rule was retired (dedicated consolidation
    scheduler handles daily runs). capability_scan is now next
    after goal_propose.
    """
    # Seed state
    sessions_path = tmp_path / "sessions.json"
    sessions_path.write_text(json.dumps({
        "current_id": "x",
        "sessions": [{
            "id": "s1",
            "started": "2026-04-14 00:00:00",
            "ended": "2026-04-14 01:00:00",
            "title": "planning",
            "archived": False,
            "turns": [
                {"ts": "2026-04-14 00:00:00", "user": "how is the agent built?", "answer": "FastAPI + pytest", "intent": "task"},
            ],
        }],
    }), encoding="utf-8")
    (tmp_path / "gaps.json").write_text(json.dumps({
        "rust": {"topic": "rust", "count": 3, "last": "2026-04-15"},
    }), encoding="utf-8")
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    user_md_path = tmp_path / "identity" / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"

    tick, lever_log, tick_log = _build_tick(tmp_path)

    # Isolate each lever's filesystem access via params injection
    from backend.autonomic.levers.integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from backend.autonomic.levers.goal_propose import FIRE_GOAL_PROPOSE
    from backend.autonomic.levers.capability_scan import FIRE_CAPABILITY_SCAN

    orig_int = FIRE_INTEGRITY_HEARTBEAT.run
    orig_goal = FIRE_GOAL_PROPOSE.run
    orig_cap = FIRE_CAPABILITY_SCAN.run

    def int_run(self, params, context):
        p = dict(params)
        p.setdefault("knowledge_root", str(tmp_path))
        return orig_int(self, p, context)

    def goal_run(self, params, context):
        p = dict(params)
        p.setdefault("gaps_path", str(tmp_path / "gaps.json"))
        return orig_goal(self, p, context)

    def cap_run(self, params, context):
        # capability_scan reads / writes under cwd; safe to call as-is.
        return orig_cap(self, params, context)

    class _FakeGoal:
        def __init__(self, description: str):
            self.description = description
            self.id = "id_" + description[:5]

    with patch.object(FIRE_INTEGRITY_HEARTBEAT, "run", int_run), \
         patch.object(FIRE_GOAL_PROPOSE, "run", goal_run), \
         patch.object(FIRE_CAPABILITY_SCAN, "run", cap_run), \
         patch("backend.autonomic.levers.goal_propose.GOALS") as mock_goals:
        mock_goals.suggest_from_gaps.side_effect = lambda gaps, max_goals=3: [_FakeGoal(f"Learn about: {g['topic']}") for g in gaps[:max_goals]]

        # Four consecutive ticks. Since 2026-06-12,
        # scheduled_messages_tick LEADS the periodic block (reminder
        # starvation fix) — it takes the first tick, then the
        # housekeeping fall-through proceeds in list order.
        tick()
        tick()
        tick()
        tick()

    lever_lines = lever_log.read_text(encoding="utf-8").splitlines()
    fired_levers = [LeverReport.from_jsonl(line).lever for line in lever_lines]

    assert fired_levers == [
        "FIRE_SCHEDULED_MESSAGES",
        "FIRE_INTEGRITY_HEARTBEAT",
        "FIRE_GOAL_PROPOSE",
        "FIRE_CAPABILITY_SCAN",
    ]

    # Audit T3.2 (2026-05-27): consolidation_tick was retired, so
    # the per-tier write-assertions for sessions/user.md/facts moved
    # to the dedicated consolidation-pipeline tests. The order check
    # above is the meaningful invariant for this scheduler test.


def test_reactive_rule_wins_over_scheduled(tmp_path: Path):
    """When an error is present, errors_present rule (reactive) fires before
    scheduled rules, even on a tick where scheduled rules are ready."""
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 10}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log, tick_log = _build_tick(tmp_path)
    tick()

    lever_lines = lever_log.read_text(encoding="utf-8").splitlines()
    fired = [LeverReport.from_jsonl(line).lever for line in lever_lines]
    # Only ERROR_TRIAGE fires (reactive wins over scheduled)
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_cooldown_fall_through_allows_other_scheduled_rules(tmp_path: Path):
    """After integrity fires on tick 1, on tick 2 (still within cooldown)
    goal_propose should still get a chance to fire via fall-through."""
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "gaps.json").write_text(json.dumps({
        "rust": {"topic": "rust", "count": 3, "last": "2026-04-15"},
    }), encoding="utf-8")

    tick, lever_log, tick_log = _build_tick(tmp_path)

    from backend.autonomic.levers.integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from backend.autonomic.levers.goal_propose import FIRE_GOAL_PROPOSE
    orig_int = FIRE_INTEGRITY_HEARTBEAT.run
    orig_goal = FIRE_GOAL_PROPOSE.run

    def int_run(self, params, context):
        p = dict(params)
        p.setdefault("knowledge_root", str(tmp_path))
        return orig_int(self, p, context)

    def goal_run(self, params, context):
        p = dict(params)
        p.setdefault("gaps_path", str(tmp_path / "gaps.json"))
        return orig_goal(self, p, context)

    class _FakeGoal:
        def __init__(self, description: str):
            self.description = description
            self.id = "id_" + description[:5]

    with patch.object(FIRE_INTEGRITY_HEARTBEAT, "run", int_run), \
         patch.object(FIRE_GOAL_PROPOSE, "run", goal_run), \
         patch("backend.autonomic.levers.goal_propose.GOALS") as mock_goals:
        mock_goals.suggest_from_gaps.side_effect = lambda gaps, max_goals=3: [_FakeGoal(f"Learn about: {g['topic']}") for g in gaps[:max_goals]]

        tick()  # scheduled_messages leads the periodic block (2026-06-12)
        tick()  # scheduled in cooldown; integrity fires
        tick()  # both in cooldown; goal_propose fires via fall-through

    lever_lines = lever_log.read_text(encoding="utf-8").splitlines()
    fired = [LeverReport.from_jsonl(line).lever for line in lever_lines]
    assert fired == [
        "FIRE_SCHEDULED_MESSAGES",
        "FIRE_INTEGRITY_HEARTBEAT",
        "FIRE_GOAL_PROPOSE",
    ]
