from datetime import datetime, timezone

from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule
from backend.autonomic.types import StateSnapshot, TickDecisionSource


def _snapshot(**overrides) -> StateSnapshot:
    base = dict(
        taken_at=datetime.now(timezone.utc),
        uptime_seconds=10.0,
        disk_free_gb=100.0,
        memory_free_gb=8.0,
        cpu_load_1m=0.5,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )
    base.update(overrides)
    return StateSnapshot(**base)


def test_engine_with_no_rules_returns_idle():
    engine = Layer0Engine(rules=[])
    decision = engine.evaluate(_snapshot())
    assert decision.source == TickDecisionSource.L0_REFLEX
    assert decision.lever is None
    assert decision.reason == "idle_no_rules_matched"


def test_rule_fires_when_predicate_true():
    rule = LayerZeroRule(
        name="disk_low",
        predicate=lambda s: s.disk_free_gb < 5.0,
        lever="FIRE_SERVER_HEALTH",
        params={"reason": "disk"},
        cooldown_seconds=10.0,
    )
    engine = Layer0Engine(rules=[rule])
    decision = engine.evaluate(_snapshot(disk_free_gb=1.0))
    assert decision.lever == "FIRE_SERVER_HEALTH"
    assert decision.rule_name == "disk_low"
    assert decision.params == {"reason": "disk"}


def test_rule_does_not_fire_when_predicate_false():
    rule = LayerZeroRule(
        name="disk_low",
        predicate=lambda s: s.disk_free_gb < 5.0,
        lever="FIRE_SERVER_HEALTH",
        params={},
    )
    engine = Layer0Engine(rules=[rule])
    decision = engine.evaluate(_snapshot(disk_free_gb=100.0))
    assert decision.lever is None
    assert decision.reason == "idle_no_rules_matched"


def test_first_matching_rule_wins():
    rule_a = LayerZeroRule(name="a", predicate=lambda s: True, lever="FIRE_A", params={})
    rule_b = LayerZeroRule(name="b", predicate=lambda s: True, lever="FIRE_B", params={})
    engine = Layer0Engine(rules=[rule_a, rule_b])
    decision = engine.evaluate(_snapshot())
    assert decision.lever == "FIRE_A"
    assert decision.rule_name == "a"


def test_cooldown_blocks_re_fire():
    rule = LayerZeroRule(
        name="noisy",
        predicate=lambda s: True,
        lever="FIRE_X",
        params={},
        cooldown_seconds=60.0,
    )
    engine = Layer0Engine(rules=[rule])
    first = engine.evaluate(_snapshot())
    assert first.lever == "FIRE_X"
    second = engine.evaluate(_snapshot())
    assert second.lever is None
    assert "cooldown" in second.reason


def test_cooldown_expires_allows_re_fire(monkeypatch):
    import backend.autonomic.layer0 as layer0
    clock = [1000.0]
    monkeypatch.setattr(layer0.time, "monotonic", lambda: clock[0])
    rule = LayerZeroRule(
        name="noisy",
        predicate=lambda s: True,
        lever="FIRE_X",
        params={},
        cooldown_seconds=60.0,
    )
    engine = Layer0Engine(rules=[rule])
    engine.evaluate(_snapshot())
    clock[0] += 61.0
    second = engine.evaluate(_snapshot())
    assert second.lever == "FIRE_X"


def test_predicate_exception_is_swallowed_and_rule_skipped():
    boom = LayerZeroRule(
        name="boom",
        predicate=lambda s: 1 / 0,
        lever="FIRE_BOOM",
        params={},
    )
    ok = LayerZeroRule(
        name="ok",
        predicate=lambda s: True,
        lever="FIRE_OK",
        params={},
    )
    engine = Layer0Engine(rules=[boom, ok])
    decision = engine.evaluate(_snapshot())
    assert decision.lever == "FIRE_OK"
