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


def test_default_rules_includes_server_and_error_rules():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    names = {r.name for r in rules}
    assert "disk_low" in names
    assert "memory_low" in names
    assert "cpu_high" in names
    assert "errors_present" in names


def test_default_rules_disk_fires_when_low():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    disk_rule = next(r for r in rules if r.name == "disk_low")
    assert disk_rule.lever == "FIRE_SERVER_HEALTH"
    assert disk_rule.predicate(_snapshot(disk_free_gb=0.5)) is True
    assert disk_rule.predicate(_snapshot(disk_free_gb=50.0)) is False


def test_default_rules_errors_fires_only_when_nonempty():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    rule = next(r for r in rules if r.name == "errors_present")
    assert rule.lever == "FIRE_ERROR_TRIAGE"
    assert rule.predicate(_snapshot(recent_errors=[])) is False
    assert rule.predicate(_snapshot(recent_errors=[{"message": "x"}])) is True


def test_default_rules_reactive_rules_come_first():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    reactive_names = {"disk_low", "memory_low", "cpu_high", "errors_present"}
    first_four = {r.name for r in rules[:4]}
    assert first_four == reactive_names


def test_default_rules_schedule_tick_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["integrity_tick"].lever == "FIRE_INTEGRITY_HEARTBEAT"
    assert rules["integrity_tick"].cooldown_seconds == 300.0
    assert rules["goal_propose_tick"].lever == "FIRE_GOAL_PROPOSE"
    assert rules["goal_propose_tick"].cooldown_seconds == 3600.0
    assert rules["consolidation_tick"].lever == "FIRE_MEMORY_CONSOLIDATION"
    assert rules["consolidation_tick"].cooldown_seconds == 86400.0


def test_default_rules_schedule_ticks_predicate_always_true():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    snap = _snapshot()
    assert rules["integrity_tick"].predicate(snap) is True
    assert rules["goal_propose_tick"].predicate(snap) is True
    assert rules["consolidation_tick"].predicate(snap) is True


def test_default_rules_d04_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["capability_scan_tick"].lever == "FIRE_CAPABILITY_SCAN"
    assert rules["capability_scan_tick"].cooldown_seconds == 21600.0
    assert rules["self_study_tick"].lever == "FIRE_SELF_STUDY"
    assert rules["self_study_tick"].cooldown_seconds == 86400.0


def test_default_rules_count_after_phase11():
    """D-07 added 18; Phase 11 added scheduled_messages_tick → 19.
    2026-05-27 audit T2.5 added log_rotation_tick → 20."""
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    assert len(rules) == 20


def test_default_rules_d07_scheduled_rules_present():
    """D-07 added self_reflection / finetune_qc / gap_detection in
    that order. After the audit T2.5 cleanup the tail is:
    self_reflection / finetune_qc / gap_detection / log_rotation /
    scheduled_messages. The D-07 triplet stays contiguous at -5..-2."""
    from backend.autonomic.layer0 import default_rules
    names = [r.name for r in default_rules()]
    assert names[-5:-2] == [
        "self_reflection_tick", "finetune_qc_tick", "gap_detection_tick",
    ]
    assert names[-1] == "scheduled_messages_tick"


def test_default_rules_d07_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["self_reflection_tick"].lever == "FIRE_SELF_REFLECTION"
    assert rules["self_reflection_tick"].cooldown_seconds == 86400.0
    assert rules["finetune_qc_tick"].lever == "FIRE_FINETUNE_QC"
    assert rules["finetune_qc_tick"].cooldown_seconds == 86400.0
    assert rules["gap_detection_tick"].lever == "FIRE_GAP_DETECTION"
    assert rules["gap_detection_tick"].cooldown_seconds == 86400.0


def test_default_rules_d06_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["model_eval_tick"].lever == "FIRE_MODEL_EVAL"
    assert rules["model_eval_tick"].cooldown_seconds == 86400.0
    assert rules["session_archive_tick"].lever == "FIRE_SESSION_ARCHIVE"
    assert rules["session_archive_tick"].cooldown_seconds == 86400.0
    assert rules["cost_audit_tick"].lever == "FIRE_COST_AUDIT"
    assert rules["cost_audit_tick"].cooldown_seconds == 3600.0


def test_default_rules_d05_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["graph_maintenance_tick"].lever == "FIRE_GRAPH_MAINTENANCE"
    assert rules["graph_maintenance_tick"].cooldown_seconds == 86400.0
    assert rules["proactive_learn_tick"].lever == "FIRE_PROACTIVE_LEARN"
    assert rules["proactive_learn_tick"].cooldown_seconds == 3600.0
    assert rules["note_curation_tick"].lever == "FIRE_NOTE_CURATION"
    assert rules["note_curation_tick"].cooldown_seconds == 604800.0
