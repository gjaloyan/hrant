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
        # A HEALTHY graph by default (2026-08-09). It used to be 0, which is
        # the collapsed-graph condition the new `graph_collapsed` recovery
        # rule exists to catch — so every test that meant to exercise the
        # scheduled-lever rotation was silently running against a snapshot
        # that says "the knowledge graph has been wiped". Tests that want the
        # collapsed case now ask for it explicitly.
        kb_graph_nodes=6877,
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
    monkeypatch.setattr(layer0.time, "time", lambda: clock[0])
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
    # service_failed joined the reactive block on 2026-08-10 — a unit systemd
    # has given up on is a safety condition, and like its neighbours its
    # predicate is FALSE in steady state, so sitting this high starves
    # nothing.
    reactive_names = {"disk_low", "memory_low", "cpu_high",
                      "service_failed", "errors_present"}
    first_five = {r.name for r in rules[:5]}
    assert first_five == reactive_names


def test_default_rules_schedule_tick_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["integrity_tick"].lever == "FIRE_INTEGRITY_HEARTBEAT"
    assert rules["integrity_tick"].cooldown_seconds == 300.0
    assert rules["goal_propose_tick"].lever == "FIRE_GOAL_PROPOSE"
    assert rules["goal_propose_tick"].cooldown_seconds == 3600.0
    # Audit T3.2 (2026-05-27): consolidation_tick retired — the
    # dedicated `backend/consolidation/scheduler.py` runs daily +
    # idle-aware, the layer-0 rule was 99% no-ops.
    assert "consolidation_tick" not in rules


def test_default_rules_schedule_ticks_predicate_always_true():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    snap = _snapshot()
    assert rules["integrity_tick"].predicate(snap) is True
    assert rules["goal_propose_tick"].predicate(snap) is True


def test_default_rules_d04_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["capability_scan_tick"].lever == "FIRE_CAPABILITY_SCAN"
    assert rules["capability_scan_tick"].cooldown_seconds == 21600.0
    assert rules["self_study_tick"].lever == "FIRE_SELF_STUDY"
    assert rules["self_study_tick"].cooldown_seconds == 86400.0


def test_default_rules_count_after_phase11():
    """D-07 added 18; Phase 11 added scheduled_messages_tick → 19.
    Audit T2.5 added log_rotation_tick → 20.
    Audit T2.2 added goal_executor_tick → 21.
    Audit T3.2 dropped consolidation_tick → 20.
    Audit T3.3 added fact_embedding_backfill_tick → 21."""
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    # +graph_collapsed +note_embedding_backfill_tick (2026-08-09): two levers
    # that had been registered since May with no rule naming them.
    # +channel_watch_tick (2026-08-20): polls followed public channels so a
    # daily digest reads a ledger rather than whatever fits on one page.
    assert len(rules) == 30  # +FIRE_PROPOSAL_DIGEST (2026-09-01)


def test_default_rules_d07_scheduled_rules_present():
    """D-07 added self_reflection / finetune_qc / gap_detection.
    Since 2026-06-12 scheduled_messages_tick sits at position 3
    (right after the safety triad), not LAST: the first-match
    engine starved due reminders behind housekeeping rules after a
    restart (caught live — a due message waited 5+ minutes).
    User-facing delivery preempts housekeeping; safety preempts
    all. The D-07 triplet therefore moved from -7..-4 to -6..-3."""
    from backend.autonomic.layer0 import default_rules
    names = [r.name for r in default_rules()]
    # Reactive quartet first (unchanged), then reminders lead the
    # always-true periodic block.
    assert names[:4] == [
        "disk_low", "memory_low", "cpu_high", "service_failed",
    ]
    # index 5 now: service_failed joined the reactive block at index 3.
    assert names[5] == "scheduled_messages_tick"
    # channel_watch_tick sits at index 6, right after scheduled_messages: it
    # first shipped LAST and measured zero fires in 80 lever selections, so
    # the tail indices below are unchanged from before it existed.
    assert names[6] == "channel_watch_tick"
    assert names[-9:-5] == [
        "self_reflection_tick", "character_reflection_tick",
        "finetune_qc_tick", "gap_detection_tick",
    ]
    # note_embedding_backfill_tick was appended after it on 2026-08-09 —
    # NOTE embeddings had no rule while FACT embeddings did.
    # unconsolidated_sessions_tick was inserted between them on 2026-08-10.
    assert names[-3] == "fact_embedding_backfill_tick"
    assert names[-2] == "unconsolidated_sessions_tick"
    assert names[-1] == "note_embedding_backfill_tick"


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


def test_tail_rule_is_reachable_when_head_rules_keep_recurring(monkeypatch):
    """A rule at the end of the list must get a turn.

    Free ticks were handed out strictly by list index, so a rule whose
    cooldown expires every minute could reclaim the slot before a rule
    that had been waiting a day was ever reached. The list index is an
    accident of when each rule was written; it was acting as a priority,
    and twice had to be corrected by hand when that starved something.

    Three rules and a 24-hour simulation are enough to show it: two
    frequent rules alternate forever and the third never runs.
    """
    import backend.autonomic.layer0 as layer0
    clock = [1000.0]
    monkeypatch.setattr(layer0.time, "time", lambda: clock[0])

    frequent = [
        LayerZeroRule(name=n, predicate=lambda s: True, lever="FIRE_" + n.upper(),
                      params={}, cooldown_seconds=60.0)
        for n in ("a", "b")
    ]
    daily = LayerZeroRule(name="daily", predicate=lambda s: True,
                          lever="FIRE_DAILY", params={}, cooldown_seconds=86400.0)
    engine = Layer0Engine(rules=frequent + [daily])

    fired = []
    for _ in range(2880):  # 24 hours of 30-second ticks
        d = engine.evaluate(_snapshot())
        if d.lever:
            fired.append(d.lever)
        clock[0] += 30.0

    assert "FIRE_DAILY" in fired


def test_most_overdue_rule_wins_over_a_lower_index_one(monkeypatch):
    """Between two eligible periodic rules, the one that has waited
    longest relative to its own cooldown goes first -- not the one that
    happens to sit higher in the list."""
    import backend.autonomic.layer0 as layer0
    clock = [1000.0]
    monkeypatch.setattr(layer0.time, "time", lambda: clock[0])

    first = LayerZeroRule(name="first", predicate=lambda s: True, lever="FIRE_FIRST",
                          params={}, cooldown_seconds=60.0)
    second = LayerZeroRule(name="second", predicate=lambda s: True, lever="FIRE_SECOND",
                           params={}, cooldown_seconds=60.0)
    engine = Layer0Engine(rules=[first, second])

    assert engine.evaluate(_snapshot()).lever == "FIRE_FIRST"
    clock[0] += 30.0
    assert engine.evaluate(_snapshot()).lever == "FIRE_SECOND"
    # Both are off cooldown now. `first` last ran 90s ago and `second`
    # 60s ago, so `first` is the more overdue and goes again -- which is
    # also what list order would have said. Push `second` further behind
    # and the answer has to change.
    clock[0] += 60.0
    assert engine.evaluate(_snapshot()).lever == "FIRE_FIRST"
    clock[0] += 3600.0
    # first: 3600s behind on a 60s cooldown. second: 3690s behind.
    assert engine.evaluate(_snapshot()).lever == "FIRE_SECOND"


def test_reflex_preempts_a_badly_overdue_periodic_rule(monkeypatch):
    """A fault is not housekeeping. A reflex rule -- one whose predicate
    reads real state rather than saying "every N seconds" -- fires ahead
    of any periodic rule no matter how long that one has waited."""
    import backend.autonomic.layer0 as layer0
    clock = [1000.0]
    monkeypatch.setattr(layer0.time, "time", lambda: clock[0])

    starved = LayerZeroRule(name="starved", predicate=lambda s: True,
                            lever="FIRE_STARVED", params={},
                            cooldown_seconds=86400.0)
    reflex = LayerZeroRule(name="disk_low", predicate=lambda s: s.disk_free_gb < 2.0,
                           lever="FIRE_SERVER_HEALTH", params={},
                           cooldown_seconds=300.0, reflex=True)
    engine = Layer0Engine(rules=[starved, reflex])

    engine.evaluate(_snapshot())          # starved fires once
    clock[0] += 86400.0 * 10              # ten days overdue
    decision = engine.evaluate(_snapshot(disk_free_gb=0.5))
    assert decision.lever == "FIRE_SERVER_HEALTH"


def test_default_rules_reflexes_are_the_fault_and_delivery_rules():
    """Which rules are reflexes is a decision, so it is pinned here.

    Everything else is periodic maintenance and gets scheduled by how
    overdue it is. `scheduled_messages_tick` is on this list because of
    2026-06-12: a due reminder waited five minutes behind housekeeping,
    and user-facing delivery is not housekeeping.
    """
    from backend.autonomic.layer0 import default_rules
    reflexes = {r.name for r in default_rules() if r.reflex}
    assert reflexes == {
        "disk_low", "memory_low", "cpu_high", "service_failed",
        "errors_present", "scheduled_messages_tick", "graph_collapsed",
    }


def test_last_fired_survives_a_restart(tmp_path):
    """A daily lever must not run again just because the process did.

    Cooldowns lived in memory and were keyed off `time.monotonic()`,
    which resets per process. Every deploy therefore re-armed all thirty
    rules, and the expensive daily ones re-ran -- on a day with several
    deploys, several times.
    """
    state = tmp_path / "layer0_state.json"
    rule = LayerZeroRule(name="daily", predicate=lambda s: True,
                         lever="FIRE_DAILY", params={},
                         cooldown_seconds=86400.0)

    first = Layer0Engine(rules=[rule], state_path=state)
    assert first.evaluate(_snapshot()).lever == "FIRE_DAILY"

    restarted = Layer0Engine(rules=[rule], state_path=state)
    decision = restarted.evaluate(_snapshot())
    assert decision.lever is None
    assert "cooldown" in decision.reason


def test_a_clock_that_jumped_backwards_does_not_freeze_a_rule(monkeypatch, tmp_path):
    """A persisted timestamp from the future would otherwise read as
    "fired -3 hours ago", which is inside every cooldown forever."""
    import backend.autonomic.layer0 as layer0
    clock = [1000.0]
    monkeypatch.setattr(layer0.time, "time", lambda: clock[0])

    state = tmp_path / "layer0_state.json"
    rule = LayerZeroRule(name="daily", predicate=lambda s: True,
                         lever="FIRE_DAILY", params={},
                         cooldown_seconds=3600.0)
    engine = Layer0Engine(rules=[rule], state_path=state)
    engine.evaluate(_snapshot())

    clock[0] -= 10000.0                    # clock corrected backwards
    restarted = Layer0Engine(rules=[rule], state_path=state)
    # The stale future timestamp is rewritten to now on the tick that
    # notices it, so the rule stays quiet for one more cooldown...
    assert restarted.evaluate(_snapshot()).lever is None
    # ...and then runs, instead of being frozen for the ~3 hours the
    # clock moved.
    clock[0] += 3601.0
    assert restarted.evaluate(_snapshot()).lever == "FIRE_DAILY"


def test_service_failed_ignores_a_unit_repair_may_not_touch():
    """A rule must not fire a lever that can only answer "not mine".

    The comment on this rule says it may sit among the reflexes because
    prod had zero failed units, so it is false in steady state. Prod
    2026-09-02 broke that assumption: `systemd-networkd-wait-online`
    fails at boot, stays failed until reboot, and is not on the repair
    lever's whitelist -- so the rule was true on every tick and spent a
    restart attempt every ten minutes declining to act.
    """
    from backend.autonomic.layer0 import default_rules
    rule = next(r for r in default_rules() if r.name == "service_failed")
    unrepairable = _snapshot(
        failed_services=["system:systemd-networkd-wait-online.service"])
    assert rule.predicate(unrepairable) is False


def test_service_failed_still_fires_for_a_unit_repair_owns():
    from backend.autonomic.layer0 import default_rules
    rule = next(r for r in default_rules() if r.name == "service_failed")
    assert rule.predicate(_snapshot(failed_services=["user:lightrag.service"])) is True
    # And a mixed list is still actionable because of the one it owns.
    assert rule.predicate(_snapshot(failed_services=[
        "system:systemd-networkd-wait-online.service",
        "user:lightrag.service",
    ])) is True
