import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule, default_rules
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


def _build_tick(tmp_path: Path, rules=None):
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
    engine = Layer0Engine(rules=rules if rules is not None else default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )
    return tick, lever_log


def test_three_d07_ticks_fire_in_order(tmp_path: Path):
    d07_only = [
        LayerZeroRule(name="self_reflection_tick", predicate=lambda s: True,
                      lever="FIRE_SELF_REFLECTION", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="finetune_qc_tick", predicate=lambda s: True,
                      lever="FIRE_FINETUNE_QC", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="gap_detection_tick", predicate=lambda s: True,
                      lever="FIRE_GAP_DETECTION", params={}, cooldown_seconds=86400.0),
    ]

    sr_log = tmp_path / "self_reflection_log.jsonl"
    fq_log = tmp_path / "finetune_qc_log.jsonl"
    gd_log = tmp_path / "gap_detection_log.jsonl"
    queue = tmp_path / "finetune_queue.jsonl"
    gaps = tmp_path / "gaps.json"

    queue.write_text(json.dumps({
        "id": "p1",
        "messages": [
            {"role": "user", "content": "what is python"},
            {"role": "assistant", "content": "A programming language used widely."},
        ],
        "metadata": {
            "source_notes": ["python"], "confidence": 90, "project": None,
            "timestamp": "2026-04-20T12:00:00", "verified": True,
            "category": "factual_qa", "boosted": False, "original_wrong_answer": None,
        },
    }) + "\n", encoding="utf-8")

    gaps.write_text(json.dumps({
        "rust": {"topic": "rust", "count": 3, "last": "2026-04-19 12:00"},
    }), encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path, rules=d07_only)

    from backend.autonomic.levers.self_reflection import FIRE_SELF_REFLECTION
    from backend.autonomic.levers.finetune_qc import FIRE_FINETUNE_QC
    from backend.autonomic.levers.gap_detection import FIRE_GAP_DETECTION

    orig_sr = FIRE_SELF_REFLECTION.run
    orig_fq = FIRE_FINETUNE_QC.run
    orig_gd = FIRE_GAP_DETECTION.run

    def sr_wrap(self, params, context):
        p = dict(params); p.setdefault("log_path", str(sr_log))
        return orig_sr(self, p, context)

    def fq_wrap(self, params, context):
        p = dict(params); p.setdefault("queue_path", str(queue)); p.setdefault("log_path", str(fq_log))
        return orig_fq(self, p, context)

    def gd_wrap(self, params, context):
        p = dict(params); p.setdefault("gaps_path", str(gaps)); p.setdefault("log_path", str(gd_log))
        return orig_gd(self, p, context)

    fake_meta = MagicMock()
    fake_meta.stats.return_value = {
        "total_failures": 5, "by_root_cause": {"x": 3, "y": 2},
        "by_domain": {"py": 5}, "avg_severity": 5.0, "patterns_count": 1, "patterns": [],
    }
    fake_meta.extract_patterns.return_value = [
        {"pattern": "p1", "priority": 7, "frequency": 3, "suggested_fix": "fix"},
    ]

    with patch.object(FIRE_SELF_REFLECTION, "run", sr_wrap), \
         patch.object(FIRE_FINETUNE_QC, "run", fq_wrap), \
         patch.object(FIRE_GAP_DETECTION, "run", gd_wrap), \
         patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        tick()
        tick()
        tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_SELF_REFLECTION", "FIRE_FINETUNE_QC", "FIRE_GAP_DETECTION"]

    assert sr_log.read_text(encoding="utf-8").count("\n") == 1
    assert fq_log.read_text(encoding="utf-8").count("\n") == 1
    assert gd_log.read_text(encoding="utf-8").count("\n") == 1


def test_reactive_rule_preempts_d07_scheduled(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path)
    tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_default_levers_registered_via_startup(tmp_path: Path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    clear_registry()
    from backend.autonomic.startup import build_scheduler
    build_scheduler()
    names = LeverRegistry.instance().names()
    # D-09 (4 immune + 15 autonomic) + Phase 11 (FIRE_SCHEDULED_MESSAGES) = 20
    # 2026-05-27 audit T1 added FIRE_EMBEDDING_BACKFILL → 21.
    assert len(names) == 21
    clear_registry()
