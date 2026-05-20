import json
from pathlib import Path
from unittest.mock import patch

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


class _FakeNote:
    class frontmatter:
        topic = "fake_topic"


def test_three_d05_ticks_fire_in_expected_order(tmp_path: Path):
    d05_only = [
        LayerZeroRule(name="graph_maintenance_tick", predicate=lambda s: True,
                      lever="FIRE_GRAPH_MAINTENANCE", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="proactive_learn_tick", predicate=lambda s: True,
                      lever="FIRE_PROACTIVE_LEARN", params={}, cooldown_seconds=3600.0),
        LayerZeroRule(name="note_curation_tick", predicate=lambda s: True,
                      lever="FIRE_NOTE_CURATION", params={}, cooldown_seconds=604800.0),
    ]

    (tmp_path / "graph.json").write_text(json.dumps({"edges": {
        "x": [{"target": "y", "relation": "r", "note": "known_note", "weight": 1.0}],
        "y": [{"target": "x", "relation": "inverse:r", "note": "known_note", "weight": 0.5}],
    }}), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps({
        "known_note": {"topic": "known_note", "category": "profession"},
    }), encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path, rules=d05_only)

    from backend.autonomic.levers.graph_maintenance import FIRE_GRAPH_MAINTENANCE
    orig_gm = FIRE_GRAPH_MAINTENANCE.run

    def gm_wrap(self, params, context):
        p = dict(params)
        p.setdefault("graph_path", str(tmp_path / "graph.json"))
        p.setdefault("index_path", str(tmp_path / "index.json"))
        return orig_gm(self, p, context)

    class _GoalStub:
        def __init__(self, gid, desc, gtype, status="active"):
            self.id = gid
            self.description = desc
            self.goal_type = gtype
            self.status = status
            self.progress_notes = []

        def add_progress(self, n):
            self.progress_notes.append(n)

    class _GoalsStub:
        def __init__(self):
            self._goals = [_GoalStub("p1", "Learn about: rust", "proactive")]

        def active_goals(self):
            return [g for g in self._goals if g.status == "active"]

        def get(self, gid):
            for g in self._goals:
                if g.id == gid:
                    return g
            return None

        def complete_goal(self, gid, note=""):
            for g in self._goals:
                if g.id == gid:
                    g.status = "completed"
                    return True
            return False

    goals_stub = _GoalsStub()

    from backend.autonomic.levers.note_curation import FIRE_NOTE_CURATION
    orig_nc = FIRE_NOTE_CURATION.run

    def nc_wrap(self, params, context):
        p = dict(params)
        p.setdefault("index_path", str(tmp_path / "index.json"))
        return orig_nc(self, p, context)

    with patch.object(FIRE_GRAPH_MAINTENANCE, "run", gm_wrap), \
         patch.object(FIRE_NOTE_CURATION, "run", nc_wrap), \
         patch("backend.autonomic.levers.proactive_learn.GOALS", goals_stub), \
         patch("backend.autonomic.levers.proactive_learn.learn_topic", return_value=_FakeNote()), \
         patch("backend.autonomic.levers.note_curation.learn_topic", return_value=_FakeNote()):
        tick()
        tick()
        tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_GRAPH_MAINTENANCE", "FIRE_PROACTIVE_LEARN", "FIRE_NOTE_CURATION"]
    assert goals_stub._goals[0].status == "completed"


def test_reactive_rule_preempts_d05_scheduled(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path)
    tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_background_py_is_deleted():
    from pathlib import Path as _P
    import backend
    bg = _P(backend.__file__).parent / "background.py"
    assert not bg.exists(), "backend/background.py must be deleted in D-05"


def test_app_has_no_legacy_background_routes():
    """D-05 cleanup pin: the LEGACY `backend/background.py` module
    (autonomic prototype) is gone, so any `/api/background-*` route
    it used to mount must also be gone. Audit T6 introduced a
    different concept — `/api/background-jobs/*` for the long-
    running subprocess registry — which is explicitly allowed."""
    from backend.main import app
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    # Allow only the audit T6 `background-jobs` family. Anything
    # else under /api/background-* would be a regression to the
    # deleted prototype.
    bad = [
        p for p in paths
        if p.startswith("/api/background")
        and not p.startswith("/api/background-jobs")
    ]
    assert bad == [], (
        f"legacy background routes still mounted: {bad}"
    )
