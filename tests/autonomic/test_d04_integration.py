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


def _fake_study() -> dict:
    return {
        "purpose": "test module purpose",
        "public_interface": [{"name": "X", "kind": "constant", "one_line": "ok"}],
        "dependencies": [],
        "notes": "ok",
    }


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


def _isolate_paths(tmp_path: Path):
    from backend.autonomic.levers.capability_scan import FIRE_CAPABILITY_SCAN
    from backend.autonomic.levers.self_study import FIRE_SELF_STUDY

    backend_root = tmp_path / "fake_backend"
    tools_dir = backend_root / "tools"
    skills_dir = backend_root / "skills"
    tools_dir.mkdir(parents=True)
    skills_dir.mkdir(parents=True)
    (backend_root / "__init__.py").write_text("", encoding="utf-8")
    (backend_root / "sample.py").write_text('"""sample module."""\n', encoding="utf-8")
    (tools_dir / "my_tool.py").write_text('"""my tool."""\n', encoding="utf-8")
    channels_path = tmp_path / "channels.json"
    channels_path.write_text(json.dumps({"channels": []}), encoding="utf-8")
    self_root = tmp_path / "self"

    orig_scan = FIRE_CAPABILITY_SCAN.run
    orig_study = FIRE_SELF_STUDY.run

    def scan_wrap(self, params, context):
        p = dict(params)
        p.setdefault("tools_dir", str(tools_dir))
        p.setdefault("skills_dir", str(skills_dir))
        p.setdefault("channels_path", str(channels_path))
        p.setdefault("self_root", str(self_root))
        return orig_scan(self, p, context)

    def study_wrap(self, params, context):
        p = dict(params)
        p.setdefault("backend_root", str(backend_root))
        p.setdefault("self_root", str(self_root))
        p.setdefault("max_modules", 3)
        return orig_study(self, p, context)

    return FIRE_CAPABILITY_SCAN, FIRE_SELF_STUDY, scan_wrap, study_wrap, self_root


def test_two_ticks_fire_capability_scan_then_self_study(tmp_path: Path):
    scan_cls, study_cls, scan_wrap, study_wrap, self_root = _isolate_paths(tmp_path)

    # Use only D-04 scheduled rules so D-01..D-03 levers don't interleave.
    d04_only = [
        LayerZeroRule(name="capability_scan_tick", predicate=lambda s: True,
                      lever="FIRE_CAPABILITY_SCAN", params={}, cooldown_seconds=21600.0),
        LayerZeroRule(name="self_study_tick", predicate=lambda s: True,
                      lever="FIRE_SELF_STUDY", params={}, cooldown_seconds=86400.0),
    ]
    tick, lever_log = _build_tick(tmp_path, rules=d04_only)

    with patch.object(scan_cls, "run", scan_wrap), \
         patch.object(study_cls, "run", study_wrap), \
         patch("backend.autonomic.levers.self_study.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_study()
        tick()
        tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_CAPABILITY_SCAN", "FIRE_SELF_STUDY"]

    assert (self_root / "server_inventory.md").exists()
    assert (self_root / "tools" / "my_tool.md").exists()
    module_notes = list((self_root / "modules").glob("*.md"))
    assert len(module_notes) >= 1


def test_reactive_rule_preempts_d04_scheduled(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path)
    tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_capability_scan_produces_all_four_artifacts_in_end_to_end(tmp_path: Path):
    scan_cls, _study_cls, scan_wrap, _study_wrap, self_root = _isolate_paths(tmp_path)

    scan_only = [
        LayerZeroRule(name="capability_scan_tick", predicate=lambda s: True,
                      lever="FIRE_CAPABILITY_SCAN", params={}, cooldown_seconds=21600.0),
    ]
    tick, lever_log = _build_tick(tmp_path, rules=scan_only)

    with patch.object(scan_cls, "run", scan_wrap):
        tick()

    assert (self_root / "tools").exists()
    assert (self_root / "server_inventory.md").exists()
    assert (self_root / "mcp_servers" / "channels.md").exists()
