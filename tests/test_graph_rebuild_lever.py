"""Autonomic lever that keeps the knowledge graph in sync with
memory_facts.jsonl.

The v2 graph (`backend/graph/builder.py`) is populated incrementally
from consolidation runs only. Facts written via other paths (legacy
memory_extractor, manual imports, schema migrations) never reach the
graph. The 2026-05-27 audit found 12 graph nodes vs 1453 facts on
prod — a 99% gap.

`FIRE_GRAPH_REBUILD` detects drift (graph noticeably smaller than
the facts file) and calls `graph.builder.rebuild()` to rederive from
source. Rebuild is idempotent and fast (<1s at ~1k facts).
"""
from __future__ import annotations


def test_lever_registered_in_autonomic_defaults():
    from backend.autonomic.levers import (
        register_default_autonomic_levers, clear_registry, list_levers,
    )
    clear_registry()
    try:
        register_default_autonomic_levers()
        assert "FIRE_GRAPH_REBUILD" in list_levers()
    finally:
        clear_registry()


def test_lever_skip_when_no_drift(monkeypatch):
    """Graph already covers the facts (node count ≈ fact count).
    No work needed."""
    from backend.autonomic.levers.graph_rebuild import FIRE_GRAPH_REBUILD
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_facts",
        lambda: 100,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_graph_nodes",
        lambda: 95,  # within tolerance
    )
    rebuild_called = {"n": 0}
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._do_rebuild",
        lambda: rebuild_called.update({"n": rebuild_called["n"] + 1}),
    )

    lever = FIRE_GRAPH_REBUILD()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert rebuild_called["n"] == 0


def test_lever_rebuild_when_graph_smaller_than_facts(monkeypatch):
    """Massive drift (12 vs 1453 — the 2026-05-27 prod state) →
    rebuild fires."""
    from backend.autonomic.levers.graph_rebuild import FIRE_GRAPH_REBUILD
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._is_legacy_schema",
        lambda: False,  # v2 schema present — drift gate is the only check
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_facts",
        lambda: 1453,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_graph_nodes",
        lambda: 12,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._do_rebuild",
        lambda: {"facts": 1453, "topics": 230, "edges": 2900,
                 "skills": 4, "projects": 0},
    )

    lever = FIRE_GRAPH_REBUILD()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["facts_before"] == 1453
    assert report.outcome["graph_nodes_before"] == 12
    assert report.outcome["stats"]["facts"] == 1453


def test_lever_force_param_skips_drift_check(monkeypatch):
    """Operator can force a rebuild via params even when graph and
    facts are aligned — used after a model swap or schema change."""
    from backend.autonomic.levers.graph_rebuild import FIRE_GRAPH_REBUILD
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_facts",
        lambda: 100,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_graph_nodes",
        lambda: 95,
    )
    called: dict = {}
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._do_rebuild",
        lambda: called.update({"fired": True}) or {"facts": 100, "edges": 250},
    )

    lever = FIRE_GRAPH_REBUILD()
    report = lever.run({"force": True}, {})
    assert report.status == LeverStatus.SUCCESS
    assert called.get("fired") is True


def test_lever_handles_rebuild_exception(monkeypatch):
    """A rebuild failure shouldn't crash the autonomic loop —
    surface it via FAILURE status with the exception text."""
    from backend.autonomic.levers.graph_rebuild import FIRE_GRAPH_REBUILD
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._is_legacy_schema",
        lambda: False,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_facts",
        lambda: 1453,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_graph_nodes",
        lambda: 12,
    )

    def _explode():
        raise RuntimeError("disk full")
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._do_rebuild", _explode,
    )

    lever = FIRE_GRAPH_REBUILD()
    report = lever.run({}, {})
    assert report.status == LeverStatus.FAILURE
    assert "disk full" in report.reason or "disk full" in str(report.outcome)


def test_lever_skip_when_no_facts(monkeypatch):
    """Cold-start case: no facts file → no drift, no rebuild."""
    from backend.autonomic.levers.graph_rebuild import FIRE_GRAPH_REBUILD
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_facts",
        lambda: 0,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._count_graph_nodes",
        lambda: 0,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._do_rebuild",
        lambda: {"facts": 0, "edges": 0},
    )

    lever = FIRE_GRAPH_REBUILD()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED


def test_drift_threshold_proportional(monkeypatch):
    """The drift check is proportional, not absolute. 50 graph nodes
    vs 60 facts is fine (16% gap), but 12 vs 1453 (99%) isn't.
    Threshold lives in module so test pins the contract."""
    from backend.autonomic.levers import graph_rebuild as gr
    # _is_drifted should be: graph_nodes < 0.5 * fact_count (less than
    # half coverage triggers rebuild). Exact threshold can move; the
    # invariant being pinned is "linear-relationship, not constant".
    assert gr._is_drifted(facts=1000, nodes=100) is True   # 10% — drift
    assert gr._is_drifted(facts=1000, nodes=900) is False  # 90% — fine
    assert gr._is_drifted(facts=10, nodes=2) is True       # 20% — drift
    assert gr._is_drifted(facts=0, nodes=0) is False       # cold start


def test_lever_skips_on_legacy_schema(monkeypatch, tmp_path):
    """Prod 2026-05-27 graph.json is the legacy KnowledgeGraph format
    (only `edges` dict, no `nodes`). The v2 rebuild would WIPE the
    legacy edges produced by memory_extractor / knowledge_manager —
    not all of which are derivable from memory_facts.jsonl. Until
    writers are reconciled, the lever must skip on legacy schema."""
    import json
    from backend.autonomic.levers.graph_rebuild import FIRE_GRAPH_REBUILD
    from backend.autonomic.types import LeverStatus

    legacy_graph = tmp_path / "graph.json"
    legacy_graph.write_text(json.dumps({
        "edges": {
            "user wife": [
                {"target": "wife", "relation": "has",
                 "note": "_memory", "weight": 0.9},
            ],
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        "backend.paths.knowledge_dir", lambda: tmp_path,
    )
    # Rebuild itself should NOT be called.
    rebuild_called = {"n": 0}
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._do_rebuild",
        lambda: rebuild_called.update({"n": rebuild_called["n"] + 1}),
    )

    lever = FIRE_GRAPH_REBUILD()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "legacy_schema" in report.reason
    assert rebuild_called["n"] == 0


def test_lever_force_overrides_legacy_schema_skip(monkeypatch, tmp_path):
    """Operator can explicitly opt into the migration via
    `params={"force": True}` — the lever runs the rebuild even on
    a legacy file. This is the migration trigger."""
    import json
    from backend.autonomic.levers.graph_rebuild import FIRE_GRAPH_REBUILD
    from backend.autonomic.types import LeverStatus

    legacy_graph = tmp_path / "graph.json"
    legacy_graph.write_text(
        json.dumps({"edges": {"x": [{"target": "y", "note": "_m"}]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.paths.knowledge_dir", lambda: tmp_path,
    )
    monkeypatch.setattr(
        "backend.autonomic.levers.graph_rebuild._do_rebuild",
        lambda: {"facts": 100, "edges": 250},
    )

    lever = FIRE_GRAPH_REBUILD()
    report = lever.run({"force": True}, {})
    assert report.status == LeverStatus.SUCCESS
