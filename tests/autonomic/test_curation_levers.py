import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.autonomic.levers.graph_maintenance import FIRE_GRAPH_MAINTENANCE
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, StateSnapshot


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


def test_graph_maintenance_metadata():
    lever = FIRE_GRAPH_MAINTENANCE()
    assert lever.name == "FIRE_GRAPH_MAINTENANCE"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_graph_maintenance_empty_graph_skips(tmp_path: Path):
    (tmp_path / "graph.json").write_text(json.dumps({"edges": {}}), encoding="utf-8")
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "empty_graph"


def test_graph_maintenance_missing_graph_skips(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "nope.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "empty_graph"


def test_graph_maintenance_prunes_edges_with_missing_note(tmp_path: Path):
    graph = {
        "edges": {
            "python": [
                {"target": "asyncio", "relation": "related_to", "note": "python_async", "weight": 1.0},
                {"target": "gil", "relation": "related_to", "note": "deleted_note", "weight": 1.0},
            ],
            "asyncio": [
                {"target": "python", "relation": "inverse:related_to", "note": "python_async", "weight": 0.5},
            ],
        }
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps({
        "python_async": {"topic": "python_async", "category": "profession"},
    }), encoding="utf-8")

    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["edges_removed"] == 1
    saved = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    edges = saved["edges"]["python"]
    assert len(edges) == 1
    assert edges[0]["note"] == "python_async"


def test_graph_maintenance_prunes_entities_with_no_edges(tmp_path: Path):
    graph = {
        "edges": {
            "orphan_entity": [
                {"target": "other", "relation": "rel", "note": "deleted", "weight": 1.0},
            ],
            "other": [
                {"target": "orphan_entity", "relation": "inverse:rel", "note": "deleted", "weight": 0.5},
            ],
        }
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})

    assert report.outcome["edges_removed"] == 2
    assert report.outcome["entities_removed"] == 2
    saved = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    assert saved["edges"] == {}


def test_graph_maintenance_idempotent(tmp_path: Path):
    graph = {
        "edges": {
            "python": [
                {"target": "asyncio", "relation": "related_to", "note": "python_async", "weight": 1.0},
            ],
            "asyncio": [
                {"target": "python", "relation": "inverse:related_to", "note": "python_async", "weight": 0.5},
            ],
        }
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps({
        "python_async": {"topic": "python_async", "category": "profession"},
    }), encoding="utf-8")

    lever = FIRE_GRAPH_MAINTENANCE()
    first = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})
    second = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})

    assert first.outcome["edges_removed"] == 0
    assert second.outcome["edges_removed"] == 0
    assert second.outcome["entities_removed"] == 0
