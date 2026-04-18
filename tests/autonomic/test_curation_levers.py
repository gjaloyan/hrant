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


from backend.autonomic.levers.note_curation import FIRE_NOTE_CURATION


def _index_entry(topic: str, category: str, confidence: str = "verified",
                 updated: str = "2026-04-18 12:00", access_count: int = 0) -> dict:
    return {
        "topic": topic,
        "category": category,
        "path": f"knowledge/{category}/{topic}.md",
        "keywords": [],
        "access_count": access_count,
        "updated": updated,
        "project": None,
        "confidence": confidence,
    }


def test_note_curation_metadata():
    lever = FIRE_NOTE_CURATION()
    assert lever.name == "FIRE_NOTE_CURATION"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_note_curation_empty_index_skips(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_NOTE_CURATION()
    report = lever.run({
        "index_path": str(tmp_path / "index.json"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_stale_notes"


def test_note_curation_picks_partial_confidence_first(tmp_path: Path):
    idx = {
        "verified_note": _index_entry("verified_note", "profession", "verified"),
        "partial_note": _index_entry("partial_note", "profession", "partial"),
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured_topics: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured_topics.append(topic)
        class _Note:
            class frontmatter:
                pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 2,
        }, {})

    assert report.status == LeverStatus.SUCCESS
    assert "partial_note" in captured_topics


def test_note_curation_picks_stale_hot_notes(tmp_path: Path):
    idx = {
        "cold_old": _index_entry("cold_old", "profession", "verified",
                                 updated="2024-01-01 00:00", access_count=1),
        "hot_old": _index_entry("hot_old", "profession", "verified",
                                updated="2024-01-01 00:00", access_count=10),
        "hot_fresh": _index_entry("hot_fresh", "profession", "verified",
                                  updated="2026-04-18 12:00", access_count=10),
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append(topic)
        class _Note:
            class frontmatter: pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 5,
        }, {})

    assert captured == ["hot_old"]


def test_note_curation_excludes_personal_and_projects(tmp_path: Path):
    idx = {
        "personal_partial": _index_entry("personal_partial", "personal", "partial"),
        "projects_partial": _index_entry("projects_partial", "projects", "partial"),
        "profession_partial": _index_entry("profession_partial", "profession", "partial"),
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append(topic)
        class _Note:
            class frontmatter: pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 5,
        }, {})

    assert captured == ["profession_partial"]
    assert report.outcome["candidates"] == 1


def test_note_curation_caps_at_max_per_tick(tmp_path: Path):
    idx = {
        f"partial_{i}": _index_entry(f"partial_{i}", "profession", "partial")
        for i in range(5)
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append(topic)
        class _Note:
            class frontmatter: pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 2,
        }, {})

    assert len(captured) == 2
    assert report.outcome["refreshed"] == 2
    assert report.outcome["candidates"] == 5
