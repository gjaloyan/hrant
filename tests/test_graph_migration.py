"""Migrate the legacy KnowledgeGraph schema (edges-only dict) into
the v2 Graph schema (version+nodes+edges lists).

Audit 2026-05-27 found graph.json on prod is in legacy format: only
an `edges` dict keyed by subject string. Both writers
(`knowledge_graph.py` legacy, `graph/store.py` v2) target the same
file with incompatible schemas. The legacy writer overwrites any
v2 fields on save, so a one-shot rebuild would be undone by the
next memory_extractor / knowledge_manager call.

This migration:
  1. Reads the legacy edges dict.
  2. Emits v2 entity nodes (one per subject + one per unique target).
  3. Emits v2 edges with kind = "legacy_{relation}".
  4. Preserves the legacy `edges` field so legacy callers keep
     working. The v2 fields ride alongside.

Plus a tweak to `knowledge_graph.KnowledgeGraph._save` to preserve
unknown fields on round-trip — so once migrated, legacy writes
don't strip the v2 data again.
"""
from __future__ import annotations

import json


def test_migration_emits_v2_nodes_for_each_subject(tmp_path):
    from backend.graph.migration import migrate_legacy_to_v2

    legacy = tmp_path / "graph.json"
    legacy.write_text(json.dumps({
        "edges": {
            "user wife": [
                {"target": "wife", "relation": "has_relation",
                 "note": "_memory", "weight": 0.9},
            ],
            "wife job": [
                {"target": "pharmaceutical lab", "relation": "works_in",
                 "note": "_memory", "weight": 0.85},
            ],
        },
    }), encoding="utf-8")

    stats = migrate_legacy_to_v2(legacy)
    data = json.loads(legacy.read_text(encoding="utf-8"))
    assert "version" in data
    assert isinstance(data.get("nodes"), list)
    assert isinstance(data.get("edges"), (list, dict))
    # Subjects + unique targets become entity nodes (3 here:
    # 'user wife', 'wife job', 'wife', 'pharmaceutical lab' — wait,
    # 'wife' is both a subject substring and a target. Migration is
    # by exact label match so they're separate. 4 entity nodes total.)
    labels = {n["label"] for n in data["nodes"]}
    assert "user wife" in labels
    assert "wife job" in labels
    assert "wife" in labels
    assert "pharmaceutical lab" in labels
    assert stats["entity_nodes"] >= 4
    assert stats["v2_edges"] == 2


def test_migration_preserves_legacy_edges_dict(tmp_path):
    """The legacy `edges` dict must survive so existing legacy
    callers (knowledge_manager, memory_extractor) keep working."""
    from backend.graph.migration import migrate_legacy_to_v2

    legacy = tmp_path / "graph.json"
    original = {
        "edges": {
            "user wife": [
                {"target": "wife", "relation": "has_relation",
                 "note": "_memory", "weight": 0.9},
            ],
        },
    }
    legacy.write_text(json.dumps(original), encoding="utf-8")

    migrate_legacy_to_v2(legacy)
    data = json.loads(legacy.read_text(encoding="utf-8"))
    # The legacy dict (or its equivalent) lives under a preserved
    # key so the round-trip is non-destructive.
    assert "edges_legacy" in data or isinstance(data.get("edges"), dict)


def test_migration_idempotent(tmp_path):
    """Running migration twice produces the same result — safe to
    re-run after a backup or after the operator forgot they did it."""
    from backend.graph.migration import migrate_legacy_to_v2

    legacy = tmp_path / "graph.json"
    legacy.write_text(json.dumps({
        "edges": {"a": [{"target": "b", "relation": "r", "note": "_n",
                         "weight": 1.0}]},
    }), encoding="utf-8")

    s1 = migrate_legacy_to_v2(legacy)
    body1 = legacy.read_text(encoding="utf-8")
    s2 = migrate_legacy_to_v2(legacy)
    body2 = legacy.read_text(encoding="utf-8")
    assert body1 == body2
    assert s2["already_migrated"] is True


def test_migration_skip_when_no_file(tmp_path):
    """If the legacy file doesn't exist, return clean stats — never
    crash."""
    from backend.graph.migration import migrate_legacy_to_v2

    stats = migrate_legacy_to_v2(tmp_path / "missing.json")
    assert stats["entity_nodes"] == 0
    assert stats["v2_edges"] == 0


def test_legacy_save_preserves_v2_fields(tmp_path, monkeypatch):
    """After migration, a subsequent legacy `_save()` must NOT strip
    the v2 `nodes` / `version` fields — otherwise the migration
    gets undone immediately."""
    from backend import paths as _paths
    monkeypatch.setattr(_paths, "knowledge_dir", lambda: tmp_path)

    legacy_path = tmp_path / "graph.json"
    legacy_path.write_text(json.dumps({
        "version": 2,
        "nodes": [{"id": "n1", "kind": "entity", "label": "x",
                   "weight": 1.0, "metadata": {}}],
        "edges": {
            "x": [{"target": "y", "relation": "r", "note": "_n",
                   "weight": 1.0}],
        },
    }), encoding="utf-8")

    from backend.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph(path=legacy_path)
    # add a new edge → triggers _save()
    kg.add_relations(
        [("z", "r2", "y")], source_note="_test",
    )
    data = json.loads(legacy_path.read_text(encoding="utf-8"))
    # v2 fields survived the legacy write.
    assert data.get("version") == 2
    assert any(n.get("label") == "x" for n in data.get("nodes", []))
    # Legacy edges dict also has the new entry.
    assert "z" in data.get("edges", {})
