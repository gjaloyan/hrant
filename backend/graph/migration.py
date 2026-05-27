"""One-shot migration from legacy KnowledgeGraph schema to v2.

Audit 2026-05-27 found `knowledge/graph.json` on prod is in the
legacy format: only an `edges` dict keyed by subject string.

`backend.graph.builder.rebuild()` writes a different shape
(version+nodes+edges as lists). Without coordination, the two
writers stomp on each other. This module bridges them:

  1. `migrate_legacy_to_v2(path)` reads a legacy file and rewrites
     it with v2 fields alongside the legacy `edges` dict. The
     result is readable by BOTH legacy and v2 paths.
  2. `KnowledgeGraph._save()` is updated (in knowledge_graph.py)
     to preserve v2 fields when round-tripping the file.

Idempotent: running migrate twice produces an identical result.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


SCHEMA_VERSION = 2


def _entity_id(label: str) -> str:
    """Stable id derived from the label. Matches the convention in
    graph.model.entity_id without the import dance — we want this
    migration to work even if the v2 graph module isn't loaded."""
    import hashlib
    h = hashlib.sha1((label or "").strip().lower().encode("utf-8"))
    return f"e_{h.hexdigest()[:12]}"


def migrate_legacy_to_v2(legacy_path: Path) -> dict:
    """Read a legacy graph.json (only `edges` dict) and rewrite it
    in a hybrid shape that both schemas understand:
      - `version`: 2
      - `nodes`: list[GraphNode-ish dicts] — one per unique label
      - `edges` (legacy dict): preserved untouched
      - `edges_v2`: list of v2 edges derived from the legacy dict

    Returns stats: {entity_nodes, v2_edges, already_migrated}.
    Idempotent — second call returns already_migrated=True with no
    file change.
    """
    stats = {"entity_nodes": 0, "v2_edges": 0, "already_migrated": False}
    if not legacy_path.exists():
        return stats

    try:
        data = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("migrate_legacy_to_v2: read %s failed: %s", legacy_path, exc)
        return stats

    if not isinstance(data, dict):
        return stats

    legacy_edges = data.get("edges")
    if not isinstance(legacy_edges, dict):
        # Already migrated (edges is a list) or empty file.
        if data.get("version") == SCHEMA_VERSION:
            stats["already_migrated"] = True
        return stats

    # If already-migrated metadata is present AND we'd produce the
    # same content, signal idempotence without rewriting.
    if (
        data.get("version") == SCHEMA_VERSION
        and isinstance(data.get("nodes"), list)
        and isinstance(data.get("edges_v2"), list)
    ):
        stats["already_migrated"] = True
        stats["entity_nodes"] = len(data["nodes"])
        stats["v2_edges"] = len(data["edges_v2"])
        return stats

    # Build v2 nodes: one per subject + one per unique target label.
    labels: set[str] = set()
    v2_edges: list[dict] = []
    for subject, edge_list in legacy_edges.items():
        if not isinstance(edge_list, list):
            continue
        labels.add(subject)
        for edge in edge_list:
            if not isinstance(edge, dict):
                continue
            target = edge.get("target") or ""
            relation = edge.get("relation") or "related_to"
            if not target:
                continue
            labels.add(target)
            v2_edges.append({
                "source": _entity_id(subject),
                "target": _entity_id(target),
                "kind": f"legacy_{relation}",
                "weight": float(edge.get("weight") or 1.0),
                "metadata": {
                    "note": edge.get("note", ""),
                    "predicate": relation,
                },
            })

    nodes = [
        {
            "id": _entity_id(label),
            "kind": "entity",
            "label": label,
            "weight": 1.0,
            "metadata": {"source": "legacy_migration"},
        }
        for label in sorted(labels)
    ]

    data["version"] = SCHEMA_VERSION
    data["nodes"] = nodes
    data["edges_v2"] = v2_edges
    # `edges` (legacy dict) stays untouched.

    tmp = legacy_path.with_suffix(legacy_path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(legacy_path)
    except OSError as exc:
        log.warning("migrate_legacy_to_v2: write %s failed: %s", legacy_path, exc)
        return stats

    stats["entity_nodes"] = len(nodes)
    stats["v2_edges"] = len(v2_edges)
    return stats
