"""FIRE_GRAPH_MAINTENANCE — prune dead edges and orphan entities from knowledge/graph.json."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

log = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = Path("knowledge/graph.json")
DEFAULT_INDEX_PATH = Path("knowledge/index.json")


class FIRE_GRAPH_MAINTENANCE(Lever):
    name = "FIRE_GRAPH_MAINTENANCE"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.3)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        graph_path = Path(params.get("graph_path") or DEFAULT_GRAPH_PATH)
        index_path = Path(params.get("index_path") or DEFAULT_INDEX_PATH)

        if not graph_path.exists():
            return self._skip(params, started, "empty_graph")
        try:
            graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._skip(params, started, "empty_graph")
        edges = graph_data.get("edges", {})
        if not edges:
            return self._skip(params, started, "empty_graph")

        known_slugs: set[str] = set()
        if index_path.exists():
            try:
                idx = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(idx, dict):
                    known_slugs = set(idx.keys())
            except json.JSONDecodeError:
                known_slugs = set()

        edges_before = sum(len(v) for v in edges.values())
        entities_before = len(edges)

        pruned_edges: dict[str, list[dict]] = {}
        for entity, edge_list in edges.items():
            kept = [e for e in edge_list if e.get("note") in known_slugs]
            if kept:
                pruned_edges[entity] = kept

        referenced_targets: set[str] = set()
        for edge_list in pruned_edges.values():
            for e in edge_list:
                referenced_targets.add(e.get("target", ""))
        surviving_entities = {
            entity: edge_list
            for entity, edge_list in pruned_edges.items()
            if edge_list or entity in referenced_targets
        }

        edges_after = sum(len(v) for v in surviving_entities.values())
        entities_after = len(surviving_entities)
        edges_removed = edges_before - edges_after
        entities_removed = entities_before - entities_after

        if edges_removed > 0 or entities_removed > 0:
            graph_data["edges"] = surviving_entities
            graph_path.write_text(
                json.dumps(graph_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "edges_before": edges_before,
                "edges_after": edges_after,
                "edges_removed": edges_removed,
                "entities_before": entities_before,
                "entities_after": entities_after,
                "entities_removed": entities_removed,
            },
            reason=f"graph_pruned:{edges_removed}_edges",
        )

    def _skip(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
