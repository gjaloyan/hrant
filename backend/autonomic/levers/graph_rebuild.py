"""FIRE_GRAPH_REBUILD — re-derive the knowledge graph from
memory_facts.jsonl when graph coverage drifts.

Audit 2026-05-27 found 12 graph nodes vs 1453 facts on prod. The
v2 graph (`backend/graph/builder.py`) only grows during
consolidation; facts written via legacy paths never appear. This
lever runs `graph.builder.rebuild()` periodically and on detected
drift, keeping the graph in sync.
"""
from __future__ import annotations

import json
import logging
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


# Below this coverage ratio (graph_nodes / fact_count), the graph
# is considered drifted. 0.5 = "less than half the facts have a
# corresponding node". The graph includes topic + entity nodes
# in addition to fact nodes, so a healthy graph usually has >1x
# nodes-to-facts; <0.5 is unambiguous drift.
DRIFT_RATIO_THRESHOLD = 0.5


def _is_drifted(*, facts: int, nodes: int) -> bool:
    """True when the graph has measurably fewer nodes than facts —
    suggesting the rebuild source has grown past what the graph
    captured. Cold start (0 facts) is never drifted."""
    if facts <= 0:
        return False
    return nodes < (facts * DRIFT_RATIO_THRESHOLD)


def _count_facts() -> int:
    """Number of fact records in memory_facts.jsonl. Returns 0 if
    the file is missing or unreadable."""
    from backend import paths
    p = paths.knowledge_dir() / "memory_facts.jsonl"
    if not p.exists():
        return 0
    try:
        return sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                   if line.strip())
    except OSError:
        return 0


def _count_graph_nodes() -> int:
    """Number of nodes in the current v2 graph file. Returns 0 if
    the file is missing/empty/unparseable."""
    from backend import paths
    p = paths.knowledge_dir() / "graph.json"
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    nodes = data.get("nodes")
    if isinstance(nodes, dict):
        return len(nodes)
    if isinstance(nodes, list):
        return len(nodes)
    return 0


def _do_rebuild() -> dict:
    """Re-derive the graph from current sources. Wraps the
    builder so the lever can be mocked in tests."""
    from backend.graph import builder
    return builder.rebuild()


class FIRE_GRAPH_REBUILD(Lever):
    name = "FIRE_GRAPH_REBUILD"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "local"  # no LLM
    estimated_cost = Cost(seconds=2.0, tokens_in=0, tokens_out=0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        force = bool(params.get("force"))

        facts_before = _count_facts()
        nodes_before = _count_graph_nodes()

        if not force and not _is_drifted(facts=facts_before, nodes=nodes_before):
            return self._skip(
                params, started,
                f"no_drift (facts={facts_before}, nodes={nodes_before})",
            )

        try:
            stats = _do_rebuild() or {}
        except Exception as exc:
            log.warning("graph_rebuild: rebuild raised: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"exception": str(exc)},
                reason=f"rebuild_failed: {exc}",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "facts_before": facts_before,
                "graph_nodes_before": nodes_before,
                "stats": dict(stats),
                "forced": force,
            },
            reason=(
                f"rebuilt:{stats.get('facts')}_facts_"
                f"{stats.get('topics', 0) + stats.get('skills', 0)}_topics+skills"
                + (" (forced)" if force else "")
            ),
        )

    def _skip(
        self, params: dict[str, Any], started, reason: str,
    ) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
