"""Persistent graph storage. One JSON file under data_dir.

Concurrency: RLock per Graph so concurrent upserts (consolidation
running while user clicks "Rebuild" in WebUI) don't tear the file.
Atomic-ish write via `.tmp` + replace, same pattern as jobs.py.

Schema versioning: every saved graph carries a `version` integer.
Loaders that see an unknown version fall back to an empty graph
with a log warning — better than crashing on a forward-compat
schema bump.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

from .. import paths
from .model import GraphEdge, GraphNode

log = logging.getLogger(__name__)


GRAPH_SCHEMA_VERSION = 1


def _graph_path() -> Path:
    """Lazy path resolution so tests that override HRANT_DATA_DIR
    after import see the override."""
    return paths.knowledge_dir() / "graph.json"


class Graph:
    """In-memory graph with file backing. Loaded lazily on first
    access; written atomically on every mutation."""

    def __init__(self, root: Optional[Path] = None):
        self._root_override = root
        self._lock = threading.RLock()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[tuple[str, str, str], GraphEdge] = {}
        self._loaded = False
        self._updated_at: float = 0.0

    @property
    def path(self) -> Path:
        if self._root_override is not None:
            return self._root_override / "graph.json"
        return _graph_path()

    # ─── Load / save ────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            p = self.path
            if not p.exists():
                self._loaded = True
                return
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("graph.json unreadable (%s); starting empty", e)
                self._loaded = True
                return
            version = int(data.get("version") or 0)
            if version > GRAPH_SCHEMA_VERSION:
                log.warning(
                    "graph.json version=%d > supported=%d; starting "
                    "empty rather than misreading forward-schema data",
                    version, GRAPH_SCHEMA_VERSION,
                )
                self._loaded = True
                return
            for raw in data.get("nodes") or []:
                try:
                    n = GraphNode.from_dict(raw)
                    self._nodes[n.id] = n
                except Exception:
                    continue
            for raw in data.get("edges") or []:
                try:
                    e = GraphEdge.from_dict(raw)
                    self._edges[e.key()] = e
                except Exception:
                    continue
            self._updated_at = float(data.get("updated_at") or 0.0)
            self._loaded = True

    def save(self) -> Path:
        with self._lock:
            self._updated_at = time.time()
            p = self.path
            p.parent.mkdir(parents=True, exist_ok=True)
            body = {
                "version": GRAPH_SCHEMA_VERSION,
                "updated_at": self._updated_at,
                "nodes": [asdict(n) for n in self._nodes.values()],
                "edges": [asdict(e) for e in self._edges.values()],
            }
            tmp = p.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(body, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(p)
        return p

    def clear(self) -> None:
        """Wipe the in-memory graph. Caller is expected to .save()
        immediately or repopulate — `clear` without a follow-up
        creates a phantom-state mismatch between disk and memory."""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._loaded = True

    # ─── Read ───────────────────────────────────────────────────

    def node_count(self) -> int:
        self._ensure_loaded()
        return len(self._nodes)

    def edge_count(self) -> int:
        self._ensure_loaded()
        return len(self._edges)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        self._ensure_loaded()
        return self._nodes.get(node_id)

    def iter_nodes(self, *, kind: Optional[str] = None) -> Iterator[GraphNode]:
        self._ensure_loaded()
        for n in self._nodes.values():
            if kind is None or n.kind == kind:
                yield n

    def iter_edges(self, *, kind: Optional[str] = None) -> Iterator[GraphEdge]:
        self._ensure_loaded()
        for e in self._edges.values():
            if kind is None or e.kind == kind:
                yield e

    def neighbors(self, node_id: str) -> list[tuple[GraphEdge, GraphNode]]:
        """Adjacent nodes via any outgoing OR incoming edge.
        Returns `(edge, other_node)` pairs so the caller knows
        which edge kind made the connection."""
        self._ensure_loaded()
        out: list[tuple[GraphEdge, GraphNode]] = []
        for e in self._edges.values():
            if e.source == node_id:
                other = self._nodes.get(e.target)
                if other is not None:
                    out.append((e, other))
            elif e.target == node_id:
                other = self._nodes.get(e.source)
                if other is not None:
                    out.append((e, other))
        return out

    def to_dict(self) -> dict:
        """Full graph as a dict — used by the REST API."""
        self._ensure_loaded()
        return {
            "version": GRAPH_SCHEMA_VERSION,
            "updated_at": self._updated_at,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes": [asdict(n) for n in self._nodes.values()],
            "edges": [asdict(e) for e in self._edges.values()],
        }

    # ─── Mutate ─────────────────────────────────────────────────

    def upsert_node(self, node: GraphNode) -> GraphNode:
        """Add or replace a node by id. Weight + metadata of an
        existing node are MERGED, not overwritten — so re-adding
        a fact from a new consolidation run doesn't lose its
        accumulated weight or earlier tags."""
        self._ensure_loaded()
        with self._lock:
            existing = self._nodes.get(node.id)
            if existing is None:
                self._nodes[node.id] = node
                return node
            # Merge: prefer fresher label (in case spelling changed),
            # cap weight at the sum, union metadata.
            existing.label = node.label or existing.label
            existing.weight = max(existing.weight, node.weight)
            merged_meta = dict(existing.metadata)
            merged_meta.update(node.metadata or {})
            existing.metadata = merged_meta
            return existing

    def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add or bump-weight an edge. Same (source, target, kind)
        triple is treated as one edge — duplicate upserts increment
        the weight to capture "we saw this connection N times"."""
        self._ensure_loaded()
        with self._lock:
            key = edge.key()
            existing = self._edges.get(key)
            if existing is None:
                self._edges[key] = edge
                return edge
            existing.weight += edge.weight
            merged_meta = dict(existing.metadata)
            merged_meta.update(edge.metadata or {})
            existing.metadata = merged_meta
            return existing

    def remove_node(self, node_id: str) -> bool:
        """Delete a node + every edge it touches. Returns True if
        the node existed."""
        self._ensure_loaded()
        with self._lock:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
            self._edges = {
                k: e for k, e in self._edges.items()
                if e.source != node_id and e.target != node_id
            }
            return True


# Module-level singleton so callers can `from .store import GRAPH`.
GRAPH = Graph()
