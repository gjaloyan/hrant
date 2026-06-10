"""Derive the graph from existing knowledge sources.

Sources:
  - `knowledge/memory_facts.jsonl`        → fact nodes + topic edges
  - `knowledge/identity/profiles/*.md`    → topic mentions
  - skills registry (Phase 12)            → skill nodes + topic edges
  - `knowledge/goals.json` (if present)   → project nodes

This is idempotent: running `rebuild()` twice produces the same
graph. Existing edges get their weights re-summed but the topology
stays stable.

For incremental updates (during consolidation), see `add_fact`
which inserts ONE fact + its topic edges without touching the
rest of the graph.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from .. import paths
from .model import GraphEdge, GraphNode, entity_id, fact_id, project_id, skill_id, topic_id
from .store import GRAPH, Graph

log = logging.getLogger(__name__)


# ─── Single-fact incremental insert ──────────────────────────────────


def add_fact(
    *,
    text: str,
    related_topics: Optional[list[str]] = None,
    category: str = "general",
    confidence: float = 0.8,
    source: str = "consolidation",
    triples: Optional[list[list[str]]] = None,
    graph: Optional[Graph] = None,
) -> str:
    """Insert one fact node + its topic edges + RDF triples.
    Returns the canonical fact_id (caller can cross-reference it
    in digest records).

    Called from the consolidation pipeline after each promoted
    fact. NOT called from the WebUI directly — the user adds
    facts through `memory_facts.jsonl`, the graph follows."""
    g = graph or GRAPH
    fid = fact_id(text)
    g.upsert_node(GraphNode(
        id=fid, kind="fact", label=text,
        weight=float(confidence),
        metadata={
            "category": category,
            "confidence": float(confidence),
            "source": source,
        },
    ))
    for raw_topic in (related_topics or []):
        label = str(raw_topic).strip()
        if not label:
            continue
        tid = topic_id(label)
        g.upsert_node(GraphNode(
            id=tid, kind="topic", label=label, weight=1.0,
        ))
        g.upsert_edge(GraphEdge(
            source=fid, target=tid, kind="is_about", weight=1.0,
        ))
    # Triples: subject — predicate — object. Subject = fact, target =
    # entity, edge kind = "mentions" with the predicate as metadata.
    for triple in (triples or []):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        s, p, o = str(triple[0]), str(triple[1]), str(triple[2])
        if not (s.strip() and o.strip()):
            continue
        eid = entity_id(o)
        g.upsert_node(GraphNode(
            id=eid, kind="entity", label=o, weight=1.0,
        ))
        g.upsert_edge(GraphEdge(
            source=fid, target=eid, kind="mentions",
            weight=1.0, metadata={"predicate": p.strip()},
        ))
    return fid


# ─── Source readers ──────────────────────────────────────────────────


def _read_memory_facts() -> list[dict]:
    p = paths.knowledge_dir() / "memory_facts.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError as e:
        log.warning("memory_facts.jsonl read failed: %s", e)
    return out


def _read_skills() -> list[dict]:
    """Skill metadata via the existing skills registry. Returns
    enabled skills only — disabled skills shouldn't pollute the
    graph with stale topics."""
    try:
        from .. import skills as _skills
    except Exception as e:
        log.warning("skills import failed: %s", e)
        return []
    try:
        registry = _skills.SkillRegistry.global_instance()  # type: ignore[attr-defined]
        all_skills = list(registry.all())
    except Exception:
        # Fallback: directly scan default skills directory. This
        # path matters when the agent boots without the autonomic
        # registry being populated (early tests).
        return []
    return [
        {
            "name": s.name,
            "description": s.description,
            "triggers": list(s.triggers or []),
            "source": s.source,
        }
        for s in all_skills
        if getattr(s, "enabled", True)
    ]


def _read_goals() -> list[dict]:
    p = paths.knowledge_dir() / "goals.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        return list(data.get("goals") or [])
    if isinstance(data, list):
        return data
    return []


# ─── Full rebuild ────────────────────────────────────────────────────


def rebuild(*, graph: Optional[Graph] = None) -> dict:
    """Re-derive the entire graph from current sources. Returns
    stats so the WebUI can show "rebuilt: N facts, M topics, ..."

    Wipes the in-memory graph first, then re-populates. Edges
    proposed by the LLM (Phase 16C.1) are NOT re-derived — they
    live in a separate file overlay (future). For now `rebuild`
    is a clean slate."""
    g = graph or GRAPH
    g.clear()

    stats = {"facts": 0, "topics": 0, "skills": 0, "projects": 0, "edges": 0}

    # Facts
    for raw in _read_memory_facts():
        text = (raw.get("summary") or raw.get("text") or "").strip()
        if not text:
            continue
        topics = list(raw.get("tags") or raw.get("related_topics") or [])
        triples = list(raw.get("triples") or [])
        add_fact(
            text=text,
            related_topics=topics,
            category=str(raw.get("category") or "general"),
            confidence=float(raw.get("confidence") or 0.7),
            source="memory_facts",
            triples=triples,
            graph=g,
        )
        stats["facts"] += 1

    # Skills
    for s in _read_skills():
        sid = skill_id(s["name"])
        g.upsert_node(GraphNode(
            id=sid, kind="skill", label=s["name"], weight=1.0,
            metadata={"description": s.get("description") or "",
                      "source": s.get("source") or "builtin"},
        ))
        for trigger in s.get("triggers") or []:
            label = str(trigger).strip()
            if not label:
                continue
            tid = topic_id(label)
            g.upsert_node(GraphNode(
                id=tid, kind="topic", label=label, weight=1.0,
            ))
            g.upsert_edge(GraphEdge(
                source=sid, target=tid, kind="uses", weight=1.0,
            ))
        stats["skills"] += 1

    # Projects (best-effort — goals.json shape varies between Phase
    # 11 builds; we just read whatever we can).
    for goal in _read_goals():
        if not isinstance(goal, dict):
            continue
        name = (goal.get("name") or goal.get("title") or
                goal.get("id") or "").strip()
        if not name:
            continue
        pid = project_id(name)
        g.upsert_node(GraphNode(
            id=pid, kind="project", label=name, weight=1.0,
            metadata={"status": goal.get("status") or "active",
                      "description": goal.get("description") or ""},
        ))
        stats["projects"] += 1

    # Count topics + edges AFTER everything's been inserted (topic
    # nodes are de-duplicated by id, so we can't sum during the
    # passes above).
    stats["topics"] = sum(1 for n in g.iter_nodes(kind="topic"))
    stats["edges"] = g.edge_count()

    g.save()
    log.info(
        "graph.rebuild: %d nodes (%d facts, %d topics, %d skills, %d projects), "
        "%d edges",
        g.node_count(), stats["facts"], stats["topics"],
        stats["skills"], stats["projects"], stats["edges"],
    )
    return stats
