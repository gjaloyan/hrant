"""Intelligence panel: graph, meta-learner, eval, usage, memory, analogies, self-modifier."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..analogy_engine import ANALOGIES
from ..evaluator import EVALUATOR
from ..knowledge_graph import GRAPH, reindex_all_notes
from ..llm import TOKENS
from ..memory_extractor import MEMORY
from ..meta_learner import META_LEARNER
from ..self_modifier import SELF_MODIFIER

router = APIRouter()


# ---- graph ----
@router.get("/api/graph")
def graph_stats():
    return GRAPH.stats()


@router.post("/api/graph/reindex")
def graph_reindex():
    return reindex_all_notes()


@router.get("/api/graph/entities")
def graph_entities():
    """List all entities with their edge counts."""
    entities: dict[str, int] = {}
    for entity, edges in GRAPH._edges.items():
        entities[entity] = len(edges)
    sorted_ents = sorted(entities.items(), key=lambda x: -x[1])
    return {"entities": [{"name": n, "edges": c} for n, c in sorted_ents]}


@router.get("/api/graph/neighbors/{entity}")
def graph_neighbors(entity: str):
    return {"entity": entity, "neighbors": GRAPH.get_neighbors(entity)}


@router.get("/api/graph/full")
def graph_full():
    """Return the entire graph as nodes + links for force-directed visualization."""
    all_entities: set[str] = set()
    links = []
    for source, edges in GRAPH._edges.items():
        all_entities.add(source)
        for edge in edges:
            target = edge["target"]
            all_entities.add(target)
            if edge["relation"].startswith("inverse:"):
                continue
            links.append({
                "source": source,
                "target": target,
                "relation": edge["relation"],
                "note": edge.get("note", ""),
                "weight": edge.get("weight", 1.0),
            })
    conn_count: dict[str, int] = {e: 0 for e in all_entities}
    for link in links:
        conn_count[link["source"]] = conn_count.get(link["source"], 0) + 1
        conn_count[link["target"]] = conn_count.get(link["target"], 0) + 1
    nodes = [
        {"id": e, "name": e, "connections": conn_count.get(e, 0)}
        for e in sorted(all_entities)
    ]
    return {"nodes": nodes, "links": links}


# ---- meta-learner ----
@router.get("/api/meta-learner")
def meta_learner_stats():
    return META_LEARNER.stats()


@router.get("/api/meta-learner/failures")
def meta_learner_failures():
    return {"failures": META_LEARNER.recent_failures(limit=30)}


@router.post("/api/meta-learner/extract-patterns")
def meta_learner_extract():
    return {"patterns": META_LEARNER.extract_patterns()}


# ---- evaluator ----
@router.get("/api/eval")
def eval_stats():
    return EVALUATOR.stats()


@router.get("/api/eval/today")
def eval_today():
    return EVALUATOR.daily_report()


@router.get("/api/eval/trend")
def eval_trend():
    return {"trend": EVALUATOR.weekly_trend()}


@router.get("/api/eval/regressions")
def eval_regressions():
    return {"regressions": EVALUATOR.detect_regression()}


@router.get("/api/eval/suggestions")
def eval_suggestions():
    return {"suggestions": EVALUATOR.suggest_priorities()}


# ---- token usage ----
@router.get("/api/usage")
def usage_stats():
    return TOKENS.stats()


@router.get("/api/usage/calls")
def usage_calls(limit: int = 50):
    return {"calls": TOKENS.recent_calls(limit=limit)}


@router.get("/api/usage/traces")
def usage_traces(limit: int = 20):
    return {"traces": TOKENS.recent_traces(limit=limit)}


# ---- memory (conversation facts) ----
@router.get("/api/memory")
def memory_stats():
    return MEMORY.stats()


@router.get("/api/memory/facts")
def memory_facts(limit: int = 50):
    return {"facts": MEMORY.recent_facts(limit=limit)}


class MemoryRecallRequest(BaseModel):
    query: str
    limit: int = 10


@router.post("/api/memory/recall")
def memory_recall(body: MemoryRecallRequest):
    facts = MEMORY.recall(body.query, limit=body.limit)
    block = MEMORY.recall_block(body.query, max_facts=body.limit)
    return {"facts": facts, "block": block}


# ---- analogy engine ----
@router.get("/api/analogies")
def analogy_list():
    return {"patterns": ANALOGIES.all_patterns(), "stats": ANALOGIES.stats()}


# ---- self-modifier ----
@router.get("/api/self-modifier")
def self_modifier_stats():
    return {**SELF_MODIFIER.stats(), "modules": SELF_MODIFIER.available_modules()}


@router.get("/api/self-modifier/proposals")
def self_modifier_proposals(status: str | None = None):
    return {"proposals": SELF_MODIFIER.list_proposals(status)}


class AnalyzeModuleRequest(BaseModel):
    module: str


@router.post("/api/self-modifier/analyze")
def self_modifier_analyze(body: AnalyzeModuleRequest):
    proposals = SELF_MODIFIER.analyze_module(body.module)
    return {"proposals": [p.to_dict() for p in proposals]}


class ReviewRequest(BaseModel):
    note: str = ""


@router.post("/api/self-modifier/proposals/{proposal_id}/approve")
def self_modifier_approve(proposal_id: str, body: ReviewRequest):
    if not SELF_MODIFIER.approve(proposal_id, body.note):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


@router.post("/api/self-modifier/proposals/{proposal_id}/reject")
def self_modifier_reject(proposal_id: str, body: ReviewRequest):
    if not SELF_MODIFIER.reject(proposal_id, body.note):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


@router.post("/api/self-modifier/proposals/{proposal_id}/apply")
def self_modifier_apply(proposal_id: str):
    result = SELF_MODIFIER.apply(proposal_id)
    if not result["ok"]:
        raise HTTPException(400, result["message"])
    return result


@router.delete("/api/self-modifier/proposals/{proposal_id}")
def self_modifier_delete(proposal_id: str):
    if not SELF_MODIFIER.delete_proposal(proposal_id):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}
