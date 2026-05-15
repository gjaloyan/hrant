"""Intelligence panel: graph, meta-learner, eval, usage, memory, analogies, self-modifier."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..analogy_engine import ANALOGIES
from ..embedder import EMBEDDER, load_config as load_embedder_config, save_config as save_embedder_config
from ..embedding_backfill import backfill_embeddings, missing_count
from ..evaluator import EVALUATOR
from ..knowledge_graph import GRAPH, reindex_all_notes
from ..llm import TOKENS
from ..memory_extractor import MEMORY
from ..meta_learner import META_LEARNER
from ..self_modifier import SELF_MODIFIER
from ..vector_store import VECTOR_STORE
from ._auth import require_owner_for_writes

router = APIRouter()


# ---- graph ----
@router.get("/api/graph")
def graph_stats():
    return GRAPH.stats()


@router.post("/api/graph/reindex")
def graph_reindex():
    require_owner_for_writes(action="reindexing the notes graph")
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
    require_owner_for_writes(action="extracting meta-learner patterns")
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
    # Owner-only: memory_facts.jsonl can contain sensitive personal
    # info (preferences, relationships, credentials the user voiced
    # to the agent). Brute-force probing via this endpoint is a
    # memory-exfiltration vector if exposed.
    require_owner_for_writes(action="recalling memory")
    facts = MEMORY.recall(body.query, limit=body.limit)
    block = MEMORY.recall_block(body.query, max_facts=body.limit)
    return {"facts": facts, "block": block}


# ---- analogy engine ----
@router.get("/api/analogies")
def analogy_list():
    return {"patterns": ANALOGIES.all_patterns(), "stats": ANALOGIES.stats()}


# ---- embeddings / vector store ----
@router.get("/api/memory/embeddings/status")
def embeddings_status():
    return {
        "embedder": EMBEDDER.status(),
        "vector_store": VECTOR_STORE.stats(),
        "coverage": missing_count(),
    }


@router.post("/api/memory/embeddings/backfill")
def embeddings_backfill(force: bool = False):
    require_owner_for_writes(action="backfilling embeddings")
    return backfill_embeddings(force=force)


@router.get("/api/memory/embeddings/config")
def embeddings_config_get():
    return {"config": load_embedder_config()}


class EmbedderConfigUpdate(BaseModel):
    backend: str | None = None  # "auto" | "llama_cpp" | "ollama" | "openai" | "cohere" | "disabled"
    llama_cpp: dict | None = None    # {url, model}
    ollama: dict | None = None       # {url, model}
    openai: dict | None = None       # {provider_id, model}
    cohere: dict | None = None       # {model}


@router.put("/api/memory/embeddings/config")
def embeddings_config_put(body: EmbedderConfigUpdate):
    """Persist embedder config and re-probe. Returns the live status."""
    require_owner_for_writes(action="changing embedder config")
    cfg = body.model_dump(exclude_none=True)
    save_embedder_config(cfg)
    EMBEDDER.reset()
    return {
        "ok": True,
        "config": load_embedder_config(),
        "embedder": EMBEDDER.status(),
        "vector_store": VECTOR_STORE.stats(),
        "coverage": missing_count(),
    }


@router.post("/api/memory/embeddings/reset")
def embeddings_reset():
    """Drop cached backend so the next embed re-probes — call after a
    server outage to retry auto-fallback without restarting the agent."""
    require_owner_for_writes(action="resetting the embedder")
    EMBEDDER.reset()
    return {
        "ok": True,
        "embedder": EMBEDDER.status(),
        "vector_store": VECTOR_STORE.stats(),
        "coverage": missing_count(),
    }


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
    require_owner_for_writes(action="running self-modifier analysis")
    proposals = SELF_MODIFIER.analyze_module(body.module)
    return {"proposals": [p.to_dict() for p in proposals]}


class ReviewRequest(BaseModel):
    note: str = ""


@router.post("/api/self-modifier/proposals/{proposal_id}/approve")
def self_modifier_approve(proposal_id: str, body: ReviewRequest):
    require_owner_for_writes(action="approving a self-modifier proposal")
    if not SELF_MODIFIER.approve(proposal_id, body.note):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


@router.post("/api/self-modifier/proposals/{proposal_id}/reject")
def self_modifier_reject(proposal_id: str, body: ReviewRequest):
    require_owner_for_writes(action="rejecting a self-modifier proposal")
    if not SELF_MODIFIER.reject(proposal_id, body.note):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


@router.post("/api/self-modifier/proposals/{proposal_id}/apply")
def self_modifier_apply(proposal_id: str):
    # Critical: applying a self-modifier proposal patches the agent's
    # OWN PYTHON CODE. Unauth would be a remote-code-execution
    # primitive on the agent process. Owner-only, full stop.
    require_owner_for_writes(action="applying a self-modifier patch")
    result = SELF_MODIFIER.apply(proposal_id)
    if not result["ok"]:
        raise HTTPException(400, result["message"])
    return result


# ---- self-modifications (patch overlay; survives `hrant update`) ----

@router.get("/api/self-mods")
def self_mods_list():
    """Return all locally-applied patches in order. Each entry:
    {id, slug, file, title, created, status, patch_filename, last_error}.

    `status` is one of:
      - "applied"      — currently active on the engine
      - "needs_review" — was applied before, conflicts with a newer
                         engine after `hrant update`; engine is at
                         the official version for this file's
                         touched lines
      - "reverted"     — user revert-clicked; kept in manifest for
                         audit trail (entry pruned on next manifest
                         rewrite)
    """
    from dataclasses import asdict
    from .. import self_mods
    return {"patches": [asdict(e) for e in self_mods.list_patches()]}


@router.post("/api/self-mods/{patch_id}/revert")
def self_mods_revert_one(patch_id: str):
    """Reverse-apply a single patch and remove it from the manifest.
    Reverting a non-tip patch may surface conflicts because later
    patches were stacked on top — see the warning in the UI."""
    require_owner_for_writes(action="reverting a self-mod patch")
    from .. import self_mods
    ok, err = self_mods.revert_one(patch_id)
    if not ok:
        raise HTTPException(400, err)
    return {"ok": True, "reverted": patch_id}


@router.post("/api/self-mods/revert-all")
def self_mods_revert_all():
    """Reset engine to official: `git reset --hard origin/master` +
    wipe top-level patch files. History (data_dir/self_mods/history/)
    is preserved — the user can still re-apply archived patches from
    Settings → Self-Modifications → History.

    User data (knowledge, workspace, settings) is untouched."""
    require_owner_for_writes(action="reverting all self-mods")
    from .. import self_mods
    ok, err = self_mods.revert_all_to_official()
    if not ok:
        raise HTTPException(400, err)
    return {"ok": True}


@router.get("/api/self-mods/history")
def self_mods_history():
    """Every archive bundle (newest first). Each entry:
    {archive_id, archived_at, patch_count, entries: [PatchEntry...]}.

    Archives are created automatically by `hrant update` (it moves
    active patches into a timestamped bundle before pulling new
    engine code) so the user can re-apply what they want afterward."""
    from dataclasses import asdict
    from .. import self_mods
    archives = self_mods.list_history()
    return {
        "archives": [
            {
                "archive_id": a.archive_id,
                "archived_at": a.archived_at,
                "patch_count": a.patch_count,
                "entries": [asdict(e) for e in a.entries],
            }
            for a in archives
        ]
    }


@router.post("/api/self-mods/history/{archive_id}/{patch_filename}/restore")
def self_mods_restore_one(archive_id: str, patch_filename: str):
    """Re-apply ONE archived patch as a fresh active modification.
    The archive bundle is unchanged; restoring is non-destructive."""
    require_owner_for_writes(action="restoring an archived self-mod")
    from dataclasses import asdict
    from .. import self_mods
    entry, err = self_mods.restore_from_history(archive_id, patch_filename)
    if entry is None:
        raise HTTPException(400, err)
    return {"ok": True, "entry": asdict(entry)}


@router.post("/api/self-mods/history/{archive_id}/restore")
def self_mods_restore_batch(archive_id: str):
    """Re-apply EVERY patch in an archive bundle, in order. Stops at
    the first conflict and reports which patch failed — user can
    then restore the rest one-by-one if needed."""
    require_owner_for_writes(action="restoring a self-mod archive batch")
    from .. import self_mods
    result = self_mods.restore_archive_batch(archive_id)
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return {"ok": True, **result}


@router.delete("/api/self-modifier/proposals/{proposal_id}")
def self_modifier_delete(proposal_id: str):
    require_owner_for_writes(action="deleting a self-modifier proposal")
    if not SELF_MODIFIER.delete_proposal(proposal_id):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}
