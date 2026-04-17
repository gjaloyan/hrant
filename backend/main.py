"""FastAPI entry point."""
from __future__ import annotations
import asyncio
import json
import logging
import secrets
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent import Agent, _capabilities_block
from .config import CONFIG
from .conversation import CONVERSATION
from .core_memory import CORE
from .finetune import store as finetune_store
from .finetune_curator import FinetuneDataCurator
from .identity import IDENTITY
from .knowledge_graph import GRAPH, reindex_all_notes
from .knowledge_manager import KM
from .llm import TOKENS, router
from .model_versions import VERSIONS
from .models import (
    ChatRequest,
    CorrectionRequest,
    CoreFactDelete,
    CoreFactRequest,
    FinetuneEdit,
    ImportGgufRequest,
    LearnRequest,
)
from .analogy_engine import ANALOGIES
from .autonomic.startup import (
    build_scheduler,
    start_autonomic_scheduler,
    stop_autonomic_scheduler,
)
from .background import BACKGROUND
from .channels import CHANNELS, get_channels, get_channel, save_channel, delete_channel
from .providers import (
    get_providers, get_provider, save_provider, delete_provider,
    get_api_key, PROVIDER_TYPES, KNOWN_PRICING, AUTH_TYPES,
    OAUTH_PRESETS, OAUTH_TOKENS, PROVIDER_CONNECT_INFO,
    ACTIVE_MODEL, get_available_models,
    generate_pkce, _pkce_store,
)
from .evaluator import EVALUATOR
from .goals import GOALS
from .memory_extractor import MEMORY
from .meta_learner import META_LEARNER
from .note_creator import learn_topic
from .project_mode import PROJECTS
from .self_modifier import SELF_MODIFIER
from .sessions import SESSIONS

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application: FastAPI):
    # --- startup ---
    log.info("Server starting — auto-starting channels...")
    try:
        channels_list = get_channels()
        auto = [c for c in channels_list if c.get("enabled") and c.get("auto_start")]
        log.info("Found %d channel(s) with auto_start", len(auto))
        for ch in auto:
            try:
                result = CHANNELS.start_channel(ch["id"])
                log.info("Auto-start channel %s: %s", ch["id"], result)
            except Exception as e:
                log.error("Failed to auto-start channel %s: %s", ch["id"], e)
    except Exception as e:
        log.warning("Channel auto-start error: %s", e)

    scheduler = build_scheduler()
    application.state.autonomic_scheduler = scheduler
    await start_autonomic_scheduler(scheduler)

    yield
    # --- shutdown ---
    log.info("Server shutting down — stopping channels...")
    CHANNELS.stop_all()
    await stop_autonomic_scheduler(application.state.autonomic_scheduler)

app = FastAPI(title="Self-Learning Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- chat (SSE) ----------------
@app.post("/api/chat")
async def chat(req: ChatRequest):
    queue: asyncio.Queue = asyncio.Queue()

    def progress(event: str, msg: str) -> None:
        queue.put_nowait({"type": "progress", "event": event, "message": msg})

    agent = Agent(progress=progress)

    async def runner():
        try:
            res = await asyncio.to_thread(agent.run, req.message, req.project or PROJECTS.current)
            # Record turn in session (full answer — no truncation for UI)
            turn = {
                "ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": req.message,
                "answer": res.answer or "",
                "intent": "chat" if res.is_chat else "task",
                "is_chat": bool(res.is_chat),
                "confidence": res.verification.confidence if res.verification else 0,
                "topics": res.used_topics or [],
            }
            SESSIONS.add_turn(turn)
            # Save thinking trace for Usage page
            if res.thinking_trace:
                TOKENS.save_request_trace(
                    question=req.message,
                    trace=[s.model_dump() for s in res.thinking_trace],
                    usage=res.token_usage.model_dump() if res.token_usage else {},
                )
            queue.put_nowait({"type": "answer", "data": res.model_dump()})
            # Trigger background proactive learning if due
            try:
                await BACKGROUND.process_proactive_goals()
            except Exception:
                pass
        except Exception as e:
            queue.put_nowait({"type": "error", "message": str(e)})
        finally:
            queue.put_nowait(None)

    async def stream() -> AsyncIterator[dict]:
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield {"data": json.dumps(item, ensure_ascii=False)}
        finally:
            await task

    return EventSourceResponse(stream())


# ---------------- knowledge ----------------
@app.get("/api/knowledge")
def list_knowledge():
    return {
        "topics": [t.model_dump() for t in KM.list_topics()],
        "by_category": {
            c: [t.model_dump() for t in items]
            for c, items in KM.all_categories().items()
        },
    }


@app.get("/api/knowledge/{topic}")
def get_knowledge(topic: str):
    note = KM.get_note(topic)
    if not note:
        raise HTTPException(404, "not found")
    return note.model_dump()


@app.post("/api/knowledge/learn")
def api_learn(req: LearnRequest):
    note = learn_topic(
        req.topic,
        depth=req.depth,
        category=req.category,
        project=PROJECTS.current,
    )
    return note.model_dump()


@app.delete("/api/knowledge/{topic}")
def delete_knowledge(topic: str):
    ok = KM.delete_note(topic)
    return {"ok": ok}


# ---------------- core memory ----------------
@app.get("/api/core-memory")
def get_core():
    return {"content": CORE.read(), "tokens": CORE.tokens(), "max": CORE.max_tokens}


@app.post("/api/core-memory")
def add_core(req: CoreFactRequest):
    msg = CORE.add_fact(req.fact, req.source)
    return {"message": msg}


@app.delete("/api/core-memory")
def delete_core(req: CoreFactDelete):
    return {"message": CORE.remove_fact(req.search_text)}


# ---------------- projects ----------------
@app.get("/api/projects")
def list_projects():
    return {"current": PROJECTS.current, "all": PROJECTS.list_projects()}


@app.post("/api/projects")
def create_project(body: dict):
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name required")
    return {"message": PROJECTS.start(name)}


@app.get("/api/projects/{name}")
def project_detail(name: str):
    return {"overview": PROJECTS.read_overview(name)}


# ---------------- finetune ----------------
@app.get("/api/finetune/status")
def finetune_status():
    ft = finetune_store()
    examples = ft.list_all()
    curated = FinetuneDataCurator().curate(examples)
    return {
        "total": len(examples),
        "curated": len(curated),
        "ready": ft.ready(),
        "min_required": ft.min_required,
        "by_category": ft.count_by_category(),
    }


@app.get("/api/finetune/examples")
def finetune_examples():
    ft = finetune_store()
    curator = FinetuneDataCurator()
    items = []
    for sp in curator.score_all(ft.list_all()):
        items.append({
            "id": sp.pair.id,
            "score": sp.score,
            "user": sp.pair.user_text(),
            "assistant": sp.pair.assistant_text(),
            "metadata": sp.pair.metadata.model_dump(),
        })
    return {"items": items}


@app.put("/api/finetune/examples/{pair_id}")
def finetune_edit(pair_id: str, body: FinetuneEdit):
    ok = finetune_store().edit(pair_id, assistant=body.assistant, boosted=body.boosted)
    if not ok:
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.delete("/api/finetune/examples/{pair_id}")
def finetune_delete(pair_id: str):
    ok = finetune_store().delete(pair_id)
    if not ok:
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.post("/api/finetune/examples/{pair_id}/boost")
def finetune_boost(pair_id: str):
    ok = finetune_store().boost(pair_id)
    if not ok:
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.post("/api/finetune/correction")
def finetune_correction(body: CorrectionRequest):
    pair = finetune_store().add_correction(
        question=body.question,
        wrong_answer=body.wrong_answer,
        corrected_answer=body.corrected_answer,
        project=body.project,
    )
    return {"ok": True, "id": pair.id}


@app.post("/api/finetune/start")
def finetune_start():
    """Запуск пайплайна с потоком прогресса."""
    from .finetune_pipeline import FineTunePipeline

    queue: asyncio.Queue = asyncio.Queue()

    def progress(stage: str, msg: str) -> None:
        queue.put_nowait({"type": "progress", "stage": stage, "message": msg})

    pipe = FineTunePipeline(progress=progress)

    async def runner():
        try:
            name = await asyncio.to_thread(pipe.run_full_pipeline)
            queue.put_nowait({"type": "done", "model": name})
        except Exception as e:
            queue.put_nowait({"type": "error", "message": str(e)})
        finally:
            queue.put_nowait(None)

    async def stream() -> AsyncIterator[dict]:
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield {"data": json.dumps(item, ensure_ascii=False)}
        finally:
            await task

    return EventSourceResponse(stream())


@app.post("/api/finetune/switch")
def finetune_switch(body: dict):
    tag = body.get("tag", "").strip()
    if not tag:
        raise HTTPException(400, "tag required")
    return {"message": VERSIONS.switch(tag)}


@app.post("/api/finetune/rollback")
def finetune_rollback():
    return {"message": VERSIONS.rollback()}


@app.get("/api/finetune/export")
def finetune_export():
    return {"jsonl": finetune_store().export_jsonl()}


@app.get("/api/model/versions")
def model_versions():
    return VERSIONS.list().model_dump()


# ---------------- status ----------------
@app.get("/api/status")
def status():
    topics = KM.list_topics()
    by_cat: dict[str, int] = {}
    for t in topics:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
    cur = VERSIONS.current()
    try:
        router_state = router().stats()
    except Exception as e:
        router_state = {"error": str(e)}
    return {
        "topics_total": len(topics),
        "by_category": by_cat,
        "core_tokens": CORE.tokens(),
        "core_max": CORE.max_tokens,
        "finetune_count": finetune_store().count(),
        "current_project": PROJECTS.current,
        "mode": CONFIG.mode,
        "finetune_enabled": CONFIG.finetune_enabled,
        "training_location": CONFIG.training_location,
        "model_a": CONFIG.model_a.get("model"),
        "model_b": (CONFIG.model_b or {}).get("model") if CONFIG.model_b else None,
        "model_version": cur.tag if cur else None,
        "router": router_state,
    }


@app.get("/api/router/stats")
def router_stats():
    try:
        return router().stats()
    except Exception as e:
        return {"error": str(e)}


# ---------------- mode ----------------
@app.get("/api/mode")
def get_mode():
    return {
        "mode": CONFIG.mode,
        "finetune_enabled": CONFIG.finetune_enabled,
        "training_location": CONFIG.training_location,
        "model_a": CONFIG.model_a.get("model"),
        "model_b": (CONFIG.model_b or {}).get("model") if CONFIG.model_b else None,
    }


# ---------------- cloud fine-tune ----------------
@app.post("/api/finetune/export-cloud")
def finetune_export_cloud():
    from .finetune_pipeline import FineTunePipeline

    try:
        pipe = FineTunePipeline()
        pkg = pipe.export_for_cloud()
        files = sorted(p.name for p in pkg.iterdir() if p.is_file())
        return {
            "package_dir": str(pkg),
            "files": files,
            "tag": pkg.name.replace("cloud_export_", ""),
            "instructions": f"Скачай {pkg.name}/, залей на GPU, запусти train_script.py",
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/finetune/import-gguf")
def finetune_import_gguf(body: ImportGgufRequest):
    from .finetune_pipeline import FineTunePipeline

    try:
        pipe = FineTunePipeline()
        name = pipe.import_gguf(body.path, tag=body.tag)
        return {"ok": True, "model": name}
    except Exception as e:
        raise HTTPException(400, str(e))


# ---------------- gaps ----------------
@app.get("/api/gaps")
def get_gaps():
    gaps = KM.hot_gaps(threshold=1)
    return {
        "gaps": gaps,
        "open": [g for g in gaps if not g["has_note_now"]],
        "closed": [g for g in gaps if g["has_note_now"]],
    }


# ---------------- capabilities ----------------
@app.get("/api/capabilities")
def get_capabilities():
    return {"block": _capabilities_block()}


# ---------------- conversation ----------------
@app.get("/api/conversation")
def get_conversation():
    return {
        "turns": CONVERSATION.recent(20),
        "count": CONVERSATION.count(),
    }


@app.delete("/api/conversation")
def clear_conversation():
    CONVERSATION.clear()
    return {"ok": True}


# ---------------- identity ----------------
@app.get("/api/identity")
def get_identity():
    return {
        "soul": IDENTITY.soul(),
        "identity": IDENTITY.identity(),
        "user_profile": IDENTITY.user_profile(),
    }


class IdentityUpdate(BaseModel):
    file: str  # "soul" | "identity" | "user"
    content: str


@app.put("/api/identity")
def update_identity(body: IdentityUpdate):
    path_map = {
        "soul": IDENTITY.soul_path,
        "identity": IDENTITY.identity_path,
        "user": IDENTITY.user_path,
    }
    p = path_map.get(body.file)
    if not p:
        raise HTTPException(400, "file must be soul, identity, or user")
    if body.file == "user":
        IDENTITY._snapshot_user_profile()
    p.write_text(body.content, encoding="utf-8")
    return {"ok": True}


@app.get("/api/identity/history")
def identity_history():
    return {"versions": IDENTITY.list_user_versions()}


# ---------------- quick note ----------------
class QuickNoteRequest(BaseModel):
    text: str


@app.post("/api/knowledge/quick-note")
def quick_note(req: QuickNoteRequest):
    note = KM.save_note(
        topic=req.text[:40],
        body=req.text,
        category="personal",
        keywords=[req.text.split()[0].lower()] if req.text.strip() else [],
        source="user_quick_note",
        confidence="verified",
    )
    return {"topic": note.frontmatter.topic, "path": str(note.path)}


# ---------------- projects (extended) ----------------
@app.post("/api/projects/{name}/end")
def end_project(name: str):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.end()}


class ProjectContextRequest(BaseModel):
    text: str


@app.post("/api/projects/{name}/context")
def add_project_context(name: str, body: ProjectContextRequest):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_context(body.text)}


class ProjectDecisionRequest(BaseModel):
    what: str
    why: str


@app.post("/api/projects/{name}/decision")
def add_project_decision(name: str, body: ProjectDecisionRequest):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_decision(body.what, body.why)}


class ProjectIssueRequest(BaseModel):
    problem: str
    fix: str


@app.post("/api/projects/{name}/issue")
def add_project_issue(name: str, body: ProjectIssueRequest):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_issue(body.problem, body.fix)}


# ---------------- add last Q&A to finetune ----------------
class AddToFinetuneRequest(BaseModel):
    question: str
    answer: str
    used_topics: list[str] = []
    confidence: int = 100
    project: str | None = None


@app.post("/api/finetune/add-from-chat")
def add_from_chat(body: AddToFinetuneRequest):
    finetune_store().add(
        question=body.question,
        answer=body.answer,
        source_notes=body.used_topics,
        confidence=body.confidence,
        project=body.project,
        verified=True,
    )
    return {"ok": True}


# ---------------- finetune compare ----------------
@app.post("/api/finetune/compare")
def finetune_compare():
    from .model_evaluator import ModelEvaluator

    state = VERSIONS.list()
    if len(state.versions) < 2:
        raise HTTPException(400, "need at least 2 model versions")
    old = state.versions[-2].model_id
    new = state.versions[-1].model_id
    return ModelEvaluator().compare(old, new).model_dump()


# ---------------- knowledge graph ----------------
@app.get("/api/graph")
def graph_stats():
    return GRAPH.stats()


@app.post("/api/graph/reindex")
def graph_reindex():
    stats = reindex_all_notes()
    return stats


@app.get("/api/graph/entities")
def graph_entities():
    """List all entities with their edge counts."""
    entities: dict[str, int] = {}
    for entity, edges in GRAPH._edges.items():
        entities[entity] = len(edges)
    # Sort by edge count descending
    sorted_ents = sorted(entities.items(), key=lambda x: -x[1])
    return {"entities": [{"name": n, "edges": c} for n, c in sorted_ents]}


@app.get("/api/graph/neighbors/{entity}")
def graph_neighbors(entity: str):
    neighbors = GRAPH.get_neighbors(entity)
    return {"entity": entity, "neighbors": neighbors}


@app.get("/api/graph/full")
def graph_full():
    """Return the entire graph as nodes + links for force-directed visualization."""
    all_entities: set[str] = set()
    links = []
    for source, edges in GRAPH._edges.items():
        all_entities.add(source)
        for edge in edges:
            target = edge["target"]
            all_entities.add(target)
            # Skip inverse edges for cleaner visualization
            if edge["relation"].startswith("inverse:"):
                continue
            links.append({
                "source": source,
                "target": target,
                "relation": edge["relation"],
                "note": edge.get("note", ""),
                "weight": edge.get("weight", 1.0),
            })

    # Count connections per entity for sizing
    conn_count: dict[str, int] = {e: 0 for e in all_entities}
    for link in links:
        conn_count[link["source"]] = conn_count.get(link["source"], 0) + 1
        conn_count[link["target"]] = conn_count.get(link["target"], 0) + 1

    nodes = [
        {"id": e, "name": e, "connections": conn_count.get(e, 0)}
        for e in sorted(all_entities)
    ]
    return {"nodes": nodes, "links": links}


# ---------------- meta-learner ----------------
@app.get("/api/meta-learner")
def meta_learner_stats():
    return META_LEARNER.stats()


@app.get("/api/meta-learner/failures")
def meta_learner_failures():
    return {"failures": META_LEARNER.recent_failures(limit=30)}


@app.post("/api/meta-learner/extract-patterns")
def meta_learner_extract():
    patterns = META_LEARNER.extract_patterns()
    return {"patterns": patterns}


# ---------------- evaluator ----------------
@app.get("/api/eval")
def eval_stats():
    return EVALUATOR.stats()


@app.get("/api/eval/today")
def eval_today():
    return EVALUATOR.daily_report()


@app.get("/api/eval/trend")
def eval_trend():
    return {"trend": EVALUATOR.weekly_trend()}


@app.get("/api/eval/regressions")
def eval_regressions():
    return {"regressions": EVALUATOR.detect_regression()}


@app.get("/api/eval/suggestions")
def eval_suggestions():
    return {"suggestions": EVALUATOR.suggest_priorities()}


# ---------------- token usage ----------------
@app.get("/api/usage")
def usage_stats():
    return TOKENS.stats()


@app.get("/api/usage/calls")
def usage_calls(limit: int = 50):
    return {"calls": TOKENS.recent_calls(limit=limit)}


@app.get("/api/usage/traces")
def usage_traces(limit: int = 20):
    return {"traces": TOKENS.recent_traces(limit=limit)}


# ---------------- memory (conversation facts) ----------------
@app.get("/api/memory")
def memory_stats():
    return MEMORY.stats()


@app.get("/api/memory/facts")
def memory_facts(limit: int = 50):
    return {"facts": MEMORY.recent_facts(limit=limit)}


class MemoryRecallRequest(BaseModel):
    query: str
    limit: int = 10


@app.post("/api/memory/recall")
def memory_recall(body: MemoryRecallRequest):
    facts = MEMORY.recall(body.query, limit=body.limit)
    block = MEMORY.recall_block(body.query, max_facts=body.limit)
    return {"facts": facts, "block": block}


# ---------------- analogy engine ----------------
@app.get("/api/analogies")
def analogy_list():
    return {"patterns": ANALOGIES.all_patterns(), "stats": ANALOGIES.stats()}


# ---------------- self-modifier ----------------
@app.get("/api/self-modifier")
def self_modifier_stats():
    return {**SELF_MODIFIER.stats(), "modules": SELF_MODIFIER.available_modules()}


@app.get("/api/self-modifier/proposals")
def self_modifier_proposals(status: str | None = None):
    return {"proposals": SELF_MODIFIER.list_proposals(status)}


class AnalyzeModuleRequest(BaseModel):
    module: str


@app.post("/api/self-modifier/analyze")
def self_modifier_analyze(body: AnalyzeModuleRequest):
    proposals = SELF_MODIFIER.analyze_module(body.module)
    return {"proposals": [p.to_dict() for p in proposals]}


class ReviewRequest(BaseModel):
    note: str = ""


@app.post("/api/self-modifier/proposals/{proposal_id}/approve")
def self_modifier_approve(proposal_id: str, body: ReviewRequest):
    if not SELF_MODIFIER.approve(proposal_id, body.note):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


@app.post("/api/self-modifier/proposals/{proposal_id}/reject")
def self_modifier_reject(proposal_id: str, body: ReviewRequest):
    if not SELF_MODIFIER.reject(proposal_id, body.note):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


@app.post("/api/self-modifier/proposals/{proposal_id}/apply")
def self_modifier_apply(proposal_id: str):
    result = SELF_MODIFIER.apply(proposal_id)
    if not result["ok"]:
        raise HTTPException(400, result["message"])
    return result


@app.delete("/api/self-modifier/proposals/{proposal_id}")
def self_modifier_delete(proposal_id: str):
    if not SELF_MODIFIER.delete_proposal(proposal_id):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


# ---------------- goals ----------------
class GoalRequest(BaseModel):
    description: str
    priority: int = 5
    goal_type: str = "user"
    context: str = ""
    subtasks: list[str] = []


@app.get("/api/goals")
def list_goals():
    return {
        "goals": [g.to_dict() for g in GOALS.all_goals()],
        "stats": GOALS.stats(),
    }


@app.post("/api/goals")
def add_goal(body: GoalRequest):
    goal = GOALS.add(
        description=body.description,
        priority=body.priority,
        goal_type=body.goal_type,
        context=body.context,
        source="user",
        subtasks=body.subtasks if body.subtasks else None,
    )
    return {"goal": goal.to_dict()}


@app.post("/api/goals/{goal_id}/complete")
def complete_goal(goal_id: str):
    if not GOALS.complete_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@app.post("/api/goals/{goal_id}/pause")
def pause_goal(goal_id: str):
    if not GOALS.pause_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@app.post("/api/goals/{goal_id}/resume")
def resume_goal(goal_id: str):
    if not GOALS.resume_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@app.post("/api/goals/{goal_id}/fail")
def fail_goal(goal_id: str):
    if not GOALS.fail_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@app.delete("/api/goals/{goal_id}")
def delete_goal(goal_id: str):
    if not GOALS.delete_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


class PriorityUpdate(BaseModel):
    priority: int


@app.put("/api/goals/{goal_id}/priority")
def update_goal_priority(goal_id: str, body: PriorityUpdate):
    if not GOALS.update_priority(goal_id, body.priority):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


# ---------------- background tasks & autonomic ----------------
# Router-based: endpoints live next to their module of origin.
from .background import router as background_router  # noqa: E402
from .autonomic.api import router as autonomic_router  # noqa: E402

app.include_router(background_router)
app.include_router(autonomic_router)


# ---------------- sessions ----------------
@app.get("/api/sessions")
def list_sessions(include_archived: bool = False):
    return {
        "sessions": SESSIONS.list_sessions(include_archived=include_archived),
        "current_id": SESSIONS._current_id,
    }


@app.get("/api/sessions/stats")
def session_stats():
    return SESSIONS.stats()


@app.get("/api/sessions/current")
def current_session():
    session = SESSIONS.current
    if not session:
        return {"session": None}
    return {"session": session.to_dict()}


@app.post("/api/sessions/new")
def new_session():
    session = SESSIONS.new_session()
    return {"session": session.to_dict()}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = SESSIONS.get_session(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return {"session": session.to_dict()}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not SESSIONS.delete_session(session_id):
        raise HTTPException(404, "session not found")
    return {"ok": True}


class ArchiveRequest(BaseModel):
    days: int = 90


@app.post("/api/sessions/archive")
def archive_sessions(body: ArchiveRequest):
    count = SESSIONS.archive_old(days=body.days)
    return {"archived": count}


# ---------------- providers (multi-LLM) ----------------
@app.get("/api/providers")
def list_providers():
    providers = get_providers()
    # Mask API keys and OAuth secrets in response
    for p in providers:
        key = p.get("api_key", "")
        if key and len(key) > 8:
            p["api_key_masked"] = "••••" + key[-6:]
        else:
            p["api_key_masked"] = "(env)" if get_api_key(p) else "(not set)"
        p.pop("api_key", None)
        # Mask OAuth secrets
        oauth = p.get("oauth", {})
        if oauth.get("client_secret"):
            oauth["client_secret_masked"] = "••••" + oauth["client_secret"][-4:]
            del oauth["client_secret"]
        # Add OAuth status
        if p.get("auth_type") == "oauth":
            p["oauth_status"] = OAUTH_TOKENS.status(p["id"])
    return {"providers": providers, "types": PROVIDER_TYPES}


@app.get("/api/providers/types")
def provider_types():
    return {"types": PROVIDER_TYPES, "pricing": KNOWN_PRICING}


@app.get("/api/providers/connect-info")
def provider_connect_info():
    return {"connect_info": PROVIDER_CONNECT_INFO}


@app.get("/api/providers/auth-types")
def get_auth_types():
    return {"auth_types": AUTH_TYPES, "oauth_presets": OAUTH_PRESETS}


# ---- OAuth callback (must be before {provider_id} routes) ----
@app.get("/api/providers/oauth/callback")
async def oauth_callback(code: str = "", state: str = "", error: str = ""):
    """OAuth callback endpoint — receives the authorization code."""
    if error:
        from fastapi.responses import HTMLResponse
        html = f"""
        <html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
        <h2>OAuth Error</h2><p>{error}</p>
        <script>setTimeout(()=>window.close(),5000)</script>
        </body></html>
        """
        return HTMLResponse(html, status_code=400)

    if not code or not state:
        raise HTTPException(400, "Missing code or state")

    provider_id = state.rsplit("_", 1)[0] if "_" in state else state
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")

    oauth = p.get("oauth", {})
    redirect_uri = oauth.get(
        "redirect_uri",
        f"http://localhost:{CONFIG.server['port']}/api/providers/oauth/callback",
    )

    pkce_verifier = _pkce_store.pop(state, None)
    result = OAUTH_TOKENS.exchange_code(provider_id, code, redirect_uri, pkce_verifier=pkce_verifier)

    from fastapi.responses import HTMLResponse
    if result.get("ok"):
        html = """
        <html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
        <h2 style="color:#34d399">Connected!</h2>
        <p>OAuth token received. You can close this tab.</p>
        <script>setTimeout(()=>window.close(),2000)</script>
        </body></html>
        """
        return HTMLResponse(html)
    html = f"""
    <html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
    <h2 style="color:#f87171">Token Exchange Failed</h2>
    <p>{result.get('error', 'Unknown error')}</p>
    <p style="font-size:12px;opacity:0.6">Copy this page URL and paste it in the agent settings to retry.</p>
    </body></html>
    """
    return HTMLResponse(html, status_code=400)


# ---- Ollama local models (must be before {provider_id} routes) ----
@app.get("/api/providers/ollama/models")
def ollama_models():
    """List locally available Ollama models."""
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "name": m.get("name", ""),
                "size": m.get("size", 0),
                "modified": m.get("modified_at", ""),
                "family": m.get("details", {}).get("family", ""),
                "parameters": m.get("details", {}).get("parameter_size", ""),
                "quantization": m.get("details", {}).get("quantization_level", ""),
            })
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


class OllamaPullRequest(BaseModel):
    model: str


@app.post("/api/providers/ollama/pull")
async def ollama_pull(body: OllamaPullRequest):
    """Pull (download) an Ollama model."""
    try:
        r = httpx.post(
            "http://localhost:11434/api/pull",
            json={"name": body.model, "stream": False},
            timeout=600.0,
        )
        r.raise_for_status()
        return {"ok": True, "message": f"Model '{body.model}' pulled successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.delete("/api/providers/ollama/models/{model_name:path}")
def ollama_delete_model(model_name: str):
    """Delete an Ollama model."""
    try:
        r = httpx.delete(
            "http://localhost:11434/api/delete",
            json={"name": model_name},
            timeout=30.0,
        )
        r.raise_for_status()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------- active model selection ----------------
@app.get("/api/active-model")
def get_active_model():
    active = ACTIVE_MODEL.get()
    models = get_available_models()
    return {"active": active, "models": models}


class SetActiveModelRequest(BaseModel):
    provider_id: str
    model: str


@app.put("/api/active-model")
def set_active_model(req: SetActiveModelRequest):
    try:
        result = ACTIVE_MODEL.set(req.provider_id, req.model)
        return {"ok": True, "active": result}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/active-model")
def clear_active_model():
    ACTIVE_MODEL.clear()
    return {"ok": True}


@app.get("/api/providers/{provider_id}")
def get_provider_api(provider_id: str):
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    key = p.get("api_key", "")
    if key and len(key) > 8:
        p["api_key_masked"] = "••••" + key[-6:]
    else:
        p["api_key_masked"] = "(env)" if get_api_key(p) else "(not set)"
    p.pop("api_key", None)
    return p


class ProviderCreateRequest(BaseModel):
    id: str
    name: str
    type: str
    enabled: bool = True
    api_key: str = ""
    api_key_env: str = ""
    base_url: str = ""
    models: list[str] = []
    default_model: str = ""
    max_tokens: int = 2000
    temperature: float = 0.3
    auth_type: str = "api_key"
    oauth: dict | None = None


@app.post("/api/providers")
def create_provider(body: ProviderCreateRequest):
    p = save_provider(body.model_dump())
    return {"ok": True, "provider": p}


class ProviderUpdateRequest(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    models: list[str] | None = None
    default_model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@app.put("/api/providers/{provider_id}")
def update_provider_api(provider_id: str, body: ProviderUpdateRequest):
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    updates = body.model_dump(exclude_none=True)
    p.update(updates)
    save_provider(p)
    return {"ok": True}


@app.delete("/api/providers/{provider_id}")
def delete_provider_api(provider_id: str):
    if not delete_provider(provider_id):
        raise HTTPException(404, "provider not found")
    return {"ok": True}


@app.post("/api/providers/{provider_id}/test")
async def test_provider(provider_id: str):
    """Test provider connection by making a minimal API call."""
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")

    api_key = get_api_key(p)
    ptype = p.get("type", "")
    model = p.get("default_model") or (p.get("models", [None]) or [None])[0]

    if ptype == "anthropic":
        if not api_key:
            return {"ok": False, "error": "No API key"}
        try:
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                json={"model": model or "claude-sonnet-4-5", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "ping"}]},
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                timeout=15.0,
            )
            if r.status_code in (200, 201):
                return {"ok": True, "model": model, "message": "Connected"}
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif ptype in ("openai", "groq", "deepseek", "mistral", "openai_compatible", "together", "openrouter"):
        if not api_key:
            return {"ok": False, "error": "No API key"}
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "mistral": "https://api.mistral.ai/v1",
            "together": "https://api.together.xyz/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        base = p.get("base_url") or base_urls.get(ptype, "")
        if not base:
            return {"ok": False, "error": "No base_url configured"}
        try:
            r = httpx.get(
                f"{base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            if r.status_code == 200:
                data = r.json()
                models = [m.get("id", "") for m in data.get("data", [])[:10]]
                return {"ok": True, "models": models, "message": f"Connected, {len(data.get('data', []))} models"}
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif ptype == "google":
        if not api_key:
            return {"ok": False, "error": "No API key"}
        try:
            base = p.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
            r = httpx.get(f"{base}/models?key={api_key}", timeout=15.0)
            if r.status_code == 200:
                data = r.json()
                models = [m.get("name", "").split("/")[-1] for m in data.get("models", [])[:10]]
                return {"ok": True, "models": models, "message": "Connected"}
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif ptype == "ollama":
        base = p.get("base_url", "http://localhost:11434").rstrip("/")
        try:
            r = httpx.get(f"{base}/api/tags", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return {"ok": True, "models": models, "message": f"Connected, {len(models)} models"}
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": f"Unknown type: {ptype}"}


# ---- OAuth endpoints ----
class OAuthConfigUpdate(BaseModel):
    auth_type: str  # "api_key", "oauth", "none"
    oauth: dict = {}  # {client_id, client_secret, token_url, authorize_url, scope, grant_type}


@app.put("/api/providers/{provider_id}/auth")
def update_provider_auth(provider_id: str, body: OAuthConfigUpdate):
    """Update auth type and OAuth config for a provider."""
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    p["auth_type"] = body.auth_type
    if body.auth_type == "oauth":
        p["oauth"] = body.oauth
    save_provider(p)
    return {"ok": True}


@app.get("/api/providers/{provider_id}/oauth/status")
def oauth_status(provider_id: str):
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    return OAUTH_TOKENS.status(provider_id)


@app.post("/api/providers/{provider_id}/oauth/authorize-url")
def oauth_authorize_url(provider_id: str):
    """Build the OAuth authorization URL for redirect-based flows.

    Supports PKCE (S256) for providers like OpenAI Codex.
    The PKCE verifier is stored in memory keyed by state parameter.
    """
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    oauth = p.get("oauth", {})
    authorize_url = oauth.get("authorize_url", "")
    client_id = oauth.get("client_id", "")
    scope = oauth.get("scope", "")
    audience = oauth.get("audience", "")
    _pkce_val = oauth.get("pkce", False)
    use_pkce = _pkce_val is True or str(_pkce_val).lower() == "true"
    redirect_uri = oauth.get(
        "redirect_uri",
        f"http://localhost:{CONFIG.server['port']}/api/providers/oauth/callback",
    )

    if not authorize_url:
        raise HTTPException(400, "Missing authorize_url in OAuth config")

    import urllib.parse

    state = f"{provider_id}_{secrets.token_urlsafe(8)}"
    params: dict[str, str] = {
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if client_id:
        params["client_id"] = client_id
    if scope:
        params["scope"] = scope
    if audience:
        params["audience"] = audience

    # Extra provider-specific params (e.g. OpenAI simplified flow flags)
    extra_params = oauth.get("extra_params", {})
    if isinstance(extra_params, dict):
        params.update(extra_params)

    # PKCE support
    if use_pkce:
        verifier, challenge = generate_pkce()
        _pkce_store[state] = verifier
        params["code_challenge"] = challenge
        params["code_challenge_method"] = "S256"

    url = f"{authorize_url}?{urllib.parse.urlencode(params)}"

    # If redirect_uri is on a different port, start a mini callback listener
    parsed_redir = urllib.parse.urlparse(redirect_uri)
    redir_port = parsed_redir.port or 80
    if redir_port != CONFIG.server.get("port", 8000):
        _start_oauth_callback_listener(redir_port, parsed_redir.path or "/auth/callback", provider_id)

    return {"url": url, "redirect_uri": redirect_uri, "state": state, "pkce": use_pkce}


def _start_oauth_callback_listener(port: int, path: str, provider_id: str):
    """Start a temporary HTTP server on the given port to catch OAuth callback."""
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            import urllib.parse as up
            parsed = up.urlparse(self.path)
            if not parsed.path.rstrip("/").endswith(path.rstrip("/")):
                self.send_response(404)
                self.end_headers()
                return

            params = up.parse_qs(parsed.query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            error = params.get("error", [None])[0]

            if error:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"""<html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
                <h2 style="color:#f87171">OAuth Error</h2><p>{error}</p>
                <script>setTimeout(()=>window.close(),5000)</script></body></html>""".encode())
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            if not code or not state:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"Missing code or state")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            # Exchange code for token
            pid = state.rsplit("_", 1)[0] if "_" in state else state
            p = get_provider(pid)
            if not p:
                self.send_response(404)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"Provider not found")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            oauth = p.get("oauth", {})
            redir = oauth.get("redirect_uri", f"http://localhost:{port}{path}")
            pkce_verifier = _pkce_store.pop(state, None)
            result = OAUTH_TOKENS.exchange_code(pid, code, redir, pkce_verifier=pkce_verifier)

            if result.get("ok"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"""<html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
                <h2 style="color:#34d399">Connected!</h2>
                <p>OAuth token received. You can close this tab.</p>
                <script>setTimeout(()=>window.close(),2000)</script></body></html>""")
            else:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                err_msg = result.get("error", "Unknown error")
                self.wfile.write(f"""<html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
                <h2 style="color:#f87171">Token Exchange Failed</h2><p>{err_msg}</p>
                <p style="font-size:12px;opacity:0.6">Copy the redirect URL and paste it in settings.</p></body></html>""".encode())

            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, fmt, *args):
            pass  # suppress logs

    def run_server():
        try:
            server = HTTPServer(("127.0.0.1", port), CallbackHandler)
            server.timeout = 300  # 5 min timeout
            server.handle_request()  # handle single request then stop
        except OSError:
            pass  # port already in use — another listener is running

    t = threading.Thread(target=run_server, daemon=True)
    t.start()


class ClientCredentialsRequest(BaseModel):
    pass


@app.post("/api/providers/{provider_id}/oauth/client-credentials")
def oauth_client_credentials(provider_id: str):
    """Authenticate with client_credentials grant (no browser redirect)."""
    result = OAUTH_TOKENS.client_credentials_auth(provider_id)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Auth failed"))
    return result


@app.post("/api/providers/{provider_id}/oauth/exchange-url")
def oauth_exchange_url(provider_id: str, body: dict):
    """Extract code from a pasted redirect URL and exchange it for a token."""
    redirect_url = body.get("url", "").strip()
    if not redirect_url:
        raise HTTPException(400, "Missing redirect URL")

    import urllib.parse
    parsed = urllib.parse.urlparse(redirect_url)
    params = urllib.parse.parse_qs(parsed.query)

    code = params.get("code", [None])[0]
    if not code:
        raise HTTPException(400, "No ?code= parameter found in the URL")

    state = params.get("state", [None])[0]
    pkce_verifier = _pkce_store.pop(state, None) if state else None

    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")

    oauth = p.get("oauth", {})
    redirect_uri = oauth.get("redirect_uri",
        f"http://localhost:{CONFIG.server['port']}/api/providers/oauth/callback")

    result = OAUTH_TOKENS.exchange_code(provider_id, code, redirect_uri, pkce_verifier=pkce_verifier)
    if result.get("ok"):
        return result
    raise HTTPException(400, result.get("error", "Token exchange failed"))


class ManualTokenRequest(BaseModel):
    access_token: str
    refresh_token: str = ""
    expires_in: int = 86400  # default 24h


@app.post("/api/providers/{provider_id}/oauth/manual-token")
def oauth_manual_token(provider_id: str, body: ManualTokenRequest):
    """Manually set an OAuth token."""
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")

    oauth = p.get("oauth", {})
    OAUTH_TOKENS._store_token(provider_id, {
        "access_token": body.access_token,
        "refresh_token": body.refresh_token,
        "expires_in": body.expires_in,
        "token_type": "Bearer",
    }, oauth)
    return {"ok": True, "message": "Token saved"}


@app.post("/api/providers/{provider_id}/oauth/revoke")
def oauth_revoke(provider_id: str):
    OAUTH_TOKENS.revoke(provider_id)
    return {"ok": True}


# ---------------- channels (Telegram, etc.) ----------------
@app.get("/api/channels")
def list_channels():
    channels = get_channels()
    runtime = CHANNELS.status_all()
    for ch in channels:
        ch["runtime_status"] = runtime.get(ch["id"], "stopped")
    return {"channels": channels}


@app.get("/api/channels/{channel_id}")
def get_channel_api(channel_id: str):
    ch = get_channel(channel_id)
    if not ch:
        raise HTTPException(404, "channel not found")
    ch["runtime_status"] = CHANNELS.channel_status(channel_id)
    return ch


class ChannelCreateRequest(BaseModel):
    id: str
    name: str
    type: str  # "telegram"
    enabled: bool = False
    auto_start: bool = False
    config: dict = {}


@app.post("/api/channels")
def create_channel(body: ChannelCreateRequest):
    ch = save_channel(body.model_dump())
    return ch


class ChannelUpdateRequest(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    auto_start: bool | None = None
    config: dict | None = None


@app.put("/api/channels/{channel_id}")
def update_channel(channel_id: str, body: ChannelUpdateRequest):
    ch = get_channel(channel_id)
    if not ch:
        raise HTTPException(404, "channel not found")
    if body.name is not None:
        ch["name"] = body.name
    if body.enabled is not None:
        ch["enabled"] = body.enabled
    if body.auto_start is not None:
        ch["auto_start"] = body.auto_start
    if body.config is not None:
        ch["config"] = body.config
    save_channel(ch)
    return ch


@app.delete("/api/channels/{channel_id}")
def delete_channel_api(channel_id: str):
    CHANNELS.stop_channel(channel_id)
    if not delete_channel(channel_id):
        raise HTTPException(404, "channel not found")
    return {"ok": True}


@app.post("/api/channels/{channel_id}/start")
def start_channel(channel_id: str):
    result = CHANNELS.start_channel(channel_id)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/channels/{channel_id}/stop")
def stop_channel(channel_id: str):
    return CHANNELS.stop_channel(channel_id)


@app.post("/api/channels/{channel_id}/test")
async def test_channel(channel_id: str):
    """Test channel connection (validate token, etc.)."""
    ch = get_channel(channel_id)
    if not ch:
        raise HTTPException(404, "channel not found")

    if ch["type"] == "telegram":
        token = ch.get("config", {}).get("bot_token", "")
        if not token:
            return {"ok": False, "error": "No bot token configured"}
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
                data = resp.json()
                if data.get("ok"):
                    bot_info = data["result"]
                    return {
                        "ok": True,
                        "bot_name": bot_info.get("first_name", ""),
                        "bot_username": bot_info.get("username", ""),
                    }
                else:
                    return {"ok": False, "error": data.get("description", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    else:
        return {"ok": False, "error": f"Unknown channel type: {ch['type']}"}


# ---------------- frontend static files ----------------
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    # Serve static assets (JS, CSS, etc.)
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="static")

    # Catch-all: serve index.html for any non-API route (SPA routing)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # If file exists in dist, serve it (favicon, etc.)
        file_path = _FRONTEND_DIST / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA client-side routing
        return FileResponse(_FRONTEND_DIST / "index.html")


def serve() -> None:
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=CONFIG.server["host"],
        port=CONFIG.server["port"],
        reload=False,
    )


if __name__ == "__main__":
    serve()
