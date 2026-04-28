"""Knowledge base + core memory + gaps + capabilities + quick-note."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..agent import _capabilities_block
from ..core_memory import CORE
from ..knowledge_manager import KM
from ..models import CoreFactDelete, CoreFactRequest, LearnRequest
from ..note_creator import learn_topic
from ..project_mode import PROJECTS

router = APIRouter()


# ---- knowledge ----
@router.get("/api/knowledge")
def list_knowledge():
    return {
        "topics": [t.model_dump() for t in KM.list_topics()],
        "by_category": {
            c: [t.model_dump() for t in items]
            for c, items in KM.all_categories().items()
        },
    }


@router.get("/api/knowledge/{topic}")
def get_knowledge(topic: str):
    note = KM.get_note(topic)
    if not note:
        raise HTTPException(404, "not found")
    return note.model_dump()


@router.post("/api/knowledge/learn")
def api_learn(req: LearnRequest):
    note = learn_topic(
        req.topic,
        depth=req.depth,
        category=req.category,
        project=PROJECTS.current,
    )
    return note.model_dump()


@router.delete("/api/knowledge/{topic}")
def delete_knowledge(topic: str):
    ok = KM.delete_note(topic)
    return {"ok": ok}


# ---- core memory ----
@router.get("/api/core-memory")
def get_core():
    return {"content": CORE.read(), "tokens": CORE.tokens(), "max": CORE.max_tokens}


@router.post("/api/core-memory")
def add_core(req: CoreFactRequest):
    msg = CORE.add_fact(req.fact, req.source)
    return {"message": msg}


@router.delete("/api/core-memory")
def delete_core(req: CoreFactDelete):
    return {"message": CORE.remove_fact(req.search_text)}


# ---- gaps ----
@router.get("/api/gaps")
def get_gaps():
    gaps = KM.hot_gaps(threshold=1)
    return {
        "gaps": gaps,
        "open": [g for g in gaps if not g["has_note_now"]],
        "closed": [g for g in gaps if g["has_note_now"]],
    }


# ---- capabilities ----
@router.get("/api/capabilities")
def get_capabilities():
    return {"block": _capabilities_block()}


# ---- quick-note ----
class QuickNoteRequest(BaseModel):
    text: str


@router.post("/api/knowledge/quick-note")
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
