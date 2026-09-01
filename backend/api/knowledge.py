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
from ._auth import require_owner_for_writes

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
    require_owner_for_writes(action="learning a new topic")
    note = learn_topic(
        req.topic,
        depth=req.depth,
        category=req.category,
        project=PROJECTS.current,
    )
    return note.model_dump()


@router.delete("/api/knowledge/{topic}")
def delete_knowledge(topic: str):
    require_owner_for_writes(action="deleting a knowledge note")
    ok = KM.delete_note(topic)
    return {"ok": ok}


# ---- core memory ----
@router.get("/api/core-memory")
def get_core():
    return {"content": CORE.read(), "tokens": CORE.tokens(), "max": CORE.max_tokens}


@router.post("/api/core-memory")
def add_core(req: CoreFactRequest):
    require_owner_for_writes(action="adding a core fact")
    msg = CORE.add_fact(req.fact, req.source)
    return {"message": msg}


@router.delete("/api/core-memory")
def delete_core(req: CoreFactDelete):
    require_owner_for_writes(action="deleting a core fact")
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


@router.get("/api/capabilities/tools")
def get_capability_tools():
    """Every tool, with its FULL description and how it is reached.

    `/api/capabilities` returns the block that goes into the system
    prompt, where descriptions are clipped to 100 characters because every
    one of them is billed on every turn. The WebUI was rendering that
    prompt artifact, so a reader browsing what the agent can do got
    sentences cut mid-word ("...to read J").

    Truncation is a prompt concern. A person reading the list wants the
    whole sentence, plus the thing the block cannot say: whether the tool
    is always available or sits behind a bundle the model must load first.
    """
    from ..builtin_tools import get_registry
    from ..skills import SKILLS
    from ..tool_bundles import BASE_TOOLS, TOOL_BUNDLES

    bundle_of: dict[str, str] = {}
    for bundle, names in TOOL_BUNDLES.items():
        for n in names:
            bundle_of[n] = bundle

    registry = get_registry()
    tools = []
    for name, tool in sorted(registry.tools.items()):
        tools.append({
            "name": name,
            "description": tool.description or "",
            "origin": getattr(tool, "origin", "builtin"),
            "always_on": name in BASE_TOOLS,
            "bundle": bundle_of.get(name, ""),
        })

    SKILLS.ensure_loaded()
    skills = [{
        "name": sk.name,
        "description": sk.description or "",
        "enabled": bool(getattr(sk, "enabled", True)),
        "triggers": list(getattr(sk, "triggers", []) or [])[:8],
    } for sk in SKILLS.skills]

    return {"tools": tools, "skills": skills}


# ---- quick-note ----
class QuickNoteRequest(BaseModel):
    text: str


@router.post("/api/knowledge/quick-note")
def quick_note(req: QuickNoteRequest):
    require_owner_for_writes(action="saving a quick note")
    note = KM.save_note(
        topic=req.text[:40],
        body=req.text,
        category="personal",
        keywords=[req.text.split()[0].lower()] if req.text.strip() else [],
        source="user_quick_note",
        confidence="verified",
    )
    return {"topic": note.frontmatter.topic, "path": str(note.path)}
