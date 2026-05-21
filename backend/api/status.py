"""Status / router-stats / mode summary endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import CONFIG
from ..core_memory import CORE
from ..finetune import store as finetune_store
from ..knowledge_manager import KM
from ..llm import router as llm_router
from ..model_versions import VERSIONS
from ..project_mode import PROJECTS

router = APIRouter()


@router.get("/api/status")
def status():
    topics = KM.list_topics()
    by_cat: dict[str, int] = {}
    for t in topics:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
    cur = VERSIONS.current()
    try:
        router_state = llm_router().stats()
    except Exception as e:
        router_state = {"error": str(e)}
    # Phase 3a follow-up: surface the REAL active model (the pinned
    # one from active_model.json) instead of the legacy
    # `CONFIG.model_a` field which is just the mode-preset default.
    # When `active_model.json` is set, that's what every LLM call
    # actually routes to — the status bar lied otherwise.
    real_model_a = CONFIG.model_a.get("model")
    real_provider_a = CONFIG.model_a.get("provider") or "anthropic"
    try:
        from ..providers import ACTIVE_MODEL
        pinned = ACTIVE_MODEL.get() or {}
        if pinned.get("model"):
            real_model_a = pinned["model"]
        if pinned.get("provider_type"):
            real_provider_a = pinned["provider_type"]
    except Exception:
        pass
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
        "model_a": real_model_a,
        "model_a_provider": real_provider_a,
        "model_b": (CONFIG.model_b or {}).get("model") if CONFIG.model_b else None,
        "model_version": cur.tag if cur else None,
        "router": router_state,
    }


@router.get("/api/router/stats")
def router_stats():
    try:
        return llm_router().stats()
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/mode")
def get_mode():
    return {
        "mode": CONFIG.mode,
        "finetune_enabled": CONFIG.finetune_enabled,
        "training_location": CONFIG.training_location,
        "model_a": CONFIG.model_a.get("model"),
        "model_b": (CONFIG.model_b or {}).get("model") if CONFIG.model_b else None,
    }
