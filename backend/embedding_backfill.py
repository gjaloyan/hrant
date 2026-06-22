"""Embed all knowledge notes (or just missing ones) into the vector store.

Used by:
  - the bootstrap path when an embedder backend first becomes available
  - the `/api/memory/embeddings/backfill` admin endpoint
  - the bench harness before measuring vector retrieval

Designed to be safe to re-run: skips notes already embedded with the
current (backend, model, dim) tuple. Pass `force=True` to rebuild
everything from scratch (used after the user switches embedder).
"""
from __future__ import annotations

import logging
from typing import Optional

from .embedder import EMBEDDER
from .knowledge_manager import KM, _slug
from .vector_store import VECTOR_STORE

log = logging.getLogger(__name__)


def missing_count() -> dict:
    """Return how many KB notes lack a current vector.

    "Missing" = note exists in the index but its slug isn't in
    VECTOR_STORE *and* the store's stamp matches the current embedder
    (so we don't false-alarm on a stale-store mismatch).
    """
    status = EMBEDDER.status()
    backend = status.get("backend")
    dim = status.get("dim") or 0
    model = status.get("model") or ""
    notes = KM.list_topics()
    total = len(notes)
    if backend in (None, "disabled"):
        return {
            "total_notes": total,
            "embedded": VECTOR_STORE.count(),
            "missing": total,
            "stale_store": False,
            "reason": "embedder_disabled",
        }
    if not VECTOR_STORE.is_compatible(dim, backend, model):
        return {
            "total_notes": total,
            "embedded": VECTOR_STORE.count(),
            "missing": total,
            "stale_store": True,
            "reason": "store_dim_or_model_mismatch",
        }
    have = sum(1 for entry in notes if VECTOR_STORE.has(_slug(entry.topic)))
    return {
        "total_notes": total,
        "embedded": have,
        "missing": total - have,
        "stale_store": False,
        "reason": "ok" if total == have else "missing_notes",
    }


def backfill_embeddings(*, force: bool = False, limit: Optional[int] = None) -> dict:
    """Embed every note that's missing from VECTOR_STORE.

    Returns a stats dict — useful for surfacing in the UI / API response.
    """
    status = EMBEDDER.status()
    if status["backend"] in (None, "disabled"):
        return {
            "ok": False,
            "reason": "no_embedder_available",
            "backend": status["backend"],
        }

    dim = status["dim"] or 0
    backend = status["backend"]
    model = status["model"] or ""

    if force or not VECTOR_STORE.is_compatible(dim, backend, model):
        # Stale or forced: clear and re-stamp (one write, not per-slug).
        VECTOR_STORE.clear()
        VECTOR_STORE.stamp(dim, backend, model)

    stats = {
        "ok": True,
        "backend": backend,
        "model": model,
        "dim": dim,
        "embedded": 0,
        "skipped": 0,
        "errors": 0,
        "total": 0,
    }

    topics = KM.list_topics()
    if limit:
        topics = topics[:limit]
    stats["total"] = len(topics)

    # Warm the model before the batch. On the CPU host the FIRST embed after
    # idle can be slow enough to hit the timeout (cold load / weight page-in),
    # which left the first note erroring on every backfill while the rest
    # succeeded — so the store stuck at N-1/N and the FIRE_EMBEDDING_BACKFILL
    # lever retried (and re-logged a timeout) every tick. A tiny throwaway
    # embed pays the warm-up cost up front.
    if topics:
        try:
            EMBEDDER.embed("warmup")
        except Exception:
            pass

    for entry in topics:
        slug = _slug(entry.topic)
        if VECTOR_STORE.has(slug) and not force:
            stats["skipped"] += 1
            continue
        try:
            note = KM.get_note(entry.topic)
            if note is None:
                stats["errors"] += 1
                continue
            text = f"{note.frontmatter.topic}\n\n{note.body}".strip()
            # One retry: a single transient timeout (CPU contention) should not
            # permanently leave a note unembedded and re-trigger the lever.
            vec = EMBEDDER.embed(text) or EMBEDDER.embed(text)
            if not vec:
                stats["errors"] += 1
                continue
            VECTOR_STORE.add(slug, vec, save=False)
            stats["embedded"] += 1
        except Exception as e:
            log.warning("backfill embed failed for %s: %s", entry.topic, e)
            stats["errors"] += 1

    VECTOR_STORE.flush()  # single write for the whole batch (was O(n^2))
    return stats
