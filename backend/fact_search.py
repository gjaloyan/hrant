"""Fact-level semantic search.

Audit 2026-05-27: memory_facts.jsonl had 1453 entries with NO
vector index. `search_knowledge` only covered topic-notes, so
facts were keyword-searchable at best (and not even that —
nothing actively scanned memory_facts.jsonl from query time).

This module owns a parallel vector store keyed by `fact_id`
(SHA-1 prefix of the normalised summary text) plus three
functions:

  - `backfill_fact_embeddings()`: walk memory_facts.jsonl, embed
    each unique summary, persist to fact_embeddings.json. Safe
    to re-run; honours embedder dim/model stamp like the existing
    note backfill.
  - `search_facts(query, limit)`: embed query, top-K cosine over
    the fact store, return full fact records (summary + metadata).
  - `count_unembedded_facts()`: coverage signal for the
    FIRE_FACT_EMBEDDING_BACKFILL lever.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .embedder import EMBEDDER
from .vector_store import VectorStore

log = logging.getLogger(__name__)


def _facts_path() -> Path:
    from . import paths
    return paths.knowledge_dir() / "memory_facts.jsonl"


def _fact_store_path() -> Path:
    from . import paths
    return paths.knowledge_dir() / "fact_embeddings.json"


def fact_id(summary: str) -> str:
    """Stable id from a fact summary. Normalises whitespace + case
    so trivial reformatting doesn't fork the id."""
    if summary is None:
        summary = ""
    normalised = " ".join(summary.strip().lower().split())
    return "f_" + hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:12]


def _summary_of(row: dict) -> str:
    """Extract the searchable text from a memory_facts row. Some
    legacy rows have `summary`, others have `text`."""
    s = (row.get("summary") or row.get("text") or "").strip()
    return s


# ─── singleton store ──────────────────────────────────────────────


_STORE: VectorStore | None = None


def get_store() -> VectorStore:
    """Lazy singleton — the path resolves via `paths.knowledge_dir()`
    which respects HRANT_DATA_DIR for tests."""
    global _STORE
    if _STORE is None:
        _STORE = VectorStore(_fact_store_path())
    return _STORE


def _new_store_for_test() -> VectorStore:
    """Re-instantiate the singleton from the current path. Tests
    use this after monkeypatching _fact_store_path."""
    global _STORE
    _STORE = VectorStore(_fact_store_path())
    return _STORE


# ─── coverage / backfill ──────────────────────────────────────────


def _iter_facts() -> list[tuple[str, dict]]:
    """Read every row in memory_facts.jsonl and return
    [(summary, full_row), ...] skipping empty-summary rows."""
    p = _facts_path()
    if not p.exists():
        return []
    out: list[tuple[str, dict]] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            summary = _summary_of(row)
            if not summary:
                continue
            out.append((summary, row))
    except OSError as exc:
        log.warning("fact_search: read %s failed: %s", p, exc)
    return out


def count_unembedded_facts() -> int:
    store = get_store()
    rows = _iter_facts()
    if not rows:
        return 0
    n = 0
    for summary, _row in rows:
        if not store.has(fact_id(summary)):
            n += 1
    return n


def backfill_fact_embeddings(*, force: bool = False) -> dict:
    """Embed every fact summary that isn't yet in the store.
    Returns stats. On embedder unavailable, returns ok=False."""
    status = EMBEDDER.status()
    backend = status.get("backend")
    if backend in (None, "disabled"):
        return {
            "ok": False,
            "reason": f"embedder_disabled (last_error={status.get('last_error')!r})",
            "embedded": 0,
            "skipped": 0,
            "errors": 0,
        }
    dim = status.get("dim") or 0
    model = status.get("model") or ""

    store = get_store()
    if force or not store.is_compatible(dim, backend, model):
        for slug in list(store._items.keys()):  # type: ignore[attr-defined]
            store.remove(slug)
        store.stamp(dim, backend, model)
    else:
        # Stamp on cold start.
        if store._dim is None:  # type: ignore[attr-defined]
            store.stamp(dim, backend, model)

    embedded = 0
    skipped = 0
    errors = 0
    # Dedup within this run — same summary text appearing twice in
    # memory_facts.jsonl (legacy duplicates) shouldn't re-embed.
    seen_ids: set[str] = set()
    for summary, _row in _iter_facts():
        fid = fact_id(summary)
        if fid in seen_ids:
            continue
        seen_ids.add(fid)
        if store.has(fid):
            skipped += 1
            continue
        try:
            vec = EMBEDDER.embed(summary)
        except Exception as exc:
            log.debug("fact_search: embed raised: %s", exc)
            errors += 1
            continue
        if not vec:
            errors += 1
            continue
        store.add(fid, vec)
        embedded += 1
    return {
        "ok": True,
        "backend": backend,
        "model": model,
        "dim": dim,
        "embedded": embedded,
        "skipped": skipped,
        "errors": errors,
        "total_unique": len(seen_ids),
    }


# ─── search ───────────────────────────────────────────────────────


def search_facts(query: str, limit: int = 5) -> list[dict]:
    """Top-K facts by cosine similarity. Returns
    [{summary, category, score, ts, source_turn}, ...]. Returns
    [] when the embedder is unavailable or the store is empty —
    never raises."""
    if not query:
        return []
    status = EMBEDDER.status()
    if status.get("backend") in (None, "disabled"):
        return []
    try:
        qvec = EMBEDDER.embed(query)
    except Exception:
        return []
    if not qvec:
        return []

    store = get_store()
    if store.count() == 0:
        return []
    scored = store.search(qvec, k=max(1, min(int(limit) or 5, 50)))
    if not scored:
        return []
    # Map fact_id back to full row by re-scanning memory_facts.jsonl.
    # At ~1500 rows this is fast; if the corpus grows we'd build a
    # secondary id→row index.
    rows_by_id: dict[str, dict] = {}
    for summary, row in _iter_facts():
        rows_by_id[fact_id(summary)] = row
    out: list[dict] = []
    for fid, score in scored:
        row = rows_by_id.get(fid)
        if not row:
            continue
        out.append({
            "summary": _summary_of(row),
            "category": row.get("category", "general"),
            "score": round(float(score), 4),
            "ts": row.get("ts"),
            "source_turn": row.get("source_turn"),
            "tags": row.get("tags") or [],
        })
    return out
