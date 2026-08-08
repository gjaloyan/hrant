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
        store.clear()  # one write, not a per-slug remove() loop (O(n^2))
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
        store.add(fid, vec, save=False)
        embedded += 1
    store.flush()  # single write for the whole batch (was O(n^2) per-add)
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


# Speaker prefixes that are NOT the owner (2026-08-08 audit). Measured on
# prod: 555 of 2778 rows in the owner's permanent memory — 20% — were written
# by `audit:claude` and `webui:bench-harness`. A probe typed during an audit
# ("My name is Alice. Remember it.") was retrieved at rank 1 against a real
# question, because the row's speaker_id was written but never read back.
_SYNTHETIC_SPEAKER_PREFIXES = ("audit:", "bench", "webui:bench", "test:")

# Cosine floor for THIS corpus. hybrid_searcher uses 0.45 with the comment
# that bge-m3 scores 0.30-0.45 on unrelated text — but the fact corpus was
# measured at a 0.49-0.51 noise ceiling, above that constant, so copying it
# would not have helped. The separation point is real: a true hit scored
# 0.719/0.593 and the noise behind it topped out at 0.510.
FACT_SCORE_FLOOR = 0.60


def _is_synthetic_speaker(sid: "str | None") -> bool:
    low = (sid or "").strip().lower()
    if not low:
        return False
    return any(p in low for p in _SYNTHETIC_SPEAKER_PREFIXES)


def search_facts(query: str, limit: int = 5,
                 score_floor: float | None = None,
                 include_synthetic: bool = False) -> list[dict]:
    """Facts above `score_floor`, by cosine similarity, authored by a real
    speaker. Returns [{summary, category, score, ts, source_turn, speaker_id},
    ...] — possibly FEWER than `limit`, including zero. Returns [] when the
    embedder is unavailable or the store is empty — never raises.

    Returning fewer than asked is the point: the previous version returned
    top-K unconditionally, so a query with no relevant fact was answered with
    the K least-irrelevant ones and the caller could not tell the difference.
    """
    if not query:
        return []
    status = EMBEDDER.status()
    if status.get("backend") in (None, "disabled"):
        return []

    store = get_store()
    if store.count() == 0:
        return []
    # Query-time compatibility check (prod bug 2026-06-11, found via
    # trajectory_memory which copied this module's pattern): when the
    # auto-probe embedder flips backends between processes (ollama/
    # bge-m3/1024 -> openai/1536), the query vector lands in a
    # different space and every cosine comes back 0.0 — fact recall
    # silently returns nothing while looking healthy. Refuse loudly;
    # FIRE_FACT_EMBEDDING_BACKFILL's wipe-and-restamp is the write
    # path that resolves a genuine backend migration.
    if not store.is_compatible(
        status.get("dim") or 0,
        status.get("backend") or "",
        status.get("model") or "",
    ):
        log.warning(
            "fact search skipped: store stamped %s but embedder is "
            "%s/%s/%s — backend flipped?",
            store.stats(), status.get("backend"), status.get("model"),
            status.get("dim"),
        )
        return []
    try:
        qvec = EMBEDDER.embed(query)
    except Exception:
        return []
    if not qvec:
        return []
    # Over-fetch: the floor and the speaker filter both drop rows, and
    # asking for exactly `limit` would return too few real hits.
    scored = store.search(qvec, k=max(1, min((int(limit) or 5) * 4, 50)))
    if not scored:
        return []
    # Map fact_id back to full row by re-scanning memory_facts.jsonl.
    # At ~1500 rows this is fast; if the corpus grows we'd build a
    # secondary id→row index.
    rows_by_id: dict[str, dict] = {}
    for summary, row in _iter_facts():
        rows_by_id[fact_id(summary)] = row
    floor = FACT_SCORE_FLOOR if score_floor is None else float(score_floor)
    out: list[dict] = []
    for fid, score in scored:
        row = rows_by_id.get(fid)
        if not row:
            continue
        if float(score) < floor:
            continue
        if not include_synthetic and _is_synthetic_speaker(row.get("speaker_id")):
            continue
        out.append({
            "summary": _summary_of(row),
            "category": row.get("category", "general"),
            "score": round(float(score), 4),
            "ts": row.get("ts"),
            "source_turn": row.get("source_turn"),
            "speaker_id": row.get("speaker_id"),
            "tags": row.get("tags") or [],
        })
    return out
