"""Trajectory memory — case-based reasoning over past solved turns.

AGI-roadmap item #2 (2026-06-11). The workspace already persists a
full artifact per turn (task, answer, tool-call order, verification)
under `workspace/turns/<id>.json` — 185+ of them on prod — but
nothing ever read them back. A small model doesn't need to INVENT a
solution when the agent already solved a similar task: it needs to
RETRIEVE and adapt the prior trajectory.

Three surfaces:
  - `index_turn(turn_id, artifact)` — called post-turn from
    run_unified. Only successful, multi-tool, non-chat turns qualify
    (a failed or trivial turn is not an experience worth replaying).
  - `backfill(limit=...)` — walk turns/ and index anything missed;
    wired into the nightly consolidation (memory replay during sleep).
  - `past_experience_block(task)` — embed the incoming task, top-K
    cosine over indexed trajectories, render a compact PAST
    EXPERIENCE prompt block. Injected on full-path turns only.

Mirrors the fact_search.py store pattern: a VectorStore keyed by
turn_id holds only vectors; the artifacts on disk stay the source
of truth and are loaded lazily for the 1-2 winning hits. A hit whose
artifact was swept by workspace retention is dropped from the index
on sight (self-healing).

Degrades to a no-op when the embedder is disabled — never raises
into the turn.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .embedder import EMBEDDER
from .vector_store import VectorStore

log = logging.getLogger(__name__)


# Qualification gates: what makes a past turn a reusable "experience".
_MIN_CONFIDENCE = 70          # verifier said the answer held up
_MIN_TOOL_CALLS = 2           # a real composed workflow, not a one-liner
_MIN_TASK_CHARS = 20          # too-short tasks carry no retrieval signal
# Similarity floor — below this the "experience" is topically unrelated
# and would just be prompt noise.
_MIN_SCORE = 0.60
_MAX_TOOL_SEQ = 12            # rendered tool-chain cap
_TASK_RENDER_CAP = 200
_OUTCOME_RENDER_CAP = 240


def _store_path() -> Path:
    from . import paths
    return paths.knowledge_dir() / "trajectory_embeddings.json"


def _turns_dir() -> Path:
    from .workspace import get_workspace, TURNS
    return get_workspace().root / TURNS


_STORE: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore(_store_path())
    return _STORE


def _new_store_for_test() -> VectorStore:
    """Re-instantiate the singleton from the current path. Tests use
    this after monkeypatching _store_path / HRANT_DATA_DIR."""
    global _STORE
    _STORE = VectorStore(_store_path())
    return _STORE


def _ensure_stamp(store: VectorStore, status: dict) -> bool:
    """Stamp the store on first use; wipe + restamp when the embedder
    backend/model/dim changed (vectors from different models don't
    share a space). Returns False when the embedder is unusable."""
    backend = status.get("backend")
    if backend in (None, "disabled"):
        return False
    dim = status.get("dim") or 0
    model = status.get("model") or ""
    if store._dim is None:  # type: ignore[attr-defined]
        store.stamp(dim, backend, model)
        return True
    if not store.is_compatible(dim, backend, model):
        for slug in list(store._items.keys()):  # type: ignore[attr-defined]
            store.remove(slug)
        store.stamp(dim, backend, model)
    return True


# ─── Qualification + extraction ───────────────────────────────────


def tool_sequence(artifact: dict) -> list[str]:
    """Ordered tool names from the thinking trace, consecutive
    duplicates collapsed (read_file x4 → read_file), capped."""
    seq: list[str] = []
    for step in (artifact.get("thinking_trace") or []):
        if not isinstance(step, dict):
            continue
        if step.get("event") not in ("tool", "tool_error"):
            continue
        tc = step.get("tool_call") or {}
        name = tc.get("name") if isinstance(tc, dict) else None
        if not name or not isinstance(name, str):
            continue
        if seq and seq[-1] == name:
            continue
        seq.append(name)
        if len(seq) >= _MAX_TOOL_SEQ:
            break
    return seq


def qualifies(artifact: dict) -> tuple[bool, str]:
    """(ok, reason). Only turns worth replaying make the index."""
    if artifact.get("is_chat"):
        return False, "chat-turn"
    task = str(artifact.get("user") or "").strip()
    if len(task) < _MIN_TASK_CHARS:
        return False, "task-too-short"
    if not str(artifact.get("answer") or "").strip():
        return False, "empty-answer"
    conf = int(artifact.get("confidence") or 0)
    if conf < _MIN_CONFIDENCE:
        return False, f"low-confidence-{conf}"
    n_tools = artifact.get("n_tool_calls")
    if n_tools is None:
        n_tools = len(tool_sequence(artifact))
    if int(n_tools) < _MIN_TOOL_CALLS:
        return False, f"too-few-tools-{n_tools}"
    return True, "ok"


# ─── Indexing ─────────────────────────────────────────────────────


def index_turn(turn_id: str, artifact: dict) -> bool:
    """Embed the turn's task text and add to the store. Returns True
    when indexed. Never raises — indexing is post-turn best-effort."""
    try:
        ok, _reason = qualifies(artifact)
        if not ok:
            return False
        status = EMBEDDER.status()
        store = get_store()
        if not _ensure_stamp(store, status):
            return False
        if store.has(turn_id):
            return False
        task = str(artifact.get("user") or "").strip()
        vec = EMBEDDER.embed(task)
        if not vec:
            return False
        store.add(turn_id, vec)
        return True
    except Exception as e:
        log.debug("trajectory index_turn(%s) failed: %s", turn_id, e)
        return False


def backfill(limit: int = 500) -> dict:
    """Index any qualifying turn artifacts not yet in the store.
    Called from the nightly consolidation (memory replay). Newest
    first so a capped run prioritises recent experience."""
    status = EMBEDDER.status()
    store = get_store()
    if not _ensure_stamp(store, status):
        return {"ok": False, "reason": "embedder_disabled",
                "indexed": 0, "skipped": 0}
    d = _turns_dir()
    if not d.exists():
        return {"ok": True, "indexed": 0, "skipped": 0}
    indexed = 0
    skipped = 0
    files = sorted(d.glob("*.json"), reverse=True)[: max(1, int(limit))]
    for p in files:
        tid = p.stem
        if store.has(tid):
            skipped += 1
            continue
        try:
            artifact = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if index_turn(tid, artifact):
            indexed += 1
        else:
            skipped += 1
    return {"ok": True, "indexed": indexed, "skipped": skipped}


# ─── Recall ───────────────────────────────────────────────────────


def _load_artifact(turn_id: str) -> Optional[dict]:
    p = _turns_dir() / f"{turn_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def recall_similar(task: str, limit: int = 2) -> list[dict]:
    """Top similar past trajectories above the score floor. Returns
    [{turn_id, score, ts, task, tool_seq, outcome, confidence}]."""
    if not task or len(task.strip()) < _MIN_TASK_CHARS:
        return []
    status = EMBEDDER.status()
    if status.get("backend") in (None, "disabled"):
        return []
    try:
        qvec = EMBEDDER.embed(task)
    except Exception:
        return []
    if not qvec:
        return []
    store = get_store()
    if store.count() == 0:
        return []
    # Over-fetch: some hits fall below the floor or lost their
    # artifact to retention sweeps.
    scored = store.search(qvec, k=max(limit * 3, 6))
    out: list[dict] = []
    for tid, score in scored:
        if score < _MIN_SCORE:
            continue
        artifact = _load_artifact(tid)
        if artifact is None:
            # Artifact swept by retention — self-heal the index.
            try:
                store.remove(tid)
            except Exception:
                pass
            continue
        out.append({
            "turn_id": tid,
            "score": round(float(score), 4),
            "ts": artifact.get("ts") or "",
            "task": str(artifact.get("user") or "").strip(),
            "tool_seq": tool_sequence(artifact),
            "outcome": str(artifact.get("answer") or "").strip(),
            "confidence": int(artifact.get("confidence") or 0),
        })
        if len(out) >= limit:
            break
    return out


def past_experience_block(task: str, limit: int = 2) -> str:
    """Rendered PAST EXPERIENCE prompt block, or "" when nothing
    relevant. Never raises."""
    try:
        hits = recall_similar(task, limit=limit)
    except Exception as e:
        log.debug("past_experience_block failed: %s", e)
        return ""
    if not hits:
        return ""
    lines = [
        "# PAST EXPERIENCE (similar tasks you already solved — adapt "
        "the approach, don't rediscover it)",
    ]
    for h in hits:
        task_preview = h["task"][:_TASK_RENDER_CAP]
        outcome = h["outcome"][:_OUTCOME_RENDER_CAP]
        ts = (h["ts"] or "")[:10]
        lines.append("")
        lines.append(
            f"## [{ts}] \"{task_preview}\" "
            f"(confidence {h['confidence']}, similarity {h['score']:.2f})"
        )
        if h["tool_seq"]:
            lines.append("Tools used: " + " -> ".join(h["tool_seq"]))
        if outcome:
            lines.append(f"Outcome: {outcome}")
    return "\n".join(lines)
