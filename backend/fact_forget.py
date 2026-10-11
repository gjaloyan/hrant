"""Forgetting a stored fact — everywhere it was stored, and reversibly.

The owner asked for it on 2026-10-11, after a memory drill had left its
probes in his long-term memory: "favourite colour teal", "favourite apricot
Shalakh", "home server Glacier", re-added two dozen times between 10 and 31
August. There was no way to take a fact back. Deleting the journal line by
hand would not have been enough either, because one fact lives in four
places:

  * memory_facts.jsonl   — the journal; every reader re-scans it
  * fact_embeddings.json — the vectors `search_facts` ranks by
  * the knowledge graph's `_memory` edges — what `MEMORY.recall` walks
  * the v2 graph's fact node (graph.json) — what `fact_graph` traverses

A forget removes the journal rows, drops the vectors no remaining row
needs, closes the graph edges (closed, not deleted — the graph keeps
history) and removes the v2 node. The removed rows are kept in
knowledge/self/forgotten_facts.jsonl, so a wrong forget can be undone
with `restore`.

Two steps by design: `candidates` shows what matches and changes nothing;
`forget` takes the ids it returned. A query alone never deletes, so a
loose word cannot take half the memory with it.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

MAX_CANDIDATES = 30


def _archive_path() -> Path:
    from . import paths
    return paths.knowledge_dir() / "self" / "forgotten_facts.jsonl"


def _row_id(row: dict) -> str:
    from .fact_search import _summary_of, fact_id
    s = _summary_of(row)
    return fact_id(s) if s else ""


def _in_scope(row: dict, speaker_id: Optional[str]) -> bool:
    return speaker_id is None or str(row.get("speaker_id") or "").strip() == speaker_id


def candidates(query: str, *, speaker_id: Optional[str] = None,
               limit: int = 20) -> list[dict]:
    """Facts that match `query`: the phrase as a whole word anywhere in the
    fact, plus the semantic neighbours `search_facts` finds. Changes nothing.

    `speaker_id` None searches every speaker (the owner's view).
    """
    from . import fact_search
    q = " ".join((query or "").split())
    if len(q) < 2:
        return []
    limit = max(1, min(int(limit or 20), MAX_CANDIDATES))
    word = re.compile(rf"(?<!\w){re.escape(q)}(?!\w)", re.IGNORECASE)
    found: dict[str, dict] = {}

    def _add(summary: str, row: dict, how: str) -> None:
        fid = fact_search.fact_id(summary)
        hit = found.get(fid)
        if hit is None:
            found[fid] = {"id": fid, "fact": summary, "speakers": [],
                          "learned": str(row.get("ts") or "")[:19],
                          "copies": 0, "matched": how}
            hit = found[fid]
        sp = str(row.get("speaker_id") or "")
        if sp not in hit["speakers"]:
            hit["speakers"].append(sp)

    for summary, row in fact_search._iter_facts():
        if _in_scope(row, speaker_id) and word.search(summary):
            _add(summary, row, "text")
    try:
        for hit in fact_search.search_facts(q, limit=limit, include_synthetic=True,
                                            speaker_id=speaker_id):
            s = (hit.get("summary") or "").strip()
            if s:
                _add(s, hit, "meaning")
    except Exception as e:
        log.debug("semantic candidates unavailable: %s", e)

    # Count every copy, so the owner sees what a forget will remove.
    ids = set(found)
    for summary, row in fact_search._iter_facts():
        fid = fact_search.fact_id(summary)
        if fid in ids and _in_scope(row, speaker_id):
            found[fid]["copies"] += 1
    ranked = sorted(found.values(), key=lambda h: (h["matched"] != "text", h["fact"]))
    return ranked[:limit]


def forget(ids: Iterable[str], *, speaker_id: Optional[str] = None,
           by: str = "") -> dict:
    """Remove every stored copy of the given fact ids. Never raises.

    Returns {forgotten: rows removed, facts: [summary...], ids: [...]}.
    """
    from . import fact_search
    from .memory_extractor import FACTS_FILE_LOCK
    wanted = {str(i).strip() for i in (ids or []) if str(i).strip()}
    out = {"ok": True, "forgotten": 0, "facts": [], "ids": []}
    if not wanted:
        return out
    path = fact_search._facts_path()
    gone: list[dict] = []
    try:
        with FACTS_FILE_LOCK:
            if not path.exists():
                return out
            keep: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    keep.append(line)
                    continue
                if _row_id(row) in wanted and _in_scope(row, speaker_id):
                    gone.append(row)
                else:
                    keep.append(line)
            if not gone:
                return out
            arch = _archive_path()
            arch.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with arch.open("a", encoding="utf-8") as f:
                for row in gone:
                    f.write(json.dumps({"forgotten_at": stamp, "by": by, "row": row},
                                       ensure_ascii=False) + "\n")
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
            tmp.replace(path)
    except Exception as e:
        log.warning("forget failed: %s", e)
        return {**out, "ok": False, "error": str(e)}

    removed_ids = {_row_id(r) for r in gone}
    _drop_vectors(removed_ids)
    _close_graph_edges(gone)
    _drop_graph_nodes(gone)
    summaries = []
    for r in gone:
        s = fact_search._summary_of(r)
        if s not in summaries:
            summaries.append(s)
    return {"ok": True, "forgotten": len(gone), "facts": summaries,
            "ids": sorted(removed_ids)}


def _drop_vectors(ids: set[str]) -> None:
    """Vectors no remaining row needs. A duplicate kept for another speaker
    still needs its vector, so check the journal, not just the ids."""
    try:
        from . import fact_search
        still = {fact_search.fact_id(s) for s, _ in fact_search._iter_facts()}
        store = fact_search.get_store()
        dropped = [fid for fid in ids if fid not in still and store.has(fid)]
        for fid in dropped:
            del store._items[fid]
        if dropped:
            store.flush()
    except Exception as e:
        log.debug("forget: vectors not dropped: %s", e)


def _close_graph_edges(rows: list[dict]) -> None:
    try:
        from .knowledge_graph import GRAPH
        for row in rows:
            scope = str(row.get("speaker_id") or "").strip() or None
            for t in row.get("triples") or []:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    GRAPH.invalidate(str(t[0]), str(t[1]), str(t[2]), scope=scope)
    except Exception as e:
        log.debug("forget: graph edges not closed: %s", e)


def _drop_graph_nodes(rows: list[dict]) -> None:
    try:
        from .fact_search import _summary_of
        from .graph.model import fact_id as node_id
        from .graph.store import GRAPH as G2
        changed = False
        for row in rows:
            s = _summary_of(row)
            if s and G2.remove_node(node_id(s)):
                changed = True
        if changed:
            G2.save()
    except Exception as e:
        log.debug("forget: graph nodes not dropped: %s", e)


def restore(ids: Iterable[str] = (), *, last: bool = False) -> dict:
    """Put forgotten rows back into the journal. `last` restores the most
    recent forget. The journal is what every reader scans; the vectors come
    back with the next embedding backfill and the v2 node with the next graph
    rebuild, while the closed legacy-graph edges stay closed as history."""
    from .fact_search import _facts_path
    from .memory_extractor import FACTS_FILE_LOCK
    arch = _archive_path()
    if not arch.exists():
        return {"ok": True, "restored": 0}
    wanted = {str(i).strip() for i in ids if str(i).strip()}
    with FACTS_FILE_LOCK:
        entries = [json.loads(l) for l in arch.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        if last and entries:
            stamp = entries[-1].get("forgotten_at")
            pick = [e for e in entries if e.get("forgotten_at") == stamp]
        else:
            pick = [e for e in entries if _row_id(e.get("row") or {}) in wanted]
        if not pick:
            return {"ok": True, "restored": 0}
        with _facts_path().open("a", encoding="utf-8") as f:
            for e in pick:
                f.write(json.dumps(e["row"], ensure_ascii=False) + "\n")
        rest = [e for e in entries if e not in pick]
        arch.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rest),
                        encoding="utf-8")
    return {"ok": True, "restored": len(pick)}
