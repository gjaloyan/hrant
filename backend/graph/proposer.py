"""LLM-proposed `relates_to` edges between facts (Phase 16C.1).

Called from the daily consolidation pipeline after a batch of new
facts has been promoted. Surfaces SEMANTIC relationships that
shared-topic edges miss: e.g. "User uses Tailscale" + "Whisper STT
runs at 100.124.210.21" are both about home-network infrastructure
but tagged with different topics, so no `is_about` edge connects
them. This step asks the LLM to find such pairs explicitly.

Symmetry: `relates_to` is conceptually undirected — A relates to B
iff B relates to A. We store ONE edge per pair, canonicalised by
sorting the two node ids. The neighborhood query (graph.query)
already walks edges in both directions, so consumers don't need
to know which direction we picked.

Why limit to top-N existing facts: input size + LLM cost. At ~100
chars per fact and 32 facts (12 new + 20 most-connected existing),
the prompt stays under 4 KB. The "most-connected" heuristic biases
toward integrating today's facts into the graph's hubs, which is
where new links carry the most leverage.

Failures don't propagate: a missing LLM, a malformed JSON, an
unparseable response — all log a warning and return an empty list.
The graph keeps its existing edges; consolidation moves on. Worst
case the user sees one digest with `links_added=[]`.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .model import GraphEdge
from .store import GRAPH, Graph

log = logging.getLogger(__name__)


# Soft caps. The proposer is meant to be cheap — if you find
# yourself wanting more, you probably want a different pipeline
# (graph rebuild from scratch).
MAX_LINKS_PER_RUN = 6
MAX_NEW_FACTS_IN_PROMPT = 15
MAX_EXISTING_FACTS_IN_PROMPT = 20


PROPOSER_SYSTEM = """You are a memory-linking module for a personal AI agent.

You see a list of FACTS the agent knows about its user. Some are
brand new (added today), some are existing knowledge. Find PAIRS
of facts that are SEMANTICALLY RELATED but don't already share an
obvious topic tag — connections the agent might otherwise miss.

Return strictly JSON:
{
  "links": [
    {
      "from": "<exact fact text from input>",
      "to":   "<exact fact text from input>",
      "reason": "one-sentence why they relate"
    }
  ]
}

Rules:
- Each `from` and `to` must MATCH a fact text from the input
  verbatim (the agent uses string equality to resolve back to
  graph nodes).
- Max 6 links total — prefer quality over quantity.
- SKIP pairs that are obviously the same topic (the tag-based
  edges already cover those).
- SKIP pairs that contradict each other (we don't yet surface
  contradictions in the UI — they'd just look like spurious
  links).
- Russian or English to match the source.
- Each `reason` is one short sentence."""


@dataclass
class ProposedLink:
    """One edge the LLM suggested. The fact texts are kept raw so
    the caller can show them in the digest UI; the graph integration
    resolves them to canonical fact_ids before persisting."""

    from_text: str
    to_text: str
    reason: str

    def to_dict(self) -> dict:
        return {"from": self.from_text, "to": self.to_text, "reason": self.reason}


# ─── Helpers ────────────────────────────────────────────────────────


def _select_existing_top_facts(
    *,
    graph: Graph,
    limit: int = MAX_EXISTING_FACTS_IN_PROMPT,
    exclude_texts: set[str],
) -> list[tuple[str, str]]:
    """Top-N most-connected facts in the graph, sorted by degree
    desc. Returns list of `(node_id, label)` tuples. Excludes any
    fact whose lower-cased text is in `exclude_texts` — that's the
    set of fact texts the caller already passed as `new_facts`.

    Why "most-connected" and not "most recent"? New connections to
    hubs have higher information value — linking today's "uses X"
    to an existing fact that already has 5 other connections is
    more useful than linking it to an isolated fact from yesterday
    that no one references."""
    degree: Counter = Counter()
    for e in graph.iter_edges():
        degree[e.source] += 1
        degree[e.target] += 1
    candidates: list[tuple[str, str, int]] = []
    for node in graph.iter_nodes(kind="fact"):
        if (node.label or "").strip().lower() in exclude_texts:
            continue
        candidates.append((node.id, node.label, degree[node.id]))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(nid, label) for nid, label, _deg in candidates[:limit]]


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Order the two node ids so `(A, B)` and `(B, A)` become the
    same canonical (source, target). Avoids duplicate edges when
    the same pair shows up from different prompt angles across
    runs."""
    return (a, b) if a <= b else (b, a)


def _resolve_to_fact_id(text: str, *, graph: Graph) -> Optional[str]:
    """Find the graph fact node whose label matches `text` after
    lower-case + whitespace-collapse normalisation. The LLM is
    asked to echo fact texts verbatim, but it occasionally tweaks
    punctuation; we normalise on both sides to recover from minor
    drift."""
    if not text:
        return None
    norm = " ".join(text.strip().lower().split())
    for node in graph.iter_nodes(kind="fact"):
        node_norm = " ".join((node.label or "").strip().lower().split())
        if node_norm == norm:
            return node.id
    return None


def _call_llm_for_links(
    *,
    new_facts: list[str],
    existing_facts: list[tuple[str, str]],
) -> list[ProposedLink]:
    """Build the prompt + call the router, parse the JSON, return
    a list of ProposedLink. Wraps the failure path: any exception
    yields `[]` and a logged warning."""
    if not new_facts and not existing_facts:
        return []

    # Build the user message: a numbered, deduped list of fact
    # texts. We feed lines as `NEW:` and `EXISTING:` tags so the
    # LLM can prioritise new→existing pairs (where the integration
    # value is highest).
    lines: list[str] = []
    for txt in new_facts[:MAX_NEW_FACTS_IN_PROMPT]:
        lines.append(f"NEW: {txt}")
    for _node_id, txt in existing_facts[:MAX_EXISTING_FACTS_IN_PROMPT]:
        lines.append(f"EXISTING: {txt}")
    user = "FACTS:\n" + "\n".join(lines)

    try:
        from ..llm import router, TaskType
        resp = router().call_json(
            TaskType.COMPLEX_SOLVING,
            PROPOSER_SYSTEM,
            user,
            max_tokens=1200,
            temperature=0.2,
        )
    except Exception as e:
        log.warning("graph.proposer LLM call failed: %s", e)
        return []

    raw_links = resp.get("links") if isinstance(resp, dict) else None
    if not isinstance(raw_links, list):
        return []

    out: list[ProposedLink] = []
    for raw in raw_links[:MAX_LINKS_PER_RUN]:
        if not isinstance(raw, dict):
            continue
        a = str(raw.get("from") or "").strip()
        b = str(raw.get("to") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        if not a or not b or a == b:
            continue
        out.append(ProposedLink(from_text=a, to_text=b, reason=reason))
    return out


# ─── Public entry ───────────────────────────────────────────────────


def propose_links(
    *,
    new_fact_texts: list[str],
    digest_date: str,
    graph: Optional[Graph] = None,
) -> list[dict]:
    """Ask the LLM to propose relates_to edges between today's new
    facts and a slice of existing hubs. Persist any returned edges
    into the graph. Returns a list of `{from, to, reason}` records
    suitable for the digest's `links_added` field.

    `new_fact_texts` is the list of texts that were just promoted
    in this consolidation run. Empty list → return `[]` (nothing to
    integrate).

    `digest_date` is stamped onto the edge's metadata so the user
    can later see which consolidation proposed it.
    """
    g = graph or GRAPH
    if not new_fact_texts:
        return []

    exclude = {t.strip().lower() for t in new_fact_texts}
    existing = _select_existing_top_facts(graph=g, exclude_texts=exclude)

    # If there are no existing facts to relate AGAINST and only one
    # new fact, the proposer has nothing useful to do. (Multiple
    # new facts in the same batch can still be related to each
    # other, so we don't skip when there are 2+ new + 0 existing.)
    if len(existing) == 0 and len(new_fact_texts) < 2:
        return []

    proposals = _call_llm_for_links(
        new_facts=new_fact_texts,
        existing_facts=existing,
    )
    if not proposals:
        return []

    persisted: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for link in proposals:
        a_id = _resolve_to_fact_id(link.from_text, graph=g)
        b_id = _resolve_to_fact_id(link.to_text, graph=g)
        if a_id is None or b_id is None or a_id == b_id:
            continue
        src, dst = _canonical_pair(a_id, b_id)
        if (src, dst) in seen_pairs:
            continue
        seen_pairs.add((src, dst))
        g.upsert_edge(GraphEdge(
            source=src, target=dst, kind="relates_to", weight=1.0,
            metadata={
                "reason": link.reason[:300],
                "source": f"consolidation:{digest_date}",
                "proposed_at": time.time(),
            },
        ))
        persisted.append({
            "from": link.from_text,
            "to": link.to_text,
            "reason": link.reason,
            "kind": "relates_to",
        })

    if persisted:
        try:
            g.save()
        except Exception as e:
            # Persist failure isn't fatal — the in-memory graph
            # still reflects the new edges; next call to .save()
            # (e.g. from `add_fact` in the next consolidation) will
            # flush them. Log and continue.
            log.warning("graph.proposer save failed: %s", e)
        log.info("graph.proposer: persisted %d new edge(s)", len(persisted))
    return persisted
