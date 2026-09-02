"""Facts reached by their links, not by resembling the question.

The fact vector store answers "what is similar to this sentence". It
cannot answer "what do I know about my brother", because the useful facts
about a person rarely resemble the question that asks for them — they
resemble each other.

The graph can. Nightly consolidation builds 3675 fact nodes joined to
topics and entities by 21281 `is_about` / `mentions` edges, and until now
nothing traversed a single one of them during a turn: the store was
written every night and read by nobody.

Deliberately NOT a new tool. `search_knowledge` is what the agent already
reaches for, and a capability behind a tool the model must first discover
is the trap that left schedule_message and the tracker tools unreachable
for months. This hangs off the entry point that is already used.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# How many hub nodes a query is allowed to match, and how many facts to
# take from each. Small on purpose: this rides along on every
# `search_knowledge` call, and its whole value is a handful of facts the
# similarity search would have missed.
MAX_ANCHORS = 2
MAX_FACTS_PER_ANCHOR = 4


# Words too common to name anything. Kept tiny and multilingual because
# the owner writes in three languages; anything longer becomes a keyword
# list to maintain, and the degree ranking already discards weak hits.
_NOISE = frozenset("""
what who where when why how do does did you your yours i me my mine
the a an of about on in for to is are was were and or not tell know
что кто где когда почему как ты твой мой мне про о об и или не расскажи
знаешь есть был была это этот
""".split())


def _terms(text: str) -> list[str]:
    """Candidate anchors out of a sentence, longest first.

    Longest first because a specific word ("Tigran") beats a general one
    ("car") at naming what the question is about, and the first anchors
    found are the ones used.
    """
    import re

    words = [w for w in re.split(r"[^\wЀ-ӿ԰-֏-]+", text) if w]
    keep = [w for w in words if len(w) > 2 and w.lower() not in _NOISE]
    return sorted(dict.fromkeys(keep), key=len, reverse=True)[:6]


def facts_about(query: str, *, limit: int = 6) -> list[dict]:
    """Facts linked to whatever the query names. Never raises.

    Returns `[{summary, via, source: "fact_graph"}]` — `via` names the
    entity or topic the fact was reached through, which is what makes the
    result explainable rather than a bare list.
    """
    text = (query or "").strip()
    if len(text) < 3:
        return []
    try:
        from .graph import query as gq

        # `gq.search` matches the query as a SUBSTRING of a node label, so
        # handing it a whole sentence finds nothing — no label contains
        # "what do you know about Tigran". Search the terms instead, and
        # let the graph's own degree ranking pick the hub.
        anchors: list[dict] = []
        seen_ids: set[str] = set()
        for term in _terms(text):
            for n in gq.search(term, limit=8):
                if n.get("kind") not in ("entity", "topic"):
                    continue  # a matching FACT is what the vector store returns
                if not n.get("degree"):
                    continue  # nothing to traverse
                if n["id"] in seen_ids:
                    continue
                seen_ids.add(n["id"])
                anchors.append(n)
            if len(anchors) >= MAX_ANCHORS:
                break
        anchors = anchors[:MAX_ANCHORS]
        if not anchors:
            return []

        out: list[dict] = []
        seen: set[str] = set()
        for a in anchors:
            hood = gq.neighborhood(a["id"])
            if not hood:
                continue
            taken = 0
            for entry in hood.get("neighbors", []):
                node = entry.get("node") or {}
                if node.get("kind") != "fact":
                    continue
                label = (node.get("label") or "").strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                out.append({
                    "summary": label,
                    "via": a.get("label") or a.get("id"),
                    "source": "fact_graph",
                })
                taken += 1
                if taken >= MAX_FACTS_PER_ANCHOR or len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out
    except Exception as exc:
        # Retrieval is an enhancement; a broken graph must never take the
        # search down with it.
        log.debug("fact_graph lookup failed: %s", exc)
        return []
