"""Graph data model.

A Hrant knowledge graph is intentionally TINY in vocabulary —
four node kinds, five edge kinds. Anything you can express in
this vocabulary, the UI can render. Anything you can't, ask
yourself if you actually need it before extending.

Node kinds:
    fact      — a single declarative statement from memory_facts.jsonl
    topic     — a free-form tag/category (e.g. "voice", "tailscale")
    skill     — one of the installed Skills (Phase 12)
    project   — a goal/project (currently sourced from goals.json)

Edge kinds:
    is_about     — fact → topic (the fact relates to that topic)
    uses         — skill → topic (skill's trigger words are topics)
    mentions     — fact → entity (RDF-style triple object)
    relates_to   — fact ↔ fact (LLM-proposed similarity, Phase 16C.1)
    continues    — project → fact (the fact advances that project)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


NODE_KINDS: tuple[str, ...] = ("fact", "topic", "skill", "project", "entity")
EDGE_KINDS: tuple[str, ...] = (
    "is_about", "uses", "mentions", "relates_to", "continues",
)


@dataclass
class GraphNode:
    """One vertex. `id` is canonical — `<kind>:<slug>` so two
    different sources can produce the same node and the deduper
    in `store.upsert_node` recognises them."""

    id: str
    kind: str           # one of NODE_KINDS
    label: str          # human-readable name
    weight: float = 1.0  # for sizing in viz / ranking in search
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GraphNode":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class GraphEdge:
    """One directed edge. The `kind` determines the semantics —
    `is_about` is fact → topic, `uses` is skill → topic, etc.
    Edges are deduped by (source, target, kind) — a duplicate
    upsert just bumps the weight."""

    source: str
    target: str
    kind: str           # one of EDGE_KINDS
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        """Identity for dedup. Same triple = same edge regardless
        of weight or metadata."""
        return (self.source, self.target, self.kind)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GraphEdge":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


# ─── ID helpers ───────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Canonicalise a string into an id-safe slug. Lower-cases,
    strips, replaces whitespace + most punctuation with `_`.
    Two facts/topics that differ only in casing or whitespace
    collide on the same id, which is what we want for dedup."""
    if not text:
        return "_"
    s = text.strip().lower()
    out: list[str] = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "/"):
            out.append("_")
        # anything else (punctuation, emoji) gets dropped silently
    slug = "".join(out).strip("_")
    return slug or "_"


def fact_id(text: str) -> str:
    """Facts are identified by a hash of their normalised text.
    The slugifier alone would collide too easily for sentence-
    length facts; use a short blake2 instead."""
    import hashlib
    norm = " ".join((text or "").strip().lower().split())
    digest = hashlib.blake2b(norm.encode("utf-8"), digest_size=6).hexdigest()
    return f"fact:{digest}"


def topic_id(label: str) -> str:
    return f"topic:{_slugify(label)}"


def skill_id(name: str) -> str:
    return f"skill:{_slugify(name)}"


def project_id(name: str) -> str:
    return f"project:{_slugify(name)}"


def entity_id(label: str) -> str:
    return f"entity:{_slugify(label)}"
