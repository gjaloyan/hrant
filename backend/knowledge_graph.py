"""Lightweight knowledge graph: entities, relations, graph traversal.

Inspired by LightRAG's approach but without heavy dependencies.
Stores a directed graph of entity-relation-entity triples in a single JSON file.
Entities are linked to source notes, enabling graph-based retrieval:
when the user asks about X, we can find related concepts Y and Z
through the graph even if they don't share keywords.

Design:
  - Graph stored as adjacency list in knowledge/graph.json
  - Each edge: {source_entity, relation, target_entity, source_note, weight,
                valid_from, valid_to, confidence}
  - valid_from / valid_to are optional ISO-date strings; None means
    "always valid". Lets us answer "what was true on 2026-01-15" without
    rewriting history.
  - Entity extraction happens during note creation (piggybacks on existing LLM call)
  - Graph search: BFS from query entities, collect related notes within N hops
  - No embeddings, no vector DB, no external dependencies
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG


# Stop words filtered out of query entity extraction. Picking the
# question-marker subset matters most ("what is the X" → keep "X", drop
# the rest). Bilingual because notes and queries mix EN + RU.
_QUERY_STOPWORDS: frozenset[str] = frozenset({
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "when", "where", "who", "whom", "why", "how", "which",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "so", "if", "then", "than",
    "this", "that", "these", "those", "it", "its",
    "i", "me", "my", "you", "your", "we", "our", "they", "them", "their",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "not", "no", "none", "any", "some", "all", "each", "every",
    "about", "into", "over", "under", "between", "through",
    # Russian
    "что", "кто", "когда", "где", "куда", "откуда", "почему", "зачем", "как",
    "и", "или", "но", "а", "же", "ли", "не", "ни",
    "в", "во", "на", "над", "под", "за", "перед", "у", "о", "об", "к",
    "с", "со", "из", "по", "при", "до", "после", "для", "без",
    "это", "то", "тот", "та", "те", "этот", "эта",
    "я", "ты", "мы", "вы", "он", "она", "они", "его", "её", "их",
    "быть", "был", "была", "было", "были",
    "есть", "нет", "уже", "ещё",
})


def _is_valid_at(edge: dict, as_of: Optional[str]) -> bool:
    """Return True if the edge is valid at `as_of` (ISO date/datetime str).

    None bounds mean "open" (always valid in that direction). If as_of is
    None, only edges with valid_to=None pass — i.e. the "current" view.
    """
    vf = edge.get("valid_from")
    vt = edge.get("valid_to")
    if as_of is None:
        return vt is None  # current view: only open-ended edges
    if vf and as_of < vf:
        return False
    if vt and as_of >= vt:
        return False
    return True


class KnowledgeGraph:
    """In-memory knowledge graph backed by a JSON file."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "graph.json")
        # Adjacency list: entity -> [{target, relation, note, weight}]
        self._edges: dict[str, list[dict]] = {}
        # Reverse index: note_slug -> set of entities mentioned in it
        self._note_entities: dict[str, set[str]] = defaultdict(set)
        # Reverse target index: target_n -> list of (subject, edge_dict).
        # Built once at load + maintained on add/remove. Without this,
        # memory_extractor.recall scans the entire graph for every
        # query term — O(N×M). With it, "where is X mentioned as a
        # target?" is O(1).
        self._target_index: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._edges = data.get("edges", {})
                self._rebuild_indexes()
            except Exception:
                self._edges = {}
                self._note_entities = defaultdict(set)
                self._target_index = defaultdict(list)

    def _rebuild_indexes(self) -> None:
        """Rebuild both reverse indexes from `self._edges`. Called on
        load and after destructive operations (remove_note)."""
        self._note_entities = defaultdict(set)
        self._target_index = defaultdict(list)
        for entity, targets in self._edges.items():
            for edge in targets:
                note = edge.get("note", "")
                if note:
                    self._note_entities[note].add(entity)
                    self._note_entities[note].add(edge.get("target", ""))
                tgt = edge.get("target", "")
                if tgt:
                    self._target_index[tgt].append((entity, edge))

    def find_facts_by_target(self, target: str) -> list[tuple[str, dict]]:
        """O(1) reverse lookup: given an entity that appears as the
        TARGET of some edge, return the (subject, edge) pairs that
        point to it. Used by memory_extractor.recall instead of the
        old full-graph scan."""
        return list(self._target_index.get(self._normalize(target), []))

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Preserve any v2 schema fields (version / nodes / edges_v2)
            # that a previous migration wrote. Without this, the next
            # legacy save() strips them — undoing the migration and
            # making graph_rebuild lever skip again. Round-trip is the
            # cheapest reconciliation while the two writers coexist.
            preserved: dict = {}
            if self.path.exists():
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict):
                        for key in ("version", "nodes", "edges_v2"):
                            if key in existing:
                                preserved[key] = existing[key]
                except (OSError, ValueError):
                    pass
            body: dict = {"edges": self._edges}
            body.update(preserved)
            self.path.write_text(
                json.dumps(body, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # graph is best-effort

    def _normalize(self, entity: str) -> str:
        """Normalize entity name for consistent matching."""
        return entity.strip().lower()

    # Relations where the same subject can only have ONE current target
    # at a time. A new `(subject, relation, new_target)` implies any
    # earlier `(subject, relation, old_target)` is no longer "current".
    # We auto-invalidate the old edge with `valid_to = today` and stamp
    # the new edge with `valid_from = today`. Listed conservatively —
    # `brother_of`, `is_a`, `contains` are NOT here because they're
    # multi-valued or permanent.
    SINGLE_VALUED_RELATIONS: frozenset[str] = frozenset({
        "lives_in", "lives", "located_at", "is_at",
        "works_at", "works_for", "works_as", "employed_at", "manages",
        "costs", "has_price", "price", "salary",
        "status", "current_status", "current_model", "version",
        "owns", "uses",
        "married_to", "partner_of",
    })

    def add_relations(
        self,
        triples: list[tuple[str, str, str]],
        source_note: str,
        weight: float = 1.0,
        *,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        confidence: float = 1.0,
        auto_invalidate: bool = True,
    ) -> int:
        """Add entity-relation-entity triples from a note.

        Args:
            triples: list of (subject, relation, object) strings
            source_note: slug of the note these came from
            weight: edge weight (higher = stronger connection)
            valid_from / valid_to: optional ISO-date strings bounding when
                this fact was/is true. None = open-ended.
            confidence: 0..1 — how sure we are this triple holds.

        Returns:
            Number of edges added.
        """
        added = 0
        for subj, rel, obj in triples:
            subj_n = self._normalize(subj)
            obj_n = self._normalize(obj)
            if not subj_n or not obj_n or not rel.strip():
                continue

            if subj_n not in self._edges:
                self._edges[subj_n] = []

            # Temporal auto-invalidation: for single-valued relations
            # (`lives_in`, `costs`, `status`, …) close any existing OPEN
            # edge that points to a DIFFERENT target — the new fact
            # supersedes the old one. Stamp valid_from on the new edge
            # if the caller didn't supply one. Skip when caller is
            # explicitly back-filling history (valid_to set).
            rel_clean = rel.strip()
            current_vfrom = valid_from
            # Idempotence guard at the top: if a SINGLE_VALUED_RELATIONS
            # add comes in for the same (subject, target) that already
            # has an OPEN edge, the call is a no-op regardless of the
            # caller's valid_from. Skip the entire write block — both
            # forward and reverse — so we don't write a duplicate that
            # disagrees on metadata. (Earlier the auto-invalidation
            # path stamped today on a transition; a re-add then had
            # `current_vfrom = None` mismatching the existing edge's
            # stamped today, producing a duplicate.)
            if (
                auto_invalidate
                and valid_to is None
                and rel_clean in self.SINGLE_VALUED_RELATIONS
                and any(
                    e.get("relation") == rel_clean
                    and e.get("target") == obj_n
                    and e.get("valid_to") is None
                    for e in self._edges[subj_n]
                )
            ):
                continue
            if (
                auto_invalidate
                and valid_to is None
                and rel_clean in self.SINGLE_VALUED_RELATIONS
            ):
                # Reaching here means: single-valued relation, NEW
                # target. Close any existing open edges with a
                # DIFFERENT target and stamp the new edge with today's
                # date (only if we actually closed something — the
                # date is a transition timestamp, not a creation
                # timestamp).
                today = datetime.utcnow().strftime("%Y-%m-%d")
                closed_something = False
                for existing in self._edges[subj_n]:
                    if (
                        existing.get("relation") == rel_clean
                        and existing.get("target") != obj_n
                        and existing.get("valid_to") is None
                    ):
                        existing["valid_to"] = today
                        closed_something = True
                        # Mirror on the inverse edge so traversal stays consistent.
                        old_target = existing.get("target", "")
                        for inv in self._edges.get(old_target, []):
                            if (
                                inv.get("target") == subj_n
                                and inv.get("relation") == f"inverse:{rel_clean}"
                                and inv.get("valid_to") is None
                            ):
                                inv["valid_to"] = today
                if closed_something and current_vfrom is None:
                    current_vfrom = today

            # Avoid exact duplicates (same target+note+validity span).
            # Compare against the SAME values the new edge will be
            # written with, not the caller's parameter — when
            # auto-invalidation stamped `current_vfrom = today` above,
            # `valid_from` (the parameter) is still None and would
            # always mismatch the freshly-written edge, producing a
            # duplicate on every idempotent re-add.
            exists = any(
                e["target"] == obj_n
                and e["note"] == source_note
                and e.get("valid_from") == current_vfrom
                and e.get("valid_to") == valid_to
                for e in self._edges[subj_n]
            )
            if not exists:
                edge: dict = {
                    "target": obj_n,
                    "relation": rel_clean,
                    "note": source_note,
                    "weight": weight,
                }
                if current_vfrom is not None:
                    edge["valid_from"] = current_vfrom
                if valid_to is not None:
                    edge["valid_to"] = valid_to
                if confidence != 1.0:
                    edge["confidence"] = confidence
                self._edges[subj_n].append(edge)
                self._target_index[obj_n].append((subj_n, edge))
                added += 1

            # Bidirectional: add reverse edge with lower weight
            if obj_n not in self._edges:
                self._edges[obj_n] = []
            # Reverse edge dedup + write must also see `current_vfrom`,
            # otherwise forward and inverse edges drift apart on
            # temporal metadata.
            rev_exists = any(
                e["target"] == subj_n
                and e["note"] == source_note
                and e.get("valid_from") == current_vfrom
                and e.get("valid_to") == valid_to
                for e in self._edges[obj_n]
            )
            if not rev_exists:
                rev_edge: dict = {
                    "target": subj_n,
                    "relation": f"inverse:{rel.strip()}",
                    "note": source_note,
                    "weight": weight * 0.5,
                }
                if current_vfrom is not None:
                    rev_edge["valid_from"] = current_vfrom
                if valid_to is not None:
                    rev_edge["valid_to"] = valid_to
                if confidence != 1.0:
                    rev_edge["confidence"] = confidence
                self._edges[obj_n].append(rev_edge)
                self._target_index[subj_n].append((obj_n, rev_edge))

            # Update reverse index
            self._note_entities[source_note].add(subj_n)
            self._note_entities[source_note].add(obj_n)

        if added > 0:
            self._save()
        return added

    def remove_note(self, source_note: str) -> None:
        """Remove all edges from a specific note."""
        for entity in list(self._edges.keys()):
            self._edges[entity] = [
                e for e in self._edges[entity]
                if e.get("note") != source_note
            ]
            if not self._edges[entity]:
                del self._edges[entity]
        self._note_entities.pop(source_note, None)
        # Bulk-rebuild target_index — selective removal would have to
        # walk every list looking for tuples whose edge_dict's `note`
        # matches; the rebuild scan is the same cost and simpler.
        self._rebuild_indexes()
        self._save()

    def find_related_notes(
        self,
        query: str,
        max_hops: int = 2,
        max_results: int = 5,
    ) -> list[tuple[str, float]]:
        """Find notes related to query via graph traversal.

        Returns list of (note_slug, relevance_score) sorted by score desc.

        Scoring (per query term):
          - direct match (entity == term)               score 1.0
          - partial substring overlap                   score 0.7
          - per-hop decay: hop_score = 0.6 * prev       (was edge.weight*0.5,
                                                         i.e. 0.25 — too steep)
          - **topic bonus**: if a query term matches the note's slug
            verbatim, add +0.5 — fixes the "python_gil" query → "python_gil"
            note alignment that BFS-only would under-rank.
          - per-(term, note) cap of 1.0 prevents one query from scoring the
            same note multiple times via different aliases.
        """
        query_terms = self._extract_query_entities(query)
        if not query_terms:
            return []

        note_scores: dict[str, float] = defaultdict(float)
        normalized_terms = {self._normalize(t) for t in query_terms}

        # Topic-direct boost: query terms that exactly match a note slug
        # (case-insensitive, slug-style). Cheap and catches the "fuzzy and
        # graph see the same thing" case where the query is the topic.
        for note_slug in self._note_entities:
            slug_clean = note_slug.lower().replace("_", " ").replace("-", " ")
            if note_slug in normalized_terms or slug_clean in normalized_terms:
                note_scores[note_slug] += 0.5

        # BFS scoring per query term, with caps so one term contributes <= 1.0
        # to a single note.
        for term in query_terms:
            term_n = self._normalize(term)
            visited: set[str] = set()
            frontier: list[tuple[str, int, float]] = []

            if term_n in self._edges:
                frontier.append((term_n, 0, 1.0))

            for entity in self._edges:
                if entity == term_n:
                    continue
                if term_n in entity or entity in term_n:
                    frontier.append((entity, 0, 0.7))

            term_scores: dict[str, float] = defaultdict(float)
            while frontier:
                entity, hops, score = frontier.pop(0)
                if entity in visited:
                    continue
                visited.add(entity)
                for note_slug, entities in self._note_entities.items():
                    if entity in entities:
                        # Cap the per-(term, note) contribution at the
                        # current hop's score — so revisits via different
                        # entity paths don't compound.
                        if score > term_scores[note_slug]:
                            term_scores[note_slug] = score
                if hops < max_hops and entity in self._edges:
                    for edge in self._edges[entity]:
                        target = edge["target"]
                        if target in visited:
                            continue
                        # Linear decay per hop instead of weight*0.5: the
                        # first neighbor is ~0.6 instead of ~0.25, which
                        # better matches what humans expect.
                        next_score = score * 0.6
                        frontier.append((target, hops + 1, next_score))

            # Aggregate this term's contribution (capped at 1.0 per note)
            for note_slug, s in term_scores.items():
                note_scores[note_slug] += min(s, 1.0)

        ranked = sorted(note_scores.items(), key=lambda x: -x[1])
        return ranked[:max_results]

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract potential entity names from a query string.

        Strategy:
          - Keep the full lowered query as one candidate (catches multi-word
            entities like "global interpreter lock" stored verbatim as edges).
          - Drop stop words ("what is the X" → keep "X", drop the rest).
          - Keep tokens >= 4 chars and bigrams of non-stop-words.
          - Deduplicate while preserving order so direct hits stay first.
        """
        results: list[str] = []
        seen: set[str] = set()

        def _push(token: str) -> None:
            if token and token not in seen:
                seen.add(token)
                results.append(token)

        q = query.strip().lower()
        if not q:
            return []
        _push(q)

        words = [w for w in re.split(r"[\s,;.?!]+", q) if w]

        # Significant single words (filter stop words)
        for w in words:
            if len(w) >= 4 and w not in _QUERY_STOPWORDS:
                _push(w)

        # Bigrams of non-stopword pairs (rejects "is the", "of the", ...)
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if a in _QUERY_STOPWORDS or b in _QUERY_STOPWORDS:
                continue
            bigram = f"{a} {b}"
            if len(bigram) > 5:
                _push(bigram)

        return results

    def get_neighbors(self, entity: str, max_depth: int = 1) -> list[dict]:
        """Get immediate neighbors of an entity (for UI visualization)."""
        entity_n = self._normalize(entity)
        if entity_n not in self._edges:
            return []
        result = []
        for edge in self._edges[entity_n][:20]:  # cap at 20
            result.append({
                "source": entity_n,
                "target": edge["target"],
                "relation": edge["relation"],
                "note": edge["note"],
                "weight": edge["weight"],
            })
        return result

    def stats(self) -> dict:
        """Graph statistics."""
        all_entities = set(self._edges.keys())
        for edges in self._edges.values():
            for e in edges:
                all_entities.add(e["target"])
        total_edges = sum(len(edges) for edges in self._edges.values())
        return {
            "entities": len(all_entities),
            "edges": total_edges,
            "notes_indexed": len(self._note_entities),
        }

    def entity_count(self) -> int:
        return len(set(self._edges.keys()))

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._edges.values())

    # ---- temporal queries ----

    def query_entity(
        self,
        entity: str,
        as_of: Optional[str] = None,
        direction: str = "out",
    ) -> list[dict]:
        """Return edges for an entity, optionally time-filtered.

        Args:
            entity: entity name (case-insensitive)
            as_of: ISO date/datetime — if set, only edges valid at that
                instant. If None, returns currently-open edges (valid_to is None).
            direction: "out" (subject = entity), "in" (object = entity), "both".

        Returns full edge dicts including subject for caller convenience.
        """
        entity_n = self._normalize(entity)
        results: list[dict] = []
        if direction in ("out", "both") and entity_n in self._edges:
            for edge in self._edges[entity_n]:
                if not _is_valid_at(edge, as_of):
                    continue
                results.append({"subject": entity_n, **edge})
        if direction in ("in", "both"):
            for subj, edges in self._edges.items():
                if subj == entity_n:
                    continue
                for edge in edges:
                    if edge.get("target") != entity_n:
                        continue
                    if not _is_valid_at(edge, as_of):
                        continue
                    results.append({"subject": subj, **edge})
        return results

    def invalidate(
        self,
        subject: str,
        relation: str,
        target: str,
        ended_at: Optional[str] = None,
    ) -> int:
        """Close all open edges matching (subject, relation, target).

        Sets `valid_to` on edges that don't have one. `ended_at` defaults
        to now (UTC ISO date). Returns number of edges closed.
        Also closes the inverse edge so traversal stays symmetric.
        """
        subj_n = self._normalize(subject)
        target_n = self._normalize(target)
        rel = relation.strip()
        ts = ended_at or datetime.utcnow().strftime("%Y-%m-%d")

        closed = 0
        if subj_n in self._edges:
            for edge in self._edges[subj_n]:
                if (
                    edge.get("target") == target_n
                    and edge.get("relation") == rel
                    and edge.get("valid_to") is None
                ):
                    edge["valid_to"] = ts
                    closed += 1

        # Close the inverse too so query_entity from the target side stays consistent.
        inv_rel = f"inverse:{rel}"
        if target_n in self._edges:
            for edge in self._edges[target_n]:
                if (
                    edge.get("target") == subj_n
                    and edge.get("relation") == inv_rel
                    and edge.get("valid_to") is None
                ):
                    edge["valid_to"] = ts

        if closed:
            self._save()
        return closed

    def timeline(self, entity: str) -> list[dict]:
        """All edges for an entity, sorted by valid_from (None first).

        Useful for "show me Max's history" — returns every fact ever
        recorded about an entity, ordered chronologically.
        """
        entries = self.query_entity(entity, as_of=None, direction="both")
        # Include closed edges too — query_entity with as_of=None only
        # returns open ones; for timeline we want everything.
        entity_n = self._normalize(entity)
        all_edges: list[dict] = []
        if entity_n in self._edges:
            for edge in self._edges[entity_n]:
                all_edges.append({"subject": entity_n, **edge})
        for subj, edges in self._edges.items():
            if subj == entity_n:
                continue
            for edge in edges:
                if edge.get("target") == entity_n:
                    all_edges.append({"subject": subj, **edge})
        all_edges.sort(key=lambda e: e.get("valid_from") or "0000-00-00")
        return all_edges


def parse_entity_relations(text: str) -> list[tuple[str, str, str]]:
    """Parse entity-relation triples from LLM output.

    Expected format (one per line):
        subject | relation | object
    or:
        subject -> relation -> object

    Returns list of (subject, relation, object) tuples.
    """
    triples = []
    for line in text.strip().splitlines():
        line = line.strip().lstrip("-•* ")
        if not line:
            continue

        # Try pipe-separated: "Python GIL | causes | single-threaded bottleneck"
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
                continue

        # Try arrow-separated: "RS-485 -> uses -> differential signaling"
        if "->" in line:
            parts = [p.strip() for p in line.split("->")]
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
                continue

    return triples


def extract_relations_from_note_body(body: str, topic: str) -> list[tuple[str, str, str]]:
    """Extract entity relations from an existing note body.

    Looks for:
    1. [[wiki-links]] in the note → creates "related_to" edges
    2. Section headers as context for relationships
    """
    triples = []
    # Extract [[wiki-links]] as related entities
    links = re.findall(r"\[\[(.+?)\]\]", body)
    for link in links:
        triples.append((topic.lower(), "related_to", link.lower()))

    # Extract bold terms as entities connected to the topic
    bold_terms = re.findall(r"\*\*(.+?)\*\*", body)
    for term in bold_terms:
        # Skip very long or very short terms
        if 3 < len(term) < 50 and not term.startswith("Q:"):
            triples.append((topic.lower(), "mentions", term.lower()))

    return triples


def reindex_all_notes() -> dict:
    """Index all existing notes into the knowledge graph.

    Reads every note from KM, extracts entities/relations from body,
    and adds them to GRAPH. Returns stats about what was indexed.
    Safe to run multiple times — existing edges from the same note
    are deduplicated.
    """
    from .knowledge_manager import KM, _slug

    stats = {"notes": 0, "triples": 0, "errors": 0}
    for entry in KM.list_topics():
        try:
            note = KM.get_note(entry.topic)
            if not note:
                continue
            slug = _slug(entry.topic)
            triples = extract_relations_from_note_body(note.body, entry.topic)
            # Add keyword relations
            for kw in entry.keywords:
                if kw.lower() != entry.topic.lower():
                    triples.append((entry.topic.lower(), "keyword", kw.lower()))
            added = GRAPH.add_relations(triples, source_note=slug)
            stats["notes"] += 1
            stats["triples"] += added
        except Exception:
            stats["errors"] += 1
    return stats


GRAPH = KnowledgeGraph()
