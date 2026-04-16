"""Lightweight knowledge graph: entities, relations, graph traversal.

Inspired by LightRAG's approach but without heavy dependencies.
Stores a directed graph of entity-relation-entity triples in a single JSON file.
Entities are linked to source notes, enabling graph-based retrieval:
when the user asks about X, we can find related concepts Y and Z
through the graph even if they don't share keywords.

Design:
  - Graph stored as adjacency list in knowledge/graph.json
  - Each edge: {source_entity, relation, target_entity, source_note, weight}
  - Entity extraction happens during note creation (piggybacks on existing LLM call)
  - Graph search: BFS from query entities, collect related notes within N hops
  - No embeddings, no vector DB, no external dependencies
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .config import CONFIG


class KnowledgeGraph:
    """In-memory knowledge graph backed by a JSON file."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "graph.json")
        # Adjacency list: entity -> [{target, relation, note, weight}]
        self._edges: dict[str, list[dict]] = {}
        # Reverse index: note_slug -> set of entities mentioned in it
        self._note_entities: dict[str, set[str]] = defaultdict(set)
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._edges = data.get("edges", {})
                # Rebuild reverse index
                self._note_entities = defaultdict(set)
                for entity, targets in self._edges.items():
                    for edge in targets:
                        note = edge.get("note", "")
                        if note:
                            self._note_entities[note].add(entity)
                            self._note_entities[note].add(edge.get("target", ""))
            except Exception:
                self._edges = {}
                self._note_entities = defaultdict(set)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"edges": self._edges}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # graph is best-effort

    def _normalize(self, entity: str) -> str:
        """Normalize entity name for consistent matching."""
        return entity.strip().lower()

    def add_relations(
        self,
        triples: list[tuple[str, str, str]],
        source_note: str,
        weight: float = 1.0,
    ) -> int:
        """Add entity-relation-entity triples from a note.

        Args:
            triples: list of (subject, relation, object) strings
            source_note: slug of the note these came from
            weight: edge weight (higher = stronger connection)

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

            # Avoid exact duplicates
            exists = any(
                e["target"] == obj_n and e["note"] == source_note
                for e in self._edges[subj_n]
            )
            if not exists:
                self._edges[subj_n].append({
                    "target": obj_n,
                    "relation": rel.strip(),
                    "note": source_note,
                    "weight": weight,
                })
                added += 1

            # Bidirectional: add reverse edge with lower weight
            if obj_n not in self._edges:
                self._edges[obj_n] = []
            rev_exists = any(
                e["target"] == subj_n and e["note"] == source_note
                for e in self._edges[obj_n]
            )
            if not rev_exists:
                self._edges[obj_n].append({
                    "target": subj_n,
                    "relation": f"inverse:{rel.strip()}",
                    "note": source_note,
                    "weight": weight * 0.5,
                })

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
        self._save()

    def find_related_notes(
        self,
        query: str,
        max_hops: int = 2,
        max_results: int = 5,
    ) -> list[tuple[str, float]]:
        """Find notes related to query via graph traversal.

        Returns list of (note_slug, relevance_score) sorted by score desc.
        Uses BFS from entities that match the query, collecting notes
        within max_hops of graph distance.
        """
        query_terms = self._extract_query_entities(query)
        if not query_terms:
            return []

        # BFS from matching entities
        visited: set[str] = set()
        # note_slug -> accumulated score
        note_scores: dict[str, float] = defaultdict(float)

        # Start from entities that match query terms
        frontier: list[tuple[str, int, float]] = []  # (entity, hops, score)
        for term in query_terms:
            term_n = self._normalize(term)
            # Direct match
            if term_n in self._edges:
                frontier.append((term_n, 0, 1.0))
            # Partial match
            for entity in self._edges:
                if term_n in entity or entity in term_n:
                    if entity != term_n:
                        frontier.append((entity, 0, 0.7))

        while frontier:
            entity, hops, score = frontier.pop(0)
            if entity in visited:
                continue
            visited.add(entity)

            # Score all notes this entity appears in
            for note_slug, entities in self._note_entities.items():
                if entity in entities:
                    note_scores[note_slug] += score

            # Expand to neighbors if within hop limit
            if hops < max_hops and entity in self._edges:
                for edge in self._edges[entity]:
                    target = edge["target"]
                    if target not in visited:
                        decay = edge.get("weight", 1.0) * 0.5  # decay per hop
                        frontier.append((target, hops + 1, score * decay))

        # Sort by score, return top results
        ranked = sorted(note_scores.items(), key=lambda x: -x[1])
        return ranked[:max_results]

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract potential entity names from a query string.

        Simple approach: split by common delimiters, filter short words,
        also try the full query as one entity.
        """
        results = []
        # Full query as entity
        q = query.strip().lower()
        if q:
            results.append(q)

        # Split into meaningful chunks
        words = re.split(r"[\s,;.?!]+", q)
        # Single significant words (>3 chars)
        for w in words:
            if len(w) > 3:
                results.append(w)

        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 5:
                results.append(bigram)

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
