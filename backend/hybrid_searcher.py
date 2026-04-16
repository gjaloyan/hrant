"""Hybrid search: fuzzy keyword matching + knowledge graph traversal.

Inspired by LightRAG's mix mode but without embeddings or vector DB.
Combines two retrieval strategies:

  1. Fuzzy (existing) — fast keyword/topic name matching via rapidfuzz.
     Good for: exact topic lookups, keyword hits.
     Weak for: semantic connections ("single-threaded" won't find "GIL").

  2. Graph (new) — BFS traversal of entity-relation graph.
     Good for: finding related concepts across topic boundaries.
     Weak for: topics not yet indexed in the graph.

Results are merged with weighted scoring. Graph hits that weren't found
by fuzzy search are the main value-add — they surface connections the
old searcher couldn't see.
"""
from __future__ import annotations
from dataclasses import dataclass

from .knowledge_graph import GRAPH, KnowledgeGraph
from .knowledge_manager import KM, _slug
from .models import IndexEntry
from .searcher import SEARCHER, SearchHit, Searcher


@dataclass
class HybridHit:
    entry: IndexEntry
    score: float
    source: str  # "fuzzy", "graph", or "both"


class HybridSearcher:
    """Combines fuzzy search + knowledge graph traversal."""

    def __init__(
        self,
        searcher: Searcher | None = None,
        graph: KnowledgeGraph | None = None,
        fuzzy_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self._searcher = searcher or SEARCHER
        self._graph = graph or GRAPH
        self.fuzzy_weight = fuzzy_weight
        self.graph_weight = graph_weight

    def search(self, query: str, limit: int = 5) -> list[HybridHit]:
        """Search using both fuzzy matching and graph traversal.

        Returns merged results sorted by combined score.
        """
        # 1. Fuzzy search (existing)
        fuzzy_hits = self._searcher.search(query, limit=limit)
        fuzzy_by_slug: dict[str, float] = {}
        entry_by_slug: dict[str, IndexEntry] = {}
        for hit in fuzzy_hits:
            slug = _slug(hit.entry.topic)
            fuzzy_by_slug[slug] = hit.score
            entry_by_slug[slug] = hit.entry

        # 2. Graph search
        graph_hits = self._graph.find_related_notes(query, max_hops=2, max_results=limit * 2)
        graph_by_slug: dict[str, float] = {}
        for note_slug, score in graph_hits:
            graph_by_slug[note_slug] = score

        # 3. Merge — union of both result sets
        all_slugs = set(fuzzy_by_slug.keys()) | set(graph_by_slug.keys())
        merged: list[HybridHit] = []

        for slug in all_slugs:
            f_score = fuzzy_by_slug.get(slug, 0.0)
            g_score = graph_by_slug.get(slug, 0.0)

            # Normalize graph score to 0-1 range (it can exceed 1.0 from BFS accumulation)
            if graph_by_slug:
                max_g = max(graph_by_slug.values()) or 1.0
                g_score_norm = g_score / max_g
            else:
                g_score_norm = 0.0

            combined = (f_score * self.fuzzy_weight) + (g_score_norm * self.graph_weight)

            # Determine source
            if slug in fuzzy_by_slug and slug in graph_by_slug:
                source = "both"
            elif slug in graph_by_slug:
                source = "graph"
            else:
                source = "fuzzy"

            # Resolve IndexEntry
            entry = entry_by_slug.get(slug)
            if entry is None:
                # Graph found a note not in fuzzy results — look it up
                note = KM.get_note(slug)
                if note:
                    entry = IndexEntry(
                        topic=note.frontmatter.topic,
                        category=note.frontmatter.category,
                        path=str(note.path),
                        keywords=note.frontmatter.keywords,
                        access_count=note.frontmatter.access_count,
                        updated=note.frontmatter.updated,
                        project=note.frontmatter.project,
                    )

            if entry is not None:
                merged.append(HybridHit(entry=entry, score=combined, source=source))

        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:limit]

    def find_best(self, topic: str) -> IndexEntry | None:
        """Find the single best match (drop-in for Searcher.find_best)."""
        hits = self.search(topic, limit=1)
        return hits[0].entry if hits else None


HYBRID = HybridSearcher()
