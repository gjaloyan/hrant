"""Hybrid search: fuzzy keyword + knowledge-graph traversal + vector embeddings.

Three retrieval strategies merged with configurable weights:

  1. Fuzzy — rapidfuzz keyword/topic matching.
     Strong: exact lookups, keyword hits.
     Weak: synonyms, paraphrases.

  2. Graph — BFS traversal of the entity-relation graph.
     Strong: cross-topic connections via shared entities.
     Weak: notes not yet indexed.

  3. Vector — cosine similarity over embeddings (when an embedder backend
     is available — Ollama, OpenAI-compat, or Cohere).
     Strong: semantic recall ("the GIL stuff" → finds "single-threaded
     bottleneck" notes).
     Weak: requires an embedder; degrades gracefully to 0-weight when
     no backend is reachable.

When the vector backend is disabled, weights re-normalize across the two
remaining signals so behavior stays sensible.
"""
from __future__ import annotations
from dataclasses import dataclass

from .embedder import EMBEDDER
from .knowledge_graph import GRAPH, KnowledgeGraph
from .knowledge_manager import KM, _slug
from .models import IndexEntry
from .searcher import SEARCHER, SearchHit, Searcher
from .vector_store import VECTOR_STORE, VectorStore


@dataclass
class HybridHit:
    entry: IndexEntry
    score: float
    source: str  # "fuzzy", "graph", "vector", or combos like "fuzzy+vector"


class HybridSearcher:
    """Combines fuzzy + knowledge graph + vector embeddings search."""

    def __init__(
        self,
        searcher: Searcher | None = None,
        graph: KnowledgeGraph | None = None,
        vector_store: VectorStore | None = None,
        fuzzy_weight: float = 0.45,
        graph_weight: float = 0.25,
        vector_weight: float = 0.30,
        # bge-m3 (the default embedder) produces cosine 0.30-0.45 on
        # genuinely unrelated text pairs in the same language. The old
        # 0.30 floor let "mercury boiling point" pull in "Scary Movie"
        # / "German vocabulary" as matched topics. 0.45 cuts off the
        # noise band without blocking true cross-domain matches (which
        # usually land >0.55 with bge-m3).
        vector_score_floor: float = 0.45,
        graph_score_floor: float = 0.15,
    ):
        self._searcher = searcher or SEARCHER
        self._graph = graph or GRAPH
        self._vector_store = vector_store or VECTOR_STORE
        self.fuzzy_weight = fuzzy_weight
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight
        # Raw-score floors per signal. Min-max normalization scales the
        # top hit to 1.0 even when raw scores are tiny — so a query for
        # a topic the KB doesn't have ("iodine") still got the closest
        # noise back ("blood sugar") with a fully-confident normalized
        # score. Fuzzy already self-filters at CONFIG.search.fuzzy_threshold;
        # vector / graph need their own floors so noise doesn't survive.
        self.vector_score_floor = vector_score_floor
        self.graph_score_floor = graph_score_floor

    # ---- public ----

    def search(self, query: str, limit: int = 5) -> list[HybridHit]:
        """Search using all available signals. Returns merged top-K."""
        fuzzy_by_slug, entry_by_slug = self._fuzzy(query, limit)
        graph_by_slug = self._graph_search(query, limit)
        vector_by_slug = self._vector_search(query, limit)

        # Re-normalize weights when one signal is unavailable. We don't want
        # the disabled weight to silently subtract relevance from results.
        weights = self._normalized_weights(
            has_fuzzy=bool(fuzzy_by_slug),
            has_graph=bool(graph_by_slug),
            has_vector=bool(vector_by_slug),
        )

        # Min-max normalize each signal individually so they're comparable.
        fuzzy_norm = self._normalize_scores(fuzzy_by_slug)
        graph_norm = self._normalize_scores(graph_by_slug)
        vector_norm = self._normalize_scores(vector_by_slug)

        all_slugs = set(fuzzy_norm) | set(graph_norm) | set(vector_norm)
        merged: list[HybridHit] = []
        for slug in all_slugs:
            f = fuzzy_norm.get(slug, 0.0)
            g = graph_norm.get(slug, 0.0)
            v = vector_norm.get(slug, 0.0)
            combined = f * weights["fuzzy"] + g * weights["graph"] + v * weights["vector"]

            sources = []
            if slug in fuzzy_norm:
                sources.append("fuzzy")
            if slug in graph_norm:
                sources.append("graph")
            if slug in vector_norm:
                sources.append("vector")

            entry = self._resolve_entry(slug, entry_by_slug)
            if entry is not None:
                merged.append(HybridHit(entry=entry, score=combined, source="+".join(sources)))

        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:limit]

    def find_best(self, topic: str, *, min_raw_score: float = 0.0) -> IndexEntry | None:
        """Find the single best match (drop-in for Searcher.find_best).

        `min_raw_score` rejects matches where no individual signal
        cleared this floor on its raw, pre-normalized score. Pass a
        non-zero value (e.g. 0.4) when you want a hard "this is
        actually relevant" gate — the search() method's combined score
        is post-min-max-normalization, where the top hit is always 1.0
        regardless of how weak the underlying match is.
        """
        if min_raw_score > 0.0:
            fuzzy_scores, _ = self._fuzzy(topic, 1)
            graph_scores = self._graph_search(topic, 1)
            vector_scores = self._vector_search(topic, 1)
            max_raw = 0.0
            for d in (fuzzy_scores, graph_scores, vector_scores):
                if d:
                    max_raw = max(max_raw, max(d.values()))
            if max_raw < min_raw_score:
                return None
        hits = self.search(topic, limit=1)
        return hits[0].entry if hits else None

    # ---- per-signal helpers ----

    def _fuzzy(self, query: str, limit: int) -> tuple[dict[str, float], dict[str, IndexEntry]]:
        scores: dict[str, float] = {}
        entries: dict[str, IndexEntry] = {}
        for hit in self._searcher.search(query, limit=limit * 2):
            slug = _slug(hit.entry.topic)
            scores[slug] = hit.score
            entries[slug] = hit.entry
        return scores, entries

    def _graph_search(self, query: str, limit: int) -> dict[str, float]:
        scores: dict[str, float] = {}
        for note_slug, score in self._graph.find_related_notes(query, max_hops=2, max_results=limit * 2):
            if score >= self.graph_score_floor:
                scores[note_slug] = score
        return scores

    def _vector_search(self, query: str, limit: int) -> dict[str, float]:
        """Embed the query and pull top-K cosine matches. Empty dict if disabled.

        Drop hits below `vector_score_floor`. Cosine 0.05 between a
        random pair of bge-m3 embeddings is effectively noise — letting
        it through means min-max scales it to 1.0 and we ship an
        unrelated note as the "best" match.
        """
        if self._vector_store.count() == 0:
            return {}
        qvec = EMBEDDER.embed(query)
        if not qvec:
            return {}
        return {
            slug: score
            for slug, score in self._vector_store.search(qvec, k=limit * 2)
            if score >= self.vector_score_floor
        }

    # ---- merge helpers ----

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """Min-max scale to [0, 1]. Empty dict in → empty dict out."""
        if not scores:
            return {}
        values = list(scores.values())
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def _normalized_weights(
        self, *, has_fuzzy: bool, has_graph: bool, has_vector: bool
    ) -> dict[str, float]:
        raw = {
            "fuzzy": self.fuzzy_weight if has_fuzzy else 0.0,
            "graph": self.graph_weight if has_graph else 0.0,
            "vector": self.vector_weight if has_vector else 0.0,
        }
        total = sum(raw.values())
        if total <= 0:
            return raw
        return {k: v / total for k, v in raw.items()}

    def _resolve_entry(
        self, slug: str, entry_by_slug: dict[str, IndexEntry]
    ) -> IndexEntry | None:
        if slug in entry_by_slug:
            return entry_by_slug[slug]
        # Graph or vector found a note not in fuzzy results — look it up.
        note = KM.get_note(slug)
        if note is None:
            return None
        return IndexEntry(
            topic=note.frontmatter.topic,
            category=note.frontmatter.category,
            path=str(note.path),
            keywords=note.frontmatter.keywords,
            access_count=note.frontmatter.access_count,
            updated=note.frontmatter.updated,
            project=note.frontmatter.project,
        )


HYBRID = HybridSearcher()
