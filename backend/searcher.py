"""Knowledge base search: keyword + fuzzy."""
from __future__ import annotations
from dataclasses import dataclass

from rapidfuzz import fuzz

from .config import CONFIG
from .knowledge_manager import KM, _slug
from .models import IndexEntry


@dataclass
class SearchHit:
    entry: IndexEntry
    score: float


class Searcher:
    def __init__(self):
        self.threshold = CONFIG.search["fuzzy_threshold"]

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        q = query.lower().strip()
        q_slug = _slug(query)
        hits: list[SearchHit] = []
        for entry in KM.list_topics():
            score = self._score(q, q_slug, entry)
            if score >= self.threshold:
                hits.append(SearchHit(entry=entry, score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def _score(self, q: str, q_slug: str, entry: IndexEntry) -> float:
        topic_slug = _slug(entry.topic)
        if q_slug == topic_slug:
            return 1.0
        if q_slug in topic_slug or topic_slug in q_slug:
            return 0.95
        if q in (k.lower() for k in entry.keywords):
            return 0.9
        # fuzzy match on topic
        topic_ratio = fuzz.ratio(q, entry.topic.lower()) / 100.0
        # max fuzzy match on keywords
        kw_ratio = 0.0
        for kw in entry.keywords:
            kw_ratio = max(kw_ratio, fuzz.partial_ratio(q, kw.lower()) / 100.0)
        return max(topic_ratio, kw_ratio)

    def exists(self, topic: str) -> bool:
        return KM.get_note(topic) is not None

    def find_best(self, topic: str) -> IndexEntry | None:
        hits = self.search(topic, limit=1)
        return hits[0].entry if hits else None


SEARCHER = Searcher()
