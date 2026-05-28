"""Automatic curation of finetune examples: quality score + deduplication."""
from __future__ import annotations
from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import FinetunePair


@dataclass
class ScoredPair:
    pair: FinetunePair
    score: float


class FinetuneDataCurator:
    MIN_SCORE = 0.7
    DUP_THRESHOLD = 80  # fuzz ratio 0..100

    def curate(self, examples: list[FinetunePair]) -> list[FinetunePair]:
        scored = self.score_all(examples)
        good: list[FinetunePair] = []
        for sp in scored:
            if sp.score < self.MIN_SCORE:
                continue
            if self._is_duplicate(sp.pair, good):
                continue
            good.append(sp.pair)
        return good

    def score_all(self, examples: list[FinetunePair]) -> list[ScoredPair]:
        return [ScoredPair(pair=e, score=self.quality_score(e)) for e in examples]

    def quality_score(self, pair: FinetunePair) -> float:
        answer = pair.assistant_text()
        meta = pair.metadata
        score = 0.0

        # answer length
        if 50 < len(answer) < 2000:
            score += 0.2

        # confidence
        if meta.confidence >= 85:
            score += 0.2

        # grounded in notes
        if meta.source_notes:
            score += 0.2

        # category bonus
        if meta.category in ("correction", "troubleshooting"):
            score += 0.3
        elif meta.category in ("factual_qa", "procedure"):
            score += 0.2

        # manual boost — separate bonus
        if meta.boosted:
            score += 0.1

        return min(score, 1.0)

    def _is_duplicate(self, candidate: FinetunePair, accepted: list[FinetunePair]) -> bool:
        q = candidate.user_text()
        for p in accepted:
            if fuzz.token_set_ratio(q, p.user_text()) >= self.DUP_THRESHOLD:
                return True
        return False

    def apply_boosting(self, curated: list[FinetunePair]) -> list[FinetunePair]:
        """Important examples (corrections/troubleshooting/boosted) are repeated 2-3x."""
        out: list[FinetunePair] = []
        for p in curated:
            out.append(p)
            cat = p.metadata.category
            if cat in ("correction", "troubleshooting"):
                out.append(p)
                if p.metadata.boosted:
                    out.append(p)  # triple
            elif p.metadata.boosted:
                out.append(p)
        return out
