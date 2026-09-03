"""Stop the same fact being stored in a second wording.

Both writers into memory_facts.jsonl deduped on the lowercased summary,
which catches a repeat and nothing else. Measured on prod 2026-09-03
after consolidating the backlog: 3949 facts, zero character-identical
duplicates, and 587 pairs at cosine >= 0.97 -- "The backend is written
primarily in Python." beside "The backend is primarily written in
Python." at 0.999.

The vectors were already there; nothing was asking them.
"""
from unittest.mock import patch

from backend import fact_search as fs


class _Embedder:
    def status(self):
        return {"backend": "fake", "dim": 3, "model": "m"}

    def embed(self, text):
        # "python backend" phrasings land together; anything else apart.
        if "python" in text.lower():
            return [1.0, 0.0, 0.02 if "primarily written" in text.lower() else 0.0]
        return [0.0, 1.0, 0.0]


def test_a_restatement_is_recognised():
    def _search(query, limit=5, score_floor=None, **kw):
        hits = [{"summary": "The backend is written primarily in "
                 "Python.", "score": 0.999}]
        return [h for h in hits
                if score_floor is None or h["score"] >= score_floor]

    with patch.object(fs, "EMBEDDER", _Embedder()), \
         patch.object(fs, "search_facts", _search):
        hit = fs.near_duplicate_of("The backend is primarily written in Python.")
    assert hit == "The backend is written primarily in Python."


def test_a_merely_related_fact_is_not_a_duplicate():
    def _search(query, limit=5, score_floor=None, **kw):
        # A faithful stand-in: the real search applies the floor, so the fake must too.
        hits = [{"summary": "The backend is written primarily in "
                 "Python.", "score": 0.88}]
        return [h for h in hits
                if score_floor is None or h["score"] >= score_floor]

    with patch.object(fs, "EMBEDDER", _Embedder()), \
         patch.object(fs, "search_facts", _search):
        assert fs.near_duplicate_of("The frontend is written in TypeScript.") is None


def test_no_embedder_means_no_opinion():
    """Never drop a fact because the embedder was down -- a lost fact is
    worse than a duplicated one."""
    class _Down:
        def status(self):
            return {"backend": None}

    with patch.object(fs, "EMBEDDER", _Down()):
        assert fs.near_duplicate_of("anything at all") is None


def test_the_consolidator_skips_a_restatement(tmp_path):
    from backend.autonomic.levers.memory_consolidation import (
        FIRE_MEMORY_CONSOLIDATION,
    )
    lever = FIRE_MEMORY_CONSOLIDATION()
    path = tmp_path / "memory_facts.jsonl"

    with patch("backend.fact_search.near_duplicate_of",
               side_effect=lambda s: "an existing wording" if "python" in s.lower() else None):
        added = lever._append_durable_facts(
            path,
            [{"summary": "The backend is primarily written in Python.",
              "confidence": 0.9},
             {"summary": "The office is in Yerevan.", "confidence": 0.9}],
            set(), "sess1")

    assert added == 1
    body = path.read_text(encoding="utf-8")
    assert "Yerevan" in body
    assert "Python" not in body
