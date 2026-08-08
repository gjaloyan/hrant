"""Memory-integrity gates from the 2026-08-08 audit.

Measured on the owner's real prod store before the fix:

  * 555 of 2778 rows in permanent personal memory — 20% — were written by
    `audit:claude` and `webui:bench-harness`. The row carries speaker_id and
    search_facts never read it, so a probe typed during an audit ("My name is
    Alice. Remember it.") came back at rank 1 against a real question.
  * search_facts returned top-K unconditionally. Gibberish returned five hits
    scoring 0.489-0.510; a true hit scored 0.719 with the noise behind it
    topping out at 0.510. The separation is real and nothing used it.
  * _auto_recall_block injected exactly 2 facts into EVERY turn with no floor
    and printed no score, so the model got two unqualified assertions it could
    not discount. A marathon-shoes question was answered with
    "Current run progress was 38/300."
  * The note lines carried `score: 1.00` — an artifact of min-max
    normalisation, where the top hit is always 1.00 however weak the match.
"""
from __future__ import annotations

import pytest

import backend.fact_search as fs


class _Store:
    def __init__(self, scored): self._scored = scored
    def count(self): return len(self._scored)
    def is_compatible(self, *a, **k): return True
    def search(self, qvec, k): return self._scored[:k]
    def stats(self): return {}


def _install(monkeypatch, rows):
    """rows: [(summary, score, speaker_id)]"""
    scored = [(fs.fact_id(s), sc) for s, sc, _ in rows]
    monkeypatch.setattr(fs, "get_store", lambda: _Store(scored))
    monkeypatch.setattr(fs, "EMBEDDER", type("E", (), {
        "status": staticmethod(lambda: {"backend": "ollama", "dim": 1024,
                                        "model": "bge-m3"}),
        "embed": staticmethod(lambda q: [0.1]),
    })())
    monkeypatch.setattr(fs, "_iter_facts", lambda: [
        (s, {"summary": s, "category": "personal", "ts": "2026-01-01",
             "source_turn": s, "speaker_id": sid}) for s, _, sid in rows])


def test_a_fact_written_during_an_audit_is_not_recalled(monkeypatch):
    """The exact prod row: an audit probe outranking the owner's own facts."""
    _install(monkeypatch, [
        ("The user's name is Alice.", 0.90, "audit:claude"),
        ("The owner lives in Yerevan.", 0.80, "telegram:848732236"),
    ])
    got = [f["summary"] for f in fs.search_facts("where does he live", limit=5)]
    assert "The user's name is Alice." not in got
    assert "The owner lives in Yerevan." in got


def test_benchmark_harness_rows_are_not_recalled(monkeypatch):
    _install(monkeypatch, [
        ("The benchmark run is in gold sanity check mode.", 0.95,
         "webui:bench-harness"),
        ("The owner prefers answers in Russian.", 0.75, "webui:default"),
    ])
    got = [f["summary"] for f in fs.search_facts("how should I answer", limit=5)]
    assert not any("benchmark" in g for g in got)
    assert "The owner prefers answers in Russian." in got


def test_noise_below_the_floor_is_dropped_entirely(monkeypatch):
    """Gibberish must return NOTHING, not the five least-irrelevant rows."""
    _install(monkeypatch, [(f"unrelated fact {i}", 0.51 - i * 0.005,
                            "webui:default") for i in range(5)])
    assert fs.search_facts("qwertyuiop asdfghjkl", limit=5) == []


def test_a_real_hit_above_the_floor_survives(monkeypatch):
    _install(monkeypatch, [
        ("Mercury boils at 356.7 C.", 0.719, "webui:default"),
        ("Mercury is a planet.", 0.593, "webui:default"),
        ("Scary Movie is a film.", 0.460, "webui:default"),
    ])
    got = fs.search_facts("boiling point of mercury", limit=5)
    assert [f["summary"] for f in got] == ["Mercury boils at 356.7 C."]
    assert got[0]["score"] >= fs.FACT_SCORE_FLOOR


def test_the_speaker_is_returned_so_provenance_is_visible(monkeypatch):
    _install(monkeypatch, [("A real fact.", 0.9, "telegram:848732236")])
    assert fs.search_facts("q")[0]["speaker_id"] == "telegram:848732236"


def test_synthetic_can_be_included_explicitly_for_tooling(monkeypatch):
    """The filter is a recall default, not a data deletion — an inspection
    tool must still be able to see what is in the store."""
    _install(monkeypatch, [("Audit probe.", 0.9, "audit:claude")])
    assert fs.search_facts("q", include_synthetic=True)[0]["summary"] == "Audit probe."


@pytest.mark.parametrize("sid,synthetic", [
    ("audit:claude", True), ("webui:bench-harness", True),
    ("test:runner", True), ("telegram:848732236", False),
    ("webui:default", False), ("", False), (None, False),
])
def test_synthetic_speaker_classification(sid, synthetic):
    assert fs._is_synthetic_speaker(sid) is synthetic


# ── the auto-recall block ─────────────────────────────────────────────

def test_auto_recall_emits_nothing_when_nothing_clears_the_floor(monkeypatch):
    import backend.unified_agent as ua
    monkeypatch.setattr("backend.fact_search.search_facts",
                        lambda *a, **k: [])
    from backend.hybrid_searcher import HYBRID
    monkeypatch.setattr(HYBRID, "find_best", lambda *a, **k: None)
    assert ua._auto_recall_block("some completely unrelated question about nothing at all") == ""


def test_auto_recall_prints_the_fact_score(monkeypatch):
    """Two unqualified assertions were being injected into every turn."""
    import backend.unified_agent as ua
    from backend.hybrid_searcher import HYBRID
    monkeypatch.setattr(HYBRID, "find_best", lambda *a, **k: None)
    monkeypatch.setattr("backend.fact_search.search_facts",
                        lambda *a, **k: [{"summary": "The owner lives in Yerevan.",
                                          "score": 0.72}])
    out = ua._auto_recall_block("where does the owner actually live these days")
    assert "The owner lives in Yerevan." in out
    assert "0.72" in out


def test_auto_recall_no_longer_prints_a_fabricated_note_score(monkeypatch):
    """search() min-max normalises: the top hit is ALWAYS 1.00."""
    import backend.unified_agent as ua
    from backend.hybrid_searcher import HYBRID

    class _E:
        topic, category, path = "Some note", "general", "notes/x.md"

    class _H:
        entry, score, source = _E(), 1.0, "vector"

    monkeypatch.setattr(HYBRID, "find_best", lambda *a, **k: _E())
    monkeypatch.setattr(HYBRID, "search", lambda *a, **k: [_H()])
    monkeypatch.setattr("backend.fact_search.search_facts", lambda *a, **k: [])
    out = ua._auto_recall_block("a task description long enough to pass the length gate")
    assert "Some note" in out
    assert "score: 1.00" not in out


def test_notes_are_gated_on_the_raw_floor_not_the_normalised_one(monkeypatch):
    """The 0.55 guard was documented in a comment and never applied — search()
    has no min_raw_score parameter, only find_best does."""
    import backend.unified_agent as ua
    from backend.hybrid_searcher import HYBRID
    called = {}

    def _fb(topic, *, min_raw_score=0.0):
        called["floor"] = min_raw_score
        return None

    monkeypatch.setattr(HYBRID, "find_best", _fb)
    monkeypatch.setattr("backend.fact_search.search_facts", lambda *a, **k: [])
    ua._auto_recall_block("a task description long enough to pass the length gate")
    assert called["floor"] == 0.55
