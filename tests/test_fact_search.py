"""Fact-level semantic search.

Audit 2026-05-27 found 1453 facts in memory_facts.jsonl that have
no vector index — `search_knowledge` only searches topic-notes.
This module adds a parallel `FACT_VECTOR_STORE` and a
`search_facts(query, limit)` helper that surfaces the most
semantically similar fact summaries.
"""
from __future__ import annotations

import json
from pathlib import Path


def _seed_facts_file(p: Path, facts: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(f, ensure_ascii=False) for f in facts) + "\n",
        encoding="utf-8",
    )


def test_fact_id_is_deterministic_and_short():
    """The same summary always hashes to the same id; ids stay
    under ~16 chars so the vector store keys are compact."""
    from backend.fact_search import fact_id
    a = fact_id("User has a wife whose Telegram contact is Wife / 1562235884.")
    b = fact_id("User has a wife whose Telegram contact is Wife / 1562235884.")
    c = fact_id("Different summary.")
    assert a == b
    assert a != c
    assert 8 <= len(a) <= 24


def test_fact_id_normalizes_whitespace():
    """Trailing whitespace and case shouldn't fork the id."""
    from backend.fact_search import fact_id
    assert fact_id("hello") == fact_id("Hello\n")
    assert fact_id("  hello  ") == fact_id("hello")


def test_count_unembedded_facts_when_store_empty(tmp_path, monkeypatch):
    from backend.fact_search import count_unembedded_facts
    facts_path = tmp_path / "memory_facts.jsonl"
    _seed_facts_file(facts_path, [
        {"summary": "fact 1"}, {"summary": "fact 2"}, {"summary": "fact 3"},
    ])
    monkeypatch.setattr(
        "backend.fact_search._facts_path", lambda: facts_path,
    )
    monkeypatch.setattr(
        "backend.fact_search._fact_store_path",
        lambda: tmp_path / "fact_embeddings.json",
    )
    from backend.fact_search import _new_store_for_test
    _new_store_for_test()

    assert count_unembedded_facts() == 3


def test_count_unembedded_skips_empty_summary(tmp_path, monkeypatch):
    """Facts with empty / missing `summary` (some legacy rows)
    don't count — they can't be embedded."""
    from backend.fact_search import count_unembedded_facts
    facts_path = tmp_path / "memory_facts.jsonl"
    _seed_facts_file(facts_path, [
        {"summary": "real fact"},
        {"summary": ""},
        {"text": "alt text shape"},  # missing summary
        {"summary": "  "},
    ])
    monkeypatch.setattr(
        "backend.fact_search._facts_path", lambda: facts_path,
    )
    monkeypatch.setattr(
        "backend.fact_search._fact_store_path",
        lambda: tmp_path / "fact_embeddings.json",
    )
    from backend.fact_search import _new_store_for_test
    _new_store_for_test()

    # Only "real fact" and "alt text shape" (via `text` fallback) count.
    assert count_unembedded_facts() == 2


def test_backfill_embeddings_uses_embedder(tmp_path, monkeypatch):
    """Backfill iterates facts, calls EMBEDDER.embed, stores vectors."""
    from backend import fact_search as fs

    facts_path = tmp_path / "memory_facts.jsonl"
    _seed_facts_file(facts_path, [
        {"summary": "alpha"}, {"summary": "beta"}, {"summary": "gamma"},
    ])
    monkeypatch.setattr(fs, "_facts_path", lambda: facts_path)
    monkeypatch.setattr(
        fs, "_fact_store_path", lambda: tmp_path / "fact_embeddings.json",
    )
    fs._new_store_for_test()

    embed_calls = []

    def _fake_embed(text):
        embed_calls.append(text)
        return [0.1, 0.2, 0.3, 0.4]
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.embed", _fake_embed,
    )
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.status",
        lambda: {"backend": "llama_cpp", "model": "bge-m3", "dim": 4},
    )

    stats = fs.backfill_fact_embeddings()
    assert stats["embedded"] == 3
    assert stats["errors"] == 0
    assert len(embed_calls) == 3


def test_backfill_is_idempotent(tmp_path, monkeypatch):
    from backend import fact_search as fs

    facts_path = tmp_path / "memory_facts.jsonl"
    _seed_facts_file(facts_path, [
        {"summary": "alpha"}, {"summary": "beta"},
    ])
    monkeypatch.setattr(fs, "_facts_path", lambda: facts_path)
    monkeypatch.setattr(
        fs, "_fact_store_path", lambda: tmp_path / "fact_embeddings.json",
    )
    fs._new_store_for_test()
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.embed",
        lambda text: [0.1, 0.2, 0.3, 0.4],
    )
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.status",
        lambda: {"backend": "llama_cpp", "model": "bge-m3", "dim": 4},
    )

    s1 = fs.backfill_fact_embeddings()
    s2 = fs.backfill_fact_embeddings()
    assert s1["embedded"] == 2
    assert s2["embedded"] == 0
    assert s2["skipped"] == 2


def test_backfill_skips_when_embedder_disabled(tmp_path, monkeypatch):
    from backend import fact_search as fs

    monkeypatch.setattr(fs, "_facts_path", lambda: tmp_path / "memory_facts.jsonl")
    monkeypatch.setattr(
        fs, "_fact_store_path", lambda: tmp_path / "fact_embeddings.json",
    )
    fs._new_store_for_test()
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.status",
        lambda: {"backend": "disabled", "model": None, "dim": 0},
    )

    stats = fs.backfill_fact_embeddings()
    assert stats["ok"] is False
    assert "embedder" in stats.get("reason", "").lower()


def test_search_facts_returns_top_k_by_cosine(tmp_path, monkeypatch):
    """search_facts embeds the query, scores all facts by cosine,
    returns top-K with their full summary text."""
    from backend import fact_search as fs

    facts_path = tmp_path / "memory_facts.jsonl"
    _seed_facts_file(facts_path, [
        {"summary": "wife laboratory HPLC", "category": "project"},
        {"summary": "router config python", "category": "tech"},
        {"summary": "schedule message wife", "category": "rule"},
    ])
    monkeypatch.setattr(fs, "_facts_path", lambda: facts_path)
    monkeypatch.setattr(
        fs, "_fact_store_path", lambda: tmp_path / "fact_embeddings.json",
    )
    fs._new_store_for_test()

    # Stub embedder: returns vectors where "wife" texts cluster.
    vectors = {
        "wife laboratory HPLC":   [1.0, 0.0, 0.0],
        "router config python":   [0.0, 1.0, 0.0],
        "schedule message wife":  [0.9, 0.0, 0.1],
        "wife":                   [1.0, 0.0, 0.0],  # query
    }
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.embed",
        lambda text: vectors.get(text.strip(), [0.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.status",
        lambda: {"backend": "llama_cpp", "model": "bge-m3", "dim": 3},
    )
    fs.backfill_fact_embeddings()

    hits = fs.search_facts("wife", limit=2)
    assert len(hits) == 2
    # Top hit should be the laboratory fact (perfect match).
    assert "laboratory" in hits[0]["summary"]
    assert hits[0]["score"] > hits[1]["score"]


def test_search_facts_empty_when_no_embeddings(tmp_path, monkeypatch):
    """No embedded facts → empty result, never raise."""
    from backend import fact_search as fs

    monkeypatch.setattr(fs, "_facts_path", lambda: tmp_path / "memory_facts.jsonl")
    monkeypatch.setattr(
        fs, "_fact_store_path", lambda: tmp_path / "fact_embeddings.json",
    )
    fs._new_store_for_test()
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.embed",
        lambda text: [1.0, 0.0],
    )
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.status",
        lambda: {"backend": "llama_cpp", "model": "bge-m3", "dim": 2},
    )

    assert fs.search_facts("anything") == []


def test_search_facts_silent_when_embedder_down(tmp_path, monkeypatch):
    """If the embedder is unavailable at query time, search_facts
    must return [] cleanly — agents fall back to keyword search."""
    from backend import fact_search as fs

    monkeypatch.setattr(fs, "_facts_path", lambda: tmp_path / "memory_facts.jsonl")
    monkeypatch.setattr(
        fs, "_fact_store_path", lambda: tmp_path / "fact_embeddings.json",
    )
    fs._new_store_for_test()
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.embed",
        lambda text: None,  # embedder unavailable
    )
    monkeypatch.setattr(
        "backend.fact_search.EMBEDDER.status",
        lambda: {"backend": "disabled", "model": None, "dim": 0},
    )

    assert fs.search_facts("anything") == []
