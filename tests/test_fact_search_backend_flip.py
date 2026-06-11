"""search_facts refuses an embedder/store backend mismatch.

Prod bug 2026-06-11 (found via trajectory_memory, which copied this
module's pattern): the auto-probe embedder flipped from ollama/
bge-m3/1024 to openai/1536 between processes. Query vectors landed
in a different space than the 1947 stored fact vectors, every cosine
returned 0.0, and fact recall silently degraded to "no results"
while looking healthy.
"""
from __future__ import annotations

import pytest


def test_search_refuses_when_store_stamp_mismatches(tmp_path, monkeypatch):
    from backend import fact_search as fs

    monkeypatch.setattr(fs, "_fact_store_path", lambda: tmp_path / "f.json")
    monkeypatch.setattr(fs, "_facts_path", lambda: tmp_path / "facts.jsonl")
    store = fs._new_store_for_test()
    store.stamp(1024, "ollama", "bge-m3")
    store.add("f_abc123", [0.1] * 1024)

    class _FlippedEmbedder:
        def status(self):
            return {"backend": "openai", "dim": 1536,
                    "model": "text-embedding-3-small"}

        def embed(self, text):  # pragma: no cover — must not be reached
            raise AssertionError(
                "query must be refused BEFORE embedding on a flip",
            )

    monkeypatch.setattr(fs, "EMBEDDER", _FlippedEmbedder())
    assert fs.search_facts("anything worth asking") == []
    # Refusal is read-only — the 1947-vector prod store must never be
    # wiped from the query path.
    assert store.has("f_abc123")


def test_search_proceeds_when_stamp_matches(tmp_path, monkeypatch):
    from backend import fact_search as fs

    monkeypatch.setattr(fs, "_fact_store_path", lambda: tmp_path / "f.json")
    facts = tmp_path / "facts.jsonl"
    facts.write_text(
        '{"summary": "User favorite color is teal.", "category": "preference"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(fs, "_facts_path", lambda: facts)
    store = fs._new_store_for_test()
    store.stamp(3, "fake", "fake-3d")
    store.add(fs.fact_id("User favorite color is teal."), [1.0, 0.0, 0.0])

    class _Embedder:
        def status(self):
            return {"backend": "fake", "dim": 3, "model": "fake-3d"}

        def embed(self, text):
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(fs, "EMBEDDER", _Embedder())
    hits = fs.search_facts("what is the user's favorite color?")
    assert len(hits) == 1
    assert hits[0]["summary"] == "User favorite color is teal."
