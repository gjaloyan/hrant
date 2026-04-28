"""Tests for the pure-stdlib vector store (cosine + persistence)."""
from __future__ import annotations

import math

import pytest

from backend.vector_store import VectorStore, cosine


def test_cosine_orthogonal():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_identical():
    assert math.isclose(cosine([0.5, 0.5], [0.5, 0.5]), 1.0, abs_tol=1e-9)


def test_cosine_opposite():
    assert math.isclose(cosine([1.0, 0.0], [-1.0, 0.0]), -1.0, abs_tol=1e-9)


def test_cosine_empty_or_mismatched():
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0], [1.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_search_returns_top_k(tmp_path):
    store = VectorStore(tmp_path / "v.json")
    store.stamp(2, "test", "fake")
    store.add("close", [0.9, 0.1])
    store.add("far", [-1.0, 0.0])
    store.add("middle", [0.5, 0.5])

    results = store.search([1.0, 0.0], k=2)
    assert results[0][0] == "close"
    assert len(results) == 2
    # Far should be ranked last and is negative
    assert results[-1][1] < results[0][1]


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "v.json"
    a = VectorStore(p)
    a.stamp(3, "ollama", "nomic-embed-text")
    a.add("note1", [1.0, 0.0, 0.0])
    a.add("note2", [0.0, 1.0, 0.0])

    b = VectorStore(p)
    assert b.count() == 2
    assert b.has("note1")
    stats = b.stats()
    assert stats["dim"] == 3
    assert stats["backend"] == "ollama"
    assert stats["model"] == "nomic-embed-text"


def test_remove(tmp_path):
    store = VectorStore(tmp_path / "v.json")
    store.stamp(2, "test", "fake")
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])
    store.remove("a")
    assert not store.has("a")
    assert store.has("b")
    # Persists
    store2 = VectorStore(tmp_path / "v.json")
    assert not store2.has("a")
    assert store2.has("b")


def test_is_compatible(tmp_path):
    store = VectorStore(tmp_path / "v.json")
    # Empty store accepts anything
    assert store.is_compatible(384, "ollama", "any")
    store.stamp(384, "ollama", "all-minilm")
    assert store.is_compatible(384, "ollama", "all-minilm")
    assert not store.is_compatible(384, "openai", "all-minilm")
    assert not store.is_compatible(768, "ollama", "all-minilm")


def test_search_empty():
    store = VectorStore.__new__(VectorStore)
    store.path = None  # type: ignore
    store._lock = __import__("threading").Lock()
    store._dim = None
    store._backend = None
    store._model = None
    store._items = {}
    assert store.search([1.0, 0.0], k=5) == []
