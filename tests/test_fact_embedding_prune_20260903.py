"""Deleting a fact has to delete it from search too.

`backfill_fact_embeddings` only ever added. Nothing reconciled the other
direction, and `search_facts` reads the vector store rather than
memory_facts.jsonl -- so a fact removed from the file kept being returned
by search, for good. Found 2026-09-03 while merging duplicate rows: the
merge would have changed the file and left recall exactly as it was.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend import fact_search as fs


@pytest.fixture()
def kb(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    k = tmp_path / "knowledge"
    k.mkdir(parents=True, exist_ok=True)
    fs._STORE = None
    monkeypatch.setattr(fs, "_fact_store_path",
                        lambda: k / "fact_embeddings.json")
    monkeypatch.setattr(fs, "_facts_path", lambda: k / "memory_facts.jsonl",
                        raising=False)
    yield k
    fs._STORE = None


def _write_facts(k, summaries):
    (k / "memory_facts.jsonl").write_text(
        "".join(json.dumps({"summary": s, "confidence": 0.9},
                           ensure_ascii=False) + "\n" for s in summaries),
        encoding="utf-8")


class _Embedder:
    def status(self):
        return {"backend": "fake", "dim": 3, "model": "m"}

    def embed(self, text):
        return [float(len(text)), 1.0, 0.0]


def test_backfill_drops_vectors_whose_fact_is_gone(kb):
    _write_facts(kb, ["kept fact", "doomed fact"])
    with patch.object(fs, "EMBEDDER", _Embedder()):
        fs.backfill_fact_embeddings()
        assert fs.get_store().count() == 2

        _write_facts(kb, ["kept fact"])
        report = fs.backfill_fact_embeddings()

    assert fs.get_store().count() == 1, "the deleted fact still has a vector"
    assert report.get("pruned") == 1
    assert fs.get_store().has(fs.fact_id("kept fact"))
    assert not fs.get_store().has(fs.fact_id("doomed fact"))


def test_backfill_prunes_nothing_when_the_file_is_unchanged(kb):
    _write_facts(kb, ["a fact", "another fact"])
    with patch.object(fs, "EMBEDDER", _Embedder()):
        fs.backfill_fact_embeddings()
        report = fs.backfill_fact_embeddings()
    assert report.get("pruned") == 0
    assert fs.get_store().count() == 2
