"""A batched backfill must write the store ONCE, not once per vector.

Root cause (2026-06-22): VectorStore.add/remove called _save() on every call,
so re-embedding 2368 facts re-serialized the growing ~50 MB JSON 2368 times
(O(n^2)). A force-re-embed stalled at 93% CPU for 30+ minutes. add(save=False)
+ flush(), and clear() instead of a per-slug remove() loop, make it one write.
"""
from __future__ import annotations

from backend.vector_store import VectorStore


def _counting_store(path):
    vs = VectorStore(path)
    writes = {"n": 0}
    real_save = vs._save

    def counting_save():
        writes["n"] += 1
        real_save()

    vs._save = counting_save  # type: ignore[method-assign]
    return vs, writes


def test_batched_add_writes_once(tmp_path):
    vs, writes = _counting_store(tmp_path / "e.json")
    for i in range(50):
        vs.add(str(i), [float(i), 0.0], save=False)
    assert writes["n"] == 0          # nothing written during the loop
    vs.flush()
    assert writes["n"] == 1          # ONE write for 50 adds, not 50
    assert vs.count() == 50
    # persisted to disk
    assert VectorStore(tmp_path / "e.json").count() == 50


def test_default_add_still_saves(tmp_path):
    vs, writes = _counting_store(tmp_path / "e.json")
    vs.add("a", [1.0])
    assert writes["n"] == 1
    assert VectorStore(tmp_path / "e.json").has("a")


def test_clear_empties_in_one_write(tmp_path):
    vs, writes = _counting_store(tmp_path / "e.json")
    for i in range(10):
        vs.add(str(i), [1.0], save=False)
    vs.flush()
    writes["n"] = 0
    vs.clear()
    assert writes["n"] == 1          # single write, not 10 removes
    assert vs.count() == 0
    assert VectorStore(tmp_path / "e.json").count() == 0
