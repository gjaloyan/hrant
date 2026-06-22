"""The knowledge backfill must survive a single transient embed timeout.

On the CPU host the first embed of a batch can time out (cold model load),
leaving one note unembedded so the store stuck at N-1/N and the
FIRE_EMBEDDING_BACKFILL lever re-logged a timeout every tick. The backfill now
warms the model up front and retries a failed embed once.
"""
from __future__ import annotations

import backend.embedding_backfill as bf


class _Topic:
    def __init__(self, t):
        self.topic = t


class _FM:
    topic = "T"


class _Note:
    frontmatter = _FM()
    body = "some body text"


def test_backfill_retries_a_transient_embed_failure(monkeypatch):
    monkeypatch.setattr(bf.KM, "list_topics", lambda: [_Topic("T")])
    monkeypatch.setattr(bf.KM, "get_note", lambda t: _Note())
    monkeypatch.setattr(
        bf.EMBEDDER, "status",
        lambda: {"backend": "ollama", "dim": 3, "model": "bge-m3"},
    )

    # warmup returns a vector; the note's first real embed fails (None), the
    # retry succeeds.
    calls = {"n": 0}

    def fake_embed(text):
        if text == "warmup":
            return [0.0, 0.0, 0.0]
        calls["n"] += 1
        return None if calls["n"] == 1 else [0.1, 0.2, 0.3]

    monkeypatch.setattr(bf.EMBEDDER, "embed", fake_embed)

    added: dict = {}
    monkeypatch.setattr(bf.VECTOR_STORE, "is_compatible", lambda *a: True)
    monkeypatch.setattr(bf.VECTOR_STORE, "has", lambda slug: False)
    monkeypatch.setattr(bf.VECTOR_STORE, "add",
                        lambda slug, vec: added.__setitem__(slug, vec))

    stats = bf.backfill_embeddings()

    assert stats["embedded"] == 1
    assert stats["errors"] == 0
    assert added  # the note was embedded on the retry, not dropped
    assert calls["n"] == 2  # first failed, retried once
