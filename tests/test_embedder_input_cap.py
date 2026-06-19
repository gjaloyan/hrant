"""Embed input must be length-capped so a long document can't stall the
CPU embedder.

Root cause (2026-06-19 audit): the knowledge backfill embedded each note's full
body. On the CPU-only host, bge-m3 could not embed a 12 KB note within any
timeout (>120 s, then error), so VECTOR_STORE stayed empty and semantic search
silently degraded to a mis-ranking fuzzy fallback. A bounded excerpt embeds in
seconds and is enough for note-level recall.
"""
from __future__ import annotations

from backend.embedder import _cap_embed_text, _MAX_EMBED_CHARS


def test_long_text_is_capped():
    capped = _cap_embed_text("x" * (10 * _MAX_EMBED_CHARS))
    assert len(capped) == _MAX_EMBED_CHARS


def test_short_text_is_unchanged():
    assert _cap_embed_text("a short query") == "a short query"


def test_topic_prefix_survives_cap():
    # Callers prepend the topic; it must survive so the vector still
    # represents the note's subject.
    topic = "Cryptocurrency analysis methodology"
    text = f"{topic}\n\n" + ("body " * 5000)
    assert _cap_embed_text(text).startswith(topic)
