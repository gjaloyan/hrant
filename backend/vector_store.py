"""Pure-stdlib vector store for the knowledge base.

Stores per-note embeddings in `knowledge/embeddings.json` and serves
top-K cosine similarity over them.

Why pure stdlib (no numpy / faiss / chromadb):
  - At our scale (≤ a few thousand notes × 1k-dim vectors) the search
    pass is dominated by Python list-iteration overhead, not BLAS.
    Adding numpy alone is ~50 MB; chromadb pulls onnxruntime + tons of
    deps. The benefit is marginal until we cross ~10k items.
  - Pure stdlib means the vector layer ships with the rest of the
    codebase and never breaks Bedrock-style soft-deps for users who
    don't enable embeddings.

If we ever scale past ~10k notes, swapping cosine() and load() for a
faiss-backed implementation is ~30 lines.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Optional


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in pure Python. Returns 0 for empty/mismatched."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class VectorStore:
    """Single-file JSON-backed vector index.

    Schema:
      {
        "dim": int,
        "backend": "ollama" | "openai" | ...,
        "model": "...",
        "items": {slug: [v1, v2, ...]}
      }

    `dim`/`backend`/`model` are recorded so we can detect schema mismatch
    after the user switches embedder backends and trigger a rebuild.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._dim: Optional[int] = None
        self._backend: Optional[str] = None
        self._model: Optional[str] = None
        self._items: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._dim = data.get("dim")
            self._backend = data.get("backend")
            self._model = data.get("model")
            self._items = {k: list(v) for k, v in (data.get("items") or {}).items()}
        except Exception:
            self._dim = None
            self._backend = None
            self._model = None
            self._items = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "dim": self._dim,
                    "backend": self._backend,
                    "model": self._model,
                    "items": self._items,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def stamp(self, dim: int, backend: str, model: str) -> None:
        """Record which embedder produced these vectors. Call before first add()."""
        with self._lock:
            self._dim = dim
            self._backend = backend
            self._model = model

    def is_compatible(self, dim: int, backend: str, model: str) -> bool:
        """Whether the stored vectors match the current embedder config."""
        if self._dim is None:
            return True  # empty store accepts anything
        return self._dim == dim and self._backend == backend and self._model == model

    def add(self, slug: str, vector: list[float]) -> None:
        with self._lock:
            self._items[slug] = list(vector)
            self._save()

    def remove(self, slug: str) -> None:
        with self._lock:
            if slug in self._items:
                del self._items[slug]
                self._save()

    def has(self, slug: str) -> bool:
        return slug in self._items

    def count(self) -> int:
        return len(self._items)

    def search(self, query_vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Top-K (slug, similarity) by cosine similarity."""
        if not query_vector or not self._items:
            return []
        scored: list[tuple[str, float]] = []
        for slug, vec in self._items.items():
            scored.append((slug, cosine(query_vector, vec)))
        scored.sort(key=lambda p: -p[1])
        return scored[:k]

    def stats(self) -> dict:
        return {
            "count": len(self._items),
            "dim": self._dim,
            "backend": self._backend,
            "model": self._model,
        }


def get_default_path() -> Path:
    from .config import CONFIG
    return Path(CONFIG.knowledge["base_dir"]) / "embeddings.json"


VECTOR_STORE = VectorStore(get_default_path())
