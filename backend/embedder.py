"""Provider-agnostic embedder.

Tries embedding sources in fallback order:
  1. llama.cpp server (POST /v1/embeddings, OpenAI-compat) — local,
     free, configured via LLAMA_CPP_EMBEDDING_URL env var. Preferred
     when set because the user explicitly stood it up; we don't probe
     for it otherwise.
  2. Ollama (POST /api/embeddings) — local, free, fastest if running
     locally on the same machine.
  3. OpenAI-compatible providers from providers.json (OpenAI, xAI, etc.)
     using their /embeddings endpoint with the configured api_key.
  4. Cohere (/v1/embed) if a Cohere provider is configured.

If nothing is available, embedder.embed() returns None and the caller
should treat the vector path as disabled — text + graph search continue
to work.

Configuration:
  AGI_EMBEDDER_BACKEND        — "auto" (default), "llama_cpp", "ollama",
                                "openai", "cohere"
  AGI_EMBEDDER_MODEL          — backend-specific model id; defaults below
  LLAMA_CPP_EMBEDDING_URL     — base URL of a llama-server running with
                                --embedding (e.g. http://host:8080/v1)
  LLAMA_CPP_EMBEDDING_MODEL   — model name to send in the request body
                                (llama.cpp echoes this back)

We never throw on transient errors; the embedder degrades gracefully so
chat doesn't break when an embedding source is temporarily down.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

log = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"  # 768 dim, small, popular
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"  # 1536 dim
DEFAULT_COHERE_MODEL = "embed-english-v3.0"  # 1024 dim
DEFAULT_LLAMA_CPP_MODEL = "bge-m3"  # what we echo back; server runs whatever GGUF is loaded


class Embedder:
    """Lazy embedder. Picks a backend on first use and remembers it."""

    def __init__(self):
        self._backend: Optional[str] = None
        self._model: Optional[str] = None
        self._dim: Optional[int] = None
        self._provider: Optional[dict] = None  # cached provider config when applicable
        self._llama_cpp_base: Optional[str] = None  # cached base URL for llama.cpp server

    # ---- public ----

    def embed(self, text: str) -> Optional[list[float]]:
        """Return an embedding or None if no backend is available."""
        if not text:
            return None
        if self._backend is None:
            self._pick_backend()
        if self._backend == "disabled":
            return None
        try:
            if self._backend == "llama_cpp":
                return self._embed_llama_cpp(text)
            if self._backend == "ollama":
                return self._embed_ollama(text)
            if self._backend == "openai":
                return self._embed_openai_compatible(text)
            if self._backend == "cohere":
                return self._embed_cohere(text)
        except Exception as e:
            log.warning("Embedder backend %s failed: %s", self._backend, e)
            return None
        return None

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def model(self) -> Optional[str]:
        return self._model

    def status(self) -> dict:
        if self._backend is None:
            self._pick_backend()
        return {
            "backend": self._backend,
            "model": self._model,
            "dim": self._dim,
        }

    # ---- backend selection ----

    def _pick_backend(self) -> None:
        forced = os.getenv("AGI_EMBEDDER_BACKEND", "auto").lower()
        forced_model = os.getenv("AGI_EMBEDDER_MODEL", "")

        candidates: list[str]
        if forced == "auto":
            # llama.cpp goes first when the user has explicitly configured
            # LLAMA_CPP_EMBEDDING_URL — they stood up a server, use it.
            candidates = ["llama_cpp", "ollama", "openai", "cohere"]
        else:
            candidates = [forced]

        for cand in candidates:
            if cand == "llama_cpp" and self._try_llama_cpp(forced_model):
                return
            if cand == "ollama" and self._try_ollama(forced_model):
                return
            if cand == "openai" and self._try_openai(forced_model):
                return
            if cand == "cohere" and self._try_cohere(forced_model):
                return
        self._backend = "disabled"

    def _try_llama_cpp(self, forced_model: str) -> bool:
        base = os.getenv("LLAMA_CPP_EMBEDDING_URL", "").rstrip("/")
        if not base:
            return False
        # Normalise: accept both "http://host:port" and "http://host:port/v1".
        if not base.endswith("/v1"):
            base = base.rstrip("/") + "/v1"
        model = forced_model or os.getenv("LLAMA_CPP_EMBEDDING_MODEL", DEFAULT_LLAMA_CPP_MODEL)
        try:
            r = httpx.post(
                f"{base}/embeddings",
                json={"model": model, "input": "ok"},
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
            vec = data["data"][0]["embedding"]
            if not vec:
                return False
            self._backend = "llama_cpp"
            self._model = model
            self._dim = len(vec)
            self._llama_cpp_base = base
            return True
        except Exception:
            return False

    def _try_ollama(self, forced_model: str) -> bool:
        base = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE).rstrip("/")
        try:
            r = httpx.get(f"{base}/api/tags", timeout=2.0)
            r.raise_for_status()
        except Exception:
            return False
        model = forced_model or DEFAULT_OLLAMA_MODEL
        # Probe with a tiny embed call to confirm model is pulled
        try:
            r = httpx.post(
                f"{base}/api/embeddings",
                json={"model": model, "prompt": "ok"},
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding") or []
            if not vec:
                return False
            self._backend = "ollama"
            self._model = model
            self._dim = len(vec)
            return True
        except Exception:
            return False

    def _try_openai(self, forced_model: str) -> bool:
        prov = self._find_provider("openai")
        if not prov:
            # Try any openai-compatible provider with an api_key (xai, openrouter, ...)
            for ptype in ("openai_compatible", "together", "openrouter", "groq"):
                prov = self._find_provider(ptype)
                if prov:
                    break
        if not prov:
            return False
        api_key = self._resolve_api_key(prov)
        if not api_key:
            return False
        base = (prov.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        model = forced_model or DEFAULT_OPENAI_MODEL
        try:
            r = httpx.post(
                f"{base}/embeddings",
                json={"model": model, "input": "ok"},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
            vec = data["data"][0]["embedding"]
            self._backend = "openai"
            self._model = model
            self._provider = prov
            self._dim = len(vec)
            return True
        except Exception:
            return False

    def _try_cohere(self, forced_model: str) -> bool:
        prov = self._find_provider("cohere")
        if not prov:
            return False
        api_key = self._resolve_api_key(prov)
        if not api_key:
            return False
        model = forced_model or DEFAULT_COHERE_MODEL
        try:
            r = httpx.post(
                "https://api.cohere.com/v1/embed",
                json={"model": model, "texts": ["ok"], "input_type": "search_document"},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
            vec = data["embeddings"][0]
            self._backend = "cohere"
            self._model = model
            self._provider = prov
            self._dim = len(vec)
            return True
        except Exception:
            return False

    @staticmethod
    def _find_provider(ptype: str) -> Optional[dict]:
        from .providers import get_providers
        for p in get_providers():
            if p.get("type") == ptype and p.get("enabled", True):
                return p
        return None

    @staticmethod
    def _resolve_api_key(p: dict) -> str:
        from .providers import get_api_key
        return get_api_key(p) or ""

    # ---- embed implementations ----

    def _embed_llama_cpp(self, text: str) -> Optional[list[float]]:
        if not self._llama_cpp_base:
            return None
        r = httpx.post(
            f"{self._llama_cpp_base}/embeddings",
            json={"model": self._model, "input": text},
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    def _embed_ollama(self, text: str) -> Optional[list[float]]:
        base = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE).rstrip("/")
        r = httpx.post(
            f"{base}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json().get("embedding")

    def _embed_openai_compatible(self, text: str) -> Optional[list[float]]:
        if not self._provider:
            return None
        api_key = self._resolve_api_key(self._provider)
        base = (self._provider.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        r = httpx.post(
            f"{base}/embeddings",
            json={"model": self._model, "input": text},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    def _embed_cohere(self, text: str) -> Optional[list[float]]:
        if not self._provider:
            return None
        api_key = self._resolve_api_key(self._provider)
        r = httpx.post(
            "https://api.cohere.com/v1/embed",
            json={"model": self._model, "texts": [text], "input_type": "search_document"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]


EMBEDDER = Embedder()
