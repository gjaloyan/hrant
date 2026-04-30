"""Provider-agnostic speech-to-text.

Mirrors the Embedder's shape: try backends in fallback order, persist the
chosen config in `knowledge/transcriber_config.json`, give the caller a
graceful `None` (with `last_error`) when nothing is reachable so the
voice path degrades to text-only without breaking chat.

Backend chain (highest first when forced=auto):
  1. whisper_cpp     POST <base>/inference  (whisper.cpp's REST server)
  2. openai_whisper  POST <base>/audio/transcriptions  (OpenAI-compat)
  3. disabled        explicit off

Configuration precedence:
  knowledge/transcriber_config.json  (managed by Settings UI)
  AGI_TRANSCRIBER_BACKEND env var    ("auto" by default)
  WHISPER_CPP_URL env var             (back-compat)

Reset() drops the cached backend so a downed server can be replaced
without restarting the agent.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

DEFAULT_OPENAI_WHISPER_MODEL = "whisper-1"
DEFAULT_WHISPER_CPP_MODEL = "whisper"


def _config_path() -> Path:
    from .config import CONFIG
    return Path(CONFIG.knowledge["base_dir"]) / "transcriber_config.json"


def load_config() -> dict:
    p = _config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def save_config(cfg: dict) -> dict:
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


class Transcriber:
    """Lazy speech-to-text. Picks a backend on first use; reset() to re-probe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._backend: Optional[str] = None
        self._model: Optional[str] = None
        self._provider: Optional[dict] = None
        self._whisper_cpp_base: Optional[str] = None
        self._last_error: Optional[str] = None

    def reset(self) -> None:
        with self._lock:
            self._backend = None
            self._model = None
            self._provider = None
            self._whisper_cpp_base = None
            self._last_error = None

    def status(self) -> dict:
        if self._backend is None:
            self._pick_backend()
        return {
            "backend": self._backend,
            "model": self._model,
            "last_error": self._last_error,
            "config": load_config(),
        }

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str = "audio/ogg",
        filename: str = "audio.ogg",
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Return transcribed text or None if no backend is available."""
        if not audio_bytes:
            return None
        if self._backend is None:
            self._pick_backend()
        if self._backend in (None, "disabled"):
            return None
        try:
            if self._backend == "whisper_cpp":
                return self._tx_whisper_cpp(audio_bytes, filename=filename, language=language)
            if self._backend == "openai_whisper":
                return self._tx_openai_whisper(audio_bytes, mime_type=mime_type, filename=filename, language=language)
        except Exception as e:
            self._last_error = f"transcribe via {self._backend} failed: {e}"
            log.warning("transcribe failed: %s", e)
            return None
        return None

    # ---- backend selection ----

    def _pick_backend(self) -> None:
        cfg = load_config()
        forced = (cfg.get("backend") or os.getenv("AGI_TRANSCRIBER_BACKEND", "auto")).lower()
        if forced == "disabled":
            self._backend = "disabled"
            return
        candidates = ["whisper_cpp", "openai_whisper"] if forced == "auto" else [forced]
        for cand in candidates:
            if cand == "whisper_cpp" and self._try_whisper_cpp(cfg):
                return
            if cand == "openai_whisper" and self._try_openai_whisper(cfg):
                return
        self._backend = "disabled"

    def _try_whisper_cpp(self, cfg: dict) -> bool:
        cfg_w = (cfg.get("whisper_cpp") or {}) if cfg else {}
        base = (cfg_w.get("url") or os.getenv("WHISPER_CPP_URL", "")).rstrip("/")
        if not base:
            return False
        # whisper.cpp server's load endpoint is /load — we only probe with HEAD on /
        try:
            r = httpx.get(base + "/", timeout=2.0)
            if r.status_code >= 500:
                return False
        except Exception as e:
            self._last_error = f"whisper_cpp probe failed: {e}"
            return False
        self._backend = "whisper_cpp"
        self._model = cfg_w.get("model") or DEFAULT_WHISPER_CPP_MODEL
        self._whisper_cpp_base = base
        self._last_error = None
        return True

    def _try_openai_whisper(self, cfg: dict) -> bool:
        cfg_o = (cfg.get("openai_whisper") or {}) if cfg else {}
        from .providers import get_api_key, get_providers
        prov = None
        for p in get_providers():
            if p.get("type") in ("openai", "openai_compatible") and p.get("enabled", True):
                prov = p
                break
        if not prov:
            return False
        api_key = get_api_key(prov)
        if not api_key:
            return False
        # Don't probe — Whisper has no cheap "is alive" call. Mark active;
        # transcribe() will surface real failures via last_error.
        self._backend = "openai_whisper"
        self._model = cfg_o.get("model") or DEFAULT_OPENAI_WHISPER_MODEL
        self._provider = prov
        self._last_error = None
        return True

    # ---- backend impls ----

    def _tx_whisper_cpp(
        self, audio_bytes: bytes, *, filename: str, language: Optional[str]
    ) -> Optional[str]:
        if not self._whisper_cpp_base:
            return None
        files = {"file": (filename, audio_bytes, "application/octet-stream")}
        data = {"response_format": "json"}
        if language:
            data["language"] = language
        r = httpx.post(
            f"{self._whisper_cpp_base}/inference",
            files=files,
            data=data,
            timeout=300.0,
        )
        r.raise_for_status()
        body = r.json()
        return (body.get("text") or "").strip() or None

    def _tx_openai_whisper(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        filename: str,
        language: Optional[str],
    ) -> Optional[str]:
        if not self._provider:
            return None
        from .providers import get_api_key
        api_key = get_api_key(self._provider)
        if not api_key:
            return None
        base = (self._provider.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        files = {"file": (filename, audio_bytes, mime_type)}
        data = {"model": self._model or DEFAULT_OPENAI_WHISPER_MODEL}
        if language:
            data["language"] = language
        r = httpx.post(
            f"{base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=300.0,
        )
        r.raise_for_status()
        body = r.json()
        return (body.get("text") or "").strip() or None


TRANSCRIBER = Transcriber()
