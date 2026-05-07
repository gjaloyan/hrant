"""Provider-agnostic text-to-speech.

Mirrors `transcriber.py`: try backends in fallback order, persist the
chosen config in `knowledge/tts_config.json`, give the caller a
graceful `None` (with `last_error`) when nothing is reachable so the
voice-reply path degrades to text-only without breaking the chat.

Backend chain (highest first when forced=auto):
  1. local_piper     POST <base>/v1/audio/speech  (no-auth FastAPI
                     wrapper around Piper, e.g. the user's home
                     server). Probed via GET <base>/health.
  2. openai_tts      POST <base>/v1/audio/speech  (OpenAI-compatible,
                     needs api key through providers.py)
  3. disabled        explicit off

Configuration precedence:
  knowledge/tts_config.json         (managed by Settings UI)
  AGI_TTS_BACKEND env var           ("auto" by default)
  LOCAL_PIPER_URL env var           (override for local_piper backend)

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

DEFAULT_LOCAL_PIPER_VOICE = "en_US-lessac-medium"
DEFAULT_OPENAI_TTS_MODEL = "tts-1"
DEFAULT_OPENAI_TTS_VOICE = "alloy"


def _config_path() -> Path:
    from .config import CONFIG
    return Path(CONFIG.knowledge["base_dir"]) / "tts_config.json"


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


class Synthesizer:
    """Lazy text-to-speech. Picks a backend on first use; reset() to re-probe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._backend: Optional[str] = None
        self._voice: Optional[str] = None
        self._provider: Optional[dict] = None
        self._local_piper_base: Optional[str] = None
        self._last_error: Optional[str] = None

    def reset(self) -> None:
        with self._lock:
            self._backend = None
            self._voice = None
            self._provider = None
            self._local_piper_base = None
            self._last_error = None

    def status(self) -> dict:
        if self._backend is None:
            self._pick_backend()
        return {
            "backend": self._backend,
            "voice": self._voice,
            "last_error": self._last_error,
            "config": load_config(),
        }

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
    ) -> Optional[bytes]:
        """Return WAV / OGG audio bytes or None if no backend is available.

        The exact container depends on the backend:
          - local_piper returns audio/wav (16-bit PCM, mono, ~22050 Hz
            for the default `en_US-lessac-medium` voice)
          - openai_tts returns audio/mpeg (.mp3) by default
        Caller decides what to do with the bytes (write to file, ship
        through a channel, embed in an HTTP response, etc.).
        """
        if not text or not text.strip():
            return None
        if self._backend is None:
            self._pick_backend()
        if self._backend in (None, "disabled"):
            return None
        try:
            if self._backend == "local_piper":
                return self._tx_local_piper(text, voice=voice)
            if self._backend == "openai_tts":
                return self._tx_openai_tts(text, voice=voice)
        except Exception as e:
            self._last_error = f"synthesize via {self._backend} failed: {e}"
            log.warning("tts synthesize failed: %s", e)
            return None
        return None

    # ---- backend selection ----

    def _pick_backend(self) -> None:
        cfg = load_config()
        forced = (cfg.get("backend") or os.getenv("AGI_TTS_BACKEND", "auto")).lower()
        if forced == "disabled":
            self._backend = "disabled"
            return
        if forced == "auto":
            candidates = ["local_piper", "openai_tts"]
        else:
            candidates = [forced]
        for cand in candidates:
            if cand == "local_piper" and self._try_local_piper(cfg):
                return
            if cand == "openai_tts" and self._try_openai_tts(cfg):
                return
        self._backend = "disabled"

    def _try_local_piper(self, cfg: dict) -> bool:
        cfg_p = (cfg.get("local_piper") or {}) if cfg else {}
        base = (cfg_p.get("url") or os.getenv("LOCAL_PIPER_URL", "")).rstrip("/")
        if not base:
            return False
        try:
            r = httpx.get(base + "/health", timeout=2.0)
            if r.status_code != 200:
                self._last_error = f"local_piper /health returned {r.status_code}"
                return False
        except Exception as e:
            self._last_error = f"local_piper probe failed: {e}"
            return False
        self._backend = "local_piper"
        self._voice = cfg_p.get("voice") or DEFAULT_LOCAL_PIPER_VOICE
        self._local_piper_base = base
        self._last_error = None
        return True

    def _try_openai_tts(self, cfg: dict) -> bool:
        cfg_o = (cfg.get("openai_tts") or {}) if cfg else {}
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
        # No cheap probe call — OpenAI TTS doesn't expose a liveness
        # endpoint. Mark active; synthesize() surfaces real failures
        # via last_error.
        self._backend = "openai_tts"
        self._voice = cfg_o.get("voice") or DEFAULT_OPENAI_TTS_VOICE
        self._provider = prov
        self._last_error = None
        return True

    # ---- backend impls ----

    def _tx_local_piper(
        self, text: str, *, voice: Optional[str] = None,
    ) -> Optional[bytes]:
        """POST text to a no-auth Piper wrapper.
        Endpoint matches the user's local-piper-tts-api server:
        `POST /v1/audio/speech` with JSON `{"input": text, "voice":
        voice}`, response body is `audio/wav` bytes."""
        if not self._local_piper_base:
            return None
        body = {"input": text, "voice": voice or self._voice or DEFAULT_LOCAL_PIPER_VOICE}
        r = httpx.post(
            f"{self._local_piper_base}/v1/audio/speech",
            json=body,
            timeout=60.0,
        )
        r.raise_for_status()
        if not r.content:
            return None
        return r.content

    def _tx_openai_tts(
        self, text: str, *, voice: Optional[str] = None,
    ) -> Optional[bytes]:
        """POST text to OpenAI / OpenAI-compatible TTS endpoint.
        Returns audio/mpeg (.mp3) bytes by default."""
        if not self._provider:
            return None
        from .providers import get_api_key
        api_key = get_api_key(self._provider)
        if not api_key:
            return None
        base = (self._provider.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        body = {
            "model": DEFAULT_OPENAI_TTS_MODEL,
            "input": text,
            "voice": voice or self._voice or DEFAULT_OPENAI_TTS_VOICE,
        }
        r = httpx.post(
            f"{base}/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=60.0,
        )
        r.raise_for_status()
        if not r.content:
            return None
        return r.content


SYNTHESIZER = Synthesizer()
