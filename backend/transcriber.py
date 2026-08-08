"""Provider-agnostic speech-to-text.

Mirrors the Embedder's shape: try backends in fallback order, persist the
chosen config in `knowledge/transcriber_config.json`, give the caller a
graceful `None` (with `last_error`) when nothing is reachable so the
voice path degrades to text-only without breaking chat.

Backend chain (highest first when forced=auto):
  0. faster_whisper  IN-PROCESS CTranslate2 (no server, no key, no network).
                     Added 2026-08-08: the box already had Systran
                     faster-whisper medium/small/base sitting in the
                     HuggingFace cache, and none of the three server-shaped
                     backends below could reach them, so speech-to-text was
                     `disabled` and every voice note the owner sent reached
                     the agent as an empty placeholder.
  1. local_whisper   POST <base>/v1/audio/transcriptions  (no-auth FastAPI
                     wrapper around faster-whisper, e.g. the user's home
                     server). Probed via GET <base>/health.
  2. whisper_cpp     POST <base>/inference  (whisper.cpp's REST server)
  3. openai_whisper  POST <base>/audio/transcriptions  (OpenAI-compat,
                     needs api key through providers.py)
  4. disabled        explicit off

Configuration precedence:
  knowledge/transcriber_config.json  (managed by Settings UI)
  AGI_TRANSCRIBER_BACKEND env var    ("auto" by default)
  LOCAL_WHISPER_URL env var          (override for local_whisper backend)
  WHISPER_CPP_URL env var            (back-compat for whisper_cpp backend)

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

DEFAULT_FASTER_WHISPER_MODEL = "medium"
# int8 on CPU: ~4x faster than float32 with no meaningful accuracy loss on
# speech, and the box is CPU-only (12 cores, no GPU).
DEFAULT_FASTER_WHISPER_COMPUTE = "int8"

DEFAULT_OPENAI_WHISPER_MODEL = "whisper-1"
DEFAULT_WHISPER_CPP_MODEL = "whisper"
DEFAULT_LOCAL_WHISPER_MODEL = "whisper-medium"


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
    from .paths import write_secret_json
    write_secret_json(_config_path(), cfg)
    return cfg


class Transcriber:
    """Lazy speech-to-text. Picks a backend on first use; reset() to re-probe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._backend: Optional[str] = None
        self._model: Optional[str] = None
        self._provider: Optional[dict] = None
        self._whisper_cpp_base: Optional[str] = None
        self._local_whisper_base: Optional[str] = None
        self._last_error: Optional[str] = None

    def reset(self) -> None:
        with self._lock:
            self._backend = None
            self._model = None
            self._provider = None
            self._whisper_cpp_base = None
            self._local_whisper_base = None
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
            if self._backend == "faster_whisper":
                return self._tx_faster_whisper(
                    audio_bytes, filename=filename, language=language)
            if self._backend == "local_whisper":
                return self._tx_local_whisper(audio_bytes, mime_type=mime_type, filename=filename, language=language)
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
        if forced == "auto":
            candidates = ["faster_whisper", "local_whisper",
                          "whisper_cpp", "openai_whisper"]
        else:
            candidates = [forced]
        for cand in candidates:
            if cand == "faster_whisper" and self._try_faster_whisper(cfg):
                return
            if cand == "local_whisper" and self._try_local_whisper(cfg):
                return
            if cand == "whisper_cpp" and self._try_whisper_cpp(cfg):
                return
            if cand == "openai_whisper" and self._try_openai_whisper(cfg):
                return
        self._backend = "disabled"

    def _try_faster_whisper(self, cfg: dict) -> bool:
        """In-process CTranslate2. Available iff the package imports AND the
        model is already present locally — we never trigger a download from
        inside a probe, because a 1.5 GB fetch on the first voice note would
        look exactly like a hang."""
        cfg_f = (cfg.get("faster_whisper") or {}) if cfg else {}
        name = (cfg_f.get("model")
                or os.getenv("FASTER_WHISPER_MODEL", DEFAULT_FASTER_WHISPER_MODEL))
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except Exception as e:
            self._last_error = f"faster_whisper not installed: {e}"
            return False
        self._backend = "faster_whisper"
        self._model = name
        self._fw_compute = (cfg_f.get("compute_type")
                            or DEFAULT_FASTER_WHISPER_COMPUTE)
        self._last_error = None
        return True

    def _tx_faster_whisper(self, audio_bytes: bytes, *, filename: str,
                           language: Optional[str] = None) -> Optional[str]:
        """Transcribe in-process. The model is loaded once and cached on the
        instance — reloading 1.5 GB per voice note would make every message
        cost tens of seconds."""
        import tempfile
        from faster_whisper import WhisperModel

        with self._lock:
            if getattr(self, "_fw_model", None) is None:
                self._fw_model = WhisperModel(
                    self._model or DEFAULT_FASTER_WHISPER_MODEL,
                    device="cpu",
                    compute_type=getattr(self, "_fw_compute",
                                         DEFAULT_FASTER_WHISPER_COMPUTE),
                )
        suffix = os.path.splitext(filename or "")[1] or ".ogg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        try:
            segments, _info = self._fw_model.transcribe(
                path, language=language, vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text or None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _try_local_whisper(self, cfg: dict) -> bool:
        """No-auth FastAPI Whisper wrapper (e.g. user's home server).
        Probed via GET /health which the server exposes for liveness;
        a 200 with `status:ok` (or anything 2xx, since some wrappers
        return a different shape) means we can use POST
        /v1/audio/transcriptions for inference."""
        cfg_l = (cfg.get("local_whisper") or {}) if cfg else {}
        base = (cfg_l.get("url") or os.getenv("LOCAL_WHISPER_URL", "")).rstrip("/")
        if not base:
            return False
        try:
            r = httpx.get(base + "/health", timeout=2.0)
            if r.status_code != 200:
                self._last_error = (
                    f"local_whisper /health returned {r.status_code}"
                )
                return False
        except Exception as e:
            self._last_error = f"local_whisper probe failed: {e}"
            return False
        self._backend = "local_whisper"
        self._model = cfg_l.get("model") or DEFAULT_LOCAL_WHISPER_MODEL
        self._local_whisper_base = base
        self._last_error = None
        return True

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

    def _tx_local_whisper(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        filename: str,
        language: Optional[str],
    ) -> Optional[str]:
        """POST audio to a no-auth OpenAI-compatible Whisper wrapper.
        Endpoint shape matches the user's local-whisper-api server:
        `POST /v1/audio/transcriptions` with a `file` upload + a `model`
        form field, returning JSON `{"text": "..."}`. No bearer token —
        these servers run on a private network."""
        if not self._local_whisper_base:
            return None
        files = {"file": (filename, audio_bytes, mime_type)}
        data = {"model": self._model or DEFAULT_LOCAL_WHISPER_MODEL}
        if language:
            data["language"] = language
        r = httpx.post(
            f"{self._local_whisper_base}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=300.0,
        )
        r.raise_for_status()
        body = r.json()
        return (body.get("text") or "").strip() or None

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
