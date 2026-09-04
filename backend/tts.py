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
import re
import threading
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

DEFAULT_LOCAL_PIPER_VOICE = "en_US-lessac-medium"
DEFAULT_LOCAL_PIPER_VOICE_RU = "ru_RU-irina-medium"

# Edge TTS — free Microsoft online TTS, ~400 voices, multilingual.
# Defaults picked for naturalness on each language. Full voice list:
#   python -m edge_tts --list-voices
DEFAULT_EDGE_TTS_VOICE = "en-US-AriaNeural"
DEFAULT_EDGE_TTS_VOICE_RU = "ru-RU-SvetlanaNeural"
DEFAULT_OPENAI_TTS_MODEL = "tts-1"
DEFAULT_OPENAI_TTS_VOICE = "alloy"


# Cyrillic Unicode block (Russian, Ukrainian, Bulgarian, Serbian, etc.).
# `[А-яЁё]` covers the full block including the accented letter forms.
# Detection is text-content based, NOT user-language-preference based —
# the user could switch language mid-conversation and the agent might
# answer in whichever language fits the question. Picking the voice from
# the answer's actual content is the only approach that produces correct
# pronunciation regardless of `user.md` preferences.
_CYRILLIC_RE = re.compile(r"[А-яЁё]")


# Scripts a voice has to match to be able to read the text at all. Keyed
# by the language tag that every voice name carries, so the rule comes
# from the voice itself rather than from a list someone has to remember
# to extend.
_SCRIPT_LANGS = (
    ("armenian", "hy", re.compile(r"[԰-֏]")),
    ("cyrillic", "ru", re.compile(r"[Ѐ-ӿ]")),
    ("greek", "el", re.compile(r"[Ͱ-Ͽ]")),
    ("hebrew", "he", re.compile(r"[֐-׿]")),
    ("arabic", "ar", re.compile(r"[؀-ۿ]")),
    ("georgian", "ka", re.compile(r"[Ⴀ-ჿ]")),
)
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _language_not_covered(text: str, voice: Optional[str]) -> Optional[str]:
    """A plain reason when the voice in use cannot read this script.

    Prod 2026-09-03: an Armenian answer came back as "synthesize via
    edge_tts failed: No audio was received. Please verify that your
    parameters are correct." The parameters were correct. Measured the
    next day: edge-tts publishes 322 voices and not one is Armenian, and
    the local Piper has only en_US and ru_RU installed. Telling the owner
    to check settings that are right sends him looking for a fault that
    is not there.

    Only consulted after synthesis has already failed, so a wrong answer
    here costs a worse message and nothing else.

    Returns None when it cannot tell: no voice name, no text, a script it
    does not track, or a voice whose language already matches.
    """
    body = (text or "").strip()
    name = (voice or "").strip()
    if not body or not name:
        return None
    letters = _LETTER_RE.findall(body)
    if not letters:
        return None
    # The voice's language tag: "en-US-AriaNeural" and "ru_RU-irina-medium"
    # both start with it.
    tag = re.split(r"[-_]", name, maxsplit=1)[0].lower()
    for label, lang, pattern in _SCRIPT_LANGS:
        hits = pattern.findall(body)
        # A dominant script, not merely a present one: a Russian sentence
        # with one Armenian word in it synthesizes fine (measured).
        if len(hits) * 2 <= len(letters):
            continue
        if tag == lang:
            return None
        return (f"no installed voice can speak {label.capitalize()} — the "
                f"voice in use is {name}. The text was left unspoken; "
                f"nothing is misconfigured.")
    return None


def _pick_voice(text: str, *, default: str, ru: Optional[str] = None) -> str:
    """Choose Piper voice name based on the text's script.
    Currently only Russian needs explicit handling — the agent's
    English / Armenian / etc. answers all sound fine on the
    default English voice (mediocre but intelligible). Russian on
    the English voice is unintelligible garbage, hence the split."""
    if not text or not ru:
        return default
    if _CYRILLIC_RE.search(text):
        return ru
    return default


# --- Speech text cleaner (markdown/emphasis -> spoken prose) --------------
#
# The answer text is written for the eye (Telegram markdown: `*_..._*`
# emphasis, `### headers`, `` `code` ``, bullet lists, links). Fed raw to
# a synthesizer it vocalizes the MARKERS ("asterisk underscore ..."). This
# strips the formatting marks while keeping the words and ordinary
# sentence punctuation (the periods/commas that shape prosody).

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_STAR_RE = re.compile(r"\*([^*\n]+)\*")
# `_italic_` only when the underscores hug a whole word group — never
# inside identifiers like file_name or backend/builtin_tools.py.
_ITALIC_US_RE = re.compile(r"(?<![\w/])_([^_\n]+)_(?![\w/])")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_STRIKE_RE = re.compile(r"~~([^~]+)~~")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_QUOTE_RE = re.compile(r"(?m)^\s{0,3}>\s?")
_BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+")
_ORDERED_RE = re.compile(r"(?m)^\s*\d+\.\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")


def strip_markdown_for_speech(text: Optional[str]) -> str:
    """Render markdown / Telegram-emphasis answer text as plain spoken
    prose for TTS: drop the formatting markers, keep the words and
    sentence punctuation. Deterministic — runs regardless of model."""
    if not text:
        return ""
    s = _FENCE_RE.sub(" ", text)        # code blocks: not speakable
    s = s.replace("*_", "").replace("_*", "")   # Telegram bold-italic combo
    s = _LINK_RE.sub(r"\1", s)          # [text](url) -> text
    s = _BOLD_RE.sub(r"\1", s)          # **bold** -> bold
    s = _STRIKE_RE.sub(r"\1", s)        # ~~strike~~ -> strike
    s = _ITALIC_STAR_RE.sub(r"\1", s)   # *italic* -> italic
    s = _ITALIC_US_RE.sub(r"\1", s)     # _italic_ -> italic (word-bounded)
    s = _INLINE_CODE_RE.sub(r"\1", s)   # `code` -> code
    s = _HEADING_RE.sub("", s)          # ### Heading -> Heading
    s = _QUOTE_RE.sub("", s)            # > quote -> quote
    s = _BULLET_RE.sub("", s)           # - item -> item
    s = _ORDERED_RE.sub("", s)          # 1. item -> item
    s = s.replace("*", "").replace("`", "")     # stray leftover marks
    s = re.sub(r"\n{2,}", ". ", s)      # paragraph break -> sentence pause
    s = s.replace("\n", " ")
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


# --- Telegram voice format conversion (WAV -> OGG/Opus) -------------------
#
# Telegram's "voice message" bubble (reply_voice) renders correctly only
# when the file is OGG container + Opus codec at 48 kHz mono. Piper hands
# us WAV (16-bit PCM at the model's native rate). PTB v20+ accepts WAV
# bytes for upload but Telegram itself plays them as a generic audio
# attachment OR as a distorted voice bubble depending on the client
# build — neither is what the user expects.
#
# We use ffmpeg as a local tool: cheap, ubiquitous on Linux, requires a
# one-time install on Windows. If ffmpeg isn't on PATH we fall back to
# returning the original WAV bytes with `format="wav"` so the caller can
# still send something rather than going silent. The first call without
# ffmpeg also logs a one-time hint with install commands so the user
# knows what to do.

_FFMPEG_PROBED = False
_FFMPEG_AVAILABLE = False


def _ffmpeg_available() -> bool:
    """Cached probe — ffmpeg presence doesn't change at runtime, so we
    test once and remember the result. Call `reset_ffmpeg_probe()` if
    you install ffmpeg without restarting."""
    global _FFMPEG_PROBED, _FFMPEG_AVAILABLE
    if _FFMPEG_PROBED:
        return _FFMPEG_AVAILABLE
    import shutil
    _FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
    _FFMPEG_PROBED = True
    if not _FFMPEG_AVAILABLE:
        log.warning(
            "ffmpeg not found on PATH — TTS replies in Telegram will go "
            "out as raw WAV (functional but plays as a generic audio "
            "attachment, not a native voice bubble). To fix, install "
            "ffmpeg: Windows `winget install Gyan.FFmpeg` (then restart "
            "the agent), Linux `apt install ffmpeg`, macOS `brew install "
            "ffmpeg`."
        )
    return _FFMPEG_AVAILABLE


def reset_ffmpeg_probe() -> None:
    """Force re-probe on next call. Call after installing ffmpeg
    without restarting the process."""
    global _FFMPEG_PROBED, _FFMPEG_AVAILABLE
    _FFMPEG_PROBED = False
    _FFMPEG_AVAILABLE = False


def convert_wav_to_telegram_voice(wav_bytes: bytes) -> tuple[bytes, str]:
    """Convert Piper's WAV output to Telegram's native voice format
    (OGG container + Opus codec, 48 kHz mono, ~32 kbps).

    Returns `(bytes, format)` where format is either `"ogg"` (success)
    or `"wav"` (ffmpeg unavailable / conversion failed → caller sends
    the original WAV as fallback). Conversion happens entirely
    in-memory through ffmpeg's stdin/stdout — no temp files, so a
    crash in the bot doesn't leave audio cruft on disk.
    """
    if not wav_bytes:
        return wav_bytes, "wav"
    if not _ffmpeg_available():
        return wav_bytes, "wav"
    import subprocess
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-ac", "1",
                "-ar", "48000",
                "-c:a", "libopus",
                "-b:a", "32k",
                "-f", "ogg",
                "pipe:1",
            ],
            input=wav_bytes,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        # Race between the cached probe and an uninstall; surface the
        # WAV so the caller still has something to send.
        reset_ffmpeg_probe()
        return wav_bytes, "wav"
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg WAV→OGG conversion timed out after 30s")
        return wav_bytes, "wav"
    except Exception as e:
        log.warning("ffmpeg WAV→OGG conversion failed: %s", e)
        return wav_bytes, "wav"
    if proc.returncode != 0 or not proc.stdout:
        # Common cause: libopus not compiled into the user's ffmpeg
        # build (rare on prebuilt distros, can happen on stripped
        # builds). Surface the stderr in the warning so it's
        # diagnosable without re-running the bot.
        err_tail = (proc.stderr or b"")[-400:].decode("utf-8", "replace")
        log.warning(
            "ffmpeg WAV→OGG conversion exit=%s, stdout=%s bytes, stderr=%s",
            proc.returncode, len(proc.stdout or b""), err_tail,
        )
        return wav_bytes, "wav"
    return proc.stdout, "ogg"


def _config_path() -> Path:
    """Where tts_config.json lives.

    Resolved through `paths.knowledge_dir()`, which re-reads
    HRANT_DATA_DIR on every call, rather than through
    `CONFIG.knowledge["base_dir"]`, which is fixed at import. Both give
    the same directory in production -- checked on the box 2026-09-04 --
    but only one of them can be redirected by a test.

    That difference had a cost. `test_load_config_tolerates_invalid_json`
    writes the string "garbage {" to this path after setting
    HRANT_DATA_DIR to a tmp dir, and the env var did not reach here, so
    the write landed in the REAL data directory. Prod's tts_config.json
    has been exactly those nine bytes since 2026-08-07 14:11, which is
    when someone ran the suite on the box: the owner's configured voices
    were destroyed and TTS has been running on defaults ever since.
    """
    from .paths import knowledge_dir
    return knowledge_dir() / "tts_config.json"


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


class Synthesizer:
    """Lazy text-to-speech. Picks a backend on first use; reset() to re-probe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._backend: Optional[str] = None
        self._voice: Optional[str] = None
        self._voice_ru: Optional[str] = None
        self._provider: Optional[dict] = None
        self._local_piper_base: Optional[str] = None
        self._last_error: Optional[str] = None

    def reset(self) -> None:
        with self._lock:
            self._backend = None
            self._voice = None
            self._voice_ru = None
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
            if self._backend == "edge_tts":
                return self._tx_edge_tts(text, voice=voice)
            if self._backend == "local_piper":
                return self._tx_local_piper(text, voice=voice)
            if self._backend == "openai_tts":
                return self._tx_openai_tts(text, voice=voice)
        except Exception as e:
            # Name the real reason when there is one. The provider's own
            # wording ("verify that your parameters are correct") sends
            # the owner to check settings that are already right.
            unspeakable = _language_not_covered(text, voice or self._voice)
            self._last_error = (
                unspeakable
                or f"synthesize via {self._backend} failed: {e}")
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
            # Order: prefer truly zero-setup options first.
            #   edge_tts    — pip-installable, no local model files,
            #                 ~400 multilingual voices via MS Edge online TTS.
            #                 The recommended free default.
            #   local_piper — needs a separate Piper HTTP server + voice
            #                 .onnx files; better quality offline but
            #                 more setup.
            #   openai_tts  — paid.
            candidates = ["edge_tts", "local_piper", "openai_tts"]
        else:
            candidates = [forced]
        for cand in candidates:
            if cand == "edge_tts" and self._try_edge_tts(cfg):
                return
            if cand == "local_piper" and self._try_local_piper(cfg):
                return
            if cand == "openai_tts" and self._try_openai_tts(cfg):
                return
        self._backend = "disabled"

    def _try_edge_tts(self, cfg: dict) -> bool:
        """Microsoft Edge TTS — free, online, ~400 voices. Probe is
        just an import check: if `edge_tts` is installed we trust
        the network can reach Microsoft's endpoint. Real failures
        surface via last_error during synthesize."""
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            self._last_error = (
                "edge_tts package not installed — `pip install edge-tts` "
                "(free, ~400 multilingual voices, no model files needed)"
            )
            return False
        cfg_e = (cfg.get("edge_tts") or {}) if cfg else {}
        self._backend = "edge_tts"
        # Sensible neural-voice defaults. Override per config or per
        # synthesize() call.
        self._voice = cfg_e.get("voice") or DEFAULT_EDGE_TTS_VOICE
        self._voice_ru = cfg_e.get("voice_ru") or DEFAULT_EDGE_TTS_VOICE_RU
        self._last_error = None
        return True

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
        # Russian voice override — used when the synthesised text
        # contains Cyrillic. Default is irina-medium (free Piper
        # voice, female, decent quality). User must download the
        # corresponding .onnx + .onnx.json into the Piper server's
        # data_dir for it to actually work; until then synth on
        # Russian text falls back to last_error and the channel
        # surfaces "Voice reply failed".
        self._voice_ru = cfg_p.get("voice_ru") or DEFAULT_LOCAL_PIPER_VOICE_RU
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

    def _tx_edge_tts(
        self, text: str, *, voice: Optional[str] = None,
    ) -> Optional[bytes]:
        """Synthesize via Microsoft Edge's online TTS (free, no API
        key). Returns MP3 bytes — the downstream ffmpeg pipeline
        converts to OGG/Opus for Telegram voice bubbles.

        Voice selection: caller-supplied `voice` wins. Otherwise
        we auto-pick — Cyrillic text → `_voice_ru` (e.g.
        `ru-RU-SvetlanaNeural`), everything else → `_voice`
        (e.g. `en-US-AriaNeural`).

        edge_tts uses asyncio internally. We run it via
        asyncio.run on a worker thread so we don't have to care
        about the caller's event-loop state — the TTS module is
        called from both sync paths (REPL, Telegram thread pool)
        and async paths (FastAPI handlers). asyncio.run() in a
        thread is the simplest portable bridge."""
        import asyncio
        import edge_tts  # local import to keep top of module light

        chosen = voice or _pick_voice(
            text,
            default=self._voice or DEFAULT_EDGE_TTS_VOICE,
            ru=self._voice_ru,
        )

        # Speech rate, e.g. "+25%" / "-10%" / "+0%". Read live from
        # `tts_config.json` so a `set_setting('tts.rate', '+25%')`
        # call by the agent takes effect on the very next synth —
        # paired with `SYNTHESIZER.reset()` from the settings router
        # so the singleton's _voice cache also re-resolves.
        cfg_e = (load_config().get("edge_tts") or {})
        rate = cfg_e.get("rate") or "+0%"

        async def _synth() -> bytes:
            buf = bytearray()
            communicate = edge_tts.Communicate(text, chosen, rate=rate)
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    buf.extend(chunk.get("data") or b"")
            return bytes(buf)

        # Always run in a fresh event loop on this thread. Avoids
        # "asyncio.run() cannot be called from a running event loop"
        # when the caller is already inside one.
        import threading
        result: dict = {}

        def _runner() -> None:
            try:
                result["audio"] = asyncio.run(_synth())
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        # 30s budget — edge-tts is usually <2s for a short sentence,
        # but allow for slow networks.
        t.join(timeout=30)
        if t.is_alive():
            self._last_error = "edge_tts timed out after 30s"
            return None
        if "error" in result:
            raise result["error"]
        return result.get("audio") or None

    def _tx_local_piper(
        self, text: str, *, voice: Optional[str] = None,
    ) -> Optional[bytes]:
        """POST text to a no-auth Piper wrapper.
        Endpoint matches the user's local-piper-tts-api server:
        `POST /v1/audio/speech` with JSON `{"input": text, "voice":
        voice}`, response body is `audio/wav` bytes.

        Voice selection: caller-supplied `voice` wins. Otherwise
        we auto-pick — Russian text routes to `_voice_ru` (e.g.
        `ru_RU-irina-medium`), everything else to `_voice` (e.g.
        `en_US-lessac-medium`). Without this, Piper would try to
        speak Russian Cyrillic with English phonemes and the user
        would hear distorted noise."""
        if not self._local_piper_base:
            return None
        chosen = voice or _pick_voice(
            text,
            default=self._voice or DEFAULT_LOCAL_PIPER_VOICE,
            ru=self._voice_ru,
        )
        body = {"input": text, "voice": chosen}
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
