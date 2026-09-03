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
import re
import threading
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

DEFAULT_FASTER_WHISPER_MODEL = "medium"

# A second model consulted when the first one may be out of its depth.
#
# Measured 2026-08-31 on the owner's own Armenian notes. The base model
# heard English and produced "Nice to hide and has gun, miss"; large-v3
# heard Turkish; large-v3-turbo heard German and scored Armenian at 0.002.
# This fine-tune read both notes exactly: "Իս դու հայերեն հասկանում ես".
#
# Chosen over three other Armenian fine-tunes on the owner's own notes,
# 2026-09-01. It reads his name -- "Բարև հրանդ" where the previous pick
# gave "հրվերանդ" -- and turns a reminder request into something legible:
# "ինձ վաղը ժամը տասին հիշացում ... դիզայներին". EmreAkgul's build reads
# comparably but is slower and heard the name as "Ֆրանտ";
# TartarusXXX's raises a shape error on every clip.
#
# The CTranslate2 build matters as much as the accuracy. The previous
# model shipped transformers weights only, which cost a subprocess and
# 20-40s per note; this one loads straight into faster-whisper and
# transcribes in 7.
#
# It is not a replacement. Given Russian speech it transcribes the sounds
# in ARMENIAN LETTERS -- "Ե՛ ադաբրեու, մոշ պրիստուպաց" for "Я одобряю,
# можешь приступать" -- and ignores the language argument entirely. So the
# two are run together and the better result is chosen, rather than one
# being swapped for the other.
SECOND_OPINION_MODEL = "hurricup/whisper-large-v3-turbo-armenian-ct2"

# Which languages the second model is worth paying for. Empty disables it.
SECOND_OPINION_FOR = ("hy",)
# int8 on CPU: ~4x faster than float32 with no meaningful accuracy loss on
# speech, and the box is CPU-only (12 cores, no GPU).
DEFAULT_FASTER_WHISPER_COMPUTE = "int8"

DEFAULT_OPENAI_WHISPER_MODEL = "whisper-1"
DEFAULT_WHISPER_CPP_MODEL = "whisper"
DEFAULT_LOCAL_WHISPER_MODEL = "whisper-medium"


def _config_path() -> Path:
    from .config import CONFIG
    return Path(CONFIG.knowledge["base_dir"]) / "transcriber_config.json"


def expected_languages(cfg: Optional[dict] = None) -> list:
    """Languages this deployment's speakers actually use, from config.

    Whisper's automatic detection is a guess over ~99 languages, and on a
    short clip the guess is frequently wrong in a way that destroys the
    message rather than degrading it. Measured on the owner's own voice
    notes, 2026-08-19: "Я имею в виду машину" came back as "Eu tenho
    vídeo-machina" (Portuguese, p=0.585). Across 48 stored notes, 45 were
    Russian, one Armenian, and two were detected as Latvian and Polish --
    both wrong, and both transcribed correctly once the choice was
    restricted.

    Empty means no restriction, which is the old behaviour: a deployment
    that has not said which languages it hears gets the raw guess.
    """
    cfg = load_config() if cfg is None else (cfg or {})
    raw = cfg.get("languages") or cfg.get("expected_languages") or []
    if isinstance(raw, str):
        raw = [p for p in raw.replace(",", " ").split() if p]
    out, seen = [], set()
    for code in raw:
        code = str(code or "").strip().lower()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def pick_language(probabilities, allowed) -> Optional[str]:
    """Most probable language among the ones this deployment expects.

    Deliberately unconditional: no "trust detection when it is confident"
    escape hatch. The measured counter-example is a note detected as
    Polish at p=0.78 against Russian at p=0.18 -- a four-fold margin, and
    restricting still produced the correct Russian text. Confidence in the
    wrong language is exactly what this is for.

    `probabilities` is faster-whisper's (code, probability) list.
    """
    if not allowed:
        return None
    allowed_set = {str(a).lower() for a in allowed}
    best, best_p = None, -1.0
    for item in probabilities or []:
        try:
            code, p = item[0], float(item[1])
        except (TypeError, IndexError, ValueError):
            continue
        if str(code).lower() in allowed_set and p > best_p:
            best, best_p = str(code).lower(), p
    # Nothing in the distribution matched: fall back to the caller's first
    # expectation rather than to a language nobody here speaks.
    return best or (str(list(allowed)[0]).lower() if allowed else None)


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


def _avg_logprob(segments) -> float:
    """Mean per-segment confidence. -99 for an empty result so anything
    real beats silence. Kept for logging; it is NOT how the two readings
    are compared — see `_prefer_second_opinion`."""
    vals = [getattr(s, "avg_logprob", None) for s in (segments or [])]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else -99.0


_ARMENIAN = re.compile(r"[԰-֏]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

# Neither reading is obviously right, so the choice moves downstream to
# the layer that reads text with a model. Truthy on purpose: a caller that
# ignores it and treats this as "take the specialist" is wrong less often
# than one that silently keeps a Cyrillic mis-hearing.
UNDECIDED = "undecided"


def _prefer_second_opinion(base_text: str, alt_text: str,
                           detected_language: str | None = None):
    """Take the specialist's reading instead of the base model's?

    Decided on SCRIPT, not on confidence. Averaged log-probability was the
    first rule and it is measurably wrong here: the base model's Armenian
    failures are not hesitant, they are fluent English sentences with high
    confidence ("Nice to hide and has gun, miss"), and they outscore a
    correct Armenian transcript every time.

    Script is unambiguous on this pair. The base model never emits Armenian
    letters — across every note measured it produced English, Russian,
    Turkish or German. The specialist emits them always, including when it
    is wrong (it renders Russian phonetically in Armenian script). So
    Armenian out of the specialist AND none out of the base means the audio
    was Armenian and only one of them heard it.

    The specialist alone is not enough to decide, because it renders
    RUSSIAN phonetically in Armenian letters too -- "Ե՛ ադաբրեու, մոշ
    պիստուպաց" for "Я одобряю, можешь приступать". Preferring it on
    Armenian output alone would wreck every Russian note.

    Cyrillic out of the base model is the guard. When the base produces
    Cyrillic it heard Russian and heard it correctly; that is its strong
    case and it keeps it. Only when the base produced LATIN -- which on
    this deployment's Armenian audio has meant English, Turkish or German
    hallucinations, every time measured -- does the specialist's Armenian
    win.

    The assumption worth stating: English-only voice notes are rare here.
    If that changes, a real English note would be overridden, and the
    discriminator has to become something better than script.
    """
    alt = (alt_text or "").strip()
    if not alt or not _ARMENIAN.search(alt):
        return False
    base = base_text or ""
    lang = str(detected_language or "").lower()
    if lang == "hy":
        # Script cannot separate the two any more. Since 2026-08-19
        # detection is restricted to the languages heard here, so Armenian
        # audio reaches the base model with `language="hy"` FORCED -- and
        # forced, generic `medium` does emit Armenian letters. Badly:
        # "Հայդան ատկարկավոտվեց" where the specialist reads "Հայրեն էդ
        # կարգավորվե՞ց։". Being letters is not being a reading, and the
        # rule below read them as agreement, so the specialist lost every
        # Armenian note between 2026-09-01 and 2026-09-03.
        #
        # When the audio is Armenian the question is not which script came
        # out, it is which model knows the language: one is generic
        # medium, the other is large-v3-turbo fine-tuned on Armenian.
        return True
    if _ARMENIAN.search(base):
        return False
    if _CYRILLIC.search(base):
        if lang == "ru":
            # Detection says Russian and the base wrote Cyrillic: it heard
            # the right language. The specialist only transliterates
            # Russian into Armenian letters, so there is nothing to gain.
            return False
        # Cyrillic used to end the matter: "the base heard Russian, so it
        # heard right". It does not. Measured 2026-09-01 -- Armenian audio
        # came back as "Ба референт, ищь качка", Cyrillic and meaningless,
        # and this rule handed it the win over the specialist's reading.
        #
        # Deciding which of two readings is real is a judgement about
        # language, not about character ranges, and there is already a
        # layer that makes it with a model. UNDECIDED sends both on so it
        # can choose.
        return UNDECIDED
    return True


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
        # The other reading, when two models disagreed and neither was
        # obviously right. Consumed by the caller immediately after
        # `transcribe`; not state anyone should hold on to.
        self._last_alternative: str = ""

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
            source = path
            if not language:
                # Detect explicitly instead of letting `transcribe` do it.
                # They are not the same code path and they disagree: on the
                # measured failure the internal one chose Portuguese at 0.585
                # while this one ranked Russian first. Restricting the choice
                # to the configured languages then fixes the rest.
                allowed = expected_languages()
                if allowed:
                    try:
                        from faster_whisper import decode_audio
                        source = decode_audio(path, sampling_rate=16000)
                        _, _, probs = self._fw_model.detect_language(source)
                        language = pick_language(probs, allowed)
                    except Exception as e:
                        # Detection is an improvement, not a dependency: a
                        # failure here must still leave a working transcript.
                        log.debug("language detection failed, using auto: %s", e)
                        source = path
                        language = None
            segments, _info = self._fw_model.transcribe(
                source, language=language, vad_filter=True,
            )
            segments = list(segments)
            text = " ".join(seg.text.strip() for seg in segments).strip()

            # Ask the specialist when the detected language is one the base
            # model is known to mangle, and keep whichever transcript the
            # models themselves are more confident about. Averaged
            # log-probability is the only quality signal available without
            # a human, and it is the right one here: the base model's
            # Armenian failures are not near-misses, they are confident
            # nonsense in the wrong language, and score accordingly.
            # Gate on what this deployment MIGHT hear, not on what
            # detection just said. Keying it on the detected language was
            # circular and shipped that way: the specialist would only be
            # consulted once detection had already recognised Armenian,
            # which is the exact thing that never happens. Measured live —
            # the note still came back "Nice to hide and has gun, miss".
            _covered = [lg for lg in SECOND_OPINION_FOR
                        if lg in (expected_languages() or SECOND_OPINION_FOR)]
            if _covered and SECOND_OPINION_MODEL:
                alt, _ = self._second_opinion(path, _covered[0])
                # The language the base actually worked in. Dropping it
                # was what left the decision to script alone.
                _detected = language or getattr(_info, "language", None)
                verdict = _prefer_second_opinion(
                    text, alt, detected_language=_detected)
                if verdict is UNDECIDED:
                    # Both readings travel on. `render_for_prompt` reads
                    # them with a model and keeps whichever is real speech,
                    # which is a judgement about language rather than one
                    # about character ranges.
                    log.info("two candidate readings, deferring the choice")
                    self._last_alternative = alt
                elif verdict:
                    log.info("second opinion (%s) taken", _covered[0])
                    return alt
                else:
                    self._last_alternative = ""
            return text or None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _second_opinion(self, path: str, language: str):
        """Run the specialist model. Returns (text, unused_score).

        In-process again since 2026-09-01: the chosen model ships a
        CTranslate2 build, so faster-whisper opens it directly. The
        previous pick had transformers weights only and needed a
        subprocess with its own interpreter -- 20-40s a note against 7,
        for worse Armenian.

        Cached like the primary: ~1.6 GB, and a deployment that never
        hears the language it covers should never pay for it. Any
        failure returns no opinion rather than raising -- a second
        reading is an improvement, never a dependency.
        """
        try:
            from faster_whisper import WhisperModel
            with self._lock:
                if getattr(self, "_fw_alt", None) is None:
                    self._fw_alt = WhisperModel(
                        SECOND_OPINION_MODEL, device="cpu",
                        compute_type=getattr(self, "_fw_compute",
                                             DEFAULT_FASTER_WHISPER_COMPUTE),
                    )
            segs, _ = self._fw_alt.transcribe(
                path, language=language, vad_filter=True)
            text = " ".join(x.text.strip() for x in segs).strip()
            # The score is unused: the choice is made on script and, when
            # that is inconclusive, by the layer that reads with a model.
            return (text or None), 0.0
        except Exception as e:
            # WARNING, not debug. This swallow is correct -- a second
            # reading must never break a transcript -- but it hid a
            # configuration error through a whole round of live testing,
            # while the base model's English nonsense won by default. A
            # capability that is installed but never runs has to be loud.
            log.warning("second opinion unavailable (%s): %s",
                        SECOND_OPINION_MODEL, e)
            return None, -99.0

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
