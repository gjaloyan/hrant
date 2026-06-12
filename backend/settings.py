"""Single source of truth for user-mutable settings.

Pre-Phase-2 the agent had to:
  1. Know that `tts_config.json` exists (now in STATE SNAPSHOT)
  2. `read_file` the current contents
  3. Hand-edit JSON to change a voice
  4. `terminal_exec` `tee` or `run_python` to write the file back
  5. Hope the relevant singleton notices (it doesn't, until restart)

That's 4-6 LLM calls + 4-6 tool calls for a one-word user request.
And every config has its own quirks (TTS uses `backend.voice` /
`edge_tts.voice` / `edge_tts.voice_ru`; transcriber uses
`backend` + per-backend block; response language lives in
`user_profile.md` under `## Language`, not in JSON).

`SettingsRouter` is the abstraction:
  - `SETTINGS.list_settings()` enumerates every mutable setting,
    with current value + description + valid choices/range.
  - `SETTINGS.get(key)` returns the current value.
  - `SETTINGS.set(key, value)` validates + writes + resets the
    relevant subsystem (so changes apply immediately, not after
    a restart).
  - The `set_setting(key, value)` tool wraps this. The agent
    can apply user requests in ONE tool call instead of 4-6.

Keys use dotted naming (`tts.voice`, `tts.voice_ru`, `tts.backend`,
`response.language`). The registry is hard-coded — adding a new
setting is a code change (intentional: each setting needs a
validator + a reset hook, not just a JSON write).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SettingSpec:
    """Declarative definition of one mutable setting.

    `get_fn` reads current value (returns None if unset).
    `set_fn` writes a new value and is responsible for:
      - validating against `choices` / regex (called BEFORE set_fn)
      - persisting to the right file
      - calling any subsystem.reset() so the change applies live
    `description` shows up in the `set_setting` tool spec AND in
    the agent's STATE SNAPSHOT so the LLM knows what each key does.
    """

    key: str
    description: str
    get_fn: Callable[[], Any]
    set_fn: Callable[[Any], None]
    # Optional value constraints — validator runs BEFORE set_fn.
    choices: tuple[str, ...] = ()              # exact match (case-insensitive for strings)
    regex: str = ""                            # full-match regex
    value_type: str = "string"                 # "string" | "bool" | "int"


def _normalise_value(spec: SettingSpec, value: Any) -> Any:
    """Coerce raw input to the spec's expected type. Empty / None
    returns None (means "unset" for the setter)."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if spec.value_type == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on", "enable", "enabled"):
            return True
        if s in ("0", "false", "no", "off", "disable", "disabled"):
            return False
        raise ValueError(f"expected boolean, got {value!r}")
    if spec.value_type == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(f"expected integer, got {value!r}")
    return str(value).strip()


def _validate(spec: SettingSpec, value: Any) -> None:
    """Raise ValueError if `value` doesn't match the spec."""
    if value is None:
        return  # allowed — means unset
    if spec.choices:
        s = str(value).lower()
        choices_lower = [c.lower() for c in spec.choices]
        if s not in choices_lower:
            raise ValueError(
                f"value must be one of {list(spec.choices)}, got {value!r}"
            )
    if spec.regex:
        if not re.fullmatch(spec.regex, str(value)):
            raise ValueError(
                f"value {value!r} doesn't match the expected pattern"
            )


# --- TTS settings -----------------------------------------------------


def _tts_load() -> dict:
    from .tts import load_config
    try:
        return load_config() or {}
    except Exception:
        return {}


def _tts_save(cfg: dict) -> None:
    from .tts import save_config, SYNTHESIZER
    save_config(cfg)
    # Reset the singleton so the new voice/backend takes effect on
    # the NEXT synthesize() call instead of waiting for a process
    # restart. The previous behaviour was "config written, but
    # active voice didn't change until restart" — caught in the
    # production audit.
    try:
        SYNTHESIZER.reset()
    except Exception as e:
        log.debug("TTS singleton reset failed: %s", e)


def _tts_get_backend() -> str:
    return (_tts_load().get("backend") or "edge_tts")


def _tts_set_backend(value: str) -> None:
    cfg = _tts_load()
    cfg["backend"] = value
    _tts_save(cfg)


def _tts_voice_for(backend: str, lang: str = "en") -> Optional[str]:
    cfg = _tts_load()
    backend_cfg = cfg.get(backend) or {}
    if isinstance(backend_cfg, dict):
        key = "voice_ru" if lang == "ru" else "voice"
        return backend_cfg.get(key)
    return None


def _tts_set_voice(value: Optional[str], *, lang: str = "en") -> None:
    cfg = _tts_load()
    backend = cfg.get("backend") or "edge_tts"
    backend_cfg = cfg.get(backend) or {}
    if not isinstance(backend_cfg, dict):
        backend_cfg = {}
    key = "voice_ru" if lang == "ru" else "voice"
    if value is None:
        backend_cfg.pop(key, None)
    else:
        backend_cfg[key] = value
    cfg[backend] = backend_cfg
    cfg.setdefault("backend", backend)
    _tts_save(cfg)


# Convenience: gender-shorthand. The user almost always says
# "male voice" / "female voice" — we resolve to specific Edge voice
# IDs so the agent doesn't have to remember them.
_GENDER_VOICE_MAP = {
    ("edge_tts", "en", "male"):   "en-US-GuyNeural",
    ("edge_tts", "en", "female"): "en-US-AriaNeural",
    ("edge_tts", "ru", "male"):   "ru-RU-DmitryNeural",
    ("edge_tts", "ru", "female"): "ru-RU-SvetlanaNeural",
}


def _tts_set_voice_gender(value: Optional[str]) -> None:
    """Convenience setter: resolves 'male' / 'female' to concrete
    Edge voices for both EN and RU, and writes both."""
    if value is None or value == "auto":
        # Reset to defaults — remove both pinned voices so the
        # synthesiser falls back to its hardcoded defaults.
        cfg = _tts_load()
        backend = cfg.get("backend") or "edge_tts"
        backend_cfg = cfg.get(backend) or {}
        if isinstance(backend_cfg, dict):
            backend_cfg.pop("voice", None)
            backend_cfg.pop("voice_ru", None)
            cfg[backend] = backend_cfg
        _tts_save(cfg)
        return
    g = str(value).lower()
    if g not in ("male", "female"):
        raise ValueError(f"voice_gender must be 'male', 'female', or 'auto', got {value!r}")
    backend = _tts_get_backend()
    en_voice = _GENDER_VOICE_MAP.get((backend, "en", g))
    ru_voice = _GENDER_VOICE_MAP.get((backend, "ru", g))
    if not en_voice and not ru_voice:
        raise ValueError(
            f"backend {backend!r} doesn't have a registered "
            f"{g} voice — set tts.voice / tts.voice_ru directly"
        )
    if en_voice:
        _tts_set_voice(en_voice, lang="en")
    if ru_voice:
        _tts_set_voice(ru_voice, lang="ru")


def _tts_voice_gender_current() -> Optional[str]:
    """Best-effort: derive current gender from the pinned voice
    names. Returns 'male' / 'female' / 'auto' / None."""
    from .self_state import _voice_gender_hint
    voice = _tts_voice_for(_tts_get_backend(), lang="en") or ""
    g = _voice_gender_hint(voice)
    return g or "auto"


# --- TTS speed (rate) -------------------------------------------------


# Edge TTS accepts a string like "+25%" / "-10%" / "+100%". Range
# clamped at ±100% — beyond that audio becomes unintelligible.
# Regex is permissive on the user-facing edges (sign + percent
# optional) because the setter normalises bare inputs like "25" or
# "+25" into "+25%". Range still enforced: only 0-100 magnitudes.
#
# Two syntaxes accepted by the setter:
#   1. Absolute     +25% / -10% / 0% / 25 → set rate to that value.
#   2. Delta        += 25% / -= 10%       → add/subtract from CURRENT
#                                            rate, clamped to ±100%.
# The audit caught the ambiguity in "increase voice speed by 25%":
# absolute syntax set it to +25%, but the user meant +25 ON TOP of
# the current value. The delta form makes the user's intent explicit.
_TTS_RATE_RE = r"(?:[+]=|[-]=)?\s?[+-]?(?:100|[0-9]?[0-9])%?"


def _tts_get_rate() -> str:
    cfg = _tts_load()
    backend = cfg.get("backend") or "edge_tts"
    return (cfg.get(backend) or {}).get("rate") or "+0%"


def _tts_set_rate(value: Optional[str]) -> None:
    """Set speech rate. Two input syntaxes:

      Absolute  '+25%' / '-10%' / '25' / '0'  → set rate to that.
      Delta     '+=25%' / '-=10' / '+= 25%'   → add/subtract from
                                                  the CURRENT rate.

    Delta form was added after the post-Phase-B+C audit caught the
    ambiguity in 'increase voice speed by 25%'. With absolute syntax,
    that phrase set the rate to +25% (forgetting any prior value).
    With delta the LLM can express 'increase by 25%' as '+=25%' and
    get the actually-correct result.

    Range is clamped to ±100% on writes; values past that point are
    unintelligible audio.

    Empty/None clears the pin (back to backend default = +0%).
    """
    cfg = _tts_load()
    backend = cfg.get("backend") or "edge_tts"
    backend_cfg = cfg.get(backend) or {}
    if not isinstance(backend_cfg, dict):
        backend_cfg = {}
    if value is None:
        backend_cfg.pop("rate", None)
    else:
        v = str(value).strip().replace(" ", "")
        if v.startswith(("+=", "-=")):
            sign = 1 if v[0] == "+" else -1
            rest = v[2:].rstrip("%")
            try:
                delta = int(rest)
            except ValueError as e:
                raise ValueError(f"invalid tts.rate delta {value!r}") from e
            current_str = ((cfg.get(backend) or {}).get("rate") or "+0%").rstrip("%")
            try:
                current = int(current_str)
            except ValueError:
                current = 0
            new_val = max(-100, min(100, current + sign * delta))
            v = ("+" if new_val >= 0 else "") + f"{new_val}%"
        else:
            # Absolute syntax: accept "25%" / "+25" / "25" / "+25%" all as +25%.
            if not v.endswith("%"):
                v = v + "%"
            if not v.startswith(("+", "-")):
                v = "+" + v
        backend_cfg["rate"] = v
    cfg[backend] = backend_cfg
    cfg.setdefault("backend", backend)
    _tts_save(cfg)


# --- Response language ------------------------------------------------


def _tz_path():
    from . import paths
    return paths.knowledge_dir() / "user_timezone.json"


def _user_timezone_get() -> str:
    """IANA timezone name; 'UTC' when unset."""
    try:
        import json as _json
        p = _tz_path()
        if p.exists():
            tz = (_json.loads(p.read_text(encoding="utf-8")) or {}).get("tz")
            if tz:
                return str(tz)
    except Exception:
        pass
    return "UTC"


def _user_timezone_set(value: Optional[str]) -> None:
    """Validate against the IANA database before persisting — a typo
    here would silently shift every reminder the user sets."""
    tz = (value or "").strip() or "UTC"
    from zoneinfo import ZoneInfo
    try:
        ZoneInfo(tz)
    except Exception:
        raise ValueError(
            f"unknown timezone {tz!r} — use an IANA name like "
            f"'Asia/Yerevan' or 'Europe/Berlin'"
        )
    from .paths import write_atomic_json
    write_atomic_json(_tz_path(), {"tz": tz})


def user_timezone() -> str:
    """Public accessor for prompt assembly and tools."""
    return _user_timezone_get()


def _response_language_get() -> Optional[str]:
    """Pinned response language from the WebUI default user_profile.
    Returns the raw section body (one line typically) or None."""
    try:
        from .identity import IDENTITY
        profile = IDENTITY.user_profile()
        body = IDENTITY._extract_language_section(profile)
        return (body.strip().splitlines() or [None])[0]
    except Exception:
        return None


def _response_language_set(value: Optional[str]) -> None:
    """Rewrites the `## Language` section of the default
    user_profile.md with the new value. None / empty clears the pin."""
    from .identity import IDENTITY
    path = IDENTITY.user_path
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    # Find or create the section. We keep it simple: scan for the
    # section header, replace its body (until next `## ` heading or
    # EOF) with the new value.
    # Accept legacy Russian headers when locating an existing section,
    # but write new sections under the English header.
    section_headers = ("## Language", "## Язык общения", "## Язык")
    lines = text.splitlines()
    start = -1
    for i, ln in enumerate(lines):
        if ln.strip() in section_headers:
            start = i
            break
    end = len(lines)
    if start >= 0:
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("## "):
                end = j
                break
    # Preserve the existing header spelling on rewrite; default to the
    # English header when creating the section fresh.
    header = lines[start].strip() if start >= 0 else "## Language"
    new_body = []
    if value:
        new_body = [f"- {value}"]
    new_section = [header, ""] + new_body + [""]
    if start >= 0:
        out_lines = lines[:start] + new_section + lines[end:]
    else:
        # No language section — append at the end.
        out_lines = lines + [""] + new_section
    path.write_text("\n".join(out_lines), encoding="utf-8")


# --- Registry ---------------------------------------------------------


def _build_registry() -> dict[str, SettingSpec]:
    return {
        "tts.backend": SettingSpec(
            key="tts.backend",
            description=(
                "Which TTS backend to use. Default `edge_tts` uses Microsoft's "
                "free online voices (good quality, ~400 voices, multilingual)."
            ),
            get_fn=_tts_get_backend,
            set_fn=_tts_set_backend,
            choices=("edge_tts", "local_piper", "openai_tts"),
        ),
        "tts.voice": SettingSpec(
            key="tts.voice",
            description=(
                "English TTS voice ID. For edge_tts use names like "
                "`en-US-GuyNeural` (male) or `en-US-AriaNeural` (female)."
            ),
            get_fn=lambda: _tts_voice_for(_tts_get_backend(), "en"),
            set_fn=lambda v: _tts_set_voice(v, lang="en"),
        ),
        "tts.voice_ru": SettingSpec(
            key="tts.voice_ru",
            description=(
                "Russian TTS voice ID. For edge_tts use names like "
                "`ru-RU-DmitryNeural` (male) or `ru-RU-SvetlanaNeural` (female)."
            ),
            get_fn=lambda: _tts_voice_for(_tts_get_backend(), "ru"),
            set_fn=lambda v: _tts_set_voice(v, lang="ru"),
        ),
        "tts.voice_gender": SettingSpec(
            key="tts.voice_gender",
            description=(
                "Shorthand: pick 'male' or 'female' and the router resolves "
                "to concrete voice IDs for EN + RU at the current backend. "
                "'auto' clears the pinned voices."
            ),
            get_fn=_tts_voice_gender_current,
            set_fn=_tts_set_voice_gender,
            choices=("male", "female", "auto"),
        ),
        "tts.rate": SettingSpec(
            key="tts.rate",
            description=(
                "Speech rate as Edge-TTS percentage string. Accepts "
                "'+25%' (25% faster), '-10%' (slower), '+100%' (max), "
                "or '+0%' to clear. Bare numbers like '25' are auto-"
                "normalised to '+25%'. The synthesiser singleton is "
                "reset on write so the new rate applies to the very "
                "next audio reply — no service restart needed."
            ),
            get_fn=_tts_get_rate,
            set_fn=_tts_set_rate,
            regex=_TTS_RATE_RE,
        ),
        "response.language": SettingSpec(
            key="response.language",
            description=(
                "Pinned response language (free-form sentence, e.g. "
                "'Respond to user in Russian.'). Empty clears the pin "
                "and the agent mirrors the user's input language."
            ),
            get_fn=_response_language_get,
            set_fn=_response_language_set,
        ),
        "user.timezone": SettingSpec(
            key="user.timezone",
            description=(
                "IANA timezone the user lives in (e.g. 'Asia/Yerevan'). "
                "The agent shows times and parses 'tomorrow' / 'on "
                "Monday' / reminder times in THIS zone, converting to "
                "UTC only where tool arguments require it. Default UTC."
            ),
            get_fn=_user_timezone_get,
            set_fn=_user_timezone_set,
        ),
    }


@dataclass
class SettingsRouter:
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _registry: dict[str, SettingSpec] = field(default_factory=dict)

    def __post_init__(self):
        if not self._registry:
            self._registry = _build_registry()

    def list_settings(self) -> list[dict[str, Any]]:
        """Snapshot of every registered setting with its current
        value. Used by the WebUI Settings panel + the `set_setting`
        tool's description list."""
        out = []
        with self._lock:
            for spec in self._registry.values():
                try:
                    current = spec.get_fn()
                except Exception as e:
                    log.debug("settings get(%s) failed: %s", spec.key, e)
                    current = None
                out.append({
                    "key": spec.key,
                    "current": current,
                    "description": spec.description,
                    "choices": list(spec.choices),
                    "value_type": spec.value_type,
                })
        return out

    def get(self, key: str) -> Any:
        spec = self._registry.get(key)
        if spec is None:
            raise KeyError(f"unknown setting {key!r}")
        try:
            return spec.get_fn()
        except Exception:
            return None

    def set(self, key: str, value: Any) -> dict[str, Any]:
        """Apply a config change. Returns `{ok, key, old, new, note}`.
        Raises KeyError for unknown keys and ValueError for invalid
        values (the caller — e.g. the `set_setting` tool — turns
        these into structured tool-result errors)."""
        with self._lock:
            spec = self._registry.get(key)
            if spec is None:
                raise KeyError(f"unknown setting {key!r}")
            normalised = _normalise_value(spec, value)
            _validate(spec, normalised)
            try:
                old = spec.get_fn()
            except Exception:
                old = None
            spec.set_fn(normalised)
            try:
                new = spec.get_fn()
            except Exception:
                new = normalised
            note = ""
            if old == new:
                note = "value already at requested state"
            return {
                "ok": True,
                "key": key,
                "old": old,
                "new": new,
                "note": note,
            }


SETTINGS = SettingsRouter()
