"""PipelineProfile — overlay-diff config snapshot the agent reads at runtime.

A profile carries only the deviations from defaults across four
domains: engine knobs, reasoning routing, system-prompt sections,
per-module logging levels. The active profile's id lives in
`_active.json`; switching is a one-line write + a cache invalidation.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


_ID_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_HISTORY_RETENTION = 10
_CACHE_TTL_SEC = 5.0


def validate_id(pid: str) -> bool:
    return bool(pid) and bool(_ID_RE.match(pid))


@dataclass
class PipelineProfile:
    id: str
    name: str
    description: str
    created_at: float
    updated_at: float
    engine_overrides: dict = field(default_factory=dict)
    reasoning_overrides: dict = field(default_factory=dict)
    prompt_overrides: dict = field(default_factory=dict)
    logging_overrides: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineProfile":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


def _profiles_root() -> Path:
    """Lazy import of `paths` avoids a circular import on module load."""
    try:
        from . import paths
        return paths.data_dir(require=False) / "pipeline_profiles"
    except Exception:
        return Path("/tmp/_hrant_pipeline_profiles_devstub")


def _history_root_for(pid: str) -> Path:
    return _profiles_root() / "_history" / pid


def _active_path() -> Path:
    return _profiles_root() / "_active.json"


class ProfileStore:
    """File-backed store. One JSON file per profile id under the
    profiles root. Atomic writes via .tmp + rename."""

    def __init__(self):
        self._lock = threading.RLock()
        # In-process snapshot of the active profile's overrides — re-read
        # every _CACHE_TTL_SEC so config readers don't hit disk on every
        # call. Invalidated explicitly on put/delete/set_active.
        self._cache_overrides: dict = {}
        self._cache_loaded_at: float = 0.0
        self._cache_active_id: str = ""
        self._cache_valid: bool = False

    def _path(self, pid: str) -> Path:
        if not validate_id(pid):
            raise ValueError(f"invalid profile id: {pid!r}")
        return _profiles_root() / f"{pid}.json"

    def get(self, pid: str) -> Optional[PipelineProfile]:
        if not validate_id(pid):
            return None
        p = self._path(pid)
        if not p.exists():
            return None
        try:
            with self._lock:
                raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("profile load %s failed: %s", pid, e)
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return PipelineProfile.from_dict(raw)
        except Exception as e:
            log.warning("profile bad shape %s: %s", pid, e)
            return None

    def list(self) -> list[PipelineProfile]:
        root = _profiles_root()
        if not root.exists():
            return []
        out: list[PipelineProfile] = []
        for p in root.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("_"):
                continue
            if not p.name.endswith(".json"):
                continue
            pid = p.stem
            if not validate_id(pid):
                continue
            prof = self.get(pid)
            if prof is not None:
                out.append(prof)
        out.sort(key=lambda x: x.updated_at, reverse=True)
        return out

    def put(self, profile: PipelineProfile) -> None:
        if not validate_id(profile.id):
            raise ValueError(f"invalid profile id: {profile.id!r}")
        with self._lock:
            p = self._path(profile.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            # If a previous version exists, snapshot it to history first.
            if p.exists():
                try:
                    prev_raw = p.read_text(encoding="utf-8")
                    hroot = _history_root_for(profile.id)
                    hroot.mkdir(parents=True, exist_ok=True)
                    stamp = int(time.time() * 1000)
                    (hroot / f"{stamp}.json").write_text(prev_raw, encoding="utf-8")
                    self._prune_history(profile.id)
                except Exception as e:
                    log.warning("history snapshot %s failed: %s", profile.id, e)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(
                json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(p)
        self._invalidate_cache()

    def delete(self, pid: str) -> None:
        if not validate_id(pid):
            return
        with self._lock:
            p = self._path(pid)
            if p.exists():
                p.unlink()
            hroot = _history_root_for(pid)
            if hroot.exists():
                for f in hroot.iterdir():
                    try:
                        f.unlink()
                    except OSError:
                        pass
                try:
                    hroot.rmdir()
                except OSError:
                    pass
        self._invalidate_cache()

    def history(self, pid: str) -> list[PipelineProfile]:
        if not validate_id(pid):
            return []
        hroot = _history_root_for(pid)
        if not hroot.exists():
            return []
        out: list[PipelineProfile] = []
        for f in sorted(hroot.iterdir(), reverse=True):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                out.append(PipelineProfile.from_dict(raw))
            except Exception:
                continue
        return out

    def history_with_timestamps(self, pid: str) -> list[tuple[int, "PipelineProfile"]]:
        """Like history() but pairs each entry with its file timestamp.
        The timestamp is the unix millisecond used as the filename — the
        same value the restore endpoint takes."""
        if not validate_id(pid):
            return []
        hroot = _history_root_for(pid)
        if not hroot.exists():
            return []
        out: list[tuple[int, PipelineProfile]] = []
        for f in sorted(hroot.iterdir(), reverse=True):
            try:
                ts = int(f.stem)
                raw = json.loads(f.read_text(encoding="utf-8"))
                out.append((ts, PipelineProfile.from_dict(raw)))
            except Exception:
                continue
        return out

    def restore(self, pid: str, ts: int) -> Optional[PipelineProfile]:
        """Restore a snapshot atomically. Audit Important #11
        (2026-05-23): pre-fix the snapshot was read OUTSIDE the lock,
        so a concurrent `delete(pid)` between the read and the
        subsequent `put(prof)` would silently resurrect a deleted
        profile. Now everything happens under one RLock take —
        delete + restore are mutually exclusive."""
        if not validate_id(pid):
            return None
        with self._lock:
            hroot = _history_root_for(pid)
            f = hroot / f"{ts}.json"
            if not f.exists():
                return None
            raw = json.loads(f.read_text(encoding="utf-8"))
            prof = PipelineProfile.from_dict(raw)
            prof.updated_at = time.time()
            self.put(prof)  # snapshots current first; RLock re-entrant
            return prof

    def _prune_history(self, pid: str) -> None:
        hroot = _history_root_for(pid)
        if not hroot.exists():
            return
        files = sorted(hroot.iterdir(), reverse=True)
        for old in files[_HISTORY_RETENTION:]:
            try:
                old.unlink()
            except OSError:
                pass

    def active_id(self) -> str:
        p = _active_path()
        if not p.exists():
            return "default"
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return str(raw.get("active_id") or "default")
        except Exception:
            return "default"

    def set_active(self, pid: str) -> None:
        if not validate_id(pid):
            raise ValueError(f"invalid profile id: {pid!r}")
        with self._lock:
            root = _profiles_root()
            root.mkdir(parents=True, exist_ok=True)
            tmp = _active_path().with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"active_id": pid}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(_active_path())
        self._invalidate_cache()

    def active_overrides(self) -> dict:
        """Return the active profile's full overrides as a plain dict.
        Cached in-process for `_CACHE_TTL_SEC` seconds. Empty dict if
        the active profile id has no profile (e.g. "default" with no
        on-disk file, or a stale id pointing nowhere)."""
        now = time.time()
        with self._lock:
            if self._cache_valid and now - self._cache_loaded_at < _CACHE_TTL_SEC:
                return dict(self._cache_overrides)
            pid = self.active_id()
            prof = self.get(pid)
            if prof is None:
                self._cache_overrides = {}
            else:
                self._cache_overrides = {
                    "engine_overrides": prof.engine_overrides or {},
                    "reasoning_overrides": prof.reasoning_overrides or {},
                    "prompt_overrides": prof.prompt_overrides or {},
                    "logging_overrides": prof.logging_overrides or {},
                }
            self._cache_active_id = pid
            self._cache_loaded_at = now
            self._cache_valid = True
            return dict(self._cache_overrides)

    def _invalidate_cache(self) -> None:
        with self._lock:
            self._cache_overrides = {}
            self._cache_loaded_at = 0.0
            self._cache_active_id = ""
            self._cache_valid = False


PROFILES = ProfileStore()


def active_overrides() -> dict:
    """Module-level convenience for config readers."""
    return PROFILES.active_overrides()


# ─── Overlay validation ────────────────────────────────────────────


_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def validate(overlay: dict) -> list[str]:
    """Return a list of human-readable error messages. Empty list = valid.

    Reuses existing whitelists:
      - `engine_overrides` -> `runtime_config._ALLOWED` validators
      - `reasoning_overrides.routing` values -> `reasoning_routing.VALID_LEVELS`
      - `prompt_overrides.modules` keys -> `prompt_modules.MODULES` (v2)
      - `logging_overrides.root` / `.modules.*` -> stdlib log level names
    """
    errors: list[str] = []
    overlay = overlay or {}

    # Engine.
    raw_engine = overlay.get("engine_overrides")
    if raw_engine is not None and not isinstance(raw_engine, dict):
        errors.append("engine_overrides: must be a dict")
        engine = {}
    else:
        engine = raw_engine or {}
    if engine:
        try:
            from .runtime_config import _ALLOWED  # type: ignore[attr-defined]
        except Exception:
            _ALLOWED = {}
        for section, fields in engine.items():
            if section not in _ALLOWED:
                errors.append(f"engine_overrides.{section}: unknown section")
                continue
            if not isinstance(fields, dict):
                errors.append(f"engine_overrides.{section}: must be a dict")
                continue
            for key, value in fields.items():
                if key not in _ALLOWED[section]:
                    errors.append(
                        f"engine_overrides.{section}.{key}: unknown field"
                    )
                    continue
                typ, check = _ALLOWED[section][key]
                try:
                    coerced = typ(value)
                except Exception:
                    errors.append(
                        f"engine_overrides.{section}.{key}: not a {typ.__name__}"
                    )
                    continue
                if not check(coerced):
                    errors.append(
                        f"engine_overrides.{section}.{key}: value {coerced!r} out of range"
                    )

    # Reasoning.
    raw_reasoning = overlay.get("reasoning_overrides")
    if raw_reasoning is not None and not isinstance(raw_reasoning, dict):
        errors.append("reasoning_overrides: must be a dict")
        reasoning = {}
    else:
        reasoning = raw_reasoning or {}
    if reasoning:
        try:
            from .reasoning_routing import VALID_LEVELS
        except Exception:
            VALID_LEVELS = ("none", "low", "medium", "high")
        routing = reasoning.get("routing") or {}
        if not isinstance(routing, dict):
            errors.append("reasoning_overrides.routing: must be a dict")
        else:
            for task_type, level in routing.items():
                if level not in VALID_LEVELS:
                    errors.append(
                        f"reasoning_overrides.routing.{task_type}: "
                        f"{level!r} not in {VALID_LEVELS}"
                    )
        fb = reasoning.get("fallback")
        if fb is not None and fb not in VALID_LEVELS:
            errors.append(
                f"reasoning_overrides.fallback: {fb!r} not in {VALID_LEVELS}"
            )

    # Prompt.
    raw_prompt = overlay.get("prompt_overrides")
    if raw_prompt is not None and not isinstance(raw_prompt, dict):
        errors.append("prompt_overrides: must be a dict")
        prompt = {}
    else:
        prompt = raw_prompt or {}
    if prompt:
        try:
            from .prompt_modules import MODULES
        except Exception:
            MODULES = {}
        modules = prompt.get("modules") or {}
        if not isinstance(modules, dict):
            errors.append("prompt_overrides.modules: must be a dict")
        else:
            for name, body in modules.items():
                if name not in MODULES:
                    errors.append(
                        f"prompt_overrides.modules.{name}: unknown module"
                    )
                    continue
                if body is not None and not isinstance(body, str):
                    errors.append(
                        f"prompt_overrides.modules.{name}: must be string or null"
                    )

    # Logging.
    raw_logging = overlay.get("logging_overrides")
    if raw_logging is not None and not isinstance(raw_logging, dict):
        errors.append("logging_overrides: must be a dict")
        logging_overrides = {}
    else:
        logging_overrides = raw_logging or {}
    if logging_overrides:
        root = logging_overrides.get("root")
        if root is not None and root not in _VALID_LOG_LEVELS:
            errors.append(
                f"logging_overrides.root: {root!r} not in {_VALID_LOG_LEVELS}"
            )
        modules = logging_overrides.get("modules") or {}
        if not isinstance(modules, dict):
            errors.append("logging_overrides.modules: must be a dict")
        else:
            for mod, level in modules.items():
                if level not in _VALID_LOG_LEVELS:
                    errors.append(
                        f"logging_overrides.modules.{mod}: "
                        f"{level!r} not in {_VALID_LOG_LEVELS}"
                    )

    return errors


# ─── First-boot seeding ────────────────────────────────────────────


def _starter_definitions() -> list[dict]:
    """Five illustrative starter profiles seeded on first boot. The
    names + descriptions are examples; owner can rename / edit /
    delete freely. Only `default` is special (empty overlay, used
    as the fallback when active points nowhere)."""
    now = time.time()
    def _shell(pid, name, desc, **overrides):
        return {
            "id": pid, "name": name, "description": desc,
            "created_at": now, "updated_at": now,
            "engine_overrides": overrides.get("engine_overrides", {}),
            "reasoning_overrides": overrides.get("reasoning_overrides", {}),
            "prompt_overrides": overrides.get("prompt_overrides", {}),
            "logging_overrides": overrides.get("logging_overrides", {}),
        }
    return [
        _shell(
            "default", "Default", "Empty overlay — uses code defaults.",
        ),
        _shell(
            "benchmark", "Benchmark Mode",
            "Tighter token discipline, medium reasoning, debug logs on supervisor.",
            engine_overrides={"router": {"tool_loop_input_budget": 80000}},
            reasoning_overrides={
                "routing": {"complex_solving": "medium"},
            },
            logging_overrides={
                "modules": {"backend.job_supervisor": "DEBUG"},
            },
        ),
        _shell(
            "development", "Development Mode",
            "Verbose logging, high reasoning, no token caps.",
            engine_overrides={"router": {"tool_loop_input_budget": 0}},
            reasoning_overrides={
                "routing": {"chat": "medium", "classification": "medium"},
            },
            logging_overrides={
                "root": "DEBUG",
                "modules": {"backend.unified_agent": "DEBUG"},
            },
        ),
        _shell(
            "safe", "Safe Mode",
            "Higher confidence bar, low reasoning on cheap tasks.",
            engine_overrides={"verification": {"min_confidence": 90}},
            reasoning_overrides={
                "routing": {"chat": "low", "quick_answer": "low"},
            },
        ),
        _shell(
            "solver", "Autonomous Solver Mode",
            "High reasoning across the board, no budget marker.",
            reasoning_overrides={
                "routing": {
                    "complex_solving": "high",
                    "supervisor": "high",
                    "self_critic": "high",
                },
                "fallback": "high",
            },
        ),
    ]


def seed_starter_profiles() -> None:
    """Idempotent: only writes profiles that don't already exist on
    disk. Owner edits survive subsequent boots."""
    for spec in _starter_definitions():
        if PROFILES.get(spec["id"]) is None:
            try:
                PROFILES.put(PipelineProfile.from_dict(spec))
            except Exception as e:
                log.warning("seed %s failed: %s", spec["id"], e)
    # Only set active=default if no _active.json exists.
    p = _active_path()
    if not p.exists():
        try:
            PROFILES.set_active("default")
        except Exception as e:
            log.warning("seed set-active failed: %s", e)
