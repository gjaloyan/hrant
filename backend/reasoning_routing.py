"""Per-task reasoning-level routing for OpenAI Codex / Responses API.

User insight (May 2026): "GPT-5.5 on `medium` reasoning may
underperform on deep planning". OpenAI Codex models (GPT-5.x via
ChatGPT-subscription bridge) accept a `reasoning.effort` parameter
that controls how much chain-of-thought the model spends per call.
Levels: `low` / `medium` / `high`.

Trade-off:
  - `low`     — fastest, cheapest, good for chat / classification
  - `medium`  — default, balanced (the Codex CLI default)
  - `high`    — deepest thinking, slower + more tokens, ideal for
                supervisor turns + complex multi-step planning

Hybrid routing: map each `task_type` to its preferred level. Tasks
that need careful reasoning (supervisor decisions, complex_solving,
self-critique) get `high`; chat fast-path / classification stay
`low`; everything else defaults to `medium`.

The routing lives in `<data_dir>/knowledge/reasoning_routing.json`
so the user can edit it via the WebUI Settings → Providers tab
without restart. A live update is picked up by the next LLM call.

This module is intentionally provider-agnostic: any provider that
supports a reasoning-level parameter (Codex, future OpenAI
Responses, Anthropic extended-thinking, etc.) reads from here.
Providers that don't support reasoning silently ignore the lookup.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import paths


log = logging.getLogger(__name__)


# Valid Codex / Responses-API levels. `none` is the explicit
# "don't send reasoning" signal — useful when the user wants to
# force a model to skip the chain-of-thought entirely.
VALID_LEVELS = ("none", "low", "medium", "high")

# Default routing. Conservative — most tasks get `medium`, supervisor
# / deep-planning tasks bump to `high`, chat fast-path to `low` to
# save tokens.
DEFAULT_ROUTING: dict[str, str] = {
    # Light tasks — fast + cheap.
    "chat": "low",
    "quick_answer": "low",
    "classification": "low",
    "keyword_extraction": "low",
    # Default — balanced.
    "task": "medium",
    "task_analysis": "medium",
    "note_creation": "medium",
    "verification": "medium",
    "simple_lookup": "medium",
    "note_search": "medium",
    "learning": "medium",
    # Heavy reasoning — multi-step planning + judgment.
    "complex_solving": "high",
    "supervisor": "high",
    "self_critic": "high",
    "skill_reflection": "medium",
}

# When a task_type isn't in the routing table, fall back to this.
DEFAULT_FALLBACK = "medium"


@dataclass
class RoutingConfig:
    """Snapshot of the live reasoning-routing config."""
    routing: dict[str, str] = field(default_factory=dict)
    fallback: str = DEFAULT_FALLBACK
    # Per-turn override — when set (non-empty), all task_types in
    # this turn use the override level. Used by the WebUI's quick
    # reasoning selector at the top of the chat input. Lives only
    # in-memory (ContextVar pattern is overkill for one global).
    # Owner clears it via empty PUT.
    override: str = ""
    # Last edited at — surfaced in the UI so the operator knows
    # when they last touched the routing.
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RoutingConfig":
        return cls(
            routing=dict(d.get("routing") or {}),
            fallback=(d.get("fallback") or DEFAULT_FALLBACK).strip().lower(),
            override=(d.get("override") or "").strip().lower(),
            updated_at=float(d.get("updated_at") or 0.0),
        )


_LOCK = threading.RLock()
_CACHE: Optional[RoutingConfig] = None
_CACHE_LOADED_AT: float = 0.0
_CACHE_TTL_SEC = 5.0


def _profile_reasoning_overrides() -> dict:
    """Active profile's reasoning_overrides — empty dict if profile missing
    or store import fails (tests outside the FastAPI boot path)."""
    try:
        from .pipeline_profile import active_overrides
        return (active_overrides().get("reasoning_overrides") or {})
    except Exception:
        return {}


def _config_path() -> Path:
    try:
        return paths.knowledge_dir() / "reasoning_routing.json"
    except Exception:
        # Pre-init fallback — tests that don't have data_dir wired
        # use the default routing.
        return Path("/tmp/_hrant_reasoning_routing_devstub.json")


def _load_from_disk() -> RoutingConfig:
    p = _config_path()
    if not p.exists():
        return RoutingConfig(
            routing=dict(DEFAULT_ROUTING),
            fallback=DEFAULT_FALLBACK,
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("reasoning_routing: load %s failed: %s", p, e)
        return RoutingConfig(
            routing=dict(DEFAULT_ROUTING),
            fallback=DEFAULT_FALLBACK,
        )
    if not isinstance(raw, dict):
        return RoutingConfig(
            routing=dict(DEFAULT_ROUTING),
            fallback=DEFAULT_FALLBACK,
        )
    cfg = RoutingConfig.from_dict(raw)
    # Merge defaults — if the on-disk file omits a task_type we
    # know about, fill in the default. Owner edits override.
    merged: dict[str, str] = dict(DEFAULT_ROUTING)
    merged.update({
        k: v for k, v in (cfg.routing or {}).items()
        if isinstance(k, str) and v in VALID_LEVELS
    })
    cfg.routing = merged
    if cfg.fallback not in VALID_LEVELS:
        cfg.fallback = DEFAULT_FALLBACK
    if cfg.override and cfg.override not in VALID_LEVELS:
        cfg.override = ""
    return cfg


def get_config() -> RoutingConfig:
    """Cached snapshot of the live config. TTL 5s — covers a turn's
    worth of calls without re-reading the JSON on every LLM call,
    but picks up owner edits within 5s of saving.

    Phase 1 (Task 5): the active pipeline profile's reasoning_overrides
    form a second layer on top of the on-disk routing config. The
    profile's own cache (5s TTL, invalidated on switch) keeps this
    cheap. The overlay is applied per call so a profile switch shows
    up immediately without invalidating this module's disk cache.
    """
    global _CACHE, _CACHE_LOADED_AT
    with _LOCK:
        now = time.time()
        if _CACHE is None or (now - _CACHE_LOADED_AT) > _CACHE_TTL_SEC:
            _CACHE = _load_from_disk()
            _CACHE_LOADED_AT = now
        base = _CACHE
    overlay = _profile_reasoning_overrides()
    if not overlay:
        return base
    merged_routing = dict(base.routing)
    overlay_routing = overlay.get("routing") or {}
    if isinstance(overlay_routing, dict):
        for task_type, level in overlay_routing.items():
            if level in VALID_LEVELS:
                merged_routing[task_type] = level
    fallback = base.fallback
    overlay_fallback = overlay.get("fallback")
    if overlay_fallback in VALID_LEVELS:
        fallback = overlay_fallback
    return RoutingConfig(
        routing=merged_routing,
        fallback=fallback,
        override=base.override,
        updated_at=base.updated_at,
    )


def save_config(cfg: RoutingConfig) -> None:
    """Persist + invalidate the cache. Atomic via .tmp + rename."""
    global _CACHE, _CACHE_LOADED_AT
    with _LOCK:
        cfg.updated_at = time.time()
        # Drop unknown levels before persisting so the disk file
        # never carries garbage that future loads have to sanitize.
        cleaned = {
            k: v for k, v in (cfg.routing or {}).items()
            if isinstance(k, str) and v in VALID_LEVELS
        }
        cfg.routing = cleaned
        if cfg.fallback not in VALID_LEVELS:
            cfg.fallback = DEFAULT_FALLBACK
        if cfg.override and cfg.override not in VALID_LEVELS:
            cfg.override = ""
        try:
            from .paths import write_secret_json
            write_secret_json(_config_path(), cfg.to_dict())
        except Exception as e:
            log.warning("reasoning_routing: save failed: %s", e)
            return
        _CACHE = cfg
        _CACHE_LOADED_AT = time.time()


def level_for(task_type: str) -> str:
    """Resolve the reasoning level for a task_type.

    Resolution order:
      1. If a global override is set, use it (per-turn UI knob).
      2. If the task_type matches an entry in the routing, use it.
      3. Strip a `:tool_iter_N` / `:tool_synth` suffix and try again
         (these are appended by the tool loop and shouldn't fork
         routing decisions).
      4. Fall back to the config's `fallback` (default medium).
    """
    cfg = get_config()
    if cfg.override:
        return cfg.override
    tt = (task_type or "").strip()
    if tt in cfg.routing:
        return cfg.routing[tt]
    if ":" in tt:
        base = tt.split(":", 1)[0]
        if base in cfg.routing:
            return cfg.routing[base]
    return cfg.fallback


def set_override(level: str) -> None:
    """Set the global per-turn override level. Empty string clears it.

    Reads via `_load_from_disk()` (NOT `get_config()`) so the active
    profile's reasoning_overrides don't accidentally bleed into the
    persisted on-disk routing file when the override is saved."""
    norm = (level or "").strip().lower()
    if norm and norm not in VALID_LEVELS:
        raise ValueError(
            f"reasoning level must be one of {VALID_LEVELS}, got {level!r}"
        )
    cfg = _load_from_disk()
    cfg.override = norm
    save_config(cfg)
