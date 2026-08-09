"""Runtime overrides for a whitelisted subset of CONFIG.

The Settings UI exposes a few knobs (router budget, verification
strictness, …) that the user wants to change without rebooting the
process or editing config.yaml. This module is the persistent store
for those changes.

Storage:
    knowledge/runtime_overrides.json — flat JSON, hand-readable.

Boot flow (called from main.py lifespan startup):
    1. config.yaml + MODE_PRESETS produce CONFIG._data (already done
       by `backend.config.Config.__init__`).
    2. apply_overrides_from_file() merges runtime_overrides.json
       on top, IN-PLACE so the existing references to CONFIG.router
       / CONFIG.verification dicts pick up the changes.

Update flow (called from /api/engine/config PUT):
    1. validate the incoming partial dict against ALLOWED.
    2. merge into the saved overrides file.
    3. apply_overrides() patches CONFIG._data in place.

The whitelist exists because we don't want a UI bug to let the user
flip arbitrary internal flags (e.g. `mode`, `model_a.api_key_env`,
or anything in `mcp_servers`). Adding a new knob is a deliberate
two-line change here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Whitelist: section → field → (type, validator).
# Validators are simple: lambda v: bool — return True if the value
# passes. Keep them inline; no DSL.
_ALLOWED: dict[str, dict[str, tuple[type, Any]]] = {
    "router": {
        "daily_api_budget_usd": (float, lambda v: 0.0 <= v <= 1000.0),
        "estimated_cost_per_call_usd": (float, lambda v: 0.0 <= v <= 10.0),
        "api_ping_cache_seconds": (int, lambda v: 0 <= v <= 3600),
        "fallback_to_local": (bool, lambda v: True),
        "tool_synth_max_tokens": (int, lambda v: 256 <= v <= 16000),
        # 0 = disabled (no hard cap, the agent runs to natural
        # max_iterations limit). Default flipped to 0 on 2026-05-21
        # per the "no limits" directive. Operators can still set
        # 10000-2000000 to opt back into a hard cap.
        "tool_loop_input_budget": (int, lambda v: v == 0 or 10000 <= v <= 2000000),
    },
    # 2026-08-09 dead-code audit: FIVE of the six keys here had zero readers
    # anywhere in the backend, while all six were live Settings sliders — the
    # owner could tune them, watch them validate and persist, and change
    # nothing. `critic_threshold` is now genuinely read by answer_critic.
    # `min_confidence` and `require_sources` are kept because `enabled` and
    # they form one coherent block a reader would expect together, but they
    # are marked here so nobody mistakes them for wired.
    "verification": {
        "enabled": (bool, lambda v: True),
        # TODO(unwired): no backend reader. Either gate the verifier on it or
        # drop it from the UI — do not leave it looking adjustable.
        "min_confidence": (int, lambda v: 0 <= v <= 100),
        # TODO(unwired): no backend reader.
        "require_sources": (bool, lambda v: True),
        # Wired 2026-08-09 -> answer_critic.content_confidence_bar().
        "critic_threshold": (int, lambda v: 0 <= v <= 100),
        # `critic_max_retries` and `critic_retry_token_budget` were REMOVED.
        # They configure the legacy pipeline/critic.py retry loop; the unified
        # critic makes one revision pass, so there is nothing for them to
        # tune. Offering them was worse than not: the names imply a retry
        # budget that cannot exist.
    },
    # Storage retention — 0 means "never auto-sweep this subtree".
    # The sweep itself runs in the autonomic loop; values are read
    # live so a slider change applies on the next tick.
    "workspace": {
        "inbox_retention_days": (int, lambda v: 0 <= v <= 3650),
        "outbox_retention_days": (int, lambda v: 0 <= v <= 3650),
        "notes_retention_days": (int, lambda v: 0 <= v <= 3650),
        "turns_retention_days": (int, lambda v: 0 <= v <= 3650),
    },
    # Knowledge capacity / promotion knobs. note_max_tokens caps the
    # size of a single saved note; core_memory_max_tokens caps the
    # ambient context bundle.
    "knowledge": {
        "core_memory_max_tokens": (int, lambda v: 256 <= v <= 32000),
        "auto_promote_threshold": (int, lambda v: 1 <= v <= 1000),
        "finetune_min_examples": (int, lambda v: 1 <= v <= 100000),
        "note_max_tokens": (int, lambda v: 100 <= v <= 16000),
    },
}


def _overrides_path() -> Path:
    from .config import CONFIG
    return Path(CONFIG.knowledge["base_dir"]) / "runtime_overrides.json"


def _profile_engine_overrides() -> dict:
    """Active profile's engine_overrides — empty dict if profile missing
    or store import fails (tests outside the FastAPI boot path)."""
    try:
        from .pipeline_profile import active_overrides
        return (active_overrides().get("engine_overrides") or {})
    except Exception:
        return {}


def load_overrides() -> dict:
    """Read the saved overrides file. Returns `{}` if not present —
    that's the normal "no user customisations yet" state."""
    p = _overrides_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        log.warning("runtime_overrides.json unreadable (%s); ignoring", e)
        return {}


def save_overrides(cfg: dict) -> dict:
    """Write the overrides file. Caller is expected to have validated
    via `validate_partial` already."""
    p = _overrides_path()
    # C3: atomic .tmp + rename. runtime_overrides.json is read at boot
    # to patch CONFIG in place — a torn write here would zero-out the
    # user's tuned router budget / verification knobs on next start.
    from . import paths as _paths
    _paths.write_atomic_json(p, cfg)
    return cfg


def validate_partial(partial: dict) -> tuple[dict, list[str]]:
    """Filter `partial` against the whitelist.

    Returns `(clean_overrides, rejected_paths)` where `rejected_paths`
    is a list of dotted keys that didn't match a whitelist entry,
    coerced to the wrong type, or failed the range check. Callers
    typically show rejected paths back to the user as a warning.
    """
    clean: dict[str, dict] = {}
    rejected: list[str] = []
    for section, fields in (partial or {}).items():
        if section not in _ALLOWED:
            rejected.append(section)
            continue
        if not isinstance(fields, dict):
            rejected.append(section)
            continue
        sec_out: dict = {}
        for key, value in fields.items():
            spec = _ALLOWED[section].get(key)
            if spec is None:
                rejected.append(f"{section}.{key}")
                continue
            expected_type, validator = spec
            try:
                # bool is a subclass of int — accept "true"/"false"/"1"/0
                # from JSON but reject integers as bools accidentally.
                if expected_type is bool:
                    if isinstance(value, bool):
                        coerced: Any = value
                    elif isinstance(value, (int, str)):
                        coerced = str(value).lower() in ("1", "true", "yes", "on")
                    else:
                        raise TypeError
                elif expected_type is float:
                    coerced = float(value)
                elif expected_type is int:
                    coerced = int(value)
                else:
                    coerced = value
            except (TypeError, ValueError):
                rejected.append(f"{section}.{key}")
                continue
            if not validator(coerced):
                rejected.append(f"{section}.{key}")
                continue
            sec_out[key] = coerced
        if sec_out:
            clean[section] = sec_out
    return clean, rejected


def apply_overrides(overrides: dict) -> None:
    """In-place merge `overrides` into CONFIG._data so existing
    references (e.g. self.cfg_router cached in llm.py) pick up the
    change. Caller is expected to pass already-validated input."""
    from .config import CONFIG
    for section, fields in overrides.items():
        target = CONFIG._data.get(section)
        if not isinstance(target, dict):
            # Shouldn't happen if whitelist is correct, but fail soft.
            CONFIG._data[section] = dict(fields)
            continue
        target.update(fields)


def apply_overrides_from_file() -> dict:
    """Boot hook — load the file, validate it (defensive — in case
    someone hand-edited values out of range), apply. Returns the
    cleaned overrides that were actually applied."""
    raw = load_overrides()
    clean, rejected = validate_partial(raw)
    if rejected:
        log.warning(
            "runtime_overrides.json had %d invalid entries: %s",
            len(rejected), rejected,
        )
    if clean:
        apply_overrides(clean)
        log.info(
            "applied runtime overrides: %s",
            {sec: list(fields.keys()) for sec, fields in clean.items()},
        )
    return clean


def get_effective_config() -> dict:
    """Read-only snapshot of the whitelisted sections from CONFIG —
    what's actually in effect right now. Used by the GET endpoint so
    the UI can pre-fill forms."""
    from .config import CONFIG
    out: dict = {}
    for section, fields in _ALLOWED.items():
        sec_src = CONFIG._data.get(section) or {}
        out[section] = {k: sec_src.get(k) for k in fields}
    # Phase 1 (Task 5): active pipeline profile's engine_overrides
    # form a third layer on top of file overrides. Only whitelisted
    # keys are applied (mirrors validate_partial gates).
    profile_engine = _profile_engine_overrides()
    for section, fields in profile_engine.items():
        if section not in out:
            continue
        if not isinstance(fields, dict):
            continue
        allowed_keys = _ALLOWED.get(section) or {}
        for key, value in fields.items():
            if key in allowed_keys:
                out[section][key] = value
    return out


def list_allowed() -> dict[str, list[str]]:
    """Names only — useful for the UI to render a form schema."""
    return {sec: list(fields.keys()) for sec, fields in _ALLOWED.items()}
