"""Tunables for daily consolidation.

These live in code rather than a JSON file because they're behaviour-
shaping defaults — users who want to tweak `IDLE_THRESHOLD_SECONDS`
or the cooldown should edit code (or override via env), not surprise
themselves with a knob they forgot they turned.

Per the user's spec for this build:
  - Unlimited cost budget (no soft/hard cap blocks the run)
  - Minimum activity = 0 (fires even on idle days, digest just notes
    "no activity")
  - Global facts/digests; per-speaker profiles only
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import paths


# How long the agent must be quiet (no new jobs) before a daily
# consolidation is allowed to fire. Keeps the agent from cutting
# off a late-night session to do bookkeeping.
IDLE_THRESHOLD_SECONDS: float = float(
    os.environ.get("HRANT_CONSOLIDATION_IDLE_SECONDS", 15 * 60)
)

# Minimum time between two daily consolidations. 24h-ish; the
# adaptive idle check above means actual fire time floats.
COOLDOWN_SECONDS: float = float(
    os.environ.get("HRANT_CONSOLIDATION_COOLDOWN_SECONDS", 24 * 60 * 60)
)

# How often the scheduler wakes up to check the gates. Cheap —
# just a timestamp comparison + an idle-since lookup.
SCHEDULER_TICK_SECONDS: float = float(
    os.environ.get("HRANT_CONSOLIDATION_TICK_SECONDS", 60)
)

# How far back the gatherer looks. Slightly more than the cooldown
# so a slow-firing consolidation doesn't miss the gap. 26h covers
# typical adaptive lag.
GATHER_WINDOW_SECONDS: float = float(
    os.environ.get("HRANT_CONSOLIDATION_WINDOW_SECONDS", 26 * 60 * 60)
)

# Activity gate. Default 1: skip days with literally zero turns
# (no point spinning the LLM pipeline + writing an empty digest
# every quiet 24h). Set to 0 to fire even on empty days; set
# higher to require a meaningful amount of activity.
MIN_JOBS_FOR_RUN: int = int(
    os.environ.get("HRANT_CONSOLIDATION_MIN_JOBS", 1)
)

# Cost is tracked for REPORTING only. Nothing enforces a cap.
#
# The previous comment ended "Set to a positive number to enforce a soft cap",
# which was false: this constant has no readers, so a positive value changes
# nothing (2026-08-09 dead-code audit). Left in place because the pipeline
# does compute `estimated_cost_usd` per run, so the value is a reasonable
# anchor for a future enforcement point — but until something reads it, the
# comment must not promise a cap that does not exist.
DAILY_COST_CAP_USD: float = float(
    os.environ.get("HRANT_CONSOLIDATION_COST_CAP_USD", 0.0)
)


# Minimum confidence required to PROMOTE a fact from the LLM
# extraction output into memory_facts.jsonl + the knowledge graph.
# Audit 2026-05-27 found daily consolidation was producing only
# ~4 facts/day with threshold=0.8 hardcoded; lowering to 0.65 lets
# medium-confidence material through while still filtering noise.
# Override via env for experimentation.
PROMOTE_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("HRANT_CONSOLIDATION_PROMOTE_THRESHOLD", 0.65)
)


# ─── Paths ────────────────────────────────────────────────────────────


def state_path() -> Path:
    """Where last-run state is persisted."""
    return paths.knowledge_dir() / "consolidation_state.json"


def digests_dir() -> Path:
    """One JSON file per day under this directory."""
    return paths.knowledge_dir() / "memory_digests"


def digest_path_for(date_str: str) -> Path:
    return digests_dir() / f"{date_str}.json"


def user_md_path() -> Path:
    """Global user profile — WebUI-default speaker."""
    return paths.knowledge_dir() / "identity" / "user.md"


def profile_path_for_speaker(speaker_id: str) -> Path:
    """Per-Telegram-user profile. The global `user.md` is for
    `webui:default`; Telegram users get isolated profiles under
    `identity/profiles/`."""
    sanitized = speaker_id.replace(":", "_").replace("/", "_")
    return paths.knowledge_dir() / "identity" / "profiles" / f"{sanitized}.md"
