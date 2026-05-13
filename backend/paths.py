"""Single source of truth for where Hrant reads and writes files.

Two distinct buckets, deliberately separate so `hrant update` can
refresh the engine without touching the user's data:

  1. ENGINE — the git checkout: backend/, frontend/, deploy/,
     knowledge_templates/, pyproject.toml. Read-only at runtime;
     replaced by `hrant update`.

  2. DATA — the user's stuff: config.yaml, .env, knowledge/,
     workspace/, runtime_overrides.json, autonomic_settings.json,
     update history. Survives every update and rollback.

Where DATA lives is determined by, in order of precedence:

  a) `HRANT_DATA_DIR` env var (absolute path)
  b) `~/.hrant/data/` if that directory exists
  c) the repo root itself (dev fallback — the current single-tree
     layout still works, no migration required)

Functions here return Path objects, never strings. Code that needs
a specific path (knowledge/, workspace/, …) calls the named helper
rather than gluing strings — that's how this module enforces the
engine/data split.
"""
from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """The git checkout's root (where pyproject.toml lives). Engine
    code, deploy/ templates, and knowledge_templates/ live here."""
    return _REPO_ROOT


def _user_default_data_dir() -> Path:
    """Cross-platform `~/.hrant/data/`. On Windows this resolves to
    `C:\\Users\\<name>\\.hrant\\data`. The leading dot is fine on
    Windows — Explorer just shows it normally."""
    return Path.home() / ".hrant" / "data"


def data_dir() -> Path:
    """User-data root. See module docstring for resolution order.

    Note: this is a function (not a constant) so tests can monkeypatch
    HRANT_DATA_DIR and immediately get the new value. The cost is a
    few env lookups per call; nothing on the hot path uses this at
    request rate."""
    forced = os.environ.get("HRANT_DATA_DIR", "").strip()
    if forced:
        return Path(forced).expanduser().resolve()
    user_default = _user_default_data_dir()
    if user_default.exists():
        return user_default
    # Dev fallback — the legacy single-tree layout (everything in
    # the repo root). On a fresh server install, the init wizard
    # creates `~/.hrant/data/` and from that point on it takes
    # precedence over the repo root.
    return _REPO_ROOT


def is_split_install() -> bool:
    """True when DATA is separated from ENGINE — i.e. data_dir() is
    NOT the repo root. Used by the init wizard / status command to
    show the deployment shape to the user."""
    return data_dir().resolve() != _REPO_ROOT.resolve()


def config_yaml_path() -> Path:
    """Where to read/write config.yaml. Prefers data_dir, falls back
    to the repo root's config.yaml (dev) so the current single-tree
    layout still works untouched."""
    in_data = data_dir() / "config.yaml"
    if in_data.exists():
        return in_data
    in_repo = _REPO_ROOT / "config.yaml"
    if in_repo.exists():
        return in_repo
    # Neither exists yet — caller (typically the init wizard) creates
    # one. Default the WRITE target to data_dir so a fresh install
    # doesn't accidentally pollute the engine repo.
    return in_data


def env_path() -> Path:
    """Where to read/write `.env`. Same precedence as config_yaml_path."""
    in_data = data_dir() / ".env"
    if in_data.exists():
        return in_data
    in_repo = _REPO_ROOT / ".env"
    if in_repo.exists():
        return in_repo
    return in_data


def knowledge_dir() -> Path:
    """User's knowledge tree. Created on first install. Survives
    every update."""
    return data_dir() / "knowledge"


def workspace_dir() -> Path:
    """User's workspace tree (inbox/outbox/notes/turns)."""
    return data_dir() / "workspace"


def history_path() -> Path:
    """Update / rollback history. Always under data_dir so a rollback
    in the engine never wipes its own ledger."""
    return data_dir() / "update_history.json"


def templates_dir() -> Path:
    """Repo-side starter content the init wizard copies into a fresh
    data_dir. Engine-side, so updates ship newer templates."""
    return _REPO_ROOT / "knowledge_templates"


def ensure_data_dir() -> Path:
    """Create data_dir (and standard subdirs) if missing. Returns
    the resolved path. Idempotent."""
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "knowledge").mkdir(exist_ok=True)
    (d / "workspace").mkdir(exist_ok=True)
    return d
