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

  a) `HRANT_DATA_DIR` env var (absolute path) — always honoured,
     even if the target doesn't exist yet (the init wizard creates
     it).
  b) `~/.hrant/data/` if that directory exists.

If NEITHER is available, the agent refuses to start with a clear
message telling the user to run `hrant init`. There is intentionally
no fallback to the repo root: a stray run from a fresh clone would
otherwise silently write personal data into the engine tree, which
defeats the engine/data split that `hrant update` and `hrant rollback`
depend on.

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


class DataDirMissing(RuntimeError):
    """Raised when no data dir is configured AND `~/.hrant/data/`
    doesn't exist. The init wizard catches this; everyone else
    surfaces it to the user so they know to run `hrant init`."""


def data_dir(*, require: bool = True) -> Path:
    """User-data root. See module docstring for resolution order.

    Pass `require=False` to get the would-be path without raising —
    used by the init wizard (which is about to CREATE the dir) and
    by tests that build paths against a fresh tmp_path.

    Note: this is a function (not a constant) so tests can monkeypatch
    HRANT_DATA_DIR and immediately get the new value. The cost is a
    few env lookups per call; nothing on the hot path uses this at
    request rate."""
    forced = os.environ.get("HRANT_DATA_DIR", "").strip()
    if forced:
        # Env var is authoritative — return even if the dir doesn't
        # exist yet (the wizard creates it).
        return Path(forced).expanduser().resolve()
    user_default = _user_default_data_dir()
    if user_default.exists():
        return user_default
    if require:
        raise DataDirMissing(
            f"no data dir found. Run `hrant init` to bootstrap "
            f"{user_default}, or set HRANT_DATA_DIR to an absolute path."
        )
    # Caller explicitly asked for the would-be path. Always the
    # user-default location — never the repo root.
    return user_default


def is_initialised() -> bool:
    """True when a usable data dir exists. The init wizard calls
    this to decide between a fresh-install flow and a reconfigure."""
    try:
        d = data_dir(require=True)
    except DataDirMissing:
        return False
    return d.exists()


def is_split_install() -> bool:
    """Always True now — engine and data are always in separate
    directories. Kept for API stability; callers can check this
    if they care to log the layout shape."""
    return True


def config_yaml_path() -> Path:
    """Where to read/write config.yaml. Always under data_dir; never
    falls back to the engine repo so a misconfigured run can't write
    user-specific values into the shared engine tree."""
    return data_dir(require=False) / "config.yaml"


def env_path() -> Path:
    """Where to read/write `.env`. Always under data_dir."""
    return data_dir(require=False) / ".env"


def knowledge_dir() -> Path:
    """User's knowledge tree. Created on first install. Survives
    every update."""
    return data_dir(require=False) / "knowledge"


def workspace_dir() -> Path:
    """User's workspace tree (inbox/outbox/notes/turns)."""
    return data_dir(require=False) / "workspace"


def history_path() -> Path:
    """Update / rollback history. Always under data_dir so a rollback
    in the engine never wipes its own ledger."""
    return data_dir(require=False) / "update_history.json"


def templates_dir() -> Path:
    """Repo-side starter content the init wizard copies into a fresh
    data_dir. Engine-side, so updates ship newer templates."""
    return _REPO_ROOT / "knowledge_templates"


def ensure_data_dir() -> Path:
    """Create data_dir (and standard subdirs) if missing. Returns
    the resolved path. Idempotent — safe to call from anywhere."""
    d = data_dir(require=False)
    d.mkdir(parents=True, exist_ok=True)
    (d / "knowledge").mkdir(exist_ok=True)
    (d / "workspace").mkdir(exist_ok=True)
    return d
