"""Single source of truth for where Hrant reads and writes files.

Two distinct buckets, deliberately separate so `hrant update` can
refresh the engine without touching the user's data:

  1. ENGINE — the git checkout: backend/ (which now contains
     backend/knowledge_templates/), frontend/, deploy/, pyproject.toml.
     Read-only at runtime; replaced by `hrant update`.

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

import json
import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """The git checkout's root (where pyproject.toml lives). Engine
    code and deploy/ templates live here. Knowledge templates now
    live under `backend/knowledge_templates/` so the wheel includes
    them — see `templates_dir()`."""
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
    """Starter content the init wizard copies into a fresh data_dir.
    Engine-side, so updates ship newer templates.

    Audit P1 #4 fix: templates ship INSIDE the `backend/` package now
    (`backend/knowledge_templates/`), so `pip install` from a wheel
    actually includes them. The legacy repo-root location (`<repo>/
    knowledge_templates/`) is checked as a fallback for any editable
    install that still has the old layout.
    """
    pkg_side = Path(__file__).resolve().parent / "knowledge_templates"
    if pkg_side.exists():
        return pkg_side
    return _REPO_ROOT / "knowledge_templates"


def ensure_data_dir() -> Path:
    """Create data_dir (and standard subdirs) if missing. Returns
    the resolved path. Idempotent — safe to call from anywhere."""
    d = data_dir(require=False)
    d.mkdir(parents=True, exist_ok=True)
    (d / "knowledge").mkdir(exist_ok=True)
    (d / "workspace").mkdir(exist_ok=True)
    return d


# ─── Sensitive-file write helper ───────────────────────────────────
#
# Audit follow-up: configs containing API keys, OAuth tokens, paired-
# device secrets and chat IDs must land on disk as 0600 (owner-only),
# not the world-readable default umask. The old write sites used
# `path.write_text(json.dumps(...))` which inherits the umask and on
# most Linux distros yields 0644. Anyone with shell access to the box
# could `cat ~/.hrant/data/knowledge/providers.json` and walk away
# with every API key.
#
# This helper:
#   • Encodes a dict to JSON (UTF-8, indented by default).
#   • Writes atomically via `.tmp` + rename so a crash mid-write never
#     leaves the file truncated.
#   • Chmods the tempfile to 0o600 BEFORE the rename so the rename
#     atomically swaps in a file that was never world-readable.
#   • On Windows, `os.chmod` is largely a no-op (ACLs aren't POSIX
#     bits) — we still call it for consistency, but on Windows the
#     real protection is the user-profile ACL on `~\.hrant\data\`.
#
# Use this for ANY file that may contain secrets. Public, non-secret
# data (e.g. cached embeddings, public manifests) stays on `write_text`.


def write_secret_json(
    path: Path,
    payload,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Atomically write `payload` (dict / list / str) as JSON to
    `path`, with mode 0600.

    Audit-driven contract: the file MUST end up at 0o600 on POSIX,
    even if a previous version was world-readable. We chmod the
    tempfile pre-rename so the swap is atomic in both content and
    mode.
    """
    import json as _json
    import os as _os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (str, bytes)):
        data = payload if isinstance(payload, str) else payload.decode("utf-8")
    else:
        data = _json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    try:
        _os.chmod(tmp, 0o600)
    except Exception:
        # Best-effort on platforms where POSIX mode bits don't apply.
        pass
    _os.replace(tmp, path)
    # Some filesystems / replace implementations carry the mode of the
    # destination, not the source. Re-chmod the final path to be safe.
    try:
        _os.chmod(path, 0o600)
    except Exception:
        pass


# ─── Public atomic-write helper for non-secret JSON stores ──────────
#
# Audit 2026-06-10 (Bundle B, fix C3): ~13 JSON stores across the
# backend used `path.write_text(json.dumps(...))` directly. That's
# a non-atomic write: a crash or kill mid-write leaves a torn file
# (truncated JSON, zero bytes, or partial content) and the next
# `_load` either raises and resets the store to empty (losing data)
# or silently treats the partial as valid. `write_secret_json` above
# already solves the same problem for secret configs; this helper
# extracts the atomic-rename pattern for non-secret stores so every
# JSON writer in the codebase converges on one battle-tested path.


def write_atomic_json(
    path: Path,
    data,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    mode: int | None = None,
) -> None:
    """Serialize `data` to JSON and write to `path` atomically.

    Writes via `path.with_suffix(path.suffix + '.tmp')` then `replace`,
    so a crash mid-write never leaves a torn or zero-byte file —
    callers' `_load` either sees the prior version or the new one.

    Creates parent directories if missing. If `mode` is given, chmods
    after rename (useful for secrets — but secret-stores should prefer
    `write_secret_json` which always chmods to 0o600). Pass `indent=None`
    to get the most compact single-line encoding (matches the pre-fix
    behaviour of stores that called `json.dumps(...)` without indent —
    notably the vector store and the daily-eval aggregate). Best-effort:
    writers that need to fail loudly should NOT catch here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=ensure_ascii, indent=indent),
        encoding="utf-8",
    )
    if mode is not None:
        try:
            os.chmod(tmp, mode)
        except Exception:
            # POSIX mode bits don't apply on Windows; best-effort.
            pass
    tmp.replace(path)


def secure_existing_file(path: Path) -> None:
    """Tighten an existing file's mode to 0o600 if it isn't already.
    Used on boot to fix files that were created by older versions
    before `write_secret_json` existed. Silent on Windows."""
    import os as _os
    import stat as _stat

    p = Path(path)
    try:
        if not p.exists():
            return
        cur = _stat.S_IMODE(p.stat().st_mode)
        if cur != 0o600:
            _os.chmod(p, 0o600)
    except Exception:
        pass
