"""Single source of truth for the agent's runtime version string.

Scheme: `{base}.{commits}` where
  - `base`   — semantic baseline read from `pyproject.toml` (`[project].version`).
              Keep this as MAJOR.MINOR (e.g. "0.16"); the third component is
              auto-computed below so we don't have to bump it on every commit.
  - `commits` — total commit count reachable from HEAD, via
              `git rev-list --count HEAD`. Each commit on master bumps the
              displayed version by 1 with zero manual work.

Result examples:
  pyproject.toml `version = "0.16"`, repo at 232 commits → "0.16.232"
  After `hrant update` pulls 3 commits                  → "0.16.235"

Failure paths (no git, detached engine on a release tarball, …) all
fall back to the bare `base` string. Never raises.

The resolver also gathers `commit` (short SHA) and `commit_date` so the
CLI can render `hrant version` with full provenance and the updater can
print a before→after delta on every successful update.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class VersionInfo:
    """All-in-one version snapshot used by CLI + updater + WebUI status."""
    base: str          # "0.16" — baseline from pyproject.toml
    commits: int       # number of commits reachable from HEAD; 0 if unknown
    full: str          # "0.16.232" — what users see
    commit: str        # "457b3b1f" short sha; "" if unknown
    commit_date: str   # "2026-05-16" ISO date of HEAD commit; "" if unknown
    branch: str        # "master"; "" if unknown


def _read_pyproject_base() -> str:
    """Pull the `[project].version` field from pyproject.toml. We don't
    use `importlib.metadata.version("agi-agent")` because that reflects
    the WHEEL installation, which is stale after `git pull` until the
    next `pip install -e .`. Reading the file directly always matches
    the source code currently on disk."""
    pp = paths.repo_root() / "pyproject.toml"
    try:
        text = pp.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return "0.0"
    # Tiny hand-roll instead of pulling tomllib (3.11+) just for one
    # field — keeps the import side-effect-free on every backend boot.
    # Strip inline `#` comments so a `version = "0.16"  # base; runtime
    # appends ".N"` annotation doesn't leak into the parsed value.
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("version") or "=" not in s:
            continue
        _, _, val = s.partition("=")
        # Drop trailing comment, if any.
        if "#" in val:
            val = val.split("#", 1)[0]
        val = val.strip()
        # Strip surrounding quotes.
        if len(val) >= 2 and val[0] in {'"', "'"} and val[-1] == val[0]:
            val = val[1:-1]
        if val:
            return val
    return "0.0"


def _git(*args: str) -> str:
    """Run a git command in the engine repo. Returns stdout stripped,
    or '' on any failure (no git, not a repo, network errors,
    monkeypatch surprises in tests). Never raises — the version
    resolver must succeed even on a release tarball with no .git
    directory. Catches the broad `Exception` instead of the specific
    OSError tuple so future refactors that swap subprocess for
    something else don't accidentally make us raise from a hot
    import path."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=str(paths.repo_root()),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if r.returncode != 0:
            return ""
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _commit_count() -> int:
    """`git rev-list --count HEAD`. 0 if git isn't available."""
    out = _git("rev-list", "--count", "HEAD")
    try:
        return int(out) if out else 0
    except ValueError:
        return 0


def _short_sha() -> str:
    return _git("rev-parse", "--short", "HEAD")


def _commit_date() -> str:
    return _git("log", "-1", "--format=%cs")  # %cs = ISO date


def _branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def _safe(fn, default):
    """Run a resolver helper, swallow anything. CLI startup must
    never crash on a bookkeeping helper."""
    try:
        return fn()
    except Exception:
        return default


def get_version_info() -> VersionInfo:
    """Resolve the current version snapshot. Cheap (one toml read +
    up to four short `git` calls); not cached because `hrant update`
    needs a fresh read after pulling commits.

    Every helper is wrapped in `_safe` so a future refactor (or a
    test that monkey-patches at a higher level) can't break version
    resolution and cascade into `--version` / `hrant run` failure."""
    base = _safe(_read_pyproject_base, "0.0")
    commits = _safe(_commit_count, 0)
    if commits > 0:
        full = f"{base}.{commits}"
    else:
        # No git available — show the bare baseline so the rest of
        # the CLI still has something to print.
        full = base
    return VersionInfo(
        base=base,
        commits=commits,
        full=full,
        commit=_safe(_short_sha, ""),
        commit_date=_safe(_commit_date, ""),
        branch=_safe(_branch, ""),
    )


def get_version() -> str:
    """Shortcut returning just the user-visible string. Equivalent to
    `get_version_info().full` — provided for callers that don't need
    the surrounding metadata."""
    return get_version_info().full
