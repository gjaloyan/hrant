"""Engine update + rollback ledger.

The engine repo (where backend/, frontend/, deploy/ live) is just a
git checkout. `hrant update` keeps it in sync with `origin/master`
and runs the supporting dep-install / build steps. `hrant rollback`
uses the ledger this module writes to revert to a previous SHA.

Update flow:
  1. Refuse if working tree is dirty (user has uncommitted tracked
     changes — we don't want to lose them).
  2. `git fetch origin` to learn about new commits.
  3. Record current SHA + timestamp into the history ledger.
  4. `git pull --ff-only origin <branch>`. Reject on non-fast-forward
     (someone divergent-pushed; user must resolve manually).
  5. `pip install -e .` so any pyproject.toml dep changes apply.
  6. Optional `cd frontend && npm install && npm run build` (skipped
     with --skip-frontend or when --no-build is set).
  7. Print a one-line "restart hrant" reminder.

Failures at steps 4-6 are surfaced verbatim to the user; we do NOT
attempt auto-rollback because the failure may be unrelated to the
update (network, npm registry down, …) and the rollback path can
make things worse. The history entry has already been written
before the pull, so `hrant rollback` is available if needed.

Ledger format (`paths.history_path()`):
  {
    "entries": [
      {
        "sha": "abc1234...",
        "timestamp": "2026-05-13T15:42:08Z",
        "branch": "master",
        "result": "success" | "failed_at_pull" | ...
      },
      ...
    ]
  }

Newest entry last. Bounded to MAX_HISTORY_ENTRIES so the file
doesn't grow forever; older entries dropped from the front when the
cap is exceeded.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import paths

log = logging.getLogger(__name__)


MAX_HISTORY_ENTRIES = 50


@dataclass
class HistoryEntry:
    sha: str
    timestamp: str
    branch: str
    result: str  # "success" | "failed_at_<step>"
    note: str = ""


@dataclass
class UpdateResult:
    ok: bool
    old_sha: str
    new_sha: Optional[str]
    branch: str
    pulled_commits: int
    pip_ran: bool
    frontend_built: bool
    error: Optional[str] = None
    messages: Optional[list[str]] = None


# --- git helpers --------------------------------------------------------


def _git(*args: str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the engine repo. Returns the completed
    process so callers can inspect stdout/stderr. `check=True` raises
    on non-zero exit (the wrapper catches it for nicer messages)."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd or paths.repo_root()),
        capture_output=True,
        text=True,
        check=check,
    )


def current_sha() -> str:
    """`git rev-parse HEAD` — the SHA the running engine is at."""
    try:
        r = _git("rev-parse", "HEAD")
        return r.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def current_branch() -> str:
    try:
        r = _git("rev-parse", "--abbrev-ref", "HEAD")
        return r.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def is_dirty() -> bool:
    """True if the working tree has uncommitted tracked changes.
    Untracked files (e.g. gitignored knowledge/) do NOT count."""
    try:
        r = _git("diff-index", "--quiet", "HEAD", "--", check=False)
        return r.returncode != 0
    except FileNotFoundError:
        # git not installed — can't tell, treat as dirty to refuse.
        return True


def fetch_remote(branch: str) -> tuple[bool, str]:
    """`git fetch origin <branch>`. Returns (ok, error_msg)."""
    try:
        r = _git("fetch", "origin", branch, check=False)
        if r.returncode != 0:
            return False, (r.stderr or "git fetch failed").strip()
        return True, ""
    except FileNotFoundError:
        return False, "git not found on PATH"


def commits_ahead(branch: str = "master") -> list[dict]:
    """Commits on `origin/<branch>` that aren't in HEAD yet, formatted
    as `[{sha, subject}, ...]`. Newest last (git log default)."""
    try:
        r = _git(
            "log",
            f"HEAD..origin/{branch}",
            "--pretty=format:%h|%s",
            check=False,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            if "|" not in line:
                continue
            sha, _, subject = line.partition("|")
            out.append({"sha": sha.strip(), "subject": subject.strip()})
        return out
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


# --- history ledger -----------------------------------------------------


def load_history() -> list[HistoryEntry]:
    p = paths.history_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("update_history.json unreadable (%s); starting fresh", e)
        return []
    entries = raw.get("entries") or []
    out: list[HistoryEntry] = []
    for e in entries:
        try:
            out.append(HistoryEntry(**e))
        except TypeError:
            continue
    return out


def save_history(entries: list[HistoryEntry]) -> None:
    p = paths.history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Cap to most-recent MAX_HISTORY_ENTRIES so the ledger doesn't
    # grow forever. Keep newest, drop oldest.
    capped = entries[-MAX_HISTORY_ENTRIES:]
    payload = {"entries": [asdict(e) for e in capped]}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def record(sha: str, branch: str, result: str, note: str = "") -> HistoryEntry:
    """Append a history entry. Returns the entry written so the
    caller can show it."""
    entry = HistoryEntry(
        sha=sha,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        branch=branch,
        result=result,
        note=note,
    )
    entries = load_history()
    entries.append(entry)
    save_history(entries)
    return entry


# --- pip / npm steps ----------------------------------------------------


def _run(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    """Wrapper that runs a command and captures combined output for
    failure reporting. Used for pip / npm steps."""
    try:
        r = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "no output").strip()[-2000:]
        return True, (r.stdout or "").strip()[-1000:]
    except FileNotFoundError:
        return False, f"{cmd[0]} not found on PATH"


def run_pip_install() -> tuple[bool, str]:
    """`pip install -e .` from the engine repo root. Picks up any
    pyproject.toml dep changes the update brought in. Uses the
    same Python interpreter that the agent is running under."""
    import sys
    return _run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=paths.repo_root(),
    )


def run_frontend_build() -> tuple[bool, str]:
    """`npm install` + `npm run build` in frontend/. Both are
    captured into one combined result — failure at either step
    aborts and returns the stderr."""
    frontend = paths.repo_root() / "frontend"
    ok, out = _run(["npm", "install"], cwd=frontend)
    if not ok:
        return False, f"npm install: {out}"
    ok, out = _run(["npm", "run", "build"], cwd=frontend)
    if not ok:
        return False, f"npm run build: {out}"
    return True, "frontend rebuilt"


def frontend_changed(commits: list[dict]) -> bool:
    """Heuristic: did the incoming commits touch anything under
    frontend/? Used to skip the (slow) rebuild when only backend
    changed. Conservative — when in doubt, returns True."""
    if not commits:
        return False
    try:
        sha_range = f"HEAD..origin/{current_branch() or 'master'}"
        r = _git("diff", "--name-only", sha_range, check=False)
        if r.returncode != 0:
            return True
        return any(line.strip().startswith("frontend/") for line in r.stdout.splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return True


# --- top-level orchestration --------------------------------------------


def do_update(
    *,
    branch: str = "master",
    skip_frontend: bool = False,
    skip_pip: bool = False,
) -> UpdateResult:
    """One-shot update. Returns UpdateResult; the CLI prints from it."""
    messages: list[str] = []
    old = current_sha()
    br = current_branch() or branch

    if is_dirty():
        return UpdateResult(
            ok=False, old_sha=old, new_sha=None, branch=br,
            pulled_commits=0, pip_ran=False, frontend_built=False,
            error=(
                "working tree has uncommitted tracked changes. "
                "Commit or `git stash` them, then retry. "
                "(Untracked files in knowledge/ or workspace/ don't count "
                "and won't block the update.)"
            ),
        )

    ok, err = fetch_remote(br)
    if not ok:
        return UpdateResult(
            ok=False, old_sha=old, new_sha=None, branch=br,
            pulled_commits=0, pip_ran=False, frontend_built=False,
            error=f"git fetch failed: {err}",
        )

    incoming = commits_ahead(br)
    if not incoming:
        return UpdateResult(
            ok=True, old_sha=old, new_sha=old, branch=br,
            pulled_commits=0, pip_ran=False, frontend_built=False,
            messages=["already up to date"],
        )

    # Record BEFORE we change anything so a partial-fail update
    # still leaves a rollback point.
    record(old, br, "pre_update", note=f"about to pull {len(incoming)} commits")

    try:
        r = _git("pull", "--ff-only", "origin", br, check=False)
        if r.returncode != 0:
            record(old, br, "failed_at_pull", note=(r.stderr or "").strip()[:200])
            return UpdateResult(
                ok=False, old_sha=old, new_sha=None, branch=br,
                pulled_commits=0, pip_ran=False, frontend_built=False,
                error=(
                    "git pull --ff-only refused (non-fast-forward; you have "
                    "local commits that aren't on origin). Resolve manually."
                ),
            )
    except FileNotFoundError:
        return UpdateResult(
            ok=False, old_sha=old, new_sha=None, branch=br,
            pulled_commits=0, pip_ran=False, frontend_built=False,
            error="git not found on PATH",
        )

    new = current_sha()
    messages.append(f"pulled {len(incoming)} commits ({old[:8]} → {new[:8]})")

    pip_ran = False
    if not skip_pip:
        ok, out = run_pip_install()
        pip_ran = ok
        if not ok:
            record(new, br, "failed_at_pip", note=out[:200])
            return UpdateResult(
                ok=False, old_sha=old, new_sha=new, branch=br,
                pulled_commits=len(incoming), pip_ran=False, frontend_built=False,
                error=f"pip install -e . failed: {out[:500]}",
            )
        messages.append("pip install -e . ✓")

    fe_built = False
    if not skip_frontend:
        if not frontend_changed(incoming):
            messages.append("frontend unchanged — skipped npm build")
        else:
            ok, out = run_frontend_build()
            fe_built = ok
            if not ok:
                record(new, br, "failed_at_frontend", note=out[:200])
                return UpdateResult(
                    ok=False, old_sha=old, new_sha=new, branch=br,
                    pulled_commits=len(incoming), pip_ran=pip_ran,
                    frontend_built=False,
                    error=f"frontend build failed: {out[:500]}",
                )
            messages.append("frontend rebuilt ✓")

    record(new, br, "success", note=f"updated from {old[:8]}")
    return UpdateResult(
        ok=True, old_sha=old, new_sha=new, branch=br,
        pulled_commits=len(incoming), pip_ran=pip_ran,
        frontend_built=fe_built, messages=messages,
    )


def do_rollback(
    *,
    to_sha: Optional[str] = None,
    skip_frontend: bool = False,
    skip_pip: bool = False,
) -> UpdateResult:
    """Reset HEAD to a previous SHA from history (or `to_sha`).

    Strategy:
      - If `to_sha` is None: use the entry BEFORE the most recent
        successful one (i.e. one step back).
      - Refuse on dirty tree (same reasoning as update).
      - `git reset --hard` is destructive — but since the user
        invoked rollback knowing what it does, we trust the call.
      - Same pip + frontend rebuild steps as update so the engine's
        deps match the rolled-back code.
    """
    messages: list[str] = []
    if is_dirty():
        return UpdateResult(
            ok=False, old_sha=current_sha(), new_sha=None,
            branch=current_branch() or "master",
            pulled_commits=0, pip_ran=False, frontend_built=False,
            error="working tree dirty; commit or stash first",
        )

    old = current_sha()
    br = current_branch() or "master"

    if to_sha is None:
        entries = load_history()
        # Walk back to find a successful entry OLDER than the current
        # SHA. Default behaviour: roll back one step.
        successes = [e for e in entries if e.result == "success"]
        if len(successes) < 1:
            return UpdateResult(
                ok=False, old_sha=old, new_sha=None, branch=br,
                pulled_commits=0, pip_ran=False, frontend_built=False,
                error=(
                    "no rollback target — history is empty or has no "
                    "successful update entries. Pass --to <sha> to choose "
                    "a target explicitly."
                ),
            )
        # The most recent success is the CURRENT state. Want the one
        # before that.
        if len(successes) < 2:
            # Try the pre_update entry (recorded just before the most
            # recent update) — that's a valid "go back to before the
            # last update" target.
            pre = [e for e in entries if e.result == "pre_update"]
            if not pre:
                return UpdateResult(
                    ok=False, old_sha=old, new_sha=None, branch=br,
                    pulled_commits=0, pip_ran=False, frontend_built=False,
                    error="no previous version in history to roll back to",
                )
            to_sha = pre[-1].sha
        else:
            to_sha = successes[-2].sha

    try:
        r = _git("reset", "--hard", to_sha, check=False)
        if r.returncode != 0:
            return UpdateResult(
                ok=False, old_sha=old, new_sha=None, branch=br,
                pulled_commits=0, pip_ran=False, frontend_built=False,
                error=f"git reset --hard {to_sha} failed: {r.stderr.strip()[:300]}",
            )
    except FileNotFoundError:
        return UpdateResult(
            ok=False, old_sha=old, new_sha=None, branch=br,
            pulled_commits=0, pip_ran=False, frontend_built=False,
            error="git not found on PATH",
        )

    new = current_sha()
    messages.append(f"reset to {new[:8]} (was {old[:8]})")

    pip_ran = False
    if not skip_pip:
        ok, out = run_pip_install()
        pip_ran = ok
        if not ok:
            record(new, br, "rollback_partial", note=f"pip failed: {out[:200]}")
            return UpdateResult(
                ok=False, old_sha=old, new_sha=new, branch=br,
                pulled_commits=0, pip_ran=False, frontend_built=False,
                error=f"reset OK but pip install -e . failed: {out[:500]}",
            )
        messages.append("pip install -e . ✓")

    fe_built = False
    if not skip_frontend:
        ok, out = run_frontend_build()
        fe_built = ok
        if not ok:
            record(new, br, "rollback_partial", note=f"frontend failed: {out[:200]}")
            return UpdateResult(
                ok=False, old_sha=old, new_sha=new, branch=br,
                pulled_commits=0, pip_ran=pip_ran, frontend_built=False,
                error=f"reset + pip OK but frontend build failed: {out[:500]}",
            )
        messages.append("frontend rebuilt ✓")

    record(new, br, "rollback", note=f"rolled back from {old[:8]}")
    return UpdateResult(
        ok=True, old_sha=old, new_sha=new, branch=br,
        pulled_commits=0, pip_ran=pip_ran, frontend_built=fe_built,
        messages=messages,
    )
