"""Local self-modifications — patch-overlay storage + apply/revert.

The agent can modify its own code at the user's request. These
modifications are NOT pushed to the official git remote; they live
on the user's machine as unified diff files in `data_dir/self_mods/`.

The design:

  1. When `self_modifier.apply()` accepts a proposal, this module
     records the change as a `.patch` file (unified diff against the
     engine repo) and writes it via `git apply`. The in-place edit
     and the on-disk patch are kept in sync.

  2. A manifest `data_dir/self_mods/applied.json` lists patches in
     order with status ("applied", "needs_review", "reverted").

  3. `revert_one(id)` runs `git apply -R <patch>` and removes the
     entry. Subsequent `git update` flows are unaffected.

  4. `revert_all_to_official()` is `git reset --hard origin/HEAD`
     plus clearing the manifest — the nuclear option, brings the
     engine back to exactly what's on GitHub.

  5. `reapply_all()` is called by `backend/updater.py` after a
     successful `git pull`. For each non-reverted patch in the
     manifest, runs `git apply --3way`. On conflict, the patch is
     marked "needs_review" (engine stays as-is — stability over the
     user's customisation in this case; the patch file is kept so
     the user can fix it manually in the Settings → Self-Modifications
     tab).

Why patch files instead of a local git branch:
  - Cleaner audit trail (one file = one change)
  - Per-change revert without `git rebase -i`
  - Works for any file type (Python, TSX, configs, templates)
  - Engine git tree always tracks origin/master exactly; the overlay
    is layered on top, never mixed into history

Why we store in data_dir, not in the engine repo:
  - `hrant update` does `git reset --hard` then re-applies; patches
    inside the engine repo would be lost
  - User data should survive engine resets — that's the whole
    engine/data split
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import paths

log = logging.getLogger(__name__)


@dataclass
class PatchEntry:
    """One self-modification on disk.

    `id` is short (8 hex chars from a uuid4) so it's safe in URLs
    and easy to type. `file` is the path under `repo_root()` that
    was modified — used for display + for revert sanity-checks.
    """

    id: str
    slug: str                  # short human-readable identifier
    file: str                  # relative path under repo_root
    title: str                 # one-line description for the UI
    created: str               # ISO timestamp
    status: str                # "applied" | "needs_review" | "reverted"
    patch_filename: str        # NNNN-<slug>.patch — sortable
    last_error: str = ""       # populated when status == "needs_review"


@dataclass
class Manifest:
    """Wraps applied.json. Order matters — patches are applied in
    creation order so a later patch can build on an earlier one."""

    entries: list[PatchEntry] = field(default_factory=list)


def _self_mods_dir() -> Path:
    return paths.data_dir() / "self_mods"


def _manifest_path() -> Path:
    return _self_mods_dir() / "applied.json"


def _ensure_dir() -> Path:
    d = _self_mods_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_manifest() -> Manifest:
    """Read applied.json. Returns an empty Manifest when missing or
    corrupt — never raises into a caller."""
    p = _manifest_path()
    if not p.exists():
        return Manifest()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        entries = [PatchEntry(**e) for e in raw.get("entries") or []]
        return Manifest(entries=entries)
    except Exception as e:
        log.warning("self_mods/applied.json unreadable (%s); starting fresh", e)
        return Manifest()


def save_manifest(m: Manifest) -> None:
    _ensure_dir()
    p = _manifest_path()
    p.write_text(
        json.dumps(
            {"entries": [asdict(e) for e in m.entries]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# --- diff generation ---------------------------------------------------


_SLUG_CLEAN = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 40) -> str:
    """Lowercase + alphanumeric + hyphens. Used for the patch
    filename so the file is grep-able and sortable."""
    s = _SLUG_CLEAN.sub("-", text.lower()).strip("-")
    return s[:max_len] or "self-mod"


def make_patch(file_rel: str, old_text: str, new_text: str) -> str:
    """Build a unified diff suitable for `git apply`. `file_rel` is
    the path relative to the repo root (e.g. `backend/agent.py`)
    — git accepts both `a/` and `b/` prefixes, we provide the
    classic style.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    # Ensure trailing newlines so diff doesn't insert "\ No newline
    # at end of file" markers in the common case.
    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    diff_iter = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_rel}",
        tofile=f"b/{file_rel}",
        n=3,
    )
    return "".join(diff_iter)


def _next_patch_number() -> int:
    """Patches are numbered 0001, 0002, … by ORDER in the manifest.
    A new patch goes at the end."""
    return len(load_manifest().entries) + 1


# --- git wrappers ------------------------------------------------------


def _git(*args: str, cwd: Optional[Path] = None, check: bool = False, input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd or paths.repo_root()),
        capture_output=True,
        text=True,
        check=check,
        input=input_text,
    )


def _git_apply(patch_text: str, *, reverse: bool = False, three_way: bool = False) -> tuple[bool, str]:
    """Apply (or reverse-apply) a unified diff via stdin. Returns
    (ok, error_message)."""
    args = ["apply"]
    if reverse:
        args.append("-R")
    if three_way:
        args.append("--3way")
    args.append("-")  # read patch from stdin
    try:
        r = _git(*args, input_text=patch_text)
    except FileNotFoundError:
        return False, "git not found on PATH"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "git apply failed").strip()
    return True, ""


def _git_apply_check(patch_text: str) -> tuple[bool, str]:
    """Dry-run: does the patch cleanly apply against the current
    working tree? Used by `reapply_all` to decide whether to call
    `git apply --3way` or mark needs_review."""
    try:
        r = _git("apply", "--check", "-", input_text=patch_text)
    except FileNotFoundError:
        return False, "git not found on PATH"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "").strip()
    return True, ""


# --- the four public flows ---------------------------------------------


def record_and_apply(
    *,
    file_rel: str,
    old_text: str,
    new_text: str,
    title: str,
    apply_now: bool = True,
) -> tuple[Optional[PatchEntry], str]:
    """Save a patch + (optionally) apply it to the engine.

    Steps:
      1. Build the unified diff.
      2. Write it to `data_dir/self_mods/NNNN-<slug>.patch`.
      3. If `apply_now=True`: `git apply` (the file is rewritten in-place).
         If `apply_now=False`: skip — the caller has already written
         the file and just wants the patch recorded for rollback
         (e.g. `self_modifier.apply()` writes the file directly so
         it can run py_compile/tests on a real disk file BEFORE
         confirming the change should persist).
      4. Append a PatchEntry to applied.json with status="applied".

    On apply failure (the file's content drifted from `old_text`),
    the patch file is removed and no manifest entry is created —
    the caller can show the error and the user can try a fresh
    proposal against the current source.
    """
    _ensure_dir()
    patch_text = make_patch(file_rel, old_text, new_text)
    if not patch_text.strip():
        return None, "empty diff — nothing to apply"

    num = _next_patch_number()
    slug = _slugify(title)
    filename = f"{num:04d}-{slug}.patch"
    patch_path = _self_mods_dir() / filename
    patch_path.write_text(patch_text, encoding="utf-8")

    if apply_now:
        ok, err = _git_apply(patch_text)
        if not ok:
            # Don't keep an orphan patch that doesn't apply — surface
            # the error and let the caller propose a fresh diff against
            # the actual current source.
            try:
                patch_path.unlink()
            except OSError:
                pass
            return None, f"git apply failed: {err}"

    entry = PatchEntry(
        id=uuid.uuid4().hex[:8],
        slug=slug,
        file=file_rel,
        title=title,
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        status="applied",
        patch_filename=filename,
    )
    m = load_manifest()
    m.entries.append(entry)
    save_manifest(m)
    log.info("self-mod applied: %s (%s)", filename, entry.id)
    return entry, ""


def revert_one(patch_id: str) -> tuple[bool, str]:
    """Reverse-apply ONE patch and remove it from the manifest.

    Note: reverting an early patch when later patches built on top
    of it may fail or produce surprising results. The Settings UI
    should warn before reverting a patch that isn't the most recent.
    """
    m = load_manifest()
    target = None
    target_idx = -1
    for i, e in enumerate(m.entries):
        if e.id == patch_id:
            target = e
            target_idx = i
            break
    if target is None:
        return False, f"no patch with id {patch_id}"
    if target.status == "reverted":
        return False, f"patch {patch_id} is already reverted"

    patch_path = _self_mods_dir() / target.patch_filename
    if not patch_path.exists():
        # Manifest out of sync — drop the entry and surface the
        # weirdness; the on-disk state is already as if reverted.
        m.entries.pop(target_idx)
        save_manifest(m)
        return False, f"patch file missing: {target.patch_filename}; manifest entry dropped"

    patch_text = patch_path.read_text(encoding="utf-8")
    # Only the "applied" status guarantees the patch is currently
    # on top of the engine. "needs_review" patches weren't applied
    # at last update — reversing them would mess up unrelated files.
    if target.status != "applied":
        # Drop from manifest + delete the file; nothing to revert.
        m.entries.pop(target_idx)
        save_manifest(m)
        try:
            patch_path.unlink()
        except OSError:
            pass
        return True, f"patch {patch_id} was not applied; removed from manifest"

    ok, err = _git_apply(patch_text, reverse=True)
    if not ok:
        return False, f"git apply -R failed: {err}"

    # Remove from manifest + delete patch file.
    m.entries.pop(target_idx)
    save_manifest(m)
    try:
        patch_path.unlink()
    except OSError:
        pass
    log.info("self-mod reverted: %s", target.patch_filename)
    return True, ""


def revert_all_to_official(*, branch: str = "master") -> tuple[bool, str]:
    """Hard reset the engine to `origin/<branch>` and wipe every
    self-mod. The user's data dir is untouched; only the engine
    repo and the `self_mods/` directory are affected.

    Equivalent to "I want exactly what's on GitHub, nothing else."
    """
    try:
        r = _git("reset", "--hard", f"origin/{branch}")
    except FileNotFoundError:
        return False, "git not found on PATH"
    if r.returncode != 0:
        # Try local branch as fallback (e.g. no remote configured).
        r = _git("reset", "--hard", branch)
        if r.returncode != 0:
            return False, f"git reset failed: {(r.stderr or '').strip()}"

    # Wipe patches + manifest.
    d = _self_mods_dir()
    if d.exists():
        for p in d.iterdir():
            if p.suffix == ".patch" or p.name == "applied.json":
                try:
                    p.unlink()
                except OSError:
                    pass
    log.info("all self-mods cleared; engine reset to origin/%s", branch)
    return True, ""


def reapply_all() -> dict:
    """Re-apply every non-reverted patch in order. Called by
    `backend/updater.py` after a successful `git pull` so the user's
    customisations survive an engine update.

    Strategy per patch:
      - `git apply --check` dry-run first.
      - If clean: apply normally, status stays "applied".
      - If conflict: `git apply --3way` attempt.
      - If `--3way` also fails: mark "needs_review", DO NOT apply.
        Engine stays at whatever the previous patches in the chain
        produced; the conflict is surfaced in the manifest for the
        UI to highlight.

    Returns a summary dict that the CLI/UI can print:
      {
        "reapplied": [<id>, ...],
        "needs_review": [<id>, ...],
        "skipped": [<id>, ...],   # already-reverted entries
      }
    """
    m = load_manifest()
    out = {"reapplied": [], "needs_review": [], "skipped": []}
    for entry in m.entries:
        if entry.status == "reverted":
            out["skipped"].append(entry.id)
            continue
        patch_path = _self_mods_dir() / entry.patch_filename
        if not patch_path.exists():
            entry.status = "needs_review"
            entry.last_error = "patch file missing"
            out["needs_review"].append(entry.id)
            continue
        patch_text = patch_path.read_text(encoding="utf-8")
        ok, err = _git_apply_check(patch_text)
        if ok:
            applied_ok, apply_err = _git_apply(patch_text)
            if applied_ok:
                entry.status = "applied"
                entry.last_error = ""
                out["reapplied"].append(entry.id)
                continue
            err = apply_err
        # Conflict on direct apply — try 3-way merge.
        applied_ok, apply_err = _git_apply(patch_text, three_way=True)
        if applied_ok:
            entry.status = "applied"
            entry.last_error = ""
            out["reapplied"].append(entry.id)
        else:
            entry.status = "needs_review"
            entry.last_error = apply_err[:400]
            out["needs_review"].append(entry.id)
    save_manifest(m)
    return out


def list_patches() -> list[PatchEntry]:
    """Public accessor for the UI. Returns the live manifest entries."""
    return load_manifest().entries
