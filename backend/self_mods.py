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

  5. THERE IS NO REAPPLY. This spot used to describe a `reapply_all()`
     called by `backend/updater.py` after a successful `git pull`, running
     `git apply --3way` per patch and marking conflicts "needs_review".
     Corrected 2026-08-09: no such function exists anywhere in the repo,
     the updater never called it, and no code path ever assigns
     "needs_review" — so the whole conflict-handling story below was
     describing a mechanism that was never built.

     What `hrant update` ACTUALLY does (updater.py:81) is ARCHIVE every
     active self-mod into data_dir/self_mods/history/<ts>/ and let the pull
     overwrite the tree. An applied self-modification therefore does NOT
     survive an update: it must be ported into git or it is gone. That is
     the real contract, and it is the opposite of what this docstring
     promised.

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
    # "needs_review" is UNREACHABLE — nothing assigns it (2026-08-09 audit).
    # Kept in the type only so old manifests on disk still deserialise.
    status: str                # "applied" | "reverted" | (legacy: needs_review)
    patch_filename: str        # NNNN-<slug>.patch — sortable
    last_error: str = ""       # legacy field; nothing populates it today


@dataclass
class Manifest:
    """Wraps applied.json. Order matters — patches are applied in
    creation order so a later patch can build on an earlier one."""

    entries: list[PatchEntry] = field(default_factory=list)


def _self_mods_dir() -> Path:
    # Use require=False so an early read on a fresh box doesn't
    # raise — self_mods/ is lazily materialised when a patch lands.
    return paths.data_dir(require=False) / "self_mods"


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
        # The reverse failed. Either the change is already gone — someone
        # edited the file, or the same content arrived from upstream in a
        # different shape — or the patch genuinely conflicts with work
        # built on top of it. Only the first is safe to treat as done.
        #
        # Without this branch a superseded patch cannot be removed at all:
        # revert refuses because there is nothing to reverse, and the
        # patch stays active and RE-APPLIES on the next start. Measured
        # 2026-09-02: three lesson patches kept reinstating rules that had
        # been consolidated into git under different wording, so every
        # restart put the duplicates back.
        if _git_apply_check(patch_text)[0]:
            m.entries.pop(target_idx)
            save_manifest(m)
            try:
                patch_path.unlink()
            except OSError:
                pass
            log.info("self-mod retired (change already absent): %s",
                     target.patch_filename)
            return True, (
                f"patch {patch_id} was no longer applied — the change is "
                "not in the tree, so it was retired rather than reversed"
            )
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


def _history_dir() -> Path:
    return _self_mods_dir() / "history"


def patch_state(patch_path: Path) -> str:
    """Is this patch already upstream, still applicable, or in conflict?

    Returns "upstream" | "applies" | "conflicts".

    `git apply --check --reverse` succeeding means the content the patch
    adds is ALREADY in the tree: it could be undone, so it is present.
    That is the precise question to ask after a pull — did the change get
    into the repository by other means?

    Used to stop `hrant update` from throwing away work that is not in git
    yet. An agent fix that survives only until the next deploy is not a fix
    the agent made; on 2026-08-31 one survived solely because a human
    noticed and committed it by hand.
    """
    if not patch_path.exists():
        return "conflicts"
    try:
        rev = _git("apply", "--check", "--reverse", str(patch_path),
                   check=False)
        if rev.returncode == 0:
            return "upstream"
        fwd = _git("apply", "--check", str(patch_path), check=False)
        return "applies" if fwd.returncode == 0 else "conflicts"
    except Exception as e:
        log.warning("patch_state failed for %s: %s", patch_path, e)
        return "conflicts"


def reapply_patch(patch_path: Path) -> bool:
    """Put a still-needed patch back after a pull. False on any failure."""
    try:
        return _git("apply", str(patch_path), check=False).returncode == 0
    except Exception as e:
        log.warning("reapply failed for %s: %s", patch_path, e)
        return False


def patch_targets(patch_text: str) -> set[str]:
    """The repo-relative files a unified diff touches.

    Reads the `+++ b/<path>` side, because that is the file as it will
    exist; `/dev/null` there means a deletion, and the `---` side names
    it instead. Some writers append a tab and a timestamp to the path.
    """
    out: set[str] = set()
    for line in (patch_text or "").splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].split("	")[0].strip()
            if path in ("/dev/null", ""):
                continue
            if path[:2] in ("a/", "b/"):
                path = path[2:]
            out.add(path)
    return out


def active_patch_targets() -> set[str]:
    """Every file the currently applied self-mods touch.

    `hrant update` needs this to tell "a self-mod is applied here", which
    is the normal state of the tree, from hand-written work in progress,
    which must keep blocking the update. Best effort: an unreadable patch
    contributes nothing, so a broken file makes the gate stricter rather
    than looser.
    """
    out: set[str] = set()
    try:
        entries = load_manifest().entries
    except Exception as exc:
        log.warning("self-mod manifest unreadable (%s); no targets", exc)
        return out
    for entry in entries:
        if entry.status not in ("applied", "needs_review"):
            continue
        try:
            text = (_self_mods_dir() / entry.patch_filename).read_text(
                encoding="utf-8")
        except Exception:
            continue
        out |= patch_targets(text)
    return out


def archive_all_active() -> dict:
    """Move every active patch into a timestamped history bundle and
    clear the active manifest. Called by `backend/updater.py` right
    after a successful `git pull` so the engine comes out at exactly
    `origin/master`. The user re-applies what they want via the
    Settings → Self-Modifications → History panel.

    Side effects on `~/.hrant/data/self_mods/`:
      - moves every `*.patch` into `history/<timestamp>/`
      - copies the manifest snapshot into `history/<timestamp>/manifest.json`
      - rewrites the top-level `applied.json` as `{"entries": []}`

    Returns a summary:
      {"archive_id": "2026-05-13T15-22-08Z",
       "archived_count": 3,
       "archive_path": "<full path>"}

    `archive_id` is the directory name — used by the Restore API
    to address one specific archive entry. Empty active manifest
    is a no-op (returns archived_count=0, archive_id=None).
    """
    m = load_manifest()
    active = [e for e in m.entries if e.status in ("applied", "needs_review")]
    if not active:
        return {"archive_id": None, "archived_count": 0, "archive_path": "",
                "kept": [], "conflicted": []}

    # Sort the active set by what the pull actually did to it, instead of
    # archiving everything the way this used to.
    #
    #   upstream  — the change is in the repo now, so the patch is spent.
    #               Archive it: that is the healthy end of a self-mod's life.
    #   applies   — still needed and still fits. Keep it: throwing it away
    #               means the agent cannot fix anything that outlives one
    #               deploy, which is the whole reason its repairs kept
    #               evaporating.
    #   conflicts — the pull moved the same code. Archive and say so
    #               loudly; guessing a merge here would be worse than
    #               losing the patch.
    kept: list[str] = []
    conflicted: list[str] = []
    still_active: list = []
    for entry in list(active):
        state = patch_state(_self_mods_dir() / entry.patch_filename)
        if state == "upstream":
            continue
        if state == "applies" and reapply_patch(
                _self_mods_dir() / entry.patch_filename):
            kept.append(entry.id)
            still_active.append(entry)
            continue
        if state == "conflicts":
            conflicted.append(entry.id)
    active = [e for e in active if e not in still_active]
    if not active:
        # Everything was either superseded or carried forward.
        save_manifest(Manifest(entries=still_active))
        return {"archive_id": None, "archived_count": 0, "archive_path": "",
                "kept": kept, "conflicted": conflicted}

    # Timestamp safe for filesystem on every OS (no colons).
    archive_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    target = _history_dir() / archive_id
    target.mkdir(parents=True, exist_ok=True)

    for entry in active:
        src = _self_mods_dir() / entry.patch_filename
        if src.exists():
            try:
                src.replace(target / entry.patch_filename)
            except OSError as e:
                log.warning("could not archive %s: %s", entry.patch_filename, e)

    # Snapshot the active subset of the manifest (we keep `reverted`
    # entries out of the archive — they were already user-discarded).
    archived_entries = [asdict(e) for e in active]
    (target / "manifest.json").write_text(
        json.dumps(
            {"archive_id": archive_id, "entries": archived_entries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Keep the patches that were carried forward; everything else is now
    # in the archive. Reverted entries are flushed either way since they
    # no longer correspond to a patch file on disk.
    save_manifest(Manifest(entries=still_active))
    log.info("archived %d self-mod(s) to %s (kept %d, conflicted %d)",
             len(active), target, len(kept), len(conflicted))
    return {
        "archive_id": archive_id,
        "archived_count": len(active),
        "archive_path": str(target),
        "kept": kept,
        "conflicted": conflicted,
    }


@dataclass
class HistoryArchive:
    """One archived bundle. `entries` are the patches it contains
    (with their original IDs / titles / files so the UI can render
    them like the active list)."""
    archive_id: str
    archived_at: str           # parsed from archive_id (ISO 8601)
    patch_count: int
    entries: list[PatchEntry]


def list_history() -> list[HistoryArchive]:
    """Every archive bundle, newest first. Used by the History panel
    in Settings → Self-Modifications."""
    hd = _history_dir()
    if not hd.exists():
        return []
    out: list[HistoryArchive] = []
    for sub in sorted(hd.iterdir(), reverse=True):
        if not sub.is_dir():
            continue
        manifest_path = sub / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("archive %s manifest unreadable (%s); skipping", sub.name, e)
            continue
        entries_raw = raw.get("entries") or []
        entries: list[PatchEntry] = []
        for e in entries_raw:
            try:
                entries.append(PatchEntry(**e))
            except TypeError:
                continue
        # Reconstruct a readable timestamp — archive_id has hyphens
        # where the source ISO had colons; restore them.
        ts = sub.name
        readable = ts.replace("T", "T", 1)
        # Convert "2026-05-13T15-22-08Z" → "2026-05-13T15:22:08Z"
        # (only the time portion's hyphens get swapped back).
        if "T" in readable and readable.endswith("Z"):
            date_part, time_part = readable.split("T", 1)
            time_part = time_part.replace("-", ":", 2)
            readable = f"{date_part}T{time_part}"
        out.append(HistoryArchive(
            archive_id=ts,
            archived_at=readable,
            patch_count=len(entries),
            entries=entries,
        ))
    return out


def _read_archive_patch(archive_id: str, patch_filename: str) -> tuple[Optional[str], str]:
    """Locate and read a patch file inside a specific archive. Used
    by the restore endpoints. Returns (patch_text, error_msg)."""
    if "/" in archive_id or "\\" in archive_id or ".." in archive_id:
        return None, "invalid archive_id"
    if "/" in patch_filename or "\\" in patch_filename or ".." in patch_filename:
        return None, "invalid patch_filename"
    target = _history_dir() / archive_id / patch_filename
    if not target.exists():
        return None, f"patch not found: {archive_id}/{patch_filename}"
    return target.read_text(encoding="utf-8"), ""


def _read_archive_manifest(archive_id: str) -> tuple[Optional[dict], str]:
    if "/" in archive_id or "\\" in archive_id or ".." in archive_id:
        return None, "invalid archive_id"
    target = _history_dir() / archive_id / "manifest.json"
    if not target.exists():
        return None, f"archive not found: {archive_id}"
    try:
        return json.loads(target.read_text(encoding="utf-8")), ""
    except Exception as e:
        return None, f"archive manifest unreadable: {e}"


def restore_from_history(
    archive_id: str, patch_filename: str,
) -> tuple[Optional[PatchEntry], str]:
    """Re-apply ONE archived patch as a fresh active modification.

    The original `PatchEntry` is preserved as much as possible (same
    title, slug, file) but gets a new `id` and a new sequence number
    in `applied.json` — this is a fresh application, not the same
    historical event. Same git apply semantics as `record_and_apply`:
    on conflict the patch file is removed and no manifest entry
    survives, so the user can try a different historical entry.

    The archived patch file STAYS in history (re-applying doesn't
    consume the archive entry — the user may want to re-apply the
    same patch on multiple machines, or after another update).
    """
    patch_text, err = _read_archive_patch(archive_id, patch_filename)
    if patch_text is None:
        return None, err

    # Use the archived manifest entry to recover a reasonable title.
    manifest, _ = _read_archive_manifest(archive_id)
    title = patch_filename
    file_rel = ""
    if manifest:
        for e in manifest.get("entries") or []:
            if e.get("patch_filename") == patch_filename:
                title = e.get("title") or title
                file_rel = e.get("file") or ""
                break

    _ensure_dir()
    num = _next_patch_number()
    slug = _slugify(title)
    new_filename = f"{num:04d}-{slug}.patch"
    new_path = _self_mods_dir() / new_filename
    new_path.write_text(patch_text, encoding="utf-8")

    ok, apply_err = _git_apply(patch_text)
    if not ok:
        try:
            new_path.unlink()
        except OSError:
            pass
        return None, (
            f"git apply failed: {apply_err}. The engine has changed "
            "enough that this archived patch no longer applies cleanly."
        )

    entry = PatchEntry(
        id=uuid.uuid4().hex[:8],
        slug=slug,
        file=file_rel,
        title=title,
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        status="applied",
        patch_filename=new_filename,
    )
    m = load_manifest()
    m.entries.append(entry)
    save_manifest(m)
    log.info("restored from history %s/%s as %s", archive_id, patch_filename, new_filename)
    return entry, ""


def restore_archive_batch(archive_id: str) -> dict:
    """Re-apply ALL patches in one archive, in their original order.

    Stops at the first conflict (mirrors what the user would expect
    when one piece of a stack breaks). Returns:
      {
        "restored": [<new_id>, ...],
        "failed_at": "<patch_filename>" | None,
        "error": "<err message>" | None,
      }

    Restored entries are visible immediately in the active list; the
    archive is unchanged.
    """
    manifest, err = _read_archive_manifest(archive_id)
    if manifest is None:
        return {"restored": [], "failed_at": None, "error": err}

    restored: list[str] = []
    for raw in manifest.get("entries") or []:
        patch_filename = raw.get("patch_filename")
        if not patch_filename:
            continue
        entry, err = restore_from_history(archive_id, patch_filename)
        if entry is None:
            return {"restored": restored, "failed_at": patch_filename, "error": err}
        restored.append(entry.id)
    return {"restored": restored, "failed_at": None, "error": None}


def list_patches() -> list[PatchEntry]:
    """Public accessor for the UI. Returns the live manifest entries."""
    return load_manifest().entries
