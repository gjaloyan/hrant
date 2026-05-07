"""Agent workspace — a real, browseable directory tree the agent can
read from and write to.

Layout under `workspace/` at the repo root:

    workspace/
      inbox/      ← files received from the user (mirrored from the
                    sha256-keyed attachments store, but here they keep
                    their ORIGINAL filename so `read_file` works on a
                    path the model already saw in the prompt).
      outbox/     ← files the agent prepared (drafts, generated reports,
                    output artifacts). Written via the
                    `save_to_workspace` tool.
      notes/      ← agent's freeform scratch notes (markdown / txt).

Why this exists: the AttachmentStore keys files by sha256
(`knowledge/attachments/<sha>.bin`), which is great for dedup but
useless to the model — when the user uploaded `contract.pdf`, the
LLM saw a generic "[ATTACHMENT NOTICE]" and tried `read_file
("contract.pdf")` against the cwd, getting a not-found. The
workspace mirror fixes that: `contract.pdf` lives at
`workspace/inbox/contract.pdf`, and the agent's prompt now lists
that path so `read_file` succeeds first try.

Cleanup is opt-in per subtree via `inbox_retention_days` /
`outbox_retention_days` / `notes_retention_days` in CONFIG. Setting
a retention to 0 disables the sweep for that tree (default for
outbox + notes; the agent's own work shouldn't vanish on a timer).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

# Subtree names — kept as a tuple so callers stay typo-resistant.
INBOX = "inbox"
OUTBOX = "outbox"
NOTES = "notes"
TURNS = "turns"
_SUBTREES = (INBOX, OUTBOX, NOTES, TURNS)

# Filename hygiene caps. 120 leaves headroom for collision suffixes
# without hitting the typical 255-byte filename limit on most
# filesystems and the 260-char path limit on older Windows.
_FILENAME_MAX_LEN = 120
_OUTBOX_CONTENT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB safety cap on agent writes

# Tracks the wall-clock time of the last sweep so we don't run
# `os.walk` over the whole tree on every single attachment.
_SWEEP_INTERVAL_SECONDS = 24 * 3600


@dataclass
class WorkspaceFile:
    """Metadata returned to callers (mostly for tests / inbox listing)."""
    path: Path
    relative_path: str
    subtree: str
    size: int
    modified: float


class WorkspaceManager:
    """Owns the on-disk workspace tree.

    Singleton-ish: one instance per process, accessed via the module-level
    `WORKSPACE`. Tests build their own with a tmp_path root."""

    def __init__(self, root: Optional[Path] = None):
        if root is None:
            from .config import CONFIG, ROOT
            ws_cfg = CONFIG.workspace
            raw = ws_cfg.get("root", "./workspace")
            p = Path(raw)
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            root = p
        self.root: Path = root
        self._lock = threading.Lock()
        for sub in _SUBTREES:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        # Lazy sweep tracking — single timestamp file at the workspace root.
        self._sweep_marker = self.root / ".last_sweep"

    # ---------- safe filename ----------

    @staticmethod
    def safe_filename(name: str, *, sha: str = "", default: str = "untitled") -> str:
        """Produce a filesystem-safe filename from a user-supplied name.

        Strips path separators (so a malicious "../etc/passwd" can't escape
        the workspace), control chars, and surrogate pairs. Collapses
        whitespace. If the result is empty after sanitisation, falls back
        to `default + short-sha` so the file still has a stable handle.
        """
        name = name or ""
        # Drop directory components — only keep the leaf.
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        # NFC-normalise so visually-identical Unicode lands at the same path.
        name = unicodedata.normalize("NFC", name)
        # Replace control chars + the Windows-reserved set < > : " | ? *
        name = re.sub(r'[\x00-\x1f<>:"|?*]', "_", name)
        # Collapse whitespace runs to single space.
        name = re.sub(r"\s+", " ", name).strip()
        # Strip leading dots so we don't accidentally make hidden files.
        name = name.lstrip(".")
        if not name:
            short = (sha or "")[:8] or "anon"
            return f"{default}_{short}"
        if len(name) > _FILENAME_MAX_LEN:
            stem, _, ext = name.rpartition(".")
            if stem and len(ext) <= 8:
                # Preserve the extension when truncating — readers care.
                stem = stem[: _FILENAME_MAX_LEN - len(ext) - 1]
                name = f"{stem}.{ext}"
            else:
                name = name[:_FILENAME_MAX_LEN]
        return name

    # ---------- attachment mirroring ----------

    def mirror_attachment(
        self,
        *,
        sha: str,
        original_name: str,
        blob_path: Path,
        kind: str = "file",
        mime: str = "",
    ) -> Path:
        """Copy an attachment blob into `inbox/` under its original name.

        Idempotent on (filename, sha) pairs:
          - If `inbox/<safe_name>` already exists AND its bytes match the
            attachment's sha, return it (no rewrite).
          - If a file with the same safe name exists but a DIFFERENT sha,
            append a short-sha suffix so both files coexist.
          - If neither exists, write the file fresh.

        The `kind` and `mime` arguments are recorded as a sidecar
        `<filename>.meta.json` so the agent can read back what it was
        looking at without round-tripping through the attachment index.
        Image/audio mirrors are written as well — even though the model
        consumes them via the multimodal pipeline, the user may ask the
        agent to save / forward / re-attach them later, and a real path
        is the simplest handle.
        """
        if not blob_path.exists():
            raise FileNotFoundError(
                f"attachment blob missing for sha={sha}: {blob_path}"
            )
        with self._lock:
            self._sweep_if_due()
            inbox = self.root / INBOX
            inbox.mkdir(parents=True, exist_ok=True)
            safe_name = self.safe_filename(original_name, sha=sha, default=kind or "file")
            target = inbox / safe_name
            if target.exists():
                if self._sha_of(target) == sha:
                    self._write_meta(target, sha=sha, kind=kind, mime=mime, original_name=original_name)
                    return target
                # Same name, different bytes — disambiguate with a short sha suffix.
                stem = target.stem
                suffix = target.suffix
                short = (sha or "")[:8] or "dup"
                target = inbox / f"{stem}_{short}{suffix}"
                if target.exists() and self._sha_of(target) == sha:
                    self._write_meta(target, sha=sha, kind=kind, mime=mime, original_name=original_name)
                    return target
            shutil.copyfile(blob_path, target)
            self._write_meta(target, sha=sha, kind=kind, mime=mime, original_name=original_name)
            return target

    @staticmethod
    def _sha_of(p: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _write_meta(
        target: Path, *, sha: str, kind: str, mime: str, original_name: str
    ) -> None:
        meta = {
            "sha256": sha,
            "kind": kind,
            "mime": mime,
            "original_name": original_name,
            "mirrored_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        meta_path = target.with_name(target.name + ".meta.json")
        try:
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.warning("workspace: meta sidecar write failed for %s: %s", target, e)

    # ---------- per-turn persistence ----------

    def save_turn(self, turn_id: str, data: dict) -> Path:
        """Persist a turn's structured record under `workspace/turns/<id>.json`.

        P1: every Agent.run successful turn writes a TurnWorkspace
        artifact here — original task, claims, evidence, tool-call
        order, verification result, token usage by stage. The runtime
        tool-loop is unchanged (no in-flight artifact references — the
        risk profile of touching all 5 provider loops outweighed the
        token saving over Round 11's curated synthesis). What this
        gives us is a stable on-disk record per turn that the WebUI
        can query, a future evaluator can replay, and a debugger can
        inspect days later when the user asks 'what did the agent
        actually see when it answered X?'.

        Caller-supplied `turn_id` should be a stable, sortable string
        (typically a UTC timestamp + short hash). We don't generate
        one here so multiple call sites that want to refer to the
        same record from different angles can.

        Returns the absolute path written.
        """
        with self._lock:
            self._sweep_if_due()
            target_dir = self.root / TURNS
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_id = self.safe_filename(turn_id, default="turn")
            if not safe_id.endswith(".json"):
                safe_id = safe_id + ".json"
            target = target_dir / safe_id
            try:
                target.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                log.warning("workspace.save_turn: write failed for %s: %s", target, e)
            return target

    # ---------- agent-driven writes ----------

    def save_outbox(
        self, filename: str, content: str | bytes, *, subdir: str = OUTBOX,
        overwrite: bool = False,
    ) -> Path:
        """Write `content` into `<subdir>/<safe_name>`. Returns the path.

        `subdir` must be one of `outbox` / `notes`. Inbox is read-only as
        far as the agent is concerned (mirrors come from the attachment
        pipeline, not direct writes) — guarding this here means a bug in
        the tool can't clobber a user-uploaded file.
        """
        if subdir not in (OUTBOX, NOTES):
            raise ValueError(f"workspace.save_outbox: subdir must be outbox|notes, got {subdir!r}")
        with self._lock:
            self._sweep_if_due()
            target_dir = self.root / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_name = self.safe_filename(filename, default=subdir)
            target = target_dir / safe_name
            if target.exists() and not overwrite:
                # Don't silently clobber. Suffix with timestamp so both versions
                # survive — agent can compare or pick one.
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                stem = target.stem
                suffix = target.suffix
                target = target_dir / f"{stem}_{stamp}{suffix}"
            if isinstance(content, str):
                data = content.encode("utf-8")
            else:
                data = bytes(content)
            if len(data) > _OUTBOX_CONTENT_MAX_BYTES:
                raise ValueError(
                    f"workspace.save_outbox: content too large "
                    f"({len(data)} > {_OUTBOX_CONTENT_MAX_BYTES} bytes cap)"
                )
            target.write_bytes(data)
            return target

    # ---------- listing ----------

    def list_subtree(self, subtree: str) -> list[WorkspaceFile]:
        """Return non-meta files in a subtree, newest first."""
        if subtree not in _SUBTREES:
            raise ValueError(f"unknown workspace subtree: {subtree}")
        d = self.root / subtree
        if not d.exists():
            return []
        items: list[WorkspaceFile] = []
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.name.endswith(".meta.json"):
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            items.append(WorkspaceFile(
                path=p,
                relative_path=str(p.relative_to(self.root.parent)).replace("\\", "/")
                    if self.root.parent in p.parents
                    else str(p),
                subtree=subtree,
                size=stat.st_size,
                modified=stat.st_mtime,
            ))
        items.sort(key=lambda f: f.modified, reverse=True)
        return items

    def relative_to_repo(self, p: Path) -> str:
        """Render a workspace path as something useful for the prompt.
        Falls back to the absolute path if the file is outside the repo
        (e.g. tests using a tmp_path workspace)."""
        try:
            return str(p.relative_to(self.root.parent)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    # ---------- cleanup ----------

    def sweep_old(
        self,
        *,
        inbox_retention_days: int,
        outbox_retention_days: int,
        notes_retention_days: int,
        turns_retention_days: int = 0,
    ) -> dict[str, int]:
        """Delete files older than the retention cutoff per subtree.

        A retention of 0 (or negative) disables the sweep for that subtree
        — useful default for outbox / notes where the agent's own work
        shouldn't be auto-deleted. Only files (and their meta sidecars)
        are removed; the subtree directory itself stays.
        """
        cutoffs: dict[str, Optional[float]] = {
            INBOX: self._cutoff(inbox_retention_days),
            OUTBOX: self._cutoff(outbox_retention_days),
            NOTES: self._cutoff(notes_retention_days),
            TURNS: self._cutoff(turns_retention_days),
        }
        deleted = {INBOX: 0, OUTBOX: 0, NOTES: 0, TURNS: 0}
        for subtree, cutoff in cutoffs.items():
            if cutoff is None:
                continue
            d = self.root / subtree
            if not d.exists():
                continue
            for p in list(d.iterdir()):
                if not p.is_file():
                    continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if mtime >= cutoff:
                    continue
                # Drop the file and its meta sidecar (if any).
                try:
                    p.unlink()
                    deleted[subtree] += 1
                except OSError as e:
                    log.warning("workspace.sweep_old: failed to remove %s: %s", p, e)
                    continue
                meta = p.with_name(p.name + ".meta.json")
                if meta.exists():
                    try:
                        meta.unlink()
                    except OSError:
                        pass
        return deleted

    @staticmethod
    def _cutoff(retention_days: int) -> Optional[float]:
        if not retention_days or retention_days <= 0:
            return None
        return (datetime.utcnow() - timedelta(days=int(retention_days))).timestamp()

    def _sweep_if_due(self) -> None:
        """Opportunistic, rate-limited sweep. Runs at most once per
        `_SWEEP_INTERVAL_SECONDS` per process. Called from
        `mirror_attachment` and `save_outbox` so cleanup happens in the
        normal flow without a separate cron."""
        try:
            from .config import CONFIG
            ws = CONFIG.workspace
        except Exception:
            return
        try:
            if self._sweep_marker.exists():
                last = self._sweep_marker.stat().st_mtime
                if (time.time() - last) < _SWEEP_INTERVAL_SECONDS:
                    return
        except OSError:
            pass
        try:
            self.sweep_old(
                inbox_retention_days=int(ws.get("inbox_retention_days", 90) or 0),
                outbox_retention_days=int(ws.get("outbox_retention_days", 0) or 0),
                notes_retention_days=int(ws.get("notes_retention_days", 0) or 0),
                turns_retention_days=int(ws.get("turns_retention_days", 30) or 0),
            )
        except Exception as e:
            log.warning("workspace: opportunistic sweep failed: %s", e)
            return
        try:
            self._sweep_marker.touch()
        except OSError:
            pass


# Module-level singleton. Constructed lazily so importing this module
# from a test before patching CONFIG doesn't immediately materialise the
# real workspace path.
_WORKSPACE_INSTANCE: Optional[WorkspaceManager] = None


def get_workspace() -> WorkspaceManager:
    global _WORKSPACE_INSTANCE
    if _WORKSPACE_INSTANCE is None:
        _WORKSPACE_INSTANCE = WorkspaceManager()
    return _WORKSPACE_INSTANCE


def _reset_for_tests(root: Optional[Path] = None) -> WorkspaceManager:
    """Test helper — replaces the module singleton with a fresh instance
    rooted at `root`. NOT for production use."""
    global _WORKSPACE_INSTANCE
    _WORKSPACE_INSTANCE = WorkspaceManager(root=root) if root else None
    return _WORKSPACE_INSTANCE if root else get_workspace()
