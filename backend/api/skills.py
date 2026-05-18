"""Skills management — list / view / edit / enable-disable / install.

Skills are markdown plugins the agent picks up at runtime (see
`backend/skills.py`). Two tiers:
  - builtin: ships with the engine repo
  - user:    installed by the owner into `~/.hrant/data/skills/`,
             survives `hrant update`

WebUI Settings → Skills uses this router.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..roles import current_speaker, is_owner
from ..skills import SKILLS

router = APIRouter()


# --- List / get / update / delete --------------------------------------


def _skill_to_dict(sk) -> dict:
    return {
        "name": sk.name,
        "description": sk.description,
        "triggers": sk.triggers,
        "when_to_use": sk.when_to_use,
        "body": sk.body,
        "source": sk.source,
        "enabled": sk.enabled,
        "path": str(sk.path),
    }


def _read_skill_md(name: str) -> str:
    sk = SKILLS.get(name)
    if sk is None:
        raise HTTPException(404, f"skill not found: {name}")
    # Skill.path is the skill's DIRECTORY (see _parse_skill_md), not
    # the SKILL.md file itself. Reading the directory directly raises
    # IsADirectoryError → bubbled as a 500 with no helpful detail in
    # the WebUI. Resolve the actual file before reading.
    skill_dir = Path(sk.path)
    md_path = skill_dir / "SKILL.md" if skill_dir.is_dir() else skill_dir
    try:
        return md_path.read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(500, f"could not read SKILL.md ({md_path}): {e}")


def _require_owner_for_writes() -> None:
    """Mutations to skills are owner-only — adding a skill runs its
    handler.py inside the agent process (arbitrary Python). The
    WebUI's local user is implicit owner; if the role lookup
    indicates otherwise (e.g. a future multi-WebUI-user setup),
    refuse."""
    sp = current_speaker()
    if sp is None:
        # Pre-Phase-11 / non-request contexts (CLI, tests) — allow.
        return
    if not is_owner(sp):
        raise HTTPException(403, "skills mutations require owner role")


@router.get("/api/skills")
def list_skills():
    """All skills (built-in + user, enabled + disabled) with full
    metadata. Used by the Skills settings tab."""
    return {
        "skills": [_skill_to_dict(s) for s in SKILLS.list()],
        "user_dir": str(SKILLS.user_dir),
    }


@router.get("/api/skills/{name}")
def get_skill(name: str):
    sk = SKILLS.get(name)
    if sk is None:
        raise HTTPException(404, f"skill not found: {name}")
    return {
        **_skill_to_dict(sk),
        "raw_md": _read_skill_md(name),
    }


class SkillUpdate(BaseModel):
    """Full SKILL.md text. Caller has already constructed the
    frontmatter + body; the manager just writes it as-is and
    re-parses."""
    raw_md: str


@router.put("/api/skills/{name}")
def upsert_skill(name: str, body: SkillUpdate):
    """Create or update a user-tier skill. If `name` matches a
    built-in skill, the user-tier override is written (built-in
    file untouched in the engine repo)."""
    _require_owner_for_writes()
    try:
        sk = SKILLS.upsert_user_skill(name, body.raw_md)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, **_skill_to_dict(sk)}


@router.delete("/api/skills/{name}")
def delete_skill(name: str):
    """Remove a user-tier skill from disk. Built-in skills aren't
    deletable (the next `hrant update` would re-ship them anyway).
    Returns 404 when there's no user-tier file with that name."""
    _require_owner_for_writes()
    if not SKILLS.delete_user_skill(name):
        raise HTTPException(404, "no user skill with that name")
    return {"ok": True, "deleted": name}


class SkillEnableUpdate(BaseModel):
    enabled: bool


@router.post("/api/skills/{name}/enabled")
def set_enabled(name: str, body: SkillEnableUpdate):
    """Soft enable/disable — writes to skills_disabled.json without
    touching the skill's own files. Built-in and user-tier skills
    can both be disabled."""
    _require_owner_for_writes()
    sk = SKILLS.set_enabled(name, body.enabled)
    if sk is None:
        raise HTTPException(404, f"skill not found: {name}")
    return {"ok": True, **_skill_to_dict(sk)}


@router.post("/api/skills/reload")
def reload_skills():
    """Force a re-scan from disk (used after an external file edit
    or post-install)."""
    SKILLS.reload()
    return {"ok": True, "count": len(SKILLS.list())}


# --- Install from external source --------------------------------------


class SkillInstallRequest(BaseModel):
    """Source types:
      - "git":   `source` is a clone URL, optional `subdir` if the
                  skill lives in a sub-directory of the repo
      - "zip":   `source` is an HTTP URL to a .zip file
      - "local": `source` is an absolute path on the server's disk
                 (copy the whole directory into user skills)
    """
    source_type: str  # git | zip | local
    source: str
    name: Optional[str] = None       # override the on-disk name
    subdir: Optional[str] = None     # for git/zip nested layouts


def _slug(name: str) -> str:
    clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")
    return clean or "skill"


def _install_from_local(src: str, target_dir: Path) -> None:
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise HTTPException(400, f"local source not found: {src_path}")
    if not src_path.is_dir():
        raise HTTPException(400, "local source must be a directory")
    if not (src_path / "SKILL.md").exists():
        raise HTTPException(400, "local source has no SKILL.md")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(src_path, target_dir)


def _install_from_git(url: str, target_dir: Path, subdir: Optional[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="hrant_skill_") as tmp:
        clone_target = Path(tmp) / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", url, str(clone_target)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except FileNotFoundError:
            raise HTTPException(500, "git not installed on this machine")
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                400,
                f"git clone failed: {(e.stderr or e.stdout or '').strip()[:300]}",
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(400, "git clone timed out (120s)")
        skill_root = clone_target / subdir if subdir else clone_target
        if not (skill_root / "SKILL.md").exists():
            raise HTTPException(
                400,
                f"repo has no SKILL.md at {subdir or '/'} — pass a `subdir`?",
            )
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(skill_root, target_dir)


def _install_from_zip(url: str, target_dir: Path, subdir: Optional[str]) -> None:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(400, "zip source must be an http(s) URL")
    with tempfile.TemporaryDirectory(prefix="hrant_skill_") as tmp:
        zip_path = Path(tmp) / "skill.zip"
        try:
            with urlopen(url, timeout=120) as resp:
                zip_path.write_bytes(resp.read())
        except Exception as e:
            raise HTTPException(400, f"zip download failed: {e}")
        extract_dir = Path(tmp) / "extracted"
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Refuse zips with absolute/parent-traversal paths.
                for member in zf.namelist():
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise HTTPException(400, f"unsafe zip entry: {member}")
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(400, "downloaded file is not a valid zip")
        skill_root = extract_dir / subdir if subdir else extract_dir
        # Auto-flatten if the zip has a single top-level dir.
        if not (skill_root / "SKILL.md").exists():
            entries = list(extract_dir.iterdir()) if extract_dir.exists() else []
            if len(entries) == 1 and entries[0].is_dir():
                skill_root = entries[0]
        if not (skill_root / "SKILL.md").exists():
            raise HTTPException(
                400,
                f"zip has no SKILL.md at {subdir or 'top level'}",
            )
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(skill_root, target_dir)


@router.post("/api/skills/install")
def install_skill(body: SkillInstallRequest):
    """Install a skill from git / zip / local path into the user
    skills dir. Owner-only — the new skill's `handler.py` runs
    inside the agent process at next load. Warn the user before
    confirming in the UI.

    Idempotent in the sense that re-installing replaces the existing
    user-tier skill of the same name. Built-in skills with the same
    name are left in the engine repo; the user-tier override wins
    at load time.
    """
    _require_owner_for_writes()

    # Derive on-disk name.
    if body.name:
        name = _slug(body.name)
    elif body.source_type == "git":
        # Strip trailing .git / tail segment.
        tail = body.source.rstrip("/").split("/")[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        name = _slug(tail or "imported")
    elif body.source_type == "local":
        name = _slug(Path(body.source).name)
    else:
        # zip — derive from URL last segment.
        tail = body.source.rstrip("/").split("/")[-1].split("?", 1)[0]
        if tail.endswith(".zip"):
            tail = tail[:-4]
        name = _slug(tail or "imported")

    target = SKILLS.user_dir / name
    if body.source_type == "git":
        _install_from_git(body.source, target, body.subdir)
    elif body.source_type == "zip":
        _install_from_zip(body.source, target, body.subdir)
    elif body.source_type == "local":
        _install_from_local(body.source, target)
    else:
        raise HTTPException(400, f"unknown source_type: {body.source_type!r}")

    SKILLS.reload()
    sk = SKILLS.get(name)
    if sk is None:
        # Shouldn't happen if we wrote correctly + SKILL.md parses.
        raise HTTPException(500, "skill installed but did not load — check logs")
    return {"ok": True, "name": name, "skill": _skill_to_dict(sk)}
