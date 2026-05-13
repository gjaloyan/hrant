"""Fresh-install bootstrap: copy templates into data_dir.

Called by `hrant init` when the data directory looks empty. Idempotent
— re-running on an already-initialised data dir is a no-op (does NOT
overwrite the user's customisations).

Files copied from `knowledge_templates/` → `data_dir/knowledge/`:
  identity/identity.md
  identity/soul.md
  identity/user_profile.md
  core_memory.md
  goals.json
  autonomic/.gitkeep

Files copied from `repo_root/config.example.yaml` → `data_dir/config.yaml`.

Files this module does NOT touch:
  - `.env`              — written separately by the init wizard's
                          API-key Q&A.
  - `providers.json` / `channels.json` / `active_model.json` —
                          created lazily by their respective modules
                          when the user adds the first entry.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

from . import paths

log = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Returned to the wizard so it can print "copied X files" /
    "already initialised, nothing to do" / mixed messages."""

    data_dir: str
    fresh: bool
    copied_files: list[str]
    skipped_files: list[str]
    config_action: str  # "copied" | "exists" | "no_template"


def is_initialised() -> bool:
    """Heuristic: a data dir is "initialised" once the identity file
    exists. Other knowledge files appear lazily (when the user uses
    the corresponding feature), so identity is the most reliable
    marker that the wizard has already run here."""
    return (paths.knowledge_dir() / "identity" / "identity.md").exists()


def _copy_one(src, dst, copied: list[str], skipped: list[str]) -> None:
    """Copy a single file. Never overwrite an existing destination —
    the user might have customised it already."""
    if dst.exists():
        skipped.append(str(dst))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def bootstrap_data_dir(*, force: bool = False) -> BootstrapResult:
    """Materialise data_dir from the engine's templates.

    `force=False` (default): skip any file that already exists in
    data_dir. Safe to run on an already-initialised install.

    `force=True`: copy every template anyway, overwriting existing
    files. Reserved for `hrant init --reset` — not used by the
    normal wizard flow.
    """
    paths.ensure_data_dir()
    fresh = not is_initialised()
    copied: list[str] = []
    skipped: list[str] = []

    src_root = paths.templates_dir()
    dst_root = paths.knowledge_dir()
    if not src_root.exists():
        log.warning("templates dir missing at %s — engine repo broken?", src_root)
        return BootstrapResult(
            data_dir=str(paths.data_dir()),
            fresh=fresh,
            copied_files=[],
            skipped_files=[],
            config_action="no_template",
        )

    for src in src_root.rglob("*"):
        if src.is_dir():
            continue
        # Skip the templates README — it's documentation for the
        # repo, not content for the user's tree.
        if src.name == "README.md" and src.parent == src_root:
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if force and dst.exists():
            dst.unlink()
        _copy_one(src, dst, copied, skipped)

    # config.yaml: copied from repo_root/config.example.yaml, NOT
    # from templates_dir (kept at repo root for visibility).
    cfg_src = paths.repo_root() / "config.example.yaml"
    cfg_dst = paths.data_dir() / "config.yaml"
    if cfg_src.exists():
        if cfg_dst.exists() and not force:
            config_action = "exists"
        else:
            shutil.copy2(cfg_src, cfg_dst)
            config_action = "copied"
    else:
        config_action = "no_template"

    return BootstrapResult(
        data_dir=str(paths.data_dir()),
        fresh=fresh,
        copied_files=copied,
        skipped_files=skipped,
        config_action=config_action,
    )
