"""File-based kill switch for the autonomic subsystem.

A simple flag file (default: `<data_dir>/knowledge/autonomic/ENABLED`) with
content `true` or `false`. If the file is missing or the content is
unrecognised, the switch reads as DISABLED (fail-safe).
"""
from __future__ import annotations

from pathlib import Path


def _default_enabled_path() -> Path:
    """Audit P1 #3 fix: anchor the kill-switch file under
    `paths.knowledge_dir()` (~/.hrant/data/knowledge/autonomic/ENABLED
    on prod), not cwd-relative `knowledge/...`.

    The legacy cwd-relative default resolved against the engine repo
    when systemd started the service from /home/hrant/hrant. After
    the rest of the autonomic defaults moved into data_dir (commit
    aca39070), this file was still being read from the engine repo,
    so the scheduler saw it as missing → considered itself DISABLED
    even though the ENABLED file existed in data_dir.
    """
    try:
        from ..paths import knowledge_dir
        return knowledge_dir() / "autonomic" / "ENABLED"
    except Exception:
        # paths module not initialised — fall back to the legacy
        # cwd-relative default so tests pre-init still work.
        return Path("knowledge/autonomic/ENABLED")


DEFAULT_PATH = _default_enabled_path()


class KillSwitch:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_PATH

    def is_enabled(self) -> bool:
        try:
            content = self._path.read_text(encoding="utf-8").strip().lower()
        except FileNotFoundError:
            return False
        return content == "true"

    def enable(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("true", encoding="utf-8")

    def disable(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("false", encoding="utf-8")
