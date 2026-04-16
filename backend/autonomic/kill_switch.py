"""File-based kill switch for the autonomic subsystem.

A simple flag file (`knowledge/autonomic/ENABLED` by default) with content
`true` or `false`. If the file is missing or the content is unrecognised,
the switch reads as DISABLED (fail-safe).
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PATH = Path("knowledge/autonomic/ENABLED")


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
