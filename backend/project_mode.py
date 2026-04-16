"""Управление проектами: overview / decisions / issues / hardware."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from .knowledge_manager import KM, _slug


class ProjectManager:
    def __init__(self):
        self.base = KM.base / "projects"
        self.base.mkdir(parents=True, exist_ok=True)
        self.current_path = self.base / ".current"

    # ---------- current project state ----------
    @property
    def current(self) -> str | None:
        if self.current_path.exists():
            name = self.current_path.read_text(encoding="utf-8").strip()
            return name or None
        return None

    @current.setter
    def current(self, name: str | None) -> None:
        if name:
            self.current_path.write_text(name, encoding="utf-8")
        elif self.current_path.exists():
            self.current_path.unlink()

    # ---------- CRUD ----------
    def _dir(self, name: str) -> Path:
        d = self.base / _slug(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start(self, name: str) -> str:
        d = self._dir(name)
        overview = d / "overview.md"
        if not overview.exists():
            overview.write_text(
                f"# Проект: {name}\n\nСоздан: {datetime.now():%Y-%m-%d %H:%M}\n\n## Описание\n\n",
                encoding="utf-8",
            )
        for fname in ("decisions.md", "issues.md", "hardware.md"):
            p = d / fname
            if not p.exists():
                p.write_text(f"# {fname[:-3].title()}\n", encoding="utf-8")
        self.current = name
        return f"✓ проект '{name}' активен"

    def end(self) -> str:
        name = self.current
        if not name:
            return "✗ нет активного проекта"
        self.current = None
        return f"✓ проект '{name}' закрыт (файлы сохранены в knowledge/projects/)"

    def _append(self, name: str, filename: str, text: str) -> None:
        d = self._dir(name)
        p = d / filename
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with p.open("a", encoding="utf-8") as f:
            f.write(f"\n- [{stamp}] {text}\n")

    def add_context(self, text: str) -> str:
        name = self.current
        if not name:
            return "✗ нет активного проекта"
        d = self._dir(name)
        p = d / "overview.md"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with p.open("a", encoding="utf-8") as f:
            f.write(f"\n- [{stamp}] {text}\n")
        return "✓ контекст добавлен"

    def add_decision(self, what: str, why: str) -> str:
        name = self.current
        if not name:
            return "✗ нет активного проекта"
        self._append(name, "decisions.md", f"**{what}** — {why}")
        return "✓ решение записано"

    def add_issue(self, problem: str, fix: str) -> str:
        name = self.current
        if not name:
            return "✗ нет активного проекта"
        self._append(name, "issues.md", f"**{problem}** → {fix}")
        return "✓ проблема записана"

    def list_projects(self) -> list[str]:
        return [p.name for p in self.base.iterdir() if p.is_dir() and not p.name.startswith(".")]

    def read_overview(self, name: str) -> str:
        p = self._dir(name) / "overview.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""


PROJECTS = ProjectManager()
