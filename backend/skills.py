"""Skills system: декларативные плагины для агента.

Скилл — это директория `backend/skills/<name>/` с двумя возможными файлами:

  SKILL.md   (обязательно)  — фронтматтер + инструкции для LLM
  handler.py (опционально)  — Python-модуль с функцией `register(registry)`,
                              которая добавляет инструменты в ToolRegistry.

Структура SKILL.md:

  ---
  name: pdf_summary
  description: Read a local PDF and produce a structured summary.
  triggers: [pdf, summarize, документ, конспект]
  when_to_use: |
    User asks to summarize, extract, or analyze a local PDF/DOCX file.
  ---

  # PDF Summary
  Подробные инструкции для LLM в свободной форме: как пользоваться,
  какие шаги, какие инструменты звать, в каком формате отдавать ответ.

Скилл может регистрировать инструменты — тогда они становятся
доступны в обычном tool-use loop. Может и не регистрировать —
тогда это просто блок инструкций, который подмешивается в system prompt
при матче.
"""
from __future__ import annotations
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .tool_registry import ToolRegistry, get_registry


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    when_to_use: str = ""
    body: str = ""           # содержимое после frontmatter
    path: Path = field(default_factory=Path)

    def matches(self, text: str) -> bool:
        """Грубая эвристика: триггер встречается в тексте (case-insensitive)."""
        if not self.triggers:
            return False
        low = text.lower()
        return any(t.lower() in low for t in self.triggers)

    def system_block(self) -> str:
        """Блок, который подмешивается в system prompt при активации."""
        parts = [f"## SKILL: {self.name}", self.description.strip()]
        if self.when_to_use:
            parts.append(f"\n*When to use:* {self.when_to_use.strip()}")
        if self.body.strip():
            parts.append(f"\n{self.body.strip()}")
        return "\n".join(parts)


def _parse_skill_md(path: Path) -> Optional[Skill]:
    """Парсит SKILL.md. Возвращает None, если файл невалидный."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    header = text[3:end].strip()
    body = text[end + 4 :].strip()
    try:
        meta = yaml.safe_load(header) or {}
    except yaml.YAMLError:
        return None
    name = str(meta.get("name", path.parent.name)).strip()
    if not name:
        return None
    return Skill(
        name=name,
        description=str(meta.get("description", "")).strip(),
        triggers=[str(t) for t in (meta.get("triggers") or [])],
        when_to_use=str(meta.get("when_to_use", "")).strip(),
        body=body,
        path=path.parent,
    )


def _load_handler(skill_dir: Path, registry: ToolRegistry) -> None:
    """Если рядом с SKILL.md лежит handler.py с функцией register(registry) —
    вызываем её. Это даёт скиллу возможность добавить свои tools."""
    handler_path = skill_dir / "handler.py"
    if not handler_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        f"backend.skills._loaded.{skill_dir.name}", handler_path
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        # Не падаем — скилл просто не будет иметь инструментов.
        print(f"[skills] handler load error in {skill_dir.name}: {e}")
        return
    register = getattr(module, "register", None)
    if callable(register):
        try:
            register(registry)
        except Exception as e:
            print(f"[skills] handler register() failed in {skill_dir.name}: {e}")


class SkillsManager:
    def __init__(self, skills_dir: Optional[Path] = None):
        self.dir = Path(
            skills_dir or Path(__file__).resolve().parent / "skills"
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        self.skills: list[Skill] = []
        self._loaded = False

    def load(self, registry: Optional[ToolRegistry] = None) -> list[Skill]:
        """Сканирует директорию скиллов и регистрирует их инструменты.

        Идемпотентно — повторный вызов перезагружает список (полезно для тестов
        и горячей перезагрузки).
        """
        registry = registry or get_registry()
        skills: list[Skill] = []
        for child in sorted(self.dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            sk = _parse_skill_md(skill_md)
            if sk is None:
                continue
            skills.append(sk)
            _load_handler(child, registry)
        self.skills = skills
        self._loaded = True
        return skills

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def list(self) -> list[Skill]:
        self.ensure_loaded()
        return list(self.skills)

    def match(self, text: str) -> list[Skill]:
        """Возвращает скиллы, чьи триггеры встречаются в тексте."""
        self.ensure_loaded()
        return [s for s in self.skills if s.matches(text)]

    def catalog_block(self) -> str:
        """Краткий каталог всех доступных скиллов для подмешивания в system."""
        self.ensure_loaded()
        if not self.skills:
            return ""
        lines = ["# AVAILABLE SKILLS",
                 "(Подробности активируются автоматически по триггерам.)"]
        for s in self.skills:
            triggers = ", ".join(s.triggers) if s.triggers else "—"
            lines.append(f"- **{s.name}**: {s.description}  _(triggers: {triggers})_")
        return "\n".join(lines)


SKILLS = SkillsManager()
