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
    # Phase 12B: which tier this skill came from. "builtin" = ships
    # with the engine repo (backend/skills/), "user" = installed by
    # the owner into data_dir (~/.hrant/data/skills/). The latter
    # survives `hrant update`.
    source: str = "builtin"
    enabled: bool = True

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


def _user_skills_dir() -> Path:
    """Per-install user skills, separate from the engine repo so they
    survive `hrant update`. Sits inside data_dir like every other piece
    of persisted user state."""
    from . import paths as _paths
    return _paths.data_dir(require=False) / "skills"


def _disabled_path() -> Path:
    """Soft kill-switch list: skill names that exist on disk but the
    user has marked disabled in the WebUI. Stored separately so a
    disable-then-re-enable round-trip doesn't risk corrupting the
    skill's own files."""
    from . import paths as _paths
    return _paths.data_dir(require=False) / "skills_disabled.json"


def _load_disabled() -> set[str]:
    import json
    p = _disabled_path()
    if not p.exists():
        return set()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return set(raw.get("disabled") or [])
    except Exception:
        return set()


def _save_disabled(disabled: set[str]) -> None:
    import json
    p = _disabled_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"disabled": sorted(disabled)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class SkillsManager:
    """Two-tier discovery:

    - Built-in skills live in `backend/skills/` (engine repo, ships
      with `hrant update`, read-only at runtime).
    - User skills live in `~/.hrant/data/skills/` (per-install,
      writable from the WebUI, preserved across engine updates).

    When the same `name` appears in both tiers the USER tier wins —
    that's how an owner can override a built-in's body without
    touching the engine repo. Disabled skills (in `skills_disabled.json`)
    are still loaded into the manifest but `enabled=False` and they
    DO NOT contribute their triggers / handlers / system blocks.
    """

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        user_skills_dir: Optional[Path] = None,
    ):
        self.dir = Path(
            skills_dir or Path(__file__).resolve().parent / "skills"
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        # User dir resolved lazily — paths.data_dir() can raise pre-init
        # but skills aren't needed before the first turn anyway.
        self._user_dir_override = user_skills_dir
        self.skills: list[Skill] = []
        self._loaded = False

    @property
    def user_dir(self) -> Path:
        return self._user_dir_override or _user_skills_dir()

    def _scan_dir(
        self,
        d: Path,
        source: str,
        disabled: set[str],
        registry: ToolRegistry,
    ) -> list[Skill]:
        if not d.exists():
            return []
        out: list[Skill] = []
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            sk = _parse_skill_md(skill_md)
            if sk is None:
                continue
            sk.source = source
            sk.enabled = sk.name not in disabled
            out.append(sk)
            # Only register handlers for ENABLED skills. A disabled
            # skill's handler.py won't add its tools to the registry.
            if sk.enabled:
                _load_handler(child, registry)
        return out

    def load(self, registry: Optional[ToolRegistry] = None) -> list[Skill]:
        """Scan both built-in and user dirs. User overrides built-in
        by name. Idempotent — safe to re-call after edits."""
        registry = registry or get_registry()
        disabled = _load_disabled()
        builtin = self._scan_dir(self.dir, "builtin", disabled, registry)
        try:
            user = self._scan_dir(self.user_dir, "user", disabled, registry)
        except Exception:
            # data_dir may not be available in tests pre-init.
            user = []
        by_name: dict[str, Skill] = {s.name: s for s in builtin}
        # User overrides builtin.
        for s in user:
            by_name[s.name] = s
        self.skills = sorted(by_name.values(), key=lambda s: s.name)
        self._loaded = True
        return self.skills

    def reload(self, registry: Optional[ToolRegistry] = None) -> list[Skill]:
        """Force re-scan even if already loaded. Called by the WebUI
        after an edit / install / enable-toggle."""
        self._loaded = False
        return self.load(registry)

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def list(self) -> list[Skill]:
        self.ensure_loaded()
        return list(self.skills)

    def match(self, text: str) -> list[Skill]:
        """Triggers from ENABLED skills only."""
        self.ensure_loaded()
        return [s for s in self.skills if s.enabled and s.matches(text)]

    def get(self, name: str) -> Optional[Skill]:
        self.ensure_loaded()
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def catalog_block(self) -> str:
        """Catalog of ENABLED skills only."""
        self.ensure_loaded()
        active = [s for s in self.skills if s.enabled]
        if not active:
            return ""
        lines = [
            "# AVAILABLE SKILLS",
            "(Detailed instructions activate automatically on trigger match.)",
        ]
        for s in active:
            triggers = ", ".join(s.triggers) if s.triggers else "—"
            lines.append(f"- **{s.name}**: {s.description}  _(triggers: {triggers})_")
        return "\n".join(lines)

    # ---- write paths (used by the API) ----

    def set_enabled(self, name: str, enabled: bool) -> Optional[Skill]:
        """Toggle the soft kill-switch. Returns the updated skill or
        None if the name doesn't exist."""
        self.ensure_loaded()
        existing = self.get(name)
        if existing is None:
            return None
        disabled = _load_disabled()
        if enabled:
            disabled.discard(name)
        else:
            disabled.add(name)
        _save_disabled(disabled)
        self.reload()
        return self.get(name)

    def upsert_user_skill(self, name: str, content: str) -> Skill:
        """Create or update a user skill from raw SKILL.md content.
        Owner-only path — caller enforces. Writes to data_dir; never
        touches the engine repo."""
        clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")
        if not clean:
            raise ValueError("skill name must contain at least one alphanumeric")
        target = self.user_dir / clean / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.reload()
        sk = self.get(clean)
        if sk is None:
            raise ValueError("written file did not parse as a valid skill")
        return sk

    def delete_user_skill(self, name: str) -> bool:
        """Remove a user-tier skill from disk. Built-in skills are
        NOT deletable from the WebUI (would be re-shipped by the
        next `hrant update` anyway). Returns True if removed."""
        target = self.user_dir / name
        if not target.exists():
            return False
        import shutil
        shutil.rmtree(target)
        self.reload()
        return True


SKILLS = SkillsManager()
