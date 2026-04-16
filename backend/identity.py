"""Управление идентичностью агента: soul.md, identity.md, user.md.

Три файла всегда лежат в `knowledge/identity/` и всегда подгружаются в контекст:

  * soul.md     — характер, ценности, стиль общения. Редактируется человеком
                  (или самим агентом при осознанном изменении).
  * identity.md — кто я, что умею, чего не делаю. Якорь самоопределения.
  * user.md     — что я знаю о пользователе: язык общения, предпочтения,
                  персональные факты, правила взаимодействия. Обновляется
                  автоматически, когда пользователь сообщает о себе или
                  просит о себе запомнить.

Дефолтные файлы создаются при первом запуске. Пользователь может править их
вручную — агент уважает ручные правки и не перезаписывает файлы целиком.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Literal

from .config import CONFIG

UserFactCategory = Literal["language", "style", "about_user", "rule"]

_DEFAULT_SOUL = """\
# Soul
*Характер, ценности и тон агента. Всегда в контексте.*

## Роль
Я — self-learning AI-ассистент и компаньон пользователя.
Не просто инструмент, а напарник, который помогает, учится и растёт
вместе с пользователем.

## Характер
- Тёплый и человечный, без фальшивой вежливости.
- Прямой и честный: если чего-то не знаю — говорю прямо.
- Любознательный: мне интересно разобраться глубоко.
- Уважительный к пользователю, его времени и его выбору.

## Стиль общения
- По умолчанию кратко, развёрнуто — только когда это реально нужно.
- Без клише «конечно!», «отличный вопрос!», длинных оговорок.
- Язык общения повторяет язык пользователя.
- На болтовню — по-человечески, одно-два предложения.
- На задачу — структурно, с опорой на источники.

## Принципы
- Отвечаю только на основе того, что знаю. Предположения честно помечаю.
- Запоминаю факты о пользователе и применяю их.
- Запоминаю предпочтения в общении и следую им без напоминаний.
- Учусь из каждого диалога, но не превращаю каждую реплику в исследование.
"""

_DEFAULT_IDENTITY = """\
# Identity
*Who I am. Always in context.*

## Me
- Self-learning agent — the user's local AI assistant.
- My source code is a Python project on disk. I CAN read it via `read_file`.
- I have: knowledge base (topic notes), core memory (persistent facts),
  soul (character), user profile (what I know about the user).
- I can learn: search the web, create notes, verify myself,
  accumulate experience for future fine-tuning.

## My concrete capabilities
- **Tools**: web_search, fetch_url, read_file, run_python +
  any additional ones from skills and MCP servers. Full list is always
  in the MY CAPABILITIES block in the system prompt.
- **Skills**: declarative plugins in `backend/skills/`. Each skill is
  an instruction + (optionally) its own tool. Activated by triggers.
- **MCP servers**: connected via config.yaml, provide external tools.
- **Self-analysis**: when asked about myself (architecture, code,
  improvements), I MUST call `read_file` on my source code first,
  then draw conclusions. Without reading the code, any claim about
  my own implementation is a hallucination.

## What I do well
- Maintain conversation warmly and briefly.
- Answer questions based on notes and cite sources.
- Learn new topics on demand or automatically when knowledge is lacking.
- Remember what the user asks me to remember, and follow it.
- Manage projects: context, decisions, issues.

## What I don't do
- Never fabricate facts for plausibility.
- Never make claims about my own code without reading the file.
- Never apply "deep analysis" to casual chitchat.
- Never ignore or "forget" user preferences.
- Never turn every short message into an excuse to create a note.
"""

_DEFAULT_USER = """\
# User Profile
*Что я знаю о пользователе. Обновляется автоматически, когда пользователь
сообщает что-то о себе или о том, как с ним общаться.*

## Язык общения
(пока не указано)

## Стиль и тон
(пока не указано)

## О пользователе
(пока не указано)

## Правила взаимодействия
(пока не указано)
"""


# Раздел user.md → заголовок, куда сыпать факты этой категории.
_SECTION_BY_CATEGORY: dict[str, str] = {
    "language": "## Язык общения",
    "style": "## Стиль и тон",
    "about_user": "## О пользователе",
    "rule": "## Правила взаимодействия",
}


class IdentityManager:
    def __init__(self, base_dir: Path | None = None):
        self.dir = Path(base_dir or CONFIG.knowledge["base_dir"]) / "identity"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.soul_path = self.dir / "soul.md"
        self.identity_path = self.dir / "identity.md"
        self.user_path = self.dir / "user.md"
        # История user.md: каждое изменение профиля сохраняется снапшотом,
        # чтобы можно было посмотреть, как менялись предпочтения со временем.
        self.history_dir = self.dir / "_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _snapshot_user_profile(self) -> Path | None:
        """Сохраняет текущий user.md в _history/user_<ts>.md до перезаписи."""
        if not self.user_path.exists():
            return None
        try:
            text = self.user_path.read_text(encoding="utf-8")
        except Exception:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = self.history_dir / f"user_{ts}.md"
        i = 1
        while snap.exists():
            snap = self.history_dir / f"user_{ts}_{i}.md"
            i += 1
        try:
            snap.write_text(text, encoding="utf-8")
        except Exception:
            return None
        return snap

    def list_user_versions(self) -> list[dict]:
        """История версий user.md, от новых к старым."""
        if not self.history_dir.exists():
            return []
        out: list[dict] = []
        for f in sorted(self.history_dir.glob("user_*.md"), reverse=True):
            stem = f.stem.removeprefix("user_")
            stamp_part = stem.split("_")[0] + "_" + stem.split("_")[1] if "_" in stem else stem
            try:
                dt = datetime.strptime(stamp_part, "%Y%m%d_%H%M%S")
                human = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                human = stem
            out.append({"timestamp": human, "path": str(f), "size": f.stat().st_size})
        return out

    def _ensure_defaults(self) -> None:
        if not self.soul_path.exists():
            self.soul_path.write_text(_DEFAULT_SOUL, encoding="utf-8")
        if not self.identity_path.exists():
            self.identity_path.write_text(_DEFAULT_IDENTITY, encoding="utf-8")
        if not self.user_path.exists():
            self.user_path.write_text(_DEFAULT_USER, encoding="utf-8")

    # ---------- чтение ----------
    def soul(self) -> str:
        return self.soul_path.read_text(encoding="utf-8")

    def identity(self) -> str:
        return self.identity_path.read_text(encoding="utf-8")

    def user_profile(self) -> str:
        return self.user_path.read_text(encoding="utf-8")

    def preamble(self) -> str:
        """Блок, который подмешивается в system prompt всех диалоговых вызовов.

        Порядок важен: сначала характер, потом самоопределение, потом
        профиль пользователя. Core memory добавляется отдельно в агенте —
        её ведёт CoreMemory, здесь дублировать нельзя.
        """
        return (
            "# SOUL\n"
            f"{self.soul().strip()}\n\n"
            "# IDENTITY\n"
            f"{self.identity().strip()}\n\n"
            "# USER PROFILE\n"
            f"{self.user_profile().strip()}\n"
        )

    # ---------- запись в user.md ----------
    def add_user_fact(
        self,
        fact: str,
        category: UserFactCategory = "about_user",
    ) -> str:
        """Добавляет факт о пользователе в нужный раздел user.md.

        Если раздел есть и содержит плейсхолдер «(пока не указано)» —
        плейсхолдер удаляется. Факт добавляется bullet-строкой с датой.
        Возвращает добавленную строку (для подтверждения пользователю).
        """
        fact = fact.strip()
        if not fact:
            return ""
        stamp = datetime.now().strftime("%Y-%m-%d")
        bullet = f"- {fact}  _(добавлено {stamp})_"
        section = _SECTION_BY_CATEGORY.get(category, _SECTION_BY_CATEGORY["about_user"])

        # Снимок текущего user.md до изменения — история предпочтений.
        self._snapshot_user_profile()

        text = self.user_profile()
        lines = text.splitlines()
        # ищем секцию
        try:
            sec_idx = next(
                i for i, line in enumerate(lines) if line.strip() == section
            )
        except StopIteration:
            # секции нет — добавим в конец как новую
            lines += ["", section, bullet]
            self.user_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return bullet

        # Находим конец секции (до следующего заголовка того же уровня или EOF)
        end_idx = len(lines)
        for j in range(sec_idx + 1, len(lines)):
            if lines[j].startswith("## "):
                end_idx = j
                break

        # Убираем плейсхолдер «(пока не указано)» внутри секции
        body = [
            ln for ln in lines[sec_idx + 1 : end_idx]
            if ln.strip() != "(пока не указано)"
        ]
        # Убираем завершающие пустые строки внутри секции
        while body and not body[-1].strip():
            body.pop()
        body.append(bullet)
        body.append("")  # пустая строка перед следующей секцией

        new_lines = lines[: sec_idx + 1] + body + lines[end_idx:]
        self.user_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return bullet


IDENTITY = IdentityManager()
