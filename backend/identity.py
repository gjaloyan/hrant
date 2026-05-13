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


_DEFAULT_SPEAKER_FOR_LEGACY_USER_MD = "webui:default"


def _sanitize_speaker_for_path(speaker_id: str) -> str:
    """Convert a speaker_id ('telegram:123', 'webui:default', …) into
    a filesystem-safe filename stem. Colons forbidden on Windows;
    only `[a-z0-9_-]` survives, everything else becomes `_`."""
    import re as _re
    return _re.sub(r"[^a-z0-9_-]+", "_", speaker_id.lower()).strip("_") or "default"


class IdentityManager:
    def __init__(self, base_dir: Path | None = None):
        self.dir = Path(base_dir or CONFIG.knowledge["base_dir"]) / "identity"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.soul_path = self.dir / "soul.md"
        self.identity_path = self.dir / "identity.md"
        # Legacy single-user_profile path. From Phase 10 onwards the
        # WebUI default speaker (`webui:default`) lives here, every
        # other speaker gets its own file under `profiles/`.
        self.user_path = self.dir / "user.md"
        self.profiles_dir = self.dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        # История каждого user-profile файла: per-file timestamped
        # snapshots. The original layout used `_history/user_*.md` for
        # the single user.md; we keep that path for `webui:default` and
        # add `_history/<sanitized>_*.md` for other speakers.
        self.history_dir = self.dir / "_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _user_path_for(self, speaker_id: str | None) -> Path:
        """Return the per-speaker user_profile file path. The WebUI
        default keeps the legacy `user.md` location; other speakers
        (each Telegram user, future channels) get their own file
        under `profiles/<sanitized>.md`."""
        sp = speaker_id or _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD
        if sp == _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD:
            return self.user_path
        return self.profiles_dir / f"{_sanitize_speaker_for_path(sp)}.md"

    def _snapshot_user_profile(self, speaker_id: str | None = None) -> Path | None:
        """Snapshot the current per-speaker user_profile into _history/.

        Filename: `user_<ts>.md` for the WebUI default (back-compat);
        `<sanitized_speaker>_<ts>.md` for everyone else."""
        path = self._user_path_for(speaker_id)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sp = speaker_id or _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD
        if sp == _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD:
            prefix = "user"
        else:
            prefix = _sanitize_speaker_for_path(sp)
        snap = self.history_dir / f"{prefix}_{ts}.md"
        i = 1
        while snap.exists():
            snap = self.history_dir / f"{prefix}_{ts}_{i}.md"
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

    def user_profile(self, speaker_id: str | None = None) -> str:
        """Read the per-speaker user_profile.md. On first read for a
        non-default speaker, the file is auto-created from the
        _DEFAULT_USER template so the profile is editable from the
        moment a new speaker arrives."""
        path = self._user_path_for(speaker_id)
        if not path.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_DEFAULT_USER, encoding="utf-8")
            except Exception:
                return _DEFAULT_USER
        return path.read_text(encoding="utf-8")

    def list_speaker_profiles(self) -> list[dict]:
        """Every user_profile file present on disk. Returns
        [{speaker_id, path, size, modified}, …] sorted by modified
        newest-first. Used by the WebUI to list profiles in the
        User Profile tab so the user can switch between them."""
        out: list[dict] = []
        # Legacy webui:default.
        if self.user_path.exists():
            try:
                st = self.user_path.stat()
                out.append({
                    "speaker_id": _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD,
                    "path": str(self.user_path),
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
            except OSError:
                pass
        # Per-speaker files under profiles/.
        if self.profiles_dir.exists():
            for f in self.profiles_dir.glob("*.md"):
                # Re-derive a "best guess" speaker_id from the sanitized
                # filename (e.g. telegram_123 -> telegram:123). Best-effort:
                # the agent stores the canonical speaker_id at write time
                # but the legacy on-disk shape only carries the stem.
                stem = f.stem
                if "_" in stem:
                    head, _, tail = stem.partition("_")
                    speaker_id = f"{head}:{tail}"
                else:
                    speaker_id = f"unknown:{stem}"
                try:
                    st = f.stat()
                    out.append({
                        "speaker_id": speaker_id,
                        "path": str(f),
                        "size": st.st_size,
                        "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except OSError:
                    continue
        return sorted(out, key=lambda r: r["modified"], reverse=True)

    @staticmethod
    def _extract_section(text: str, headers: tuple[str, ...]) -> str:
        """Pull the body of the first matching `## <header>` section.

        Empty if no matching header is found, the section is empty, or
        the section holds only the `(пока не указано)` placeholder.
        """
        lines = text.splitlines()
        try:
            start = next(
                i for i, ln in enumerate(lines)
                if ln.strip() in headers
            )
        except StopIteration:
            return ""
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("## "):
                end = j
                break
        body_lines = [
            ln for ln in lines[start + 1 : end]
            if ln.strip() and ln.strip() != "(пока не указано)"
        ]
        return "\n".join(body_lines).strip()

    @classmethod
    def _extract_language_section(cls, profile_text: str) -> str:
        """Pull the body of the `## Язык общения` section from user.md."""
        return cls._extract_section(
            profile_text, ("## Язык общения", "## Language", "## Язык"),
        )

    @classmethod
    def _extract_name_section(cls, identity_text: str) -> str:
        """Pull the body of the `## Имя` section from identity.md."""
        return cls._extract_section(
            identity_text, ("## Имя", "## Name"),
        )

    def preamble(self, *, speaker_id: str | None = None) -> str:
        """Блок, который подмешивается в system prompt всех диалоговых вызовов.

        Order: character first, then self-definition, then user profile.
        Core memory is appended separately by the agent (CoreMemory owns it).

        `speaker_id` picks the per-speaker user_profile file. Default
        (None) → WebUI default (legacy `user.md`).

        If the user profile pins a response language, append a
        LANGUAGE OVERRIDE block at the END so it carries the most weight
        in the model's context.
        """
        identity_text = self.identity().strip()
        profile_text = self.user_profile(speaker_id=speaker_id).strip()
        out = (
            "# SOUL\n"
            f"{self.soul().strip()}\n\n"
            "# IDENTITY\n"
            f"{identity_text}\n\n"
            "# USER PROFILE\n"
            f"{profile_text}\n"
        )
        # Name override: same trick as LANGUAGE OVERRIDE — pull the
        # `## Имя` body from identity.md and re-state it at the END of
        # the preamble, where the model's attention weights it highest.
        # Inside the IDENTITY block alone the name was getting lost
        # under the longer SOUL section + recent conversation turns
        # where the agent had previously denied being called by name.
        name_body = self._extract_name_section(identity_text)
        if name_body:
            out += (
                "\n# AGENT NAME OVERRIDE\n"
                "When the user addresses you by the name(s) below, "
                "they ARE talking to YOU. Acknowledge it as your own "
                "name. Do not deny it. Do not say 'I'm not <name>'.\n"
                f"{name_body}\n"
            )
        lang_body = self._extract_language_section(profile_text)
        if lang_body:
            out += (
                "\n# LANGUAGE OVERRIDE\n"
                "User profile pins the response language below. "
                "This OVERRIDES the soul-level rule about mirroring the "
                "user's input language. Even if the current user message "
                "is in a different language, respond in the language "
                "specified here:\n"
                f"{lang_body}\n"
            )
        return out

    # ---------- запись в user.md ----------
    @staticmethod
    def _normalize_fact(s: str) -> str:
        """Canonical form for dedup comparison.

        Strips the bullet marker, the trailing `_(добавлено YYYY-MM-DD)_`
        timestamp, leading/trailing whitespace and punctuation, and
        lowercases. Two bullets that say the same thing in the same
        language collapse to the same key. Cross-language duplicates
        (EN vs RU phrasing of the same fact) still pass — semantic
        dedup is a separate problem we don't tackle here.
        """
        import re as _re
        s = s.strip()
        # drop leading '- ' or '* '
        s = _re.sub(r"^[-*]\s*", "", s)
        # drop the auto-added timestamp marker
        s = _re.sub(r"\s*_\([Дд]обавлено\s+\d{4}-\d{2}-\d{2}\)_\s*$", "", s)
        s = _re.sub(r"\s*_\(added\s+\d{4}-\d{2}-\d{2}\)_\s*$", "", s)
        # collapse whitespace
        s = _re.sub(r"\s+", " ", s).strip()
        # strip trailing punctuation that varies between phrasings
        s = s.rstrip(".!?;:,")
        return s.lower()

    def add_user_fact(
        self,
        fact: str,
        category: UserFactCategory = "about_user",
        *,
        speaker_id: str | None = None,
    ) -> str:
        """Add a fact to the per-speaker user_profile file.

        Default (speaker_id=None) targets the WebUI legacy `user.md`.
        Pass `speaker_id="telegram:123"` to write into THAT user's
        profile so facts about your wife don't leak into your own
        profile (and vice versa).

        Dedup: same fact (after canonicalisation) won't be re-added.
        Empty `(пока не указано)` placeholders are removed when the
        first real fact lands."""
        fact = fact.strip()
        if not fact:
            return ""
        stamp = datetime.now().strftime("%Y-%m-%d")
        bullet = f"- {fact}  _(добавлено {stamp})_"
        section = _SECTION_BY_CATEGORY.get(category, _SECTION_BY_CATEGORY["about_user"])
        new_key = self._normalize_fact(fact)

        target_path = self._user_path_for(speaker_id)
        text = self.user_profile(speaker_id=speaker_id)
        lines = text.splitlines()
        # find section
        try:
            sec_idx = next(
                i for i, line in enumerate(lines) if line.strip() == section
            )
        except StopIteration:
            # missing section — append at file end
            self._snapshot_user_profile(speaker_id=speaker_id)
            lines += ["", section, bullet]
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return bullet

        end_idx = len(lines)
        for j in range(sec_idx + 1, len(lines)):
            if lines[j].startswith("## "):
                end_idx = j
                break

        for existing in lines[sec_idx + 1 : end_idx]:
            stripped = existing.strip()
            if not stripped or stripped == "(пока не указано)":
                continue
            if self._normalize_fact(stripped) == new_key:
                return existing

        self._snapshot_user_profile(speaker_id=speaker_id)
        body = [
            ln for ln in lines[sec_idx + 1 : end_idx]
            if ln.strip() != "(пока не указано)"
        ]
        while body and not body[-1].strip():
            body.pop()
        body.append(bullet)
        body.append("")

        new_lines = lines[: sec_idx + 1] + body + lines[end_idx:]
        target_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return bullet

    def set_user_profile(self, content: str, *, speaker_id: str | None = None) -> Path:
        """Replace the entire per-speaker user_profile file. Snapshots
        the previous content into _history/ first."""
        self._snapshot_user_profile(speaker_id=speaker_id)
        path = self._user_path_for(speaker_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


IDENTITY = IdentityManager()
