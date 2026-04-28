"""CRUD для заметок + index.json + access_log + finetune_queue."""
from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .models import (
    Category,
    Confidence,
    IndexEntry,
    Note,
    NoteFrontmatter,
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[\s_-]+", "_", s)
    return s or "untitled"


class KnowledgeManager:
    def __init__(self, base_dir: Optional[str] = None):
        self.base = Path(base_dir or CONFIG.knowledge["base_dir"])
        self.base.mkdir(parents=True, exist_ok=True)
        for sub in ("fundamentals", "profession", "projects", "personal"):
            (self.base / sub).mkdir(exist_ok=True)
        self.index_path = self.base / "index.json"
        self.access_log_path = self.base / "access_log.json"
        # gap-лог: темы, которые пользователь спрашивал, а у нас их не было
        # в базе в момент запроса (нам пришлось учить на лету). Это сигнал
        # "чем пользователь интересуется, но мы к этому не готовы заранее".
        self.gaps_path = self.base / "gaps.json"
        self.finetune_path = self.base / "finetune_queue.jsonl"
        self.core_path = self.base / "core_memory.md"
        # История версий заметок: snapshot предыдущего содержимого перед
        # каждой перезаписью. Даёт возможность откатиться и посмотреть diff.
        self.history_dir = self.base / "_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({})
        if not self.access_log_path.exists():
            self._write_json(self.access_log_path, {})
        if not self.core_path.exists():
            self.core_path.write_text(
                "# Core Memory\n\n*Постоянные факты, всегда в контексте.*\n",
                encoding="utf-8",
            )

    # ---------- version history ----------
    def _snapshot_note(self, path: Path) -> Optional[Path]:
        """Сохраняет текущее содержимое заметки в _history/<slug>/<ts>.md.

        Возвращает путь к снапшоту или None, если заметки ещё нет на диске.
        Любая ошибка ловится — версионирование не должно ломать сохранение.
        """
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        slug = path.stem
        folder = self.history_dir / slug
        folder.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = folder / f"{ts}.md"
        # На случай микросекундных коллизий в тестах/скриптах добавляем суффикс.
        i = 1
        while snap.exists():
            snap = folder / f"{ts}_{i}.md"
            i += 1
        try:
            snap.write_text(text, encoding="utf-8")
        except Exception:
            return None
        return snap

    def list_versions(self, topic: str) -> list[dict]:
        """Список снапшотов заметки, от новых к старым.

        Возвращает [{timestamp, path, size}], где timestamp в формате
        'YYYY-MM-DD HH:MM:SS'. Пустой список, если истории нет.
        """
        slug = _slug(topic)
        folder = self.history_dir / slug
        if not folder.exists():
            return []
        out: list[dict] = []
        for f in sorted(folder.glob("*.md"), reverse=True):
            name = f.stem  # YYYYMMDD_HHMMSS[_N]
            stamp_part = name.split("_")[0] + "_" + name.split("_")[1] if "_" in name else name
            try:
                dt = datetime.strptime(stamp_part, "%Y%m%d_%H%M%S")
                human = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                human = name
            out.append({"timestamp": human, "path": str(f), "size": f.stat().st_size})
        return out

    def get_version(self, topic: str, index: int = 0) -> Optional[str]:
        """Читает N-ю сверху версию (0 = самая свежая). None, если нет."""
        versions = self.list_versions(topic)
        if not versions or index >= len(versions) or index < 0:
            return None
        try:
            return Path(versions[index]["path"]).read_text(encoding="utf-8")
        except Exception:
            return None

    # ---------- index ----------
    def _read_index(self) -> dict[str, dict]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_index(self, data: dict) -> None:
        self.index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_json(self, p: Path, data) -> None:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- note CRUD ----------
    def note_path(self, category: Category, topic: str, project: str | None = None) -> Path:
        if category == "projects" and project:
            d = self.base / "projects" / _slug(project)
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{_slug(topic)}.md"
        return self.base / category / f"{_slug(topic)}.md"

    def save_note(
        self,
        topic: str,
        body: str,
        category: Category = "profession",
        keywords: list[str] | None = None,
        source: str = "",
        confidence: Confidence = "unverified",
        project: str | None = None,
    ) -> Note:
        path = self.note_path(category, topic, project)
        existing = self.load_note_by_path(path)
        created = existing.frontmatter.created if existing else _now()
        access_count = existing.frontmatter.access_count if existing else 0
        fm = NoteFrontmatter(
            topic=topic,
            category=category,
            created=created,
            updated=_now(),
            keywords=keywords or [],
            source=source,
            confidence=confidence,
            access_count=access_count,
            project=project,
        )
        note = Note(frontmatter=fm, body=body, path=str(path))
        # Снимок старой версии до перезаписи (если это не первое сохранение).
        if existing is not None:
            self._snapshot_note(path)
        path.write_text(note.to_markdown(), encoding="utf-8")
        self._upsert_index(note)
        # Best-effort embedding refresh — never block save on a degraded
        # vector backend.
        try:
            self._embed_note(note)
        except Exception:
            pass
        return note

    def _embed_note(self, note: Note) -> None:
        """Embed `topic + body` and store in the vector index. No-op if disabled."""
        from .embedder import EMBEDDER
        from .vector_store import VECTOR_STORE
        text = f"{note.frontmatter.topic}\n\n{note.body}".strip()
        vec = EMBEDDER.embed(text)
        if not vec or EMBEDDER.dim is None or EMBEDDER.backend is None:
            return
        if not VECTOR_STORE.is_compatible(EMBEDDER.dim, EMBEDDER.backend, EMBEDDER.model or ""):
            # Backend changed; we'll reset on the next backfill explicitly.
            return
        if VECTOR_STORE.count() == 0:
            VECTOR_STORE.stamp(EMBEDDER.dim, EMBEDDER.backend, EMBEDDER.model or "")
        VECTOR_STORE.add(_slug(note.frontmatter.topic), vec)

    def _upsert_index(self, note: Note) -> None:
        idx = self._read_index()
        key = _slug(note.frontmatter.topic)
        idx[key] = IndexEntry(
            topic=note.frontmatter.topic,
            category=note.frontmatter.category,
            path=str(note.path),
            keywords=note.frontmatter.keywords,
            access_count=note.frontmatter.access_count,
            updated=note.frontmatter.updated,
            project=note.frontmatter.project,
        ).model_dump()
        self._write_index(idx)

    def load_note_by_path(self, path: Path) -> Optional[Note]:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return self._parse_note(text, path)

    def _parse_note(self, text: str, path: Path) -> Note:
        if not text.startswith("---"):
            return Note(
                frontmatter=NoteFrontmatter(
                    topic=path.stem,
                    category="profession",
                    created=_now(),
                    updated=_now(),
                ),
                body=text,
                path=str(path),
            )
        end = text.find("\n---", 3)
        header = text[3:end].strip()
        body = text[end + 4 :].strip()
        meta: dict = {}
        for line in header.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        kws = [k.strip() for k in meta.get("keywords", "").split(",") if k.strip()]
        try:
            access = int(meta.get("access_count", "0"))
        except ValueError:
            access = 0
        fm = NoteFrontmatter(
            topic=meta.get("topic", path.stem),
            category=meta.get("category", "profession"),  # type: ignore
            created=meta.get("created", _now()),
            updated=meta.get("updated", _now()),
            keywords=kws,
            source=meta.get("source", ""),
            confidence=meta.get("confidence", "unverified"),  # type: ignore
            access_count=access,
            project=meta.get("project") or None,
        )
        return Note(frontmatter=fm, body=body, path=str(path))

    def get_note(self, topic: str) -> Optional[Note]:
        idx = self._read_index()
        entry = idx.get(_slug(topic))
        if not entry:
            return None
        note = self.load_note_by_path(Path(entry["path"]))
        if note:
            self.log_access(topic)
        return note

    def delete_note(self, topic: str) -> bool:
        idx = self._read_index()
        key = _slug(topic)
        entry = idx.pop(key, None)
        if not entry:
            return False
        try:
            Path(entry["path"]).unlink(missing_ok=True)
        except Exception:
            pass
        self._write_index(idx)
        # Best-effort vector index cleanup.
        try:
            from .vector_store import VECTOR_STORE
            VECTOR_STORE.remove(key)
        except Exception:
            pass
        return True

    def list_topics(self) -> list[IndexEntry]:
        idx = self._read_index()
        return [IndexEntry(**v) for v in idx.values()]

    def all_categories(self) -> dict[str, list[IndexEntry]]:
        out: dict[str, list[IndexEntry]] = {}
        for entry in self.list_topics():
            out.setdefault(entry.category, []).append(entry)
        return out

    # ---------- access tracking ----------
    def log_access(self, topic: str) -> int:
        key = _slug(topic)
        log = json.loads(self.access_log_path.read_text(encoding="utf-8") or "{}")
        log[key] = log.get(key, 0) + 1
        self._write_json(self.access_log_path, log)

        idx = self._read_index()
        if key in idx:
            idx[key]["access_count"] = log[key]
            self._write_index(idx)

            note_path = Path(idx[key]["path"])
            note = self.load_note_by_path(note_path)
            if note:
                note.frontmatter.access_count = log[key]
                note_path.write_text(note.to_markdown(), encoding="utf-8")
        return log[key]

    def access_count(self, topic: str) -> int:
        log = json.loads(self.access_log_path.read_text(encoding="utf-8") or "{}")
        return log.get(_slug(topic), 0)

    def hot_topics(self, threshold: int) -> list[str]:
        log = json.loads(self.access_log_path.read_text(encoding="utf-8") or "{}")
        return [k for k, v in log.items() if v >= threshold]

    # ---------- gap tracking ----------
    def _read_gaps(self) -> dict[str, dict]:
        if not self.gaps_path.exists():
            return {}
        try:
            return json.loads(self.gaps_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            return {}

    def log_miss(self, topic: str) -> int:
        """Регистрирует: тему спросили — у нас её не было в базе.

        Хранит {slug: {"topic": str, "count": int, "last": "YYYY-MM-DD HH:MM"}}.
        Возвращает новый счётчик промахов для этой темы. Любая I/O ошибка
        гасится — gap-лог не должен ломать основной поток.
        """
        topic = (topic or "").strip()
        if not topic:
            return 0
        key = _slug(topic)
        try:
            data = self._read_gaps()
            entry = data.get(key) or {"topic": topic, "count": 0, "last": ""}
            entry["topic"] = topic  # обновляем «каноничную» форму
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["last"] = _now()
            data[key] = entry
            self._write_json(self.gaps_path, data)
            return entry["count"]
        except Exception:
            return 0

    def hot_gaps(self, threshold: int = 1, limit: int = 20) -> list[dict]:
        """Возвращает темы, промахнувшиеся ≥ threshold раз.

        Отсортированы по count убыванию. Каждый элемент:
        {"topic": str, "count": int, "last": str, "has_note_now": bool}.
        `has_note_now=True` означает, что тему в итоге удалось изучить и
        заметка уже в базе — пользователю можно её показать как "закрытую".
        """
        data = self._read_gaps()
        idx = self._read_index()
        items: list[dict] = []
        for slug, entry in data.items():
            count = int(entry.get("count", 0))
            if count < threshold:
                continue
            items.append({
                "topic": entry.get("topic", slug),
                "count": count,
                "last": entry.get("last", ""),
                "has_note_now": slug in idx,
            })
        items.sort(key=lambda x: (-x["count"], x["topic"]))
        return items[:limit]

    def open_gaps(self, threshold: int = 1, limit: int = 20) -> list[dict]:
        """Только незакрытые пробелы: темы, которые всё ещё не в базе.

        Полезно для "пробелов", которые стоит проактивно изучить
        (фоновый режим, /gaps команда, подсказка в UI).
        """
        return [g for g in self.hot_gaps(threshold, limit=limit * 3) if not g["has_note_now"]][:limit]


KM = KnowledgeManager()
