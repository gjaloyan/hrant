"""Управление core memory (Уровень 1) + auto-promote (Уровень 3 — финт-кью в KM)."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from .config import CONFIG
from .knowledge_manager import KM


def _approx_tokens(text: str) -> int:
    # грубая оценка: 1 токен ≈ 4 символа
    return max(1, len(text) // 4)


class CoreMemory:
    def __init__(self):
        self.path: Path = KM.core_path

    # Read CONFIG.knowledge live each call so the Engine tab's
    # "applies live" claim actually holds — Phase 5C exposes these
    # as runtime overrides, and a snapshot here would silently
    # require a restart to pick up the new value.
    @property
    def max_tokens(self) -> int:
        return int(CONFIG.knowledge["core_memory_max_tokens"])

    @property
    def promote_threshold(self) -> int:
        return int(CONFIG.knowledge["auto_promote_threshold"])

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def tokens(self) -> int:
        return _approx_tokens(self.read())

    def add_fact(self, fact: str, source: str = "user") -> str:
        text = self.read().rstrip()
        stamp = datetime.now().strftime("%Y-%m-%d")
        line = f"- {fact.strip()}  _(добавлено {stamp}, источник: {source})_"
        new = text + "\n" + line + "\n"
        if _approx_tokens(new) > self.max_tokens:
            return f"⚠️ Core memory достигла лимита {self.max_tokens} токенов. Факт не добавлен."
        self.path.write_text(new, encoding="utf-8")
        return "✓ добавлено в core memory"

    def remove_fact(self, search_text: str) -> str:
        text = self.read()
        new_lines = []
        removed = 0
        for line in text.splitlines():
            if search_text.lower() in line.lower() and line.lstrip().startswith("-"):
                removed += 1
                continue
            new_lines.append(line)
        if not removed:
            return "✗ ничего не найдено"
        self.path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return f"✓ удалено строк: {removed}"

    def suggest_promotions(self) -> list[str]:
        """Темы, доступаемые threshold+ раз — кандидаты в core memory."""
        return KM.hot_topics(self.promote_threshold)


CORE = CoreMemory()
