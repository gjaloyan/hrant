"""Core memory management (Level 1) + auto-promote (Level 3 — finetune queue in KM)."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from .config import CONFIG
from .knowledge_manager import KM


def _approx_tokens(text: str) -> int:
    # rough estimate: 1 token ≈ 4 characters
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
            return f"⚠️ Core memory has reached the limit of {self.max_tokens} tokens. Fact not added."
        self.path.write_text(new, encoding="utf-8")
        return "✓ added to core memory"

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
            return "✗ nothing found"
        self.path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return f"✓ removed lines: {removed}"

    def suggest_promotions(self) -> list[str]:
        """Topics accessed threshold+ times — candidates for core memory."""
        return KM.hot_topics(self.promote_threshold)


CORE = CoreMemory()
