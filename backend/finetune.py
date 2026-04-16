"""FinetuneStore — CRUD для finetune_queue.jsonl + детектор категорий.

Формат хранения (jsonl) — OpenAI chat-style:
{"id": "...", "messages": [...], "metadata": {...}}

ID — стабильный 12-символьный sha1 от user+assistant+timestamp, присваивается при add().
"""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .config import CONFIG
from .models import (
    ChatMessage,
    FinetuneCategory,
    FinetuneMetadata,
    FinetunePair,
)

FINETUNE_SYSTEM_PROMPT = (
    "You are an expert automation engineer. "
    "Answer precisely based on your knowledge. If unsure, say so."
)


# ------------ category detection ------------
# (patterns — regex с границами слов, чтобы "какое" не матчило "как")
_TROUBLESHOOT_PAT = re.compile(
    r"\b(не\s+работает|ошибк|проблем|падает|drops?|error|fails?|not\s+working|"
    r"debug|диагностик|чинит|сбой|зависает|пропада)",
    re.I | re.U,
)
_PROCEDURE_PAT = re.compile(
    r"(\bкак\s|\bhow\s+to\b|\bпошагов|\bstep\b|\bprocedure|\binstall|"
    r"\bнастрой|\bкалибр|\bподключить)",
    re.I | re.U,
)
_DECISION_PAT = re.compile(
    r"\b(почему|why\b|выбра(ть|ли)|chosen|преимуществ|\svs\s|сравнение|лучше\s+чем)",
    re.I | re.U,
)
_FACTUAL_PAT = re.compile(
    r"\b(что\s+так(ое|ой)|what\s+is\b|сколько|how\s+many\b|"
    r"какое|какая|какой|какие|длина|напряжен|voltage|current|max\b|"
    r"параметр|спецификац|spec\b)",
    re.I | re.U,
)


def detect_category(question: str, answer: str) -> FinetuneCategory:
    q = question or ""
    a = answer or ""
    text = q + "\n" + a

    if _TROUBLESHOOT_PAT.search(text):
        return "troubleshooting"
    if _DECISION_PAT.search(q):
        return "decision"
    if _FACTUAL_PAT.search(q):
        return "factual_qa"
    if _PROCEDURE_PAT.search(q):
        return "procedure"
    return "other"


# ------------ FinetuneStore ------------
def _hash_id(user: str, assistant: str, ts: str) -> str:
    h = hashlib.sha1(f"{user}\x1f{assistant}\x1f{ts}".encode("utf-8"))
    return h.hexdigest()[:12]


def _clean_answer(text: str) -> str:
    """Убирает ⚠️-префикс и лишние метаданные из ответа."""
    if not text:
        return text
    text = re.sub(r"^⚠️[^\n]*\n+", "", text).strip()
    return text


class FinetuneStore:
    def __init__(self, path: Optional[Path] = None):
        from .knowledge_manager import KM
        self.path: Path = path or KM.finetune_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self.min_required: int = CONFIG.knowledge["finetune_min_examples"]
        self.confidence_threshold: int = 85  # см. FINETUNE_PIPELINE.md

    # ---------- low-level I/O ----------
    def _read_all(self) -> list[FinetunePair]:
        if not self.path.exists():
            return []
        out: list[FinetunePair] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    out.append(FinetunePair(**raw))
                except Exception:
                    continue
        return out

    def _write_all(self, pairs: Iterable[FinetunePair]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")

    # ---------- public CRUD ----------
    def list_all(self) -> list[FinetunePair]:
        return self._read_all()

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.open("r", encoding="utf-8") if line.strip())

    def count_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self._read_all():
            out[p.metadata.category] = out.get(p.metadata.category, 0) + 1
        return out

    def get(self, pair_id: str) -> Optional[FinetunePair]:
        for p in self._read_all():
            if p.id == pair_id:
                return p
        return None

    def add(
        self,
        question: str,
        answer: str,
        *,
        source_notes: list[str],
        confidence: int,
        project: str | None,
        category: FinetuneCategory | None = None,
        verified: bool = True,
        original_wrong_answer: str | None = None,
        system_prompt: str = FINETUNE_SYSTEM_PROMPT,
    ) -> FinetunePair:
        answer = _clean_answer(answer)
        ts = datetime.now().isoformat(timespec="seconds")
        pair_id = _hash_id(question, answer, ts)
        if category is None:
            category = detect_category(question, answer)
        meta = FinetuneMetadata(
            source_notes=source_notes,
            confidence=confidence,
            project=project,
            timestamp=ts,
            verified=verified,
            category=category,
            original_wrong_answer=original_wrong_answer,
        )
        pair = FinetunePair(
            id=pair_id,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=question),
                ChatMessage(role="assistant", content=answer),
            ],
            metadata=meta,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pair.model_dump(), ensure_ascii=False) + "\n")
        return pair

    def maybe_add_from_agent(
        self,
        question: str,
        answer: str,
        *,
        source_notes: list[str],
        confidence: int,
        is_verified: bool,
        project: str | None,
    ) -> FinetunePair | None:
        """Автосбор по правилам Этапа 1 FINETUNE_PIPELINE.md."""
        if not is_verified:
            return None
        if confidence < self.confidence_threshold:
            return None
        if not source_notes:
            return None
        if len(question.strip()) < 10:
            return None
        return self.add(
            question=question,
            answer=answer,
            source_notes=source_notes,
            confidence=confidence,
            project=project,
            verified=True,
        )

    def add_correction(
        self,
        question: str,
        wrong_answer: str,
        corrected_answer: str,
        *,
        project: str | None = None,
        source_notes: list[str] | None = None,
    ) -> FinetunePair:
        """User-verified correction — категория 'correction', confidence = 100."""
        return self.add(
            question=question,
            answer=corrected_answer,
            source_notes=source_notes or [],
            confidence=100,
            project=project,
            category="correction",
            verified=True,
            original_wrong_answer=wrong_answer,
        )

    def edit(self, pair_id: str, *, assistant: str | None = None,
             boosted: bool | None = None) -> bool:
        pairs = self._read_all()
        found = False
        for p in pairs:
            if p.id != pair_id:
                continue
            found = True
            if assistant is not None:
                for m in p.messages:
                    if m.role == "assistant":
                        m.content = assistant
                        break
            if boosted is not None:
                p.metadata.boosted = boosted
        if found:
            self._write_all(pairs)
        return found

    def delete(self, pair_id: str) -> bool:
        pairs = self._read_all()
        before = len(pairs)
        pairs = [p for p in pairs if p.id != pair_id]
        if len(pairs) == before:
            return False
        self._write_all(pairs)
        return True

    def boost(self, pair_id: str) -> bool:
        return self.edit(pair_id, boosted=True)

    def ready(self) -> bool:
        return self.count() >= self.min_required

    def export_jsonl(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""


# ленивый синглтон
_store: FinetuneStore | None = None


def store() -> FinetuneStore:
    global _store
    if _store is None:
        _store = FinetuneStore()
    return _store
