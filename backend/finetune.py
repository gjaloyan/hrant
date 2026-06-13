"""FinetuneStore — CRUD for finetune_queue.jsonl + category detector.

Storage format (jsonl) — OpenAI chat-style:
{"id": "...", "messages": [...], "metadata": {...}}

ID — stable 12-character sha1 of user+assistant+timestamp, assigned on add().
"""
from __future__ import annotations
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .config import CONFIG
log = logging.getLogger(__name__)
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
# (patterns — regex with word boundaries so "какое" does not match "как")
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
    """Strips the ⚠️ prefix and extra metadata from the answer."""
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
        self.confidence_threshold: int = 85  # see FINETUNE_PIPELINE.md

    # Read CONFIG.knowledge live so a Settings/Engine PUT for
    # `finetune_min_examples` actually applies without restart.
    # See Phase 5C audit.
    @property
    def min_required(self) -> int:
        return int(CONFIG.knowledge["finetune_min_examples"])

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
        """Auto-collection according to Stage 1 rules in FINETUNE_PIPELINE.md."""
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
        """User-verified correction — category 'correction', confidence = 100."""
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


# lazy singleton
_store: FinetuneStore | None = None


def store() -> FinetuneStore:
    global _store
    if _store is None:
        _store = FinetuneStore()
    return _store


def collect_from_turn(
    *,
    task: str,
    answer: str,
    vr,
    tool_names: "list[str] | None" = None,
    is_chat: bool = False,
    supervisor_mode: bool = False,
    project: "str | None" = None,
) -> "FinetunePair | None":
    """Unified-loop auto-collection (AGI roadmap, 2026-06-12).

    The legacy `maybe_add_from_agent` was written for the notes-based
    pipeline and was never wired into the unified loop — the queue
    sat at 0 rows since the cutover. This is its unified-era
    replacement: the gates run off the turn's VerificationResult
    (including the 2026-06-11 grader-calibration split fields) and
    grounding comes from tool evidence, not just KB notes.

    Collected turns become distillation data for the small local
    model (finetune_pipeline: Unsloth LoRA on a 7B → Ollama). Only
    turns the verifier scored as trustworthy AND delivered make the
    queue — we distill the strong model's SUCCESSES, not its noise.

    Returns the stored pair or None (gate name is intentionally not
    surfaced — callers treat collection as fire-and-forget).
    """
    if supervisor_mode or is_chat:
        return None
    task = (task or "").strip()
    answer = (answer or "").strip()
    if len(task) < 10:
        return None
    if not (50 <= len(answer) <= 4000):
        return None
    st = store()
    confidence = int(getattr(vr, "confidence", 0) or 0)
    if confidence < st.confidence_threshold:
        return None
    if getattr(vr, "endpoint_met", None) is False:
        return None
    if getattr(vr, "contradictions", None):
        return None
    # Grounding: prefer the verifier's notes; fall back to the tool
    # evidence that produced the answer. A turn with neither is
    # ungrounded chat-shaped output — skip (curator would score it
    # down anyway).
    sources = list(getattr(vr, "notes_used", None) or [])
    if not sources:
        sources = [f"tool:{n}" for n in dict.fromkeys(tool_names or [])][:8]
    if not sources:
        return None
    return st.add(
        question=task,
        answer=answer,
        source_notes=sources,
        confidence=confidence,
        project=project,
        verified=True,
    )


_CORRECTION_JUDGE_SYSTEM = (
    "You label CORRECTION pairs for an agent's training data.\n"
    "You are given the PREVIOUS exchange (the user's question and the "
    "assistant's answer) and the CURRENT exchange (the user's reply and "
    "the assistant's new answer).\n\n"
    "A CORRECTION happened when ALL hold:\n"
    "  - the user's CURRENT reply points out that the assistant's "
    "PREVIOUS answer was wrong, incomplete, or off, AND\n"
    "  - the assistant's CURRENT answer genuinely fixes it with real, "
    "specific content (not merely 'sorry, you are right' with nothing "
    "added).\n\n"
    "It is NOT a correction when: the user simply asks a follow-up; the "
    "user adds a new unrelated request; the assistant only apologizes "
    "without delivering a better answer; the topic changed.\n\n"
    "Judge in the user's own language. Return strictly JSON: "
    '{"is_correction": true|false, "reason": "short"}'
)


def maybe_capture_correction(
    *,
    is_chat: bool = False,
    supervisor_mode: bool = False,
    speaker_id: "str | None" = None,
    session_key: "str | None" = None,
    project: "str | None" = None,
) -> "FinetunePair | None":
    """Detect a user-driven correction of the agent's PREVIOUS turn and
    store it as a high-value 'correction' training pair (AGI roadmap C,
    2026-06-13).

    Reading the real conversation history showed the richest learning
    signal — turns where the human caught the agent being wrong and the
    agent then fixed it — was being thrown away: corrections score low
    confidence, so `collect_from_turn`'s >=85 gate rejected them, and
    `add_correction` was only ever called manually from the WebUI.

    Must run AFTER the current turn is persisted to CONVERSATION (so
    `recent(2)` yields [prior, current]). An LLM CLASSIFICATION judge
    (routable to the small model) confirms the correction so we never
    keyword-match. Fail-closed: any error / non-correction → no row.

    The stored pair is (original question -> corrected answer) with the
    wrong answer kept as `original_wrong_answer`, so a future model
    learns to answer the ORIGINAL question correctly the first time.
    Curation gives 'correction' a quality bonus + 2-3x boosting, and the
    pair is human-reviewable in the Fine-Tune panel before training.
    """
    if is_chat or supervisor_mode:
        return None
    try:
        from .conversation import CONVERSATION
        turns = CONVERSATION.recent(2, session_key=session_key)
        if len(turns) < 2:
            return None
        prior, current = turns[-2], turns[-1]
        prior_q = (prior.get("user") or "").strip()
        prior_a = (prior.get("answer") or "").strip()
        corr_msg = (current.get("user") or "").strip()
        corrected_a = (current.get("answer") or "").strip()
        # Need real content on both sides to be worth a training pair.
        if len(prior_q) < 10 or len(prior_a) < 30 or len(corrected_a) < 30:
            return None

        from .llm import router, TaskType
        data = router().call_json(
            TaskType.CLASSIFICATION,
            _CORRECTION_JUDGE_SYSTEM,
            (
                f"=== PREVIOUS ===\nUSER: {prior_q[:1500]}\n"
                f"ASSISTANT: {prior_a[:1500]}\n\n"
                f"=== CURRENT ===\nUSER: {corr_msg[:1500]}\n"
                f"ASSISTANT: {corrected_a[:1500]}"
            ),
            max_tokens=120, temperature=0.0,
        )
        if not bool(data.get("is_correction", False)):
            return None
        return store().add_correction(
            question=prior_q,
            wrong_answer=prior_a,
            corrected_answer=corrected_a,
            project=project,
        )
    except Exception as e:
        log.debug("maybe_capture_correction failed (non-fatal): %s", e)
        return None
