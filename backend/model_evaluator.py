"""ModelEvaluator: runs tests on the old and new models and compares them.

Test set — JSON: [{"question": "...", "expected": "..."}, ...]
Default location: knowledge/eval_set.json
Score — fuzz.token_set_ratio between the actual and expected answer.
"""
from __future__ import annotations
import json
from pathlib import Path

from rapidfuzz import fuzz

from .config import CONFIG
from .knowledge_manager import KM
from .llm import AnthropicLLM, BaseLLM, OllamaLLM
from .models import EvaluationResult


def _llm_for(model_id: str) -> BaseLLM:
    """claude* → Anthropic (using settings from model_a), otherwise Ollama (from model_b)."""
    if model_id.lower().startswith("claude"):
        cfg = dict(CONFIG.model_a)
        cfg["model"] = model_id
        return AnthropicLLM(cfg)
    cfg = dict(CONFIG.model_b)
    cfg["model"] = model_id
    return OllamaLLM(cfg)


class ModelEvaluator:
    def __init__(self, eval_set_path: Path | None = None):
        self.path = eval_set_path or (KM.base / "eval_set.json")

    def load_test_set(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_test_set(self, items: list[dict]) -> None:
        self.path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _score(self, actual: str, expected: str) -> float:
        return fuzz.token_set_ratio(actual, expected) / 100.0

    def compare(self, old_model: str, new_model: str) -> EvaluationResult:
        tests = self.load_test_set()
        if not tests:
            return EvaluationResult(
                old_model=old_model,
                new_model=new_model,
                old_score=0.0,
                new_score=0.0,
                improvement=0.0,
                should_upgrade=False,
                details=[{"warn": "eval_set.json is empty — add tests"}],
            )
        old_llm = _llm_for(old_model)
        new_llm = _llm_for(new_model)
        details = []
        old_total = 0.0
        new_total = 0.0

        for q in tests:
            question = q["question"]
            expected = q.get("expected", "")
            try:
                old_ans = old_llm.complete("", question, max_tokens=500)
            except Exception as e:
                old_ans = f"[err: {e}]"
            try:
                new_ans = new_llm.complete("", question, max_tokens=500)
            except Exception as e:
                new_ans = f"[err: {e}]"
            s_old = self._score(old_ans, expected)
            s_new = self._score(new_ans, expected)
            old_total += s_old
            new_total += s_new
            details.append({
                "question": question,
                "old_score": s_old,
                "new_score": s_new,
                "old_answer": old_ans[:400],
                "new_answer": new_ans[:400],
            })

        n = len(tests)
        old_avg = old_total / n
        new_avg = new_total / n
        return EvaluationResult(
            old_model=old_model,
            new_model=new_model,
            old_score=old_avg,
            new_score=new_avg,
            improvement=new_avg - old_avg,
            should_upgrade=new_avg > old_avg,
            details=details,
        )
