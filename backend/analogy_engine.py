"""Analogy engine: finds structural similarities across domains.

Uses the knowledge graph to find patterns that recur in different contexts.
When the agent encounters a new problem, it checks if a structurally similar
problem was solved before in another domain and adapts that solution.

Can work with local models for cost-effective pattern matching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .knowledge_graph import GRAPH
from .knowledge_manager import KM
from .llm import LLMError, TaskType, router


PATTERN_EXTRACT_SYSTEM = """You extract abstract, reusable patterns from solutions.

Given a question and its solution, extract the abstract pattern — the general
principle that could apply to similar problems in OTHER domains.

Return strictly JSON:
{
  "pattern": "short abstract description of the reusable pattern",
  "abstract_form": "X reduces Y by doing Z" (template with variables),
  "domain": "the original domain",
  "mechanism": "the core mechanism that makes it work",
  "applicable_when": "conditions when this pattern is useful"
}"""

ANALOGY_SEARCH_SYSTEM = """You find analogies between a new problem and known patterns.

Given a problem and a list of abstract patterns from other domains,
find which patterns could be applied and HOW to adapt them.

Return strictly JSON:
{
  "analogies": [
    {
      "pattern_id": "id of the matching pattern",
      "relevance": 0.0-1.0,
      "adaptation": "how to apply this pattern to the current problem",
      "reasoning": "why this analogy works"
    }
  ]
}

Only include analogies with relevance >= 0.5. Max 3 analogies."""


class Pattern:
    """An abstract reusable pattern extracted from a solved problem."""

    def __init__(
        self,
        pattern_id: str,
        pattern: str,
        abstract_form: str,
        domain: str,
        mechanism: str,
        applicable_when: str,
        examples: list[dict] | None = None,
    ):
        self.pattern_id = pattern_id
        self.pattern = pattern
        self.abstract_form = abstract_form
        self.domain = domain
        self.mechanism = mechanism
        self.applicable_when = applicable_when
        self.examples = examples or []

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern": self.pattern,
            "abstract_form": self.abstract_form,
            "domain": self.domain,
            "mechanism": self.mechanism,
            "applicable_when": self.applicable_when,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pattern":
        return cls(
            pattern_id=data.get("pattern_id", ""),
            pattern=data.get("pattern", ""),
            abstract_form=data.get("abstract_form", ""),
            domain=data.get("domain", ""),
            mechanism=data.get("mechanism", ""),
            applicable_when=data.get("applicable_when", ""),
            examples=data.get("examples", []),
        )


class AnalogyEngine:
    """Finds and applies cross-domain analogies."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "patterns.json")
        self._patterns: list[Pattern] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._patterns = [Pattern.from_dict(p) for p in data]
            except Exception:
                self._patterns = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([p.to_dict() for p in self._patterns], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def extract_pattern(self, question: str, answer: str, domain: str = "") -> Pattern | None:
        """Extract an abstract pattern from a high-confidence answer."""
        try:
            user_prompt = (
                f"QUESTION: {question}\n\n"
                f"SOLUTION: {answer[:800]}\n\n"
                f"DOMAIN: {domain or 'general'}"
            )
            data = router().call_json(
                TaskType.LEARNING,
                PATTERN_EXTRACT_SYSTEM,
                user_prompt,
                max_tokens=400,
                temperature=0.2,
            )

            # Generate stable ID from pattern text
            pat_text = data.get("pattern", "")
            if not pat_text:
                return None
            pat_id = f"p_{abs(hash(pat_text)) % 100000:05d}"

            # Deduplicate
            for existing in self._patterns:
                if existing.pattern.lower() == pat_text.lower():
                    # Add as new example
                    existing.examples.append({"question": question[:100], "domain": domain})
                    self._save()
                    return existing

            pattern = Pattern(
                pattern_id=pat_id,
                pattern=pat_text,
                abstract_form=data.get("abstract_form", ""),
                domain=data.get("domain", domain),
                mechanism=data.get("mechanism", ""),
                applicable_when=data.get("applicable_when", ""),
                examples=[{"question": question[:100], "domain": domain}],
            )
            self._patterns.append(pattern)
            self._save()
            return pattern

        except LLMError:
            return None

    def find_analogies(self, problem: str, domain: str = "") -> list[dict]:
        """Search for applicable patterns from other domains."""
        if not self._patterns:
            return []

        # Filter patterns from different domains (cross-domain only)
        candidates = self._patterns
        if domain:
            candidates = [p for p in self._patterns if p.domain.lower() != domain.lower()]
        if not candidates:
            candidates = self._patterns  # fallback to all if no cross-domain

        try:
            patterns_text = json.dumps(
                [{"pattern_id": p.pattern_id, "pattern": p.pattern,
                  "abstract_form": p.abstract_form, "domain": p.domain,
                  "mechanism": p.mechanism, "applicable_when": p.applicable_when}
                 for p in candidates[:20]],
                ensure_ascii=False,
            )
            user_prompt = (
                f"PROBLEM: {problem}\n"
                f"PROBLEM DOMAIN: {domain or 'unknown'}\n\n"
                f"KNOWN PATTERNS:\n{patterns_text}"
            )
            data = router().call_json(
                TaskType.TASK_ANALYSIS,
                ANALOGY_SEARCH_SYSTEM,
                user_prompt,
                max_tokens=500,
                temperature=0.2,
            )
            return data.get("analogies", [])

        except LLMError:
            return []

    def context_block(self, problem: str, domain: str = "", max_analogies: int = 2) -> str:
        """Build analogy context block for injection into solver prompts."""
        analogies = self.find_analogies(problem, domain)
        if not analogies:
            return ""

        lines = [
            "# ANALOGIES FROM OTHER DOMAINS",
            "(Patterns from solved problems in other areas that may apply here.)",
            "",
        ]
        for a in analogies[:max_analogies]:
            pid = a.get("pattern_id", "?")
            pat = None
            for p in self._patterns:
                if p.pattern_id == pid:
                    pat = p
                    break
            if pat:
                lines.append(f"**Pattern**: {pat.pattern}")
                lines.append(f"  - Abstract: {pat.abstract_form}")
                lines.append(f"  - From domain: {pat.domain}")
                lines.append(f"  - Adaptation: {a.get('adaptation', '')}")
                lines.append(f"  - Why: {a.get('reasoning', '')}")
                lines.append("")

        return "\n".join(lines)

    def all_patterns(self) -> list[dict]:
        return [p.to_dict() for p in self._patterns]

    def stats(self) -> dict:
        domains: dict[str, int] = {}
        total_examples = 0
        for p in self._patterns:
            domains[p.domain] = domains.get(p.domain, 0) + 1
            total_examples += len(p.examples)
        return {
            "total_patterns": len(self._patterns),
            "by_domain": domains,
            "total_examples": total_examples,
        }


ANALOGIES = AnalogyEngine()
