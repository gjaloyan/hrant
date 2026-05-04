"""Self-modifier: proposes code improvements with user approval/reject flow.

The agent can analyze its own modules, propose changes, and track proposals.
ALL changes require explicit user approval — no automatic code modifications.

Flow:
  1. Agent (or user) requests analysis of a module
  2. SelfModifier reads the code, identifies improvements
  3. Creates a Proposal with diff, reasoning, and expected impact
  4. User reviews in UI: approve or reject
  5. If approved, applies the diff and logs the change
  6. If rejected, logs the rejection reason for learning

Persistence: proposals.json in knowledge directory.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG, ROOT
from .llm import LLMError, TaskType, router

ANALYZE_SYSTEM = """You are a code improvement analyst for a Python AI agent codebase.
Given a module's source code, identify concrete improvements.

Focus on:
- Performance bottlenecks (inefficient algorithms, unnecessary I/O)
- Missing error handling at system boundaries
- Code clarity issues that hurt maintainability
- Opportunities to reduce API calls or latency

Do NOT suggest:
- Style-only changes (formatting, renaming)
- Adding comments or docstrings
- Speculative abstractions

Return strictly JSON:
{
  "improvements": [
    {
      "title": "short title",
      "description": "what to change and why",
      "old_code": "exact code snippet to replace (max 10 lines)",
      "new_code": "replacement code",
      "impact": "performance" | "reliability" | "clarity" | "feature",
      "risk": "low" | "medium" | "high",
      "reasoning": "why this improves the code"
    }
  ]
}

Max 3 improvements per analysis. Only include changes you're confident about."""


class Proposal:
    """A code change proposal awaiting user approval."""

    def __init__(
        self,
        id: str | None = None,
        module: str = "",
        title: str = "",
        description: str = "",
        old_code: str = "",
        new_code: str = "",
        impact: str = "",
        risk: str = "low",
        reasoning: str = "",
        status: str = "pending",
        created: str | None = None,
        reviewed: str | None = None,
        review_note: str = "",
    ):
        self.id = id or uuid.uuid4().hex[:10]
        self.module = module
        self.title = title
        self.description = description
        self.old_code = old_code
        self.new_code = new_code
        self.impact = impact
        self.risk = risk
        self.reasoning = reasoning
        self.status = status  # pending, approved, rejected, applied, failed
        self.created = created or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reviewed = reviewed
        self.review_note = review_note

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "module": self.module,
            "title": self.title,
            "description": self.description,
            "old_code": self.old_code,
            "new_code": self.new_code,
            "impact": self.impact,
            "risk": self.risk,
            "reasoning": self.reasoning,
            "status": self.status,
            "created": self.created,
            "reviewed": self.reviewed,
            "review_note": self.review_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Proposal":
        return cls(**{k: data[k] for k in data if k in cls.__init__.__code__.co_varnames})


class SelfModifier:
    """Proposes and applies code improvements with user approval."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "proposals.json")
        self._proposals: list[Proposal] = []
        self._backend_dir = ROOT / "backend"
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._proposals = [Proposal.from_dict(p) for p in data]
            except Exception:
                self._proposals = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([p.to_dict() for p in self._proposals], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def analyze_module(self, module_name: str) -> list[Proposal]:
        """Read a backend module and propose improvements."""
        # Resolve module file
        if not module_name.endswith(".py"):
            module_name += ".py"
        module_path = self._backend_dir / module_name
        if not module_path.exists():
            return []

        try:
            code = module_path.read_text(encoding="utf-8")
        except Exception:
            return []

        # Truncate very large files. 8000 chars used to drop ~75% of
        # agent.py / llm.py — so suggestions only ever covered the
        # imports + first few classes. Bumped to 30000, and when a file
        # exceeds that we keep BOTH ends (head + tail) instead of just
        # the head, because module-level singletons (`X = ClassName()`),
        # `if __name__ == "__main__"` blocks, and registry hooks live at
        # the bottom and matter for self-analysis.
        max_total = 30000
        if len(code) > max_total:
            head_chars = 22000
            tail_chars = max_total - head_chars - 80  # leave room for marker
            head = code[:head_chars]
            tail = code[-tail_chars:]
            omitted = len(code) - head_chars - tail_chars
            code = (
                f"{head}\n"
                f"# ... [{omitted} chars omitted from middle of file] ...\n"
                f"{tail}"
            )

        try:
            user_prompt = f"MODULE: backend/{module_name}\n\nSOURCE CODE:\n```python\n{code}\n```"
            data = router().call_json(
                TaskType.TASK_ANALYSIS,
                ANALYZE_SYSTEM,
                user_prompt,
                max_tokens=1500,
                temperature=0.2,
            )

            proposals = []
            for imp in data.get("improvements", []):
                proposal = Proposal(
                    module=f"backend/{module_name}",
                    title=imp.get("title", ""),
                    description=imp.get("description", ""),
                    old_code=imp.get("old_code", ""),
                    new_code=imp.get("new_code", ""),
                    impact=imp.get("impact", ""),
                    risk=imp.get("risk", "low"),
                    reasoning=imp.get("reasoning", ""),
                )
                self._proposals.append(proposal)
                proposals.append(proposal)

            self._save()
            return proposals

        except LLMError:
            return []

    def approve(self, proposal_id: str, note: str = "") -> bool:
        """Approve a proposal (does not apply it yet)."""
        for p in self._proposals:
            if p.id == proposal_id:
                p.status = "approved"
                p.reviewed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                p.review_note = note
                self._save()
                return True
        return False

    def reject(self, proposal_id: str, note: str = "") -> bool:
        """Reject a proposal."""
        for p in self._proposals:
            if p.id == proposal_id:
                p.status = "rejected"
                p.reviewed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                p.review_note = note
                self._save()
                return True
        return False

    def apply(self, proposal_id: str) -> dict:
        """Apply an approved proposal to the actual source file.

        Returns {ok, message} dict. Only applies if status == 'approved'.
        """
        proposal = None
        for p in self._proposals:
            if p.id == proposal_id:
                proposal = p
                break
        if not proposal:
            return {"ok": False, "message": "Proposal not found"}
        if proposal.status != "approved":
            return {"ok": False, "message": f"Proposal status is '{proposal.status}', must be 'approved'"}
        if not proposal.old_code or not proposal.new_code:
            return {"ok": False, "message": "Proposal has no code diff"}

        # Resolve file path
        file_path = ROOT / proposal.module
        if not file_path.exists():
            proposal.status = "failed"
            self._save()
            return {"ok": False, "message": f"File not found: {proposal.module}"}

        try:
            content = file_path.read_text(encoding="utf-8")
            if proposal.old_code not in content:
                proposal.status = "failed"
                proposal.review_note += " | old_code not found in file"
                self._save()
                return {"ok": False, "message": "old_code not found in file — code may have changed"}

            new_content = content.replace(proposal.old_code, proposal.new_code, 1)
            file_path.write_text(new_content, encoding="utf-8")
            proposal.status = "applied"
            self._save()
            return {"ok": True, "message": f"Applied to {proposal.module}"}

        except Exception as e:
            proposal.status = "failed"
            proposal.review_note += f" | apply error: {e}"
            self._save()
            return {"ok": False, "message": str(e)}

    def list_proposals(self, status: str | None = None) -> list[dict]:
        """List proposals, optionally filtered by status."""
        proposals = self._proposals
        if status:
            proposals = [p for p in proposals if p.status == status]
        return [p.to_dict() for p in reversed(proposals)]

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        for p in self._proposals:
            if p.id == proposal_id:
                return p
        return None

    def delete_proposal(self, proposal_id: str) -> bool:
        for i, p in enumerate(self._proposals):
            if p.id == proposal_id:
                self._proposals.pop(i)
                self._save()
                return True
        return False

    def stats(self) -> dict:
        by_status: dict[str, int] = {}
        by_impact: dict[str, int] = {}
        for p in self._proposals:
            by_status[p.status] = by_status.get(p.status, 0) + 1
            by_impact[p.impact] = by_impact.get(p.impact, 0) + 1
        return {
            "total": len(self._proposals),
            "by_status": by_status,
            "by_impact": by_impact,
        }

    def available_modules(self) -> list[str]:
        """List backend modules available for analysis."""
        return sorted(
            f.name for f in self._backend_dir.glob("*.py")
            if not f.name.startswith("_")
        )


SELF_MODIFIER = SelfModifier()
