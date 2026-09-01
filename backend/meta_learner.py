"""Meta-learner: analyzes agent failures and learns from mistakes.

After each low-confidence answer, the meta-learner:
  1. Classifies the root cause (missing knowledge, wrong reasoning, tool misuse)
  2. Extracts error patterns across multiple failures
  3. Auto-creates goals or notes to fix recurring issues
  4. Tracks error history for regression detection

Persistence: error_log.jsonl (append-only), error_patterns.json (aggregated).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .goals import GOALS
from .llm import LLMError, TaskType, router
from .models import VerificationResult
from .paths import write_atomic_json

META_ANALYSIS_SYSTEM = """You are a failure analyst for an AI agent.
Given a question, the agent's wrong answer, and verification details,
determine WHY the answer was wrong and HOW to fix it.

Return strictly JSON:
{
  "root_cause": "missing_knowledge" | "wrong_reasoning" | "outdated_info" | "tool_misuse" | "hallucination",
  "missing_topic": "topic to learn" (if root_cause is missing_knowledge, else null),
  "error_pattern": "short description of the reasoning mistake" (if wrong_reasoning/hallucination),
  "domain": "which knowledge domain this relates to",
  "fix_action": "learn_topic" | "add_core_fact" | "update_note" | "improve_prompt" | "none",
  "fix_detail": "specific actionable fix description",
  "severity": 1-10
}"""

PATTERN_EXTRACTION_SYSTEM = """You are analyzing a batch of agent failures.
Find recurring patterns — mistakes the agent makes repeatedly.

Given a list of recent failures, return JSON:
{
  "patterns": [
    {
      "pattern": "short description of recurring mistake",
      "frequency": number of occurrences,
      "domains": ["affected domains"],
      "suggested_fix": "how to prevent this pattern",
      "priority": 1-10
    }
  ]
}

Only include patterns that appear 2+ times. Max 5 patterns."""


class MetaLearner:
    """Analyzes agent failures and creates corrective actions."""

    # Run extract_patterns() automatically every Nth analyzed failure.
    # extract_patterns is one extra LLM call (~500 tokens), so we don't
    # do it after every single failure — but waiting for a manual call
    # means recurring patterns never become goals. 5 is the agent's own
    # suggestion and matches typical "user noticed something repeats"
    # cadence.
    AUTO_EXTRACT_EVERY_N_FAILURES = 5

    def __init__(self, path: Optional[Path] = None, patterns_path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.log_path = path or (kb_dir / "error_log.jsonl")
        self.patterns_path = patterns_path or (kb_dir / "error_patterns.json")
        self._patterns: list[dict] = []
        self._failure_count: int = 0  # in-process counter for auto-extract
        self._load_patterns()

    def _load_patterns(self) -> None:
        if self.patterns_path.exists():
            try:
                self._patterns = json.loads(
                    self.patterns_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._patterns = []

    def _save_patterns(self) -> None:
        try:
            # C3: atomic .tmp + rename so a crash mid-write doesn't
            # truncate error_patterns.json — losing the patterns means
            # losing the meta-learner's deduplicated failure history.
            write_atomic_json(self.patterns_path, self._patterns)
        except Exception:
            pass

    def _append_log(self, entry: dict) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _read_log(self, limit: int = 50) -> list[dict]:
        if not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
            entries = []
            for line in lines[-limit:]:
                if line.strip():
                    entries.append(json.loads(line))
            return entries
        except Exception:
            return []

    def log_tool_error(
        self,
        *,
        tool: str,
        message: str,
        args: Optional[dict] = None,
        turn_id: str = "",
    ) -> None:
        """Record a FAILED TOOL CALL in the shape the immune matcher reads.

        Added 2026-08-10. Until now error_log.jsonl held only low-confidence
        TURN records — fields like `question`, `confidence`, `unverified` —
        and none of the three fields `immune.SignatureStore.match()` reads:
        `source`, `message`, `service`. So the matcher could never have
        matched anything even if something had called it, and the 10590 fires
        of FIRE_ERROR_TRIAGE were counting turn quality, not failures.

        Tool errors are the right first source because they are frequent,
        machine-shaped and genuinely actionable. Measured from the owner's
        2026-08-10 conversation: agent_browser and npm failed repeatedly with
        `404 Not Found` and `command not found` while the agent kept
        retrying, and nothing anywhere recorded a single one of them.
        """
        entry = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "tool",
            "service": str(tool or ""),
            "message": str(message or "")[:600],
            "severity": "error",
            "tool_args": {k: str(v)[:120] for k, v in (args or {}).items()},
            "turn_id": turn_id,
        }
        self._append_log(entry)
        # Close the self-modification loop: a tool failing AFTER the agent
        # patched it is the evidence that the patch did not work. Recorded
        # here because this is the one place every tool failure passes
        # through. Best-effort — the error path must never raise.
        try:
            from .self_mod_outcomes import OUTCOMES
            OUTCOMES.note_tool_failure(str(tool or ""), str(message or ""))
        except Exception:
            pass

    def analyze_failure(
        self,
        question: str,
        answer: str,
        verification: VerificationResult,
        intent: str = "task",
    ) -> dict | None:
        """Analyze a single failure. Returns analysis dict or None if not a failure."""
        if verification.confidence >= 60:
            return None

        # Grader calibration (2026-06-11). The endpoint cap clips
        # `confidence` to 30 on a delivery miss, which used to route
        # every such turn through the LLM failure analyst — and the
        # analyst, seeing a fine answer with a low score, guessed
        # "hallucination" (35 of 96 in the 2026-06-10 reflection
        # were this mislabel). When the CONTENT was actually fine
        # (pre-clip score >= 60) and the only failure is delivery,
        # we already KNOW the root cause — record it directly, skip
        # the LLM, and let self_reflection aggregate `endpoint_miss`
        # as its own category.
        content_conf = (
            verification.content_confidence
            if verification.content_confidence is not None
            else verification.confidence
        )
        if content_conf >= 60 and verification.endpoint_met is False:
            entry = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question[:200],
                "answer_preview": answer[:300],
                "confidence": verification.confidence,
                "content_confidence": content_conf,
                "contradictions": verification.contradictions[:5],
                "unverified": verification.unverified_claims[:5],
                "intent": intent,
                "analysis": {
                    "root_cause": "endpoint_miss",
                    "missing_topic": None,
                    "error_pattern": (
                        "action-shaped request answered without an "
                        "execute-class tool call or delivery"
                    ),
                    "domain": "task_execution_and_validation",
                    "fix_action": "none",
                    "fix_detail": (
                        "process failure — handled by self-correction "
                        "and nightly lessons, not knowledge fixes"
                    ),
                    "severity": 4,
                },
            }
            self._append_log(entry)
            self._failure_count += 1
            if self._failure_count % self.AUTO_EXTRACT_EVERY_N_FAILURES == 0:
                try:
                    self.extract_patterns()
                except Exception:
                    pass
            return entry["analysis"]

        # Log the failure first
        entry = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question[:200],
            "answer_preview": answer[:300],
            "confidence": verification.confidence,
            "contradictions": verification.contradictions[:5],
            "unverified": verification.unverified_claims[:5],
            "intent": intent,
            "analysis": None,
        }

        # Try LLM analysis
        try:
            user_prompt = (
                f"QUESTION: {question}\n\n"
                f"AGENT'S ANSWER: {answer[:500]}\n\n"
                f"CONTRADICTIONS: {json.dumps(verification.contradictions[:5], ensure_ascii=False)}\n"
                f"UNVERIFIED CLAIMS: {json.dumps(verification.unverified_claims[:5], ensure_ascii=False)}\n"
                f"CONFIDENCE: {verification.confidence}%"
            )
            analysis = router().call_json(
                TaskType.VERIFICATION,
                META_ANALYSIS_SYSTEM,
                user_prompt,
                max_tokens=500,
                temperature=0.1,
            )
            entry["analysis"] = analysis
            self._append_log(entry)
            self._auto_fix(analysis)
            # Auto-extract patterns every N analyzed failures so the
            # feedback loop closes without waiting for a manual call.
            # extract_patterns logs nothing useful for the current
            # turn's caller, so swallow its errors — best-effort.
            self._failure_count += 1
            if self._failure_count % self.AUTO_EXTRACT_EVERY_N_FAILURES == 0:
                try:
                    self.extract_patterns()
                except Exception:
                    pass
            return analysis
        except LLMError:
            # If LLM is down, still log the failure without analysis
            self._append_log(entry)
            return None

    def _auto_fix(self, analysis: dict) -> None:
        """Create goals or take action based on failure analysis."""
        action = analysis.get("fix_action", "none")
        detail = analysis.get("fix_detail", "")
        severity = analysis.get("severity", 5)

        if action == "learn_topic" and analysis.get("missing_topic"):
            topic = analysis["missing_topic"]
            GOALS.add(
                description=f"Learn: {topic}",
                priority=min(9, severity),
                goal_type="learning",
                context=f"Meta-learner: {detail}",
                source="meta_learner",
            )
        elif action == "add_core_fact" and detail:
            GOALS.add(
                description=f"Add core fact: {detail[:80]}",
                priority=min(8, severity),
                goal_type="improvement",
                context=f"Meta-learner identified missing core knowledge: {detail}",
                source="meta_learner",
            )
        elif action == "update_note" and detail:
            GOALS.add(
                description=f"Update note: {detail[:80]}",
                priority=min(7, severity),
                goal_type="improvement",
                context=f"Meta-learner: existing note needs update — {detail}",
                source="meta_learner",
            )
        elif action == "improve_prompt" and detail:
            target_module = self._guess_target_module(detail)

            # "Improve prompt" IS a prompt edit, so propose the actual
            # line instead of a goal asking someone to think of one.
            # Without a guessable module this branch used to create a
            # goal with subtasks=None, which nothing could execute: 88
            # of 97 active goals were in that state on 2026-09-01, all
            # heading for the stale sweep. Approval stays explicit — the
            # proposal carries a diff the owner reads and taps, and
            # apply() compiles it and runs the tests.
            from .lesson_proposals import propose_lesson
            proposed = propose_lesson(
                detail, evidence=f"meta-learner, severity {severity}")

            # A goal WITH a plan is still worth keeping; a goal without
            # one is the thing being removed here.
            if proposed is None and target_module:
                GOALS.add(
                    description=f"Improve prompt: {detail[:80]}",
                    priority=min(6, severity),
                    goal_type="improvement",
                    context=(
                        f"Meta-learner: prompt engineering needed — {detail}"
                        f" | target module: {target_module}"
                    ),
                    source="meta_learner",
                    subtasks=[
                        f"Run SELF_MODIFIER.analyze_module('{target_module}')",
                        "Review the resulting proposal in the WebUI",
                        "Approve or reject explicitly",
                    ],
                )
            if severity >= 7 and target_module:
                try:
                    from .self_modifier import SELF_MODIFIER
                    SELF_MODIFIER.analyze_module(target_module)
                    # The proposal lands in self_modifier's queue;
                    # apply() still requires explicit user approval.
                except Exception:
                    pass  # best-effort bridge

    @staticmethod
    def _guess_target_module(detail: str) -> str:
        """Map a free-form 'improve prompt' detail to a backend module
        name. Returns "" when the detail doesn't mention anything we
        recognize — better than picking a wrong module to patch.
        """
        d = (detail or "").lower()
        candidates = (
            ("verifier", "verifier"),
            ("verif", "verifier"),
            ("agent", "agent"),
            ("classif", "agent"),  # intent classifier lives in agent.py
            ("classify", "agent"),
            ("solver", "agent"),
            ("think", "agent"),
            ("identity", "identity"),
            ("memory_extractor", "memory_extractor"),
            ("memory extract", "memory_extractor"),
            ("knowledge_graph", "knowledge_graph"),
            ("hybrid_searcher", "hybrid_searcher"),
            ("self_modifier", "self_modifier"),
            ("goals", "goals"),
        )
        for needle, module in candidates:
            if needle in d:
                return module
        return ""

    def extract_patterns(self) -> list[dict]:
        """Analyze recent failures to find recurring error patterns."""
        recent = self._read_log(limit=30)
        failures_with_analysis = [
            e for e in recent if e.get("analysis")
        ]
        if len(failures_with_analysis) < 3:
            return self._patterns

        try:
            summaries = []
            for e in failures_with_analysis[-20:]:
                a = e["analysis"]
                summaries.append({
                    "question_preview": e.get("question", "")[:100],
                    "root_cause": a.get("root_cause", "unknown"),
                    "error_pattern": a.get("error_pattern", ""),
                    "domain": a.get("domain", ""),
                    "severity": a.get("severity", 5),
                })

            user_prompt = f"RECENT FAILURES:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}"
            result = router().call_json(
                TaskType.VERIFICATION,
                PATTERN_EXTRACTION_SYSTEM,
                user_prompt,
                max_tokens=800,
                temperature=0.1,
            )
            patterns = result.get("patterns", [])
            if patterns:
                self._patterns = patterns
                self._save_patterns()

                # Create goals for high-priority patterns
                # A recurring behavioural failure is a RULE the agent is
                # missing, not a project. These used to become goals with no
                # subtasks — nothing could execute them, so all 372 were
                # swept after 14 days. Proposing the actual prompt edit puts
                # them in the queue that has a Telegram approval flow, a
                # readable diff, and an apply path that runs the tests.
                from .lesson_proposals import propose_lesson
                for p in patterns:
                    if p.get("priority", 0) >= 7:
                        lesson = (p.get("suggested_fix") or "").strip()
                        made = propose_lesson(
                            lesson,
                            evidence=(f"seen {p.get('frequency', 0)}x: "
                                      f"{p['pattern'][:90]}"),
                        ) if lesson else None
                        if made is not None:
                            continue
                        # No usable fix text, or a duplicate. Keep the goal
                        # for visibility rather than losing the observation.
                        GOALS.add(
                            description=f"Fix pattern: {p['pattern'][:80]}",
                            priority=min(9, p["priority"]),
                            goal_type="improvement",
                            context=f"Recurring error ({p.get('frequency', 0)}x): {p.get('suggested_fix', '')}",
                            source="meta_learner_pattern",
                        )

            return self._patterns
        except LLMError:
            return self._patterns

    def recent_failures(self, limit: int = 20) -> list[dict]:
        """Return recent failure log entries."""
        return self._read_log(limit)

    def stats(self) -> dict:
        """Aggregate error statistics."""
        entries = self._read_log(limit=100)
        total = len(entries)
        by_cause: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        avg_severity = 0.0
        severities: list[int] = []

        for e in entries:
            a = e.get("analysis") or {}
            cause = a.get("root_cause", "unknown")
            by_cause[cause] = by_cause.get(cause, 0) + 1
            domain = a.get("domain", "unknown")
            by_domain[domain] = by_domain.get(domain, 0) + 1
            if a.get("severity"):
                severities.append(a["severity"])

        if severities:
            avg_severity = sum(severities) / len(severities)

        return {
            "total_failures": total,
            "by_root_cause": by_cause,
            "by_domain": by_domain,
            "avg_severity": round(avg_severity, 1),
            "patterns_count": len(self._patterns),
            "patterns": self._patterns,
        }


META_LEARNER = MetaLearner()
