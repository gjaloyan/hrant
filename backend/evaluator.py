"""Self-evaluator: tracks agent performance over time.

Logs every interaction with metrics, detects regressions, generates
daily reports, and suggests priorities for improvement.

Persistence: eval_log.jsonl (append-only), eval_daily.json (daily aggregates).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import CONFIG


class EvalEntry:
    """A single evaluation log entry."""

    def __init__(
        self,
        question: str,
        intent: str,
        confidence: int,
        topics_used: list[str],
        contradictions: int,
        unverified: int,
        verified: int,
        is_chat: bool = False,
        response_time_ms: int = 0,
        ts: str | None = None,
        # Grader calibration (2026-06-11): the split signal.
        # `confidence` stays the conservative (possibly endpoint-
        # clipped) scalar; `content_confidence` is the pre-clip
        # claim score; `endpoint_met` is the delivery judgment.
        # None on old rows / turns where the verifier didn't run —
        # readers must .get() with a default.
        content_confidence: int | None = None,
        endpoint_met: bool | None = None,
    ):
        self.ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.question = question[:200]
        self.intent = intent
        self.confidence = confidence
        self.topics_used = topics_used
        self.contradictions = contradictions
        self.unverified = unverified
        self.verified = verified
        self.is_chat = is_chat
        self.response_time_ms = response_time_ms
        self.content_confidence = content_confidence
        self.endpoint_met = endpoint_met

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "question": self.question,
            "intent": self.intent,
            "confidence": self.confidence,
            "topics_used": self.topics_used,
            "contradictions": self.contradictions,
            "unverified": self.unverified,
            "verified": self.verified,
            "is_chat": self.is_chat,
            "response_time_ms": self.response_time_ms,
            "content_confidence": self.content_confidence,
            "endpoint_met": self.endpoint_met,
        }


class SelfEvaluator:
    """Tracks agent performance and detects regressions."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.log_path = path or (kb_dir / "eval_log.jsonl")
        self.daily_path = kb_dir / "eval_daily.json"

    def log(self, entry: EvalEntry) -> None:
        """Append an evaluation entry to the log."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def prune(self, max_rows: int = 5000) -> int:
        """I8: trim oldest rows past `max_rows`.

        Background: eval_log.jsonl was append-only with no GC. The
        evaluator writes a row per agent turn → a busy box piles up
        tens of thousands of rows; every `daily_report` / `stats`
        re-reads the entire file. Keeping the newest 5k rows preserves
        the rolling window the report functions use (most read at most
        500-1000 rows back) while bounding the on-disk size.

        Returns the number of rows dropped. Atomic rewrite (.tmp +
        rename) so a crash mid-prune can't truncate the log. Not
        auto-fired from anywhere yet — exposed for a future autonomic
        FIRE_STALE_PROPOSALS-style lever.
        """
        if not self.log_path.exists():
            return 0
        try:
            text = self.log_path.read_text(encoding="utf-8")
        except Exception:
            return 0
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) <= max_rows:
            return 0
        kept = lines[-max_rows:]
        tmp = self.log_path.with_suffix(self.log_path.suffix + ".tmp")
        try:
            tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
            tmp.replace(self.log_path)
        except Exception:
            return 0
        return len(lines) - len(kept)

    def _read_log(self, limit: int = 500) -> list[dict]:
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

    def _entries_for_period(self, days: int) -> list[dict]:
        """Return entries from the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in self._read_log(1000) if e.get("ts", "") >= cutoff]

    def daily_report(self, date: str | None = None) -> dict:
        """Generate report for a specific date (default: today)."""
        target = date or datetime.now().strftime("%Y-%m-%d")
        entries = [e for e in self._read_log(500) if e.get("ts", "").startswith(target)]

        if not entries:
            return {
                "date": target,
                "total_interactions": 0,
                "tasks": 0,
                "chats": 0,
                "avg_confidence": 0,
                "total_contradictions": 0,
                "total_unverified": 0,
                "topics_used": [],
                "by_intent": {},
                "low_confidence_count": 0,
                "high_confidence_count": 0,
            }

        tasks = [e for e in entries if not e.get("is_chat")]
        chats = [e for e in entries if e.get("is_chat")]
        confidences = [e["confidence"] for e in tasks if e.get("confidence") is not None]
        all_topics: set[str] = set()
        by_intent: dict[str, int] = {}
        total_contradictions = 0
        total_unverified = 0

        for e in entries:
            intent = e.get("intent", "unknown")
            by_intent[intent] = by_intent.get(intent, 0) + 1
            total_contradictions += e.get("contradictions", 0)
            total_unverified += e.get("unverified", 0)
            for t in e.get("topics_used", []):
                all_topics.add(t)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            "date": target,
            "total_interactions": len(entries),
            "tasks": len(tasks),
            "chats": len(chats),
            "avg_confidence": round(avg_conf, 1),
            "total_contradictions": total_contradictions,
            "total_unverified": total_unverified,
            "topics_used": sorted(all_topics),
            "by_intent": by_intent,
            "low_confidence_count": sum(1 for c in confidences if c < 70),
            "high_confidence_count": sum(1 for c in confidences if c >= 90),
        }

    def weekly_trend(self) -> list[dict]:
        """Return daily aggregates for the last 7 days."""
        results = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            results.append(self.daily_report(date))
        results.reverse()
        return results

    def detect_regression(self) -> list[dict]:
        """Compare this week vs last week. Flag domains where confidence dropped."""
        this_week = self._entries_for_period(7)
        last_week = [
            e for e in self._entries_for_period(14)
            if e not in this_week
        ]

        if not this_week or not last_week:
            return []

        def avg_by_domain(entries: list[dict]) -> dict[str, list[int]]:
            domain_confs: dict[str, list[int]] = defaultdict(list)
            for e in entries:
                if e.get("is_chat"):
                    continue
                for topic in e.get("topics_used", ["_general"]):
                    domain_confs[topic].append(e.get("confidence", 0))
            return domain_confs

        this_domains = avg_by_domain(this_week)
        last_domains = avg_by_domain(last_week)

        regressions = []
        for domain, confs in this_domains.items():
            if domain in last_domains:
                this_avg = sum(confs) / len(confs)
                last_avg = sum(last_domains[domain]) / len(last_domains[domain])
                if last_avg - this_avg >= 15:  # 15% drop threshold
                    regressions.append({
                        "domain": domain,
                        "this_week_avg": round(this_avg, 1),
                        "last_week_avg": round(last_avg, 1),
                        "drop": round(last_avg - this_avg, 1),
                        "sample_size": len(confs),
                    })

        regressions.sort(key=lambda r: -r["drop"])
        return regressions

    def suggest_priorities(self) -> list[dict]:
        """Based on eval data, suggest what to improve next."""
        entries = self._entries_for_period(7)
        if not entries:
            return []

        # Find topics with lowest average confidence
        topic_confs: dict[str, list[int]] = defaultdict(list)
        intent_confs: dict[str, list[int]] = defaultdict(list)
        for e in entries:
            if e.get("is_chat"):
                continue
            conf = e.get("confidence", 0)
            for topic in e.get("topics_used", []):
                topic_confs[topic].append(conf)
            intent_confs[e.get("intent", "unknown")].append(conf)

        suggestions = []

        # Weak topics
        for topic, confs in topic_confs.items():
            avg = sum(confs) / len(confs)
            if avg < 75 and len(confs) >= 2:
                suggestions.append({
                    "type": "weak_topic",
                    "topic": topic,
                    "avg_confidence": round(avg, 1),
                    "sample_size": len(confs),
                    "suggestion": f"Strengthen knowledge on '{topic}' — avg confidence {avg:.0f}%",
                    "priority": 8 if avg < 50 else 6,
                })

        # Weak intent types
        for intent, confs in intent_confs.items():
            avg = sum(confs) / len(confs)
            if avg < 70 and len(confs) >= 3:
                suggestions.append({
                    "type": "weak_intent",
                    "intent": intent,
                    "avg_confidence": round(avg, 1),
                    "sample_size": len(confs),
                    "suggestion": f"Improve '{intent}' question handling — avg confidence {avg:.0f}%",
                    "priority": 7,
                })

        # Regressions
        for reg in self.detect_regression():
            suggestions.append({
                "type": "regression",
                "topic": reg["domain"],
                "drop": reg["drop"],
                "suggestion": f"Regression on '{reg['domain']}': {reg['last_week_avg']}% → {reg['this_week_avg']}%",
                "priority": 9,
            })

        suggestions.sort(key=lambda s: -s.get("priority", 0))
        return suggestions[:10]

    def stats(self) -> dict:
        """Overall evaluation statistics."""
        all_entries = self._read_log(500)
        today = self.daily_report()
        trend = self.weekly_trend()
        regressions = self.detect_regression()
        suggestions = self.suggest_priorities()

        total_tasks = sum(1 for e in all_entries if not e.get("is_chat"))
        total_chats = sum(1 for e in all_entries if e.get("is_chat"))
        all_confs = [e["confidence"] for e in all_entries if e.get("confidence") is not None and not e.get("is_chat")]

        # Completion-gate counters. Surfaced here because a metric nobody
        # looks at is not a metric: `first_try_pass_rate` near 1.0 means the
        # proofs are decorative, and a climbing `judge_fail_open` means the
        # completion judge is silently absent.
        try:
            from .gate_metrics import summary as _gate_summary
            gates = _gate_summary(days=7)
        except Exception:
            gates = {}

        return {
            "total_logged": len(all_entries),
            "total_tasks": total_tasks,
            "total_chats": total_chats,
            "gates": gates,
            "overall_avg_confidence": round(sum(all_confs) / len(all_confs), 1) if all_confs else 0,
            "today": today,
            "weekly_trend": trend,
            "regressions": regressions,
            "suggestions": suggestions,
        }


EVALUATOR = SelfEvaluator()
