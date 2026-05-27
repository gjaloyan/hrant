"""FIRE_MEMORY_CONSOLIDATION — daily session review, route facts to memory tiers."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from ..lever import Lever, resolve_knowledge_path
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

from backend.llm import router, TaskType

log = logging.getLogger(__name__)

CONSOLIDATION_SYSTEM = """You are the daily memory consolidation module.

Review the session transcript and extract information in THREE tiers:
1. user_profile_facts: things we learned about the user (role, location, preferences, projects).
2. durable_facts: world-facts worth remembering (prices, specs, events, entity relationships).
3. topic_threads: short phrases describing what the session was about.

Also produce a 2-3 sentence session_summary.

Return strictly JSON:
{
  "session_summary": "...",
  "user_profile_facts": [{"summary": "...", "confidence": 0-1, "category": "role|location|preference|project|general"}],
  "durable_facts": [{"summary": "...", "triples": [["e1","r","e2"]], "tags": [...], "category": "price|technical|event|location|preference|relationship|rule|general", "confidence": 0-1}],
  "topic_threads": ["topic phrase", ...]
}

Rules:
- Skip greetings, small talk, agent reasoning.
- Confidence >=0.8 for durable facts worth promoting.
- Max 8 durable_facts, max 5 user_profile_facts, max 5 topic_threads.
- Do NOT include anything about the agent's own identity or values."""

DEFAULT_SESSIONS_PATH = Path("knowledge/sessions.json")
DEFAULT_USER_MD_PATH = Path("knowledge/identity/user.md")
DEFAULT_FACTS_PATH = Path("knowledge/memory_facts.jsonl")
DEDUP_WINDOW = 200
CONFIDENCE_THRESHOLD = 0.8


class FIRE_MEMORY_CONSOLIDATION(Lever):
    name = "FIRE_MEMORY_CONSOLIDATION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=30.0, tokens_in=4000, tokens_out=1500)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        sessions_path = resolve_knowledge_path(params.get("sessions_path") or DEFAULT_SESSIONS_PATH)
        user_md_path = resolve_knowledge_path(params.get("user_md_path") or DEFAULT_USER_MD_PATH)
        facts_path = resolve_knowledge_path(params.get("memory_facts_path") or DEFAULT_FACTS_PATH)
        max_sessions = int(params.get("max_sessions", 5))

        sessions_blob = self._load_sessions(sessions_path)
        candidates = [
            s for s in sessions_blob.get("sessions", [])
            if not s.get("consolidated") and s.get("turns")
        ]
        if not candidates:
            return self._skip(params, started, "no_unconsolidated_sessions")

        targets = candidates[:max_sessions]
        existing_profile = self._load_user_md_lines(user_md_path)
        existing_fact_summaries = self._load_recent_fact_summaries(facts_path)

        profile_added = 0
        facts_added = 0
        all_threads: list[str] = []

        for session in targets:
            transcript = self._session_transcript(session)
            current_profile_snapshot = user_md_path.read_text(encoding="utf-8") if user_md_path.exists() else ""
            try:
                data = router().call_json(
                    TaskType.TASK_ANALYSIS,
                    CONSOLIDATION_SYSTEM,
                    f"SESSION TRANSCRIPT:\n{transcript}\n\nCURRENT USER PROFILE (for dedup):\n{current_profile_snapshot[:2000]}",
                    max_tokens=2000,
                    temperature=0.2,
                )
            except Exception as exc:
                log.warning("consolidation cortex call failed for session %s: %s", session.get("id"), exc)
                continue

            summary = str(data.get("session_summary", ""))
            profile_facts = data.get("user_profile_facts", []) or []
            durable_facts = data.get("durable_facts", []) or []
            threads = data.get("topic_threads", []) or []

            added_profile = self._append_profile_facts(user_md_path, profile_facts, existing_profile)
            added_facts = self._append_durable_facts(facts_path, durable_facts, existing_fact_summaries, session.get("id", ""))

            profile_added += added_profile
            facts_added += added_facts
            all_threads.extend(str(t) for t in threads if t)

            session["consolidated"] = True
            session["summary"] = summary

        self._save_sessions(sessions_path, sessions_blob)

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "sessions_processed": len(targets),
                "profile_added": profile_added,
                "facts_added": facts_added,
                "threads_queued": len(all_threads),
            },
            reason=f"consolidated_{len(targets)}_sessions",
            follow_ups=all_threads[:10],
        )

    def _skip(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )

    def _load_sessions(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"current_id": None, "sessions": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"current_id": None, "sessions": []}
        if not isinstance(data, dict):
            return {"current_id": None, "sessions": []}
        data.setdefault("sessions", [])
        return data

    def _save_sessions(self, path: Path, blob: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _session_transcript(self, session: dict[str, Any]) -> str:
        lines: list[str] = []
        for turn in session.get("turns", []):
            u = str(turn.get("user", "")).strip()
            a = str(turn.get("answer", "")).strip()
            if u:
                lines.append(f"USER: {u[:800]}")
            if a:
                lines.append(f"AGENT: {a[:800]}")
        return "\n".join(lines)[:8000]

    def _load_user_md_lines(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_recent_fact_summaries(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-DEDUP_WINDOW:]
        except OSError:
            return set()
        out: set[str] = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                s = obj.get("summary")
                if s:
                    out.add(str(s).strip().lower())
            except json.JSONDecodeError:
                continue
        return out

    def _append_profile_facts(self, path: Path, facts: list[dict], existing_lower_lines: list[str]) -> int:
        added = 0
        to_write: list[str] = []
        existing_blob = " ".join(existing_lower_lines)
        for f in facts:
            conf = float(f.get("confidence", 0.0) or 0.0)
            if conf < CONFIDENCE_THRESHOLD:
                continue
            summary = str(f.get("summary", "")).strip()
            if not summary:
                continue
            key = summary.lower()
            if key in existing_blob:
                continue
            today = date.today().isoformat()
            to_write.append(f"- {summary}  _(добавлено {today})_")
            existing_blob += " " + key
            added += 1
        if to_write:
            path.parent.mkdir(parents=True, exist_ok=True)
            prefix = "" if path.exists() else "# User Profile\n\n## О пользователе\n"
            content = prefix + ("\n".join(to_write) + "\n")
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
        return added

    def _append_durable_facts(
        self,
        path: Path,
        facts: list[dict],
        existing_summaries: set[str],
        session_id: str,
    ) -> int:
        added = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for raw in facts:
                conf = float(raw.get("confidence", 0.0) or 0.0)
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                summary = str(raw.get("summary", "")).strip()
                if not summary:
                    continue
                key = summary.lower()
                if key in existing_summaries:
                    continue
                entry = {
                    "summary": summary,
                    "triples": [list(t) for t in raw.get("triples", []) if isinstance(t, (list, tuple)) and len(t) >= 3],
                    "tags": list(raw.get("tags", []) or []),
                    "category": str(raw.get("category", "general")),
                    "confidence": conf,
                    "ts": utcnow().isoformat(),
                    "source_turn": f"consolidation:{session_id}",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing_summaries.add(key)
                added += 1
        return added
