"""Session management: groups conversation turns into sessions.

Sessions persist to disk so they survive server restarts. Each session
has a unique ID, start/end timestamps, a list of turns, and summary stats.

Sessions are never deleted automatically — only archived after a
configurable period (default 90 days). Archived sessions are moved to
a separate file to keep the active list fast.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import CONFIG


class Session:
    """A single conversation session."""

    def __init__(
        self,
        id: str | None = None,
        started: str | None = None,
        ended: str | None = None,
        turns: list[dict] | None = None,
        title: str | None = None,
        archived: bool = False,
    ):
        self.id = id or uuid.uuid4().hex[:12]
        self.started = started or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ended = ended
        self.turns = turns or []
        self.title = title or ""
        self.archived = archived

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def total_confidence(self) -> float:
        confs = [t.get("confidence", 0) for t in self.turns if t.get("confidence")]
        return sum(confs) / len(confs) if confs else 0.0

    @property
    def intents(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.turns:
            intent = t.get("intent", "unknown")
            counts[intent] = counts.get(intent, 0) + 1
        return counts

    @property
    def topics_used(self) -> list[str]:
        topics: set[str] = set()
        for t in self.turns:
            for topic in t.get("topics", []):
                topics.add(topic)
        return sorted(topics)

    @property
    def duration_seconds(self) -> float | None:
        if not self.ended:
            return None
        try:
            fmt = "%Y-%m-%d %H:%M:%S"
            s = datetime.strptime(self.started, fmt)
            e = datetime.strptime(self.ended, fmt)
            return (e - s).total_seconds()
        except Exception:
            return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started": self.started,
            "ended": self.ended,
            "title": self.title,
            "archived": self.archived,
            "turns": self.turns,
            "turn_count": self.turn_count,
            "avg_confidence": round(self.total_confidence, 1),
            "intents": self.intents,
            "topics_used": self.topics_used,
            "duration_seconds": self.duration_seconds,
        }

    def summary(self) -> dict:
        """Lightweight summary without full turns (for list view)."""
        return {
            "id": self.id,
            "started": self.started,
            "ended": self.ended,
            "title": self.title,
            "archived": self.archived,
            "turn_count": self.turn_count,
            "avg_confidence": round(self.total_confidence, 1),
            "intents": self.intents,
            "topics_used": self.topics_used,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            id=data.get("id"),
            started=data.get("started"),
            ended=data.get("ended"),
            turns=data.get("turns", []),
            title=data.get("title", ""),
            archived=data.get("archived", False),
        )


class SessionManager:
    """Manages multiple sessions with disk persistence."""

    def __init__(self, path: Optional[Path] = None, archive_path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "sessions.json")
        self.archive_path = archive_path or (kb_dir / "sessions_archive.json")
        self._sessions: list[Session] = []
        self._current_id: str | None = None
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._sessions = [Session.from_dict(s) for s in data.get("sessions", [])]
                self._current_id = data.get("current_id")
            except Exception:
                self._sessions = []
                self._current_id = None

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_id": self._current_id,
                "sessions": [s.to_dict() for s in self._sessions],
            }
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @property
    def current(self) -> Session | None:
        if self._current_id:
            for s in self._sessions:
                if s.id == self._current_id:
                    return s
        return None

    def get_or_create_current(self) -> Session:
        """Get the current active session, or create a new one."""
        session = self.current
        if session is None:
            session = Session()
            self._sessions.append(session)
            self._current_id = session.id
            self._save()
        return session

    def add_turn(self, turn: dict) -> None:
        """Add a turn to the current session."""
        session = self.get_or_create_current()
        session.turns.append(turn)
        # Auto-title from first user message
        if not session.title and turn.get("user"):
            text = turn["user"]
            session.title = text[:60] + ("..." if len(text) > 60 else "")
        self._save()

    def new_session(self) -> Session:
        """End the current session and start a new one."""
        old = self.current
        if old and not old.ended:
            old.ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session = Session()
        self._sessions.append(session)
        self._current_id = session.id
        self._save()
        return session

    def end_current(self) -> None:
        """End the current session without starting a new one."""
        session = self.current
        if session and not session.ended:
            session.ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._current_id = None
            self._save()

    def get_session(self, session_id: str) -> Session | None:
        for s in self._sessions:
            if s.id == session_id:
                return s
        # Check archive
        for s in self._load_archive():
            if s.id == session_id:
                return s
        return None

    def list_sessions(self, include_archived: bool = False) -> list[dict]:
        """List all sessions (summaries only)."""
        result = [s.summary() for s in reversed(self._sessions)]
        if include_archived:
            archived = self._load_archive()
            result.extend(s.summary() for s in reversed(archived))
        return result

    def archive_old(self, days: int = 90) -> int:
        """Move sessions older than `days` to the archive file."""
        cutoff = datetime.now() - timedelta(days=days)
        to_archive: list[Session] = []
        remaining: list[Session] = []

        for s in self._sessions:
            try:
                ended = datetime.strptime(s.ended, "%Y-%m-%d %H:%M:%S") if s.ended else None
            except Exception:
                ended = None

            if ended and ended < cutoff:
                s.archived = True
                to_archive.append(s)
            else:
                remaining.append(s)

        if not to_archive:
            return 0

        # Append to archive file
        existing = self._load_archive()
        existing.extend(to_archive)
        try:
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            self.archive_path.write_text(
                json.dumps([s.to_dict() for s in existing], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return 0

        self._sessions = remaining
        if self._current_id and not any(s.id == self._current_id for s in remaining):
            self._current_id = None
        self._save()
        return len(to_archive)

    def _load_archive(self) -> list[Session]:
        if not self.archive_path.exists():
            return []
        try:
            data = json.loads(self.archive_path.read_text(encoding="utf-8"))
            return [Session.from_dict(s) for s in data]
        except Exception:
            return []

    def stats(self) -> dict:
        """Aggregate stats across all sessions."""
        total = len(self._sessions)
        total_turns = sum(s.turn_count for s in self._sessions)
        all_intents: dict[str, int] = {}
        daily_counts: dict[str, int] = {}
        confidence_over_time: list[dict] = []

        for s in self._sessions:
            for intent, count in s.intents.items():
                all_intents[intent] = all_intents.get(intent, 0) + count
            # Count sessions per day
            day = s.started[:10] if s.started else "unknown"
            daily_counts[day] = daily_counts.get(day, 0) + 1
            # Confidence per session
            if s.turn_count > 0:
                confidence_over_time.append({
                    "session_id": s.id,
                    "date": s.started,
                    "avg_confidence": round(s.total_confidence, 1),
                    "turns": s.turn_count,
                })

        return {
            "total_sessions": total,
            "total_turns": total_turns,
            "intents": all_intents,
            "daily_counts": daily_counts,
            "confidence_over_time": confidence_over_time,
            "archived_count": len(self._load_archive()),
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        for i, s in enumerate(self._sessions):
            if s.id == session_id:
                self._sessions.pop(i)
                if self._current_id == session_id:
                    self._current_id = None
                self._save()
                return True
        return False


SESSIONS = SessionManager()
