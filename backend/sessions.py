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


DEFAULT_SPEAKER = "webui:default"


def normalize_speaker(speaker_id: str | None) -> str:
    """Coerce a speaker_id to canonical form ('<channel>:<user_id>').

    None / empty → DEFAULT_SPEAKER. Strings without a colon are
    assumed to be channels with no user (e.g. legacy 'webui') and
    get ':default' appended."""
    if not speaker_id:
        return DEFAULT_SPEAKER
    s = speaker_id.strip()
    if not s:
        return DEFAULT_SPEAKER
    if ":" not in s:
        return f"{s}:default"
    return s


class Session:
    """A single conversation session. Each session belongs to one
    `speaker_id` — the channel-qualified identity of who is talking
    to the agent. Different speakers (WebUI user vs each Telegram
    user) get their own independent sessions."""

    def __init__(
        self,
        id: str | None = None,
        speaker_id: str = DEFAULT_SPEAKER,
        started: str | None = None,
        ended: str | None = None,
        turns: list[dict] | None = None,
        title: str | None = None,
        archived: bool = False,
    ):
        self.id = id or uuid.uuid4().hex[:12]
        self.speaker_id = normalize_speaker(speaker_id)
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
            "speaker_id": self.speaker_id,
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
            "speaker_id": self.speaker_id,
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
            speaker_id=data.get("speaker_id") or DEFAULT_SPEAKER,
            started=data.get("started"),
            ended=data.get("ended"),
            turns=data.get("turns", []),
            title=data.get("title", ""),
            archived=data.get("archived", False),
        )


class SessionManager:
    """Manages multiple sessions with disk persistence.

    Sessions are partitioned by `speaker_id` ('<channel>:<user_id>'):
    each speaker (WebUI user, each Telegram user, future channels)
    has its OWN current session. The agent's run() picks the right
    one via `get_or_create_current(speaker_id=...)`.

    The on-disk format stores `current_by_speaker: {speaker_id: session_id}`
    so independent conversations don't trample each other on every
    save. The legacy top-level `current_id` field is also written
    for back-compat with consumers that look at "the" current
    session — it holds the WebUI default's current id.
    """

    def __init__(self, path: Optional[Path] = None, archive_path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "sessions.json")
        self.archive_path = archive_path or (kb_dir / "sessions_archive.json")
        self._sessions: list[Session] = []
        # speaker_id -> session_id mapping. One entry per active speaker.
        self._current_by_speaker: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._sessions = [Session.from_dict(s) for s in data.get("sessions", [])]
                self._current_by_speaker = dict(data.get("current_by_speaker") or {})
                # Back-compat: an older single `current_id` becomes the
                # WebUI default's current.
                legacy_current = data.get("current_id")
                if legacy_current and DEFAULT_SPEAKER not in self._current_by_speaker:
                    self._current_by_speaker[DEFAULT_SPEAKER] = legacy_current
            except Exception:
                self._sessions = []
                self._current_by_speaker = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_by_speaker": dict(self._current_by_speaker),
                # Back-compat surface for old readers: surface the
                # WebUI default's current as `current_id`. Always None
                # when no WebUI session exists yet.
                "current_id": self._current_by_speaker.get(DEFAULT_SPEAKER),
                "sessions": [s.to_dict() for s in self._sessions],
            }
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def current_for(self, speaker_id: str = DEFAULT_SPEAKER) -> Session | None:
        """Active session for this speaker. None if they haven't
        started a session yet."""
        speaker_id = normalize_speaker(speaker_id)
        sid = self._current_by_speaker.get(speaker_id)
        if not sid:
            return None
        for s in self._sessions:
            if s.id == sid:
                return s
        # Stale pointer (session was deleted) — clear it.
        self._current_by_speaker.pop(speaker_id, None)
        return None

    @property
    def current(self) -> Session | None:
        """Back-compat: 'the' current session is the WebUI default's."""
        return self.current_for(DEFAULT_SPEAKER)

    def get_or_create_current(self, speaker_id: str = DEFAULT_SPEAKER) -> Session:
        """Active session for this speaker, creating one if missing.
        Each speaker gets independent sessions — switching speakers
        does NOT carry context across."""
        speaker_id = normalize_speaker(speaker_id)
        session = self.current_for(speaker_id)
        if session is None:
            session = Session(speaker_id=speaker_id)
            self._sessions.append(session)
            self._current_by_speaker[speaker_id] = session.id
            self._save()
        return session

    def add_turn(self, turn: dict, *, speaker_id: str = DEFAULT_SPEAKER) -> None:
        """Append a turn to the speaker's current session."""
        session = self.get_or_create_current(speaker_id)
        session.turns.append(turn)
        if not session.title and turn.get("user"):
            text = turn["user"]
            session.title = text[:60] + ("..." if len(text) > 60 else "")
        self._save()

    def new_session(self, *, speaker_id: str = DEFAULT_SPEAKER) -> Session:
        """End this speaker's current session and start a new one."""
        speaker_id = normalize_speaker(speaker_id)
        old = self.current_for(speaker_id)
        if old and not old.ended:
            old.ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session = Session(speaker_id=speaker_id)
        self._sessions.append(session)
        self._current_by_speaker[speaker_id] = session.id
        self._save()
        return session

    def end_current(self, *, speaker_id: str = DEFAULT_SPEAKER) -> None:
        """End this speaker's session without starting a new one."""
        speaker_id = normalize_speaker(speaker_id)
        session = self.current_for(speaker_id)
        if session and not session.ended:
            session.ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._current_by_speaker.pop(speaker_id, None)
            self._save()

    def get_session(self, session_id: str) -> Session | None:
        for s in self._sessions:
            if s.id == session_id:
                return s
        for s in self._load_archive():
            if s.id == session_id:
                return s
        return None

    def list_sessions(
        self,
        include_archived: bool = False,
        *,
        speaker_id: Optional[str] = None,
    ) -> list[dict]:
        """List sessions (summaries only). When `speaker_id` is set,
        only sessions belonging to that speaker."""
        if speaker_id is not None:
            speaker_id = normalize_speaker(speaker_id)
        def _match(s: Session) -> bool:
            return speaker_id is None or s.speaker_id == speaker_id
        result = [s.summary() for s in reversed(self._sessions) if _match(s)]
        if include_archived:
            archived = self._load_archive()
            result.extend(s.summary() for s in reversed(archived) if _match(s))
        return result

    def list_speakers(self) -> list[dict]:
        """Speakers known to the system with their summary stats —
        used by the WebUI Sessions panel to group sessions per
        speaker. Returns:
          [{speaker_id, session_count, last_active}, ...]
        sorted newest-first by last_active.
        """
        agg: dict[str, dict] = {}
        for s in self._sessions:
            sp = s.speaker_id or DEFAULT_SPEAKER
            slot = agg.setdefault(sp, {"speaker_id": sp, "session_count": 0, "last_active": ""})
            slot["session_count"] += 1
            last = s.ended or s.started
            if last and last > slot["last_active"]:
                slot["last_active"] = last
        return sorted(agg.values(), key=lambda r: r["last_active"], reverse=True)

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
