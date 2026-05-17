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
        session_key: str | None = None,
    ):
        self.id = id or uuid.uuid4().hex[:12]
        self.speaker_id = normalize_speaker(speaker_id)
        self.started = started or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ended = ended
        self.turns = turns or []
        self.title = title or ""
        self.archived = archived
        # session_key isolates conversation threads — same user across
        # different bots / DMs / groups gets distinct keys. Identity
        # (speaker_id) stays the same; only the thread differs. Old
        # sessions on disk have no session_key — fall back to
        # speaker_id (one-thread-per-user, the pre-refactor behaviour).
        self.session_key = (session_key or self.speaker_id).strip() or self.speaker_id

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
            "session_key": self.session_key,
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
            session_key=data.get("session_key") or None,
        )


class SessionManager:
    """Manages multiple sessions with disk persistence.

    Sessions are partitioned by `session_key`. The default session_key
    is the `speaker_id` itself, so for callers that don't yet know
    about chat-thread isolation, behaviour is unchanged: one thread
    per speaker. For channels.py / agent.run that DO pass an explicit
    session_key (e.g. `telegram:<bot>:<chat>:<user>`), each distinct
    chat gets its own thread even though the speaker_id is the same.

    Why two keys, not one? `speaker_id` is identity (roles, profile,
    permissions — Wife is Wife in every chat). `session_key` is
    where the conversation lives (a group chat ≠ a DM ≠ a different
    bot). Conflating them would break either roles or threading.

    On disk we now store `current_by_session_key: {session_key: session_id}`.
    Legacy state with `current_by_speaker` migrates lazily on load.
    """

    def __init__(self, path: Optional[Path] = None, archive_path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "sessions.json")
        self.archive_path = archive_path or (kb_dir / "sessions_archive.json")
        self._sessions: list[Session] = []
        # session_key -> session_id mapping. One entry per active thread.
        self._current_by_session_key: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._sessions = [Session.from_dict(s) for s in data.get("sessions", [])]
                # New field. If absent, fall back to the legacy
                # speaker-keyed map (each speaker_id becomes its own
                # session_key — the pre-refactor 1:1 behaviour).
                cbsk = data.get("current_by_session_key")
                if cbsk:
                    self._current_by_session_key = dict(cbsk)
                else:
                    self._current_by_session_key = dict(data.get("current_by_speaker") or {})
                # Back-compat: an older single `current_id` becomes the
                # WebUI default's current.
                legacy_current = data.get("current_id")
                if legacy_current and DEFAULT_SPEAKER not in self._current_by_session_key:
                    self._current_by_session_key[DEFAULT_SPEAKER] = legacy_current
            except Exception:
                self._sessions = []
                self._current_by_session_key = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_by_session_key": dict(self._current_by_session_key),
                # Back-compat surface for old readers that look at the
                # legacy speaker-keyed map. For each known speaker_id,
                # surface ANY of its session_keys' current id — picking
                # the first match keeps "the" current useful for
                # single-thread consumers (the WebUI).
                "current_by_speaker": self._derive_current_by_speaker(),
                # Back-compat surface for old readers: surface the
                # WebUI default's current as `current_id`. Always None
                # when no WebUI session exists yet.
                "current_id": self._current_by_session_key.get(DEFAULT_SPEAKER),
                "sessions": [s.to_dict() for s in self._sessions],
            }
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _derive_current_by_speaker(self) -> dict[str, str]:
        """Legacy speaker_id → session_id map computed from the
        session_key-indexed source of truth. Picks the newest-started
        session per speaker so single-thread consumers (the old WebUI
        path) see the freshest one."""
        by_speaker: dict[str, tuple[str, str]] = {}
        for key, sid in self._current_by_session_key.items():
            session = next((s for s in self._sessions if s.id == sid), None)
            if session is None:
                continue
            sp = session.speaker_id
            prev = by_speaker.get(sp)
            if prev is None or session.started > prev[1]:
                by_speaker[sp] = (sid, session.started)
        return {sp: sid for sp, (sid, _) in by_speaker.items()}

    def _resolve_key(self, speaker_id: str, session_key: str | None) -> tuple[str, str]:
        """Normalise both keys. When session_key is not provided,
        fall back to speaker_id (old one-thread-per-speaker behaviour)."""
        sp = normalize_speaker(speaker_id)
        key = (session_key or "").strip() or sp
        return sp, key

    def current_for(
        self,
        speaker_id: str = DEFAULT_SPEAKER,
        *,
        session_key: str | None = None,
    ) -> Session | None:
        """Active session for this thread. None if it doesn't exist
        yet. `session_key` selects the conversation thread (defaults
        to the speaker_id for back-compat)."""
        _, key = self._resolve_key(speaker_id, session_key)
        sid = self._current_by_session_key.get(key)
        if not sid:
            return None
        for s in self._sessions:
            if s.id == sid:
                return s
        # Stale pointer (session was deleted) — clear it.
        self._current_by_session_key.pop(key, None)
        return None

    @property
    def current(self) -> Session | None:
        """Back-compat: 'the' current session is the WebUI default's."""
        return self.current_for(DEFAULT_SPEAKER)

    def get_or_create_current(
        self,
        speaker_id: str = DEFAULT_SPEAKER,
        *,
        session_key: str | None = None,
    ) -> Session:
        """Active session for this thread, creating one if missing.
        Each thread (session_key) gets independent sessions even if
        the speaker_id is the same — Wife DMing vs Wife in a group
        get distinct threads, both still owned by speaker_id `telegram:<wife>`."""
        sp, key = self._resolve_key(speaker_id, session_key)
        session = self.current_for(sp, session_key=key)
        if session is None:
            session = Session(speaker_id=sp, session_key=key)
            self._sessions.append(session)
            self._current_by_session_key[key] = session.id
            self._save()
        return session

    def add_turn(
        self,
        turn: dict,
        *,
        speaker_id: str = DEFAULT_SPEAKER,
        session_key: str | None = None,
    ) -> None:
        """Append a turn to the thread's current session."""
        session = self.get_or_create_current(speaker_id, session_key=session_key)
        session.turns.append(turn)
        if not session.title and turn.get("user"):
            text = turn["user"]
            session.title = text[:60] + ("..." if len(text) > 60 else "")
        self._save()

    def new_session(
        self,
        *,
        speaker_id: str = DEFAULT_SPEAKER,
        session_key: str | None = None,
    ) -> Session:
        """End this thread's current session and start a new one."""
        sp, key = self._resolve_key(speaker_id, session_key)
        old = self.current_for(sp, session_key=key)
        if old and not old.ended:
            old.ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session = Session(speaker_id=sp, session_key=key)
        self._sessions.append(session)
        self._current_by_session_key[key] = session.id
        self._save()
        return session

    def end_current(
        self,
        *,
        speaker_id: str = DEFAULT_SPEAKER,
        session_key: str | None = None,
    ) -> None:
        """End this thread's session without starting a new one."""
        sp, key = self._resolve_key(speaker_id, session_key)
        session = self.current_for(sp, session_key=key)
        if session and not session.ended:
            session.ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._current_by_session_key.pop(key, None)
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
