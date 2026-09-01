"""Living project tracker — structured tracker.json layered on the existing
knowledge/projects/<slug>/ dirs (coexists with the markdown journal).

See docs/superpowers/specs/2026-06-17-project-tracker-design.md.
Unified model: a project contains steps; a step with a due_at IS a check-in;
a standalone reminder is a one-step project with domain="inbox".
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .knowledge_manager import _slug
from .paths import data_dir, write_atomic_json
from .sessions import normalize_speaker

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_step(title: str, *, due_at: str = "", check_in_kind: str = "ask_status") -> dict:
    return {
        "id": "st_" + uuid.uuid4().hex[:10],
        "title": title.strip(),
        "status": "pending",          # pending | active | done | blocked
        "due_at": due_at or "",
        "check_in_kind": check_in_kind,  # ask_status | remind | none
        "note": "",
        "last_checked_at": None,
        # Follow-up state (2026-09-01). A check-in used to fire once and the
        # step stayed pending forever if nobody answered.
        "nudges": 0,             # follow-ups already sent
        "last_nudge_at": None,
        "next_check_at": "",     # when the next one fires; may differ from due_at
    }


def may_access(tracker: dict, speaker_id: str) -> bool:
    """May this speaker see/modify this tracker?

    Owners may reach anything. Everyone else reaches only what they created.

    A tracker written before 2026-09-01 carries no `owner` field. Those are
    treated as the owner's rather than as everyone's: the alternative --
    "unowned means public" -- is the very leak this function exists to
    close, and the two that existed on prod were in fact his.
    """
    from .roles import is_owner
    who = normalize_speaker(speaker_id or "")
    if not who:
        return False
    if is_owner(who):
        return True
    return who == normalize_speaker(tracker.get("owner") or "")


class TrackerStore:
    @property
    def _base(self) -> Path:
        # Resolve via data_dir() each call so that monkeypatched HRANT_DATA_DIR
        # is honoured in tests without requiring a full module reload of config.
        b = data_dir() / "knowledge" / "projects"
        b.mkdir(parents=True, exist_ok=True)
        return b

    def _path(self, tracker_id: str) -> Path | None:
        for d in self._base.iterdir():
            if d.is_dir():
                p = d / "tracker.json"
                if p.exists():
                    try:
                        if json.loads(p.read_text(encoding="utf-8")).get("id") == tracker_id:
                            return p
                    except Exception as e:
                        log.warning("tracker: unreadable %s (%s); skipping", p, e)
                        continue
        return None

    def create(self, *, title: str, domain: str = "work",
               steps: list[dict] | None = None,
               requested_by: str = "webui:default") -> dict:
        tid = "trk_" + uuid.uuid4().hex[:10]
        # Dir = slug (readable, coexists with the markdown journal). On a slug
        # COLLISION with an existing tracker, suffix the id so two same-titled
        # projects never overwrite each other. The chosen dir is stored as
        # `slug` for O(1) saves.
        dirname = _slug(title or tid)
        if (self._base / dirname / "tracker.json").exists():
            dirname = f"{dirname}-{tid[-8:]}"
        d = self._base / dirname
        d.mkdir(parents=True, exist_ok=True)
        tracker = {
            "id": tid,
            "title": title.strip(),
            "domain": domain,
            "status": "active",
            "created_at": _now(),
            "owner": normalize_speaker(requested_by or ""),
            "slug": dirname,
            "steps": [
                _new_step(
                    s["title"],
                    due_at=s.get("due_at", ""),
                    check_in_kind=s.get("check_in_kind", "ask_status"),
                )
                for s in (steps or []) if s.get("title")
            ],
            "notes": "",
        }
        write_atomic_json(d / "tracker.json", tracker)
        return tracker

    def create_inbox_reminder(self, *, title: str, due_at: str) -> dict:
        return self.create(
            title=title, domain="inbox",
            steps=[{"title": title, "due_at": due_at, "check_in_kind": "remind"}],
        )

    def get(self, tracker_id: str) -> dict | None:
        p = self._path(tracker_id)
        if not p:
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self, status: str = "active", *,
             requested_by: str | None = None) -> list[dict]:
        """Trackers, newest first. With `requested_by`, only the ones that
        speaker may see.

        The filter was missing until 2026-09-01: `create` recorded no owner
        and this method returned every tracker on disk, so a second user
        would have been shown the owner's whole task list. The owner's rule
        for reminders -- "notifications need to work isolated for each
        user" -- applies to the list they hang off just as much.
        """
        out = []
        for d in self._base.iterdir():
            p = d / "tracker.json" if d.is_dir() else None
            if p and p.exists():
                try:
                    t = json.loads(p.read_text(encoding="utf-8"))
                except Exception as e:
                    log.warning("tracker: unreadable %s (%s); skipping", p, e)
                    continue
                if status not in ("", "all") and t.get("status") != status:
                    continue
                if requested_by is not None and not may_access(t, requested_by):
                    continue
                out.append(t)
        return sorted(out, key=lambda t: t.get("created_at", ""), reverse=True)

    def _save(self, tracker: dict) -> None:
        # Use the dir chosen at create time (`slug`); fall back to the
        # title-derived slug for pre-existing trackers without the field.
        # No re-scan of the projects dir on every mutation.
        dirname = tracker.get("slug") or _slug(tracker["title"] or tracker["id"])
        write_atomic_json(self._base / dirname / "tracker.json", tracker)

    def set_status(self, tracker_id: str, status: str) -> dict | None:
        t = self.get(tracker_id)
        if not t:
            return None
        t["status"] = status
        self._save(t)
        return t

    def _schedule_check_in(self, tracker: dict, step: dict, requested_by: str) -> None:
        """Create/refresh the check_in row for a step with a due_at. Cancels
        any prior pending check-in for this step first (idempotent reschedule)."""
        from .scheduled_messages import schedule, _read_all, cancel
        for r in _read_all():
            if (r.get("kind") == "check_in" and r["status"] == "pending"
                    and (r.get("meta") or {}).get("step_id") == step["id"]):
                cancel(r["id"])
        # A follow-up overrides the deadline for SCHEDULING only: `due_at`
        # stays the real date so it still reads correctly to the user.
        fire_at = step.get("next_check_at") or step.get("due_at")
        if fire_at and step.get("check_in_kind") != "none" \
                and step.get("status") not in ("done", "blocked", "stalled"):
            schedule(
                target_speaker=requested_by, text="", due_at=fire_at,
                requested_by=requested_by, kind="check_in",
                meta={"tracker_id": tracker["id"], "step_id": step["id"],
                      "check_in_kind": step.get("check_in_kind", "ask_status")},
            )

    def arm_follow_up(self, tracker_id: str, step_id: str, *,
                      requested_by: str = "webui:default") -> dict | None:
        """Record that we just nudged, and schedule the next one.

        Returns the updated step, or None when the step has spent every
        follow-up it is allowed -- the caller then parks it as stalled and
        asks once whether it still matters.
        """
        from .follow_up import next_nudge_at
        t = self.get(tracker_id)
        if not t:
            return None
        for step in t["steps"]:
            if step["id"] != step_id:
                continue
            # Count FIRST, then space off that count -- incrementing before
            # the lookup skipped BACKOFF_HOURS[0] entirely, so the 1h gap
            # was unreachable and the first follow-up landed 3h out. Found
            # on a live run 2026-09-01, not by reading: the armed stamps
            # came back +3h, +8h, +24h, +48h.
            sent = int(step.get("nudges") or 0)
            when = next_nudge_at(sent)
            if not when:
                # Spent. Do NOT count this call: `nudges` is what the agent
                # tells the user ("I have asked three times"), so it must be
                # the number actually sent, not the number of attempts.
                step["next_check_at"] = ""
                self._save(t)
                return None
            step["nudges"] = sent + 1
            step["last_nudge_at"] = _now()
            step["next_check_at"] = when
            self._save(t)
            self._schedule_check_in(t, step, requested_by)
            return step
        return None

    def park_stalled(self, tracker_id: str, step_id: str) -> dict | None:
        """Stop asking. The step is not done and not refused -- it is simply
        unanswered, which is a distinct state worth being able to see."""
        t = self.get(tracker_id)
        if not t:
            return None
        for step in t["steps"]:
            if step["id"] == step_id:
                step["status"] = "stalled"
                step["next_check_at"] = ""
                self._save(t)
                return step
        return None

    def add_step(self, tracker_id: str, title: str, *, due_at: str = "",
                 check_in_kind: str = "ask_status",
                 requested_by: str = "webui:default") -> dict | None:
        t = self.get(tracker_id)
        if not t:
            return None
        step = _new_step(title, due_at=due_at, check_in_kind=check_in_kind)
        t["steps"].append(step)
        self._save(t)
        self._schedule_check_in(t, step, requested_by)
        return step

    def update_step(self, tracker_id: str, step_id: str, *, status: str | None = None,
                    note: str | None = None, due_at: str | None = None,
                    title: str | None = None,
                    requested_by: str = "webui:default") -> dict | None:
        t = self.get(tracker_id)
        if not t:
            return None
        for step in t["steps"]:
            if step["id"] == step_id:
                if status is not None:
                    step["status"] = status
                    # Closing a step must silence it: leaving next_check_at
                    # set would re-arm a nudge for something already done.
                    if status in ("done", "blocked", "stalled"):
                        step["next_check_at"] = ""
                if note is not None:
                    step["note"] = note
                if due_at is not None:
                    step["due_at"] = due_at
                if title is not None:
                    step["title"] = title.strip()
                self._save(t)
                self._schedule_check_in(t, step, requested_by)
                return step
        return None


TRACKERS = TrackerStore()
