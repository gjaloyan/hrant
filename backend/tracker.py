"""Living project tracker — structured tracker.json layered on the existing
knowledge/projects/<slug>/ dirs (coexists with the markdown journal).

See docs/superpowers/specs/2026-06-17-project-tracker-design.md.
Unified model: a project contains steps; a step with a due_at IS a check-in;
a standalone reminder is a one-step project with domain="inbox".
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from .knowledge_manager import _slug
from .paths import data_dir, write_atomic_json


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
    }


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
                        import json
                        if json.loads(p.read_text(encoding="utf-8")).get("id") == tracker_id:
                            return p
                    except Exception:
                        continue
        return None

    def create(self, *, title: str, domain: str = "work",
               steps: list[dict] | None = None) -> dict:
        tid = "trk_" + uuid.uuid4().hex[:10]
        d = self._base / _slug(title or tid)
        d.mkdir(parents=True, exist_ok=True)
        tracker = {
            "id": tid,
            "title": title.strip(),
            "domain": domain,
            "status": "active",
            "created_at": _now(),
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
        import json
        p = self._path(tracker_id)
        if not p:
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self, status: str = "active") -> list[dict]:
        import json
        out = []
        for d in self._base.iterdir():
            p = d / "tracker.json" if d.is_dir() else None
            if p and p.exists():
                try:
                    t = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if status in ("", "all") or t.get("status") == status:
                    out.append(t)
        return sorted(out, key=lambda t: t.get("created_at", ""), reverse=True)

    def _save(self, tracker: dict) -> None:
        p = self._path(tracker["id"])
        if p:
            write_atomic_json(p, tracker)

    def set_status(self, tracker_id: str, status: str) -> dict | None:
        t = self.get(tracker_id)
        if not t:
            return None
        t["status"] = status
        self._save(t)
        return t

    def add_step(self, tracker_id: str, title: str, *, due_at: str = "",
                 check_in_kind: str = "ask_status") -> dict | None:
        t = self.get(tracker_id)
        if not t:
            return None
        step = _new_step(title, due_at=due_at, check_in_kind=check_in_kind)
        t["steps"].append(step)
        self._save(t)
        return step

    def update_step(self, tracker_id: str, step_id: str, *, status: str | None = None,
                    note: str | None = None, due_at: str | None = None,
                    title: str | None = None) -> dict | None:
        t = self.get(tracker_id)
        if not t:
            return None
        for step in t["steps"]:
            if step["id"] == step_id:
                if status is not None:
                    step["status"] = status
                if note is not None:
                    step["note"] = note
                if due_at is not None:
                    step["due_at"] = due_at
                if title is not None:
                    step["title"] = title.strip()
                self._save(t)
                return step
        return None


TRACKERS = TrackerStore()
