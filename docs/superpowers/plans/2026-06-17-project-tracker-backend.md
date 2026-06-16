# Project Tracker — Backend MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend foundation of the living project tracker — a unified `tracker.json` model (project ⊇ steps), agent tools to drive it, proactive due-date check-ins that wake the agent, and tracker API endpoints.

**Architecture:** A focused `backend/tracker.py` store manages `tracker.json` files under the existing `knowledge/projects/<slug>/` dirs (coexisting with the markdown journal). Steps with a `due_at` schedule a `kind:"check_in"` row via the existing `scheduled_messages` ledger; `deliver_due()` branches on `kind` to wake the agent instead of static send. Tools live in `builtin_tools.py`; endpoints extend `backend/api/projects.py`.

**Tech Stack:** Python 3.12, pytest. Spec: `docs/superpowers/specs/2026-06-17-project-tracker-design.md`. WebUI (live tables + calendar) is a SEPARATE follow-up plan.

**Out of scope (fast-follow):** `merge_tracker`, full `complete_tracker` archival/consolidation (this plan only flips status), domain templates, WebUI.

---

### Task 1: Tracker store (`backend/tracker.py`)

**Files:**
- Create: `backend/tracker.py`
- Test: `tests/test_tracker_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tracker_store.py`:

```python
"""Tracker store — tracker.json CRUD under knowledge/projects/<slug>/."""
from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    # KM resolves paths from HRANT_DATA_DIR at import; reload to pick up tmp.
    import importlib
    from backend import knowledge_manager, tracker
    importlib.reload(knowledge_manager)
    importlib.reload(tracker)
    return tracker.TRACKERS


def test_create_and_get(store):
    t = store.create(title="Blister tooling", domain="work",
                      steps=[{"title": "Approve drawings"}])
    assert t["id"].startswith("trk_")
    assert t["title"] == "Blister tooling"
    assert t["status"] == "active"
    assert len(t["steps"]) == 1
    assert t["steps"][0]["id"].startswith("st_")
    assert t["steps"][0]["status"] == "pending"
    got = store.get(t["id"])
    assert got["title"] == "Blister tooling"


def test_list_active_excludes_archived(store):
    a = store.create(title="A", domain="work", steps=[])
    b = store.create(title="B", domain="work", steps=[])
    store.set_status(b["id"], "archived")
    ids = [t["id"] for t in store.list(status="active")]
    assert a["id"] in ids
    assert b["id"] not in ids


def test_add_and_update_step(store):
    t = store.create(title="T", domain="work", steps=[])
    s = store.add_step(t["id"], title="Pay supplier", due_at="2026-06-25T09:00:00Z")
    assert s["title"] == "Pay supplier"
    assert s["due_at"] == "2026-06-25T09:00:00Z"
    store.update_step(t["id"], s["id"], status="done", note="paid")
    got = store.get(t["id"])
    step = next(x for x in got["steps"] if x["id"] == s["id"])
    assert step["status"] == "done"
    assert step["note"] == "paid"


def test_inbox_reminder_is_a_one_step_project(store):
    t = store.create_inbox_reminder(title="call bank",
                                    due_at="2026-06-18T11:00:00Z")
    assert t["domain"] == "inbox"
    assert len(t["steps"]) == 1
    assert t["steps"][0]["due_at"] == "2026-06-18T11:00:00Z"
    assert t["steps"][0]["check_in_kind"] == "remind"


def test_unknown_tracker_returns_none(store):
    assert store.get("trk_does_not_exist") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tracker_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.tracker'`

- [ ] **Step 3: Implement `backend/tracker.py`**

```python
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

from .knowledge_manager import KM, _slug
from .paths import write_atomic_json


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
        b = KM.base / "projects"
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_tracker_store.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/tracker.py tests/test_tracker_store.py
git commit -m "feat(tracker): tracker.json store — unified project/steps model"
```

---

### Task 2: `schedule()` carries `kind` + `meta`

**Files:**
- Modify: `backend/scheduled_messages.py` (`schedule()` ~line 94, the row dict)
- Test: `tests/test_scheduled_kind_meta.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_scheduled_kind_meta.py`:

```python
"""schedule() must carry an optional kind + meta so the tick can route
check-ins to the agent instead of a static send."""
from __future__ import annotations

import pytest


@pytest.fixture
def sched(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import scheduled_messages as sm
    importlib.reload(sm)
    return sm


def test_schedule_defaults_to_message_kind(sched):
    row = sched.schedule(target_speaker="webui:default", text="hi",
                         due_at="2026-06-25T09:00:00Z", requested_by="webui:default")
    assert row["kind"] == "message"
    assert row["meta"] == {}


def test_schedule_records_kind_and_meta(sched):
    row = sched.schedule(
        target_speaker="webui:default", text="", due_at="2026-06-25T09:00:00Z",
        requested_by="webui:default", kind="check_in",
        meta={"tracker_id": "trk_1", "step_id": "st_1", "check_in_kind": "ask_status"},
    )
    assert row["kind"] == "check_in"
    assert row["meta"]["tracker_id"] == "trk_1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_scheduled_kind_meta.py -q`
Expected: FAIL — `TypeError: schedule() got an unexpected keyword argument 'kind'`

- [ ] **Step 3: Implement — add params + row fields**

In `backend/scheduled_messages.py`, change the `schedule` signature (~line 94) and row:

```python
def schedule(
    *,
    target_speaker: str,
    text: str,
    due_at: str,
    requested_by: str,
    kind: str = "message",
    meta: dict | None = None,
) -> dict:
```

Add these two keys to the `row = {...}` dict (right after `"status": "pending",`):

```python
        "kind": kind,            # message | check_in
        "meta": meta or {},      # check_in carries {tracker_id, step_id, check_in_kind}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_scheduled_kind_meta.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/scheduled_messages.py tests/test_scheduled_kind_meta.py
git commit -m "feat(scheduled): rows carry optional kind + meta for check-ins"
```

---

### Task 3: `deliver_due()` routes check-ins to the agent

**Files:**
- Create: `backend/tracker_checkin.py` (the agent-wake fn — separate file avoids a scheduled_messages→agent import cycle)
- Modify: `backend/scheduled_messages.py` (`deliver_due()` ~line 423)
- Test: `tests/test_checkin_routing.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_checkin_routing.py`:

```python
"""deliver_due() must route kind=='check_in' rows to the agent-wake path
and NOT to static deliver(); normal rows still go to deliver()."""
from __future__ import annotations

import pytest


@pytest.fixture
def sched(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import scheduled_messages as sm
    importlib.reload(sm)
    return sm


def test_checkin_row_wakes_agent_not_deliver(sched, monkeypatch):
    delivered, woken = [], []
    monkeypatch.setattr(sched, "deliver", lambda row: (delivered.append(row["id"]) or (True, "")))
    import backend.tracker_checkin as tc
    monkeypatch.setattr(tc, "run_check_in", lambda row: woken.append(row["id"]))

    sched.schedule(target_speaker="webui:default", text="status?",
                   due_at="2000-01-01T00:00:00Z", requested_by="webui:default",
                   kind="check_in", meta={"tracker_id": "trk_1", "step_id": "st_1"})
    sched.schedule(target_speaker="webui:default", text="plain",
                   due_at="2000-01-01T00:00:00Z", requested_by="webui:default")

    summary = sched.deliver_due()
    assert len(woken) == 1              # the check-in row woke the agent
    assert len(delivered) == 1          # the plain message went to deliver()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_checkin_routing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.tracker_checkin'`

- [ ] **Step 3: Create `backend/tracker_checkin.py`**

```python
"""Wake the agent for a due project check-in. Kept separate from
scheduled_messages.py to avoid a scheduled_messages -> agent import cycle
(scheduled_messages is imported by low-level delivery; the agent imports it)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_check_in(row: dict) -> None:
    """Fire an agent turn for a due check-in row. The agent reviews the
    step and sends the owner a concise status query (ask_status) or a
    reminder (remind). Best-effort: a failure here must not break the tick."""
    meta = row.get("meta") or {}
    tracker_id = meta.get("tracker_id", "")
    step_id = meta.get("step_id", "")
    check_in_kind = meta.get("check_in_kind", "ask_status")
    try:
        from .tracker import TRACKERS
        t = TRACKERS.get(tracker_id)
        if not t or t.get("status") != "active":
            return
        step = next((s for s in t["steps"] if s["id"] == step_id), None)
        if not step or step.get("status") == "done":
            return
        if check_in_kind == "remind":
            prompt = (
                f"Reminder due for project '{t['title']}': {step['title']}. "
                f"Deliver this reminder to the user concisely."
            )
        else:
            prompt = (
                f"Check-in due: step '{step['title']}' of project "
                f"'{t['title']}' has reached its date. Review what you know, "
                f"send the user ONE concise status question, and update the "
                f"step from their reply."
            )
        from .agent import Agent
        from .sessions import normalize_speaker
        speaker = normalize_speaker(row.get("target_speaker") or "webui:default")
        Agent().run(prompt, channel="telegram", speaker_id=speaker)
    except Exception as e:
        log.warning("run_check_in failed for %s: %s", row.get("id"), e)
```

- [ ] **Step 4: Branch in `deliver_due()`**

In `backend/scheduled_messages.py`, replace the loop body of `deliver_due()` (~line 433):

```python
    for row in due_now():
        if row.get("kind") == "check_in":
            try:
                from .tracker_checkin import run_check_in
                run_check_in(row)
                mark_sent(row["id"])
                summary["sent"].append(row["id"])
            except Exception as e:
                mark_failed(row["id"], str(e)[:200])
                summary["failed"].append({"id": row["id"], "error": str(e)[:200]})
            continue
        ok, err = deliver(row)
        if ok:
            summary["sent"].append(row["id"])
        else:
            summary["failed"].append({"id": row["id"], "error": err})
    return summary
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_checkin_routing.py -q`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/tracker_checkin.py backend/scheduled_messages.py tests/test_checkin_routing.py
git commit -m "feat(tracker): deliver_due routes check-ins to wake the agent"
```

---

### Task 4: `update_step`/`add_step` (re)schedule the check-in row

**Files:**
- Modify: `backend/tracker.py` (`add_step`, `update_step` — schedule/cancel a check-in row when `due_at` is set/cleared)
- Test: `tests/test_tracker_checkin_schedule.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_tracker_checkin_schedule.py`:

```python
"""Setting a step's due_at schedules a kind='check_in' row; clearing it cancels."""
from __future__ import annotations

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import knowledge_manager, scheduled_messages, tracker
    importlib.reload(knowledge_manager)
    importlib.reload(scheduled_messages)
    importlib.reload(tracker)
    return tracker, scheduled_messages


def test_add_step_with_due_schedules_checkin(env):
    tracker, sm = env
    t = tracker.TRACKERS.create(title="T", domain="work", steps=[])
    step = tracker.TRACKERS.add_step(t["id"], "Pay", due_at="2026-06-25T09:00:00Z",
                                     requested_by="webui:default")
    rows = [r for r in sm._read_all() if r.get("kind") == "check_in"]
    assert len(rows) == 1
    assert rows[0]["meta"]["tracker_id"] == t["id"]
    assert rows[0]["meta"]["step_id"] == step["id"]
    assert rows[0]["due_at"] == "2026-06-25T09:00:00Z"


def test_update_step_done_cancels_checkin(env):
    tracker, sm = env
    t = tracker.TRACKERS.create(title="T", domain="work", steps=[])
    step = tracker.TRACKERS.add_step(t["id"], "Pay", due_at="2026-06-25T09:00:00Z",
                                     requested_by="webui:default")
    tracker.TRACKERS.update_step(t["id"], step["id"], status="done")
    pending = [r for r in sm._read_all()
               if r.get("kind") == "check_in" and r["status"] == "pending"]
    assert pending == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tracker_checkin_schedule.py -q`
Expected: FAIL — `TypeError: add_step() got an unexpected keyword argument 'requested_by'`

- [ ] **Step 3: Wire scheduling into tracker.py**

Add a helper + thread `requested_by` through `add_step`/`update_step`. Add to `backend/tracker.py`:

```python
    def _schedule_check_in(self, tracker: dict, step: dict, requested_by: str) -> None:
        """Create/refresh the check_in row for a step with a due_at. Cancels
        any prior pending check-in for this step first (idempotent reschedule)."""
        from .scheduled_messages import schedule, _read_all, cancel
        for r in _read_all():
            if (r.get("kind") == "check_in" and r["status"] == "pending"
                    and (r.get("meta") or {}).get("step_id") == step["id"]):
                cancel(r["id"])
        if step.get("due_at") and step.get("check_in_kind") != "none" \
                and step.get("status") not in ("done", "blocked"):
            schedule(
                target_speaker=requested_by, text="", due_at=step["due_at"],
                requested_by=requested_by, kind="check_in",
                meta={"tracker_id": tracker["id"], "step_id": step["id"],
                      "check_in_kind": step.get("check_in_kind", "ask_status")},
            )
```

Change `add_step` to accept `requested_by` and call the helper after `self._save(t)`:

```python
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
```

Change `update_step` to accept `requested_by` and re-schedule after `self._save(t)` (replace its return branch):

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_tracker_checkin_schedule.py tests/test_tracker_store.py -q`
Expected: PASS (7 passed — store tests still green; default `requested_by` keeps them working)

- [ ] **Step 5: Commit**

```bash
git add backend/tracker.py tests/test_tracker_checkin_schedule.py
git commit -m "feat(tracker): steps (re)schedule/cancel their check-in on due_at"
```

---

### Task 5: Agent tools

**Files:**
- Modify: `backend/builtin_tools.py` (register tracker tools near `schedule_message` ~line 2160)
- Test: `tests/test_tracker_tools.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_tracker_tools.py`:

```python
"""create_tracker proposes steps from recalled experience when steps are
omitted; tools round-trip through TRACKERS."""
from __future__ import annotations

import json
import pytest


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import knowledge_manager, scheduled_messages, tracker, builtin_tools
    importlib.reload(knowledge_manager)
    importlib.reload(scheduled_messages)
    importlib.reload(tracker)
    importlib.reload(builtin_tools)
    # owner gate: make _check_owner pass
    monkeypatch.setattr(builtin_tools, "_check_owner", lambda *a, **k: (False, "webui:default"))
    return builtin_tools


def test_create_tracker_explicit_steps(tools):
    out = json.loads(tools._create_tracker_handler(
        title="Tooling", domain="work",
        steps=[{"title": "Approve drawings"}, {"title": "Pay"}]))
    assert out["ok"] is True
    assert len(out["tracker"]["steps"]) == 2


def test_create_tracker_recalls_steps_when_omitted(tools, monkeypatch):
    import backend.tracker as tk
    monkeypatch.setattr(
        "backend.trajectory_memory.recall_similar",
        lambda task, limit=2: [{"steps": ["design", "approve", "ship"]}],
    )
    out = json.loads(tools._create_tracker_handler(title="Tooling from China",
                                                   domain="work", steps=None))
    titles = [s["title"] for s in out["tracker"]["steps"]]
    assert "design" in titles and "ship" in titles


def test_list_and_update(tools):
    json.loads(tools._create_tracker_handler(title="T", domain="work",
                                             steps=[{"title": "A"}]))
    listed = json.loads(tools._list_trackers_handler())
    assert listed["count"] == 1
    tid = listed["trackers"][0]["id"]
    sid = listed["trackers"][0]["steps"][0]["id"]
    upd = json.loads(tools._update_step_handler(tracker_id=tid, step_id=sid, status="done"))
    assert upd["ok"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tracker_tools.py -q`
Expected: FAIL — `AttributeError: module 'backend.builtin_tools' has no attribute '_create_tracker_handler'`

- [ ] **Step 3: Implement handlers + register**

Add handlers to `backend/builtin_tools.py` (near the other handlers, before `register_builtin_tools`):

```python
def _propose_steps_from_experience(title: str) -> list[dict]:
    """Recall a similar past project's step template; [] if none."""
    try:
        from .trajectory_memory import recall_similar
        for hit in (recall_similar(title, limit=2) or []):
            steps = hit.get("steps") or []
            if steps:
                return [{"title": s} for s in steps if isinstance(s, str)]
    except Exception:
        pass
    return []


def _create_tracker_handler(title: str, domain: str = "work",
                            steps: list | None = None) -> str:
    refuse, _sp = _check_owner("create_tracker")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"}, ensure_ascii=False)
    from .tracker import TRACKERS
    use_steps = steps if steps else _propose_steps_from_experience(title)
    recalled = bool(not steps and use_steps)
    t = TRACKERS.create(title=title, domain=domain, steps=use_steps or [])
    return json.dumps({"ok": True, "tracker": t, "steps_recalled": recalled,
                       "note": ("proposed steps from past experience — confirm "
                                "or edit them" if recalled else "")},
                      ensure_ascii=False)


def _list_trackers_handler(status: str = "active") -> str:
    from .tracker import TRACKERS
    items = TRACKERS.list(status=status)
    return json.dumps({"ok": True, "count": len(items), "trackers": items},
                      ensure_ascii=False)


def _get_tracker_handler(tracker_id: str) -> str:
    from .tracker import TRACKERS
    t = TRACKERS.get(tracker_id)
    if not t:
        return json.dumps({"ok": False, "error": "tracker not found"}, ensure_ascii=False)
    return json.dumps({"ok": True, "tracker": t}, ensure_ascii=False)


def _add_step_handler(tracker_id: str, title: str, due_at: str = "",
                      check_in_kind: str = "ask_status") -> str:
    refuse, speaker = _check_owner("add_step")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"}, ensure_ascii=False)
    from .tracker import TRACKERS
    s = TRACKERS.add_step(tracker_id, title, due_at=due_at,
                          check_in_kind=check_in_kind, requested_by=speaker)
    if s is None:
        return json.dumps({"ok": False, "error": "tracker not found"}, ensure_ascii=False)
    return json.dumps({"ok": True, "step": s}, ensure_ascii=False)


def _update_step_handler(tracker_id: str, step_id: str, status: str = "",
                         note: str = "", due_at: str = "", title: str = "") -> str:
    refuse, speaker = _check_owner("update_step")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"}, ensure_ascii=False)
    from .tracker import TRACKERS
    s = TRACKERS.update_step(
        tracker_id, step_id,
        status=status or None, note=note or None,
        due_at=due_at or None, title=title or None, requested_by=speaker)
    if s is None:
        return json.dumps({"ok": False, "error": "tracker/step not found"}, ensure_ascii=False)
    return json.dumps({"ok": True, "step": s}, ensure_ascii=False)
```

Register them inside `register_builtin_tools` (after the `schedule_message` block ~line 2200) — one `reg.register_func(...)` per tool. Example for `create_tracker` (repeat the pattern for the other four with their handlers + schemas):

```python
    reg.register_func(
        name="create_tracker",
        description=(
            "Start a living project/tracker for systematic, multi-step work "
            "(an order, a trip, a research effort). Omit `steps` to have the "
            "agent propose them from past experience (it recalls similar "
            "projects); pass `steps` to set an explicit plan. Steps with a "
            "due_at are checked on automatically. Owner/trusted only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Project title."},
                "domain": {"type": "string",
                           "description": "work | personal | research | travel."},
                "steps": {"type": "array", "items": {"type": "object"},
                          "description": "Optional [{title, due_at?}]. Omit to "
                                         "propose from experience."},
            },
            "required": ["title"],
        },
        handler=_create_tracker_handler,
    )
    reg.register_func(
        name="list_trackers",
        description="List the agent's active projects/trackers with their steps "
                    "and check-ins — the unified view of all ongoing work.",
        input_schema={"type": "object", "properties": {
            "status": {"type": "string", "description": "active | archived | all."}}},
        handler=_list_trackers_handler,
    )
    reg.register_func(
        name="get_tracker",
        description="Read one tracker by id (full steps/status).",
        input_schema={"type": "object", "properties": {
            "tracker_id": {"type": "string"}}, "required": ["tracker_id"]},
        handler=_get_tracker_handler,
    )
    reg.register_func(
        name="add_step",
        description="Add a step to a tracker. A due_at schedules a check-in.",
        input_schema={"type": "object", "properties": {
            "tracker_id": {"type": "string"},
            "title": {"type": "string"},
            "due_at": {"type": "string",
                       "description": "UTC ISO 8601; schedules a check-in."},
            "check_in_kind": {"type": "string",
                              "description": "ask_status | remind | none."}},
            "required": ["tracker_id", "title"]},
        handler=_add_step_handler,
    )
    reg.register_func(
        name="update_step",
        description="Update a step's status/note/due_at/title. Changing due_at "
                    "reschedules its check-in; marking done cancels it.",
        input_schema={"type": "object", "properties": {
            "tracker_id": {"type": "string"},
            "step_id": {"type": "string"},
            "status": {"type": "string",
                       "description": "pending | active | done | blocked."},
            "note": {"type": "string"},
            "due_at": {"type": "string"},
            "title": {"type": "string"}},
            "required": ["tracker_id", "step_id"]},
        handler=_update_step_handler,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_tracker_tools.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/builtin_tools.py tests/test_tracker_tools.py
git commit -m "feat(tracker): agent tools — create/list/get/add_step/update_step"
```

---

### Task 6: Tracker API endpoints

**Files:**
- Modify: `backend/api/projects.py` (add tracker routes)
- Test: `tests/test_tracker_api.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_tracker_api.py`:

```python
"""Tracker API: list trackers with steps, read one, update a step, complete."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import knowledge_manager, scheduled_messages, tracker
    importlib.reload(knowledge_manager)
    importlib.reload(scheduled_messages)
    importlib.reload(tracker)
    from backend.api import projects as projects_api
    importlib.reload(projects_api)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(projects_api.router)
    return TestClient(app), tracker


def test_list_and_complete(client):
    c, tracker = client
    t = tracker.TRACKERS.create(title="T", domain="work", steps=[{"title": "A"}])
    r = c.get("/api/trackers")
    assert r.status_code == 200
    assert any(x["id"] == t["id"] for x in r.json()["trackers"])
    r2 = c.post(f"/api/trackers/{t['id']}/complete")
    assert r2.status_code == 200
    assert tracker.TRACKERS.get(t["id"])["status"] == "archived"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tracker_api.py -q`
Expected: FAIL — 404 on `/api/trackers` (route not defined)

- [ ] **Step 3: Add routes to `backend/api/projects.py`**

Append (the router is already defined in the file as `router`):

```python
@router.get("/api/trackers")
def list_trackers_api(status: str = "active"):
    from ..tracker import TRACKERS
    return {"trackers": TRACKERS.list(status=status)}


@router.get("/api/trackers/{tracker_id}")
def get_tracker_api(tracker_id: str):
    from ..tracker import TRACKERS
    t = TRACKERS.get(tracker_id)
    if not t:
        raise HTTPException(404, "tracker not found")
    return t


@router.put("/api/trackers/{tracker_id}/steps/{step_id}")
def update_step_api(tracker_id: str, step_id: str, body: dict):
    from ..tracker import TRACKERS
    s = TRACKERS.update_step(
        tracker_id, step_id,
        status=body.get("status"), note=body.get("note"),
        due_at=body.get("due_at"), title=body.get("title"))
    if s is None:
        raise HTTPException(404, "tracker/step not found")
    return {"ok": True, "step": s}


@router.post("/api/trackers/{tracker_id}/complete")
def complete_tracker_api(tracker_id: str):
    from ..tracker import TRACKERS
    t = TRACKERS.set_status(tracker_id, "archived")
    if t is None:
        raise HTTPException(404, "tracker not found")
    return {"ok": True, "tracker": t}
```

Ensure `from fastapi import HTTPException` is imported at the top of the file (add it if missing).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_tracker_api.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full tracker suite + broad regression**

Run: `python -m pytest tests/test_tracker_store.py tests/test_scheduled_kind_meta.py tests/test_checkin_routing.py tests/test_tracker_checkin_schedule.py tests/test_tracker_tools.py tests/test_tracker_api.py -q`
Expected: PASS (all green)

Run: `python -m pytest tests/ -k "scheduled or projects or builtin" -q`
Expected: PASS (no regressions in existing scheduled-message / projects / tools tests).

- [ ] **Step 6: Commit**

```bash
git add backend/api/projects.py tests/test_tracker_api.py
git commit -m "feat(tracker): tracker API — list/get/update-step/complete"
```

---

## Notes for the implementer

- **`_check_owner`** is the existing owner/trusted gate used by `schedule_message` (`backend/builtin_tools.py`). It returns `(refuse: bool, speaker_id: str)`. Reuse it verbatim — do NOT invent a new gate.
- **Check-in delivery is best-effort.** `run_check_in` swallows its own errors; `deliver_due` marks the row sent/failed so the tick never crashes on a bad tracker.
- **`requested_by` = the step's check-in target.** For now it equals the owner's speaker_id (so the agent checks in with the owner). Group/other-speaker check-ins are out of scope.
- **Experience→steps depends on archival (deferred).** `_propose_steps_from_experience` reads a `steps` template off recalled trajectories. Those templates are written by the §5 archival path (fast-follow), so until that lands, recall usually returns `[]` and `create_tracker` without explicit steps simply asks the user. The MECHANISM is built + tested (mocked) now; it lights up when archival ships. This is expected, not a bug.
- **WebUI (live status table + calendar) is the NEXT plan** — it consumes `GET /api/trackers`. Do not build it here.
- After all tasks: `superpowers:finishing-a-development-branch`.
