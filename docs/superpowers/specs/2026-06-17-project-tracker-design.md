# Project Tracker (living projects) — Design

**Date:** 2026-06-17
**Status:** approved (brainstorm), pending implementation plan

**Goal:** Turn the lightweight `Projects` journal into a **living tracker**: the
agent plans a project as steps with due dates, proactively checks in on each
step when it comes due, shows live status tables + a calendar in the WebUI, and
archives finished projects into long-term memory — getting smarter at planning
the next similar project.

**Architecture:** One unified model — a *project* contains *steps*, and steps
with a `due_at` ARE the reminders/check-ins (a standalone reminder is a one-step
"Inbox" project). Structured `tracker.json` lives alongside the existing
`knowledge/projects/<id>/` markdown journal. Proactive check-ins reuse the
existing `scheduled_messages` tick, branching on a `check_in` kind to wake the
agent instead of sending static text. Completion digests the project into the
knowledge base and drops it from the active set.

**Tech Stack:** Python (project_mode, scheduled_messages, autonomic tick,
consolidation, builtin_tools), FastAPI (`/api/projects`), React/TS
(`ProjectsPanel.tsx`), pytest.

---

## 1. Background — what exists today

- **Projects** (`backend/project_mode.py`, `PROJECTS`): a named directory under
  `knowledge/projects/<name>/` with append-only markdown (`context` /
  `decisions` / `issues`) + a `.current` pointer. No steps, no status, no due
  dates, no proactivity, no link to reminders. API at `backend/api/projects.py`
  (list/create/detail/end/context/decision/issue); UI in
  `frontend/src/components/ProjectsPanel.tsx`.
- **Reminders** (`backend/scheduled_messages.py`): rows
  `{id, target_speaker, text, due_at, status}` in
  `knowledge/scheduled_messages.jsonl`. An autonomic tick
  (`backend/autonomic/layer0.py`, lever `FIRE_SCHEDULED_MESSAGES`) calls
  `due_now()` → `deliver(row)`, which sends **static text** and marks the row
  `sent`.

The two are unaware of each other. This design unifies them.

## 2. Unified data model

A **Project** (tracker) is the container; everything else lives inside it.

```python
# tracker.json under knowledge/projects/<id>/  (atomic write)
{
  "id": "trk_<hex>",
  "title": "Blister tooling from China",
  "domain": "work",            # work | personal | research | travel | inbox
  "status": "active",          # active | done | archived | cancelled
  "created_at": "...Z",
  "steps": [
    {
      "id": "st_<hex>",
      "title": "Approve drawings",
      "status": "pending",     # pending | active | done | blocked
      "due_at": "2026-06-25T09:00:00Z",   # optional; drives the check-in
      "check_in_kind": "ask_status",      # ask_status | remind | none
      "note": "",
      "last_checked_at": null
    }
  ],
  "notes": ""
}
```

- **A standalone reminder** = a project with `domain: "inbox"` and a single step
  carrying the `due_at` + `check_in_kind: "remind"`. So "remind me at 3pm" and a
  multi-step project use the SAME model — the agent has one unified view and can
  fold a loose reminder into a real project (the "merge them together" ask).
- The existing markdown journal (`context`/`decisions`/`issues`) is kept for
  narrative; `tracker.json` is the structured truth.

## 3. Proactive check-ins (driven by step due dates)

A step with a `due_at` schedules a `scheduled_messages` row with a new
`kind: "check_in"` plus `project_id` / `step_id`. The existing
`FIRE_SCHEDULED_MESSAGES` tick already finds due rows — it branches on `kind`:

- `kind` absent / normal → today's behaviour (static `deliver`).
- `kind == "check_in"` → **wake the agent** instead of static send: fire a turn
  with a synthesized prompt on the owner's channel/speaker. The row carries the
  step's `check_in_kind` so the agent knows the intent: `ask_status` → *"Step
  'Approve drawings' of 'Blister tooling' is due — send {owner} ONE concise
  status query and update the step from their reply"*; `remind` → just deliver
  the reminder (closer to today's static send, but phrased by the agent). The
  agent composes the contextual message (and may `ask_user`).
  (Field map: `step.check_in_kind` is the *intent*; the scheduled row's
  `kind:"check_in"` is the *routing flag* that diverts the tick to this branch.)
- An **overdue** pending step (past `due_at`, still pending at the next tick)
  → the agent nudges once, then backs off (re-schedule + a `last_checked_at`
  guard so it doesn't spam).

This reuses the tick + delivery loop and the `delivering`→`sent` crash-safety
already in `scheduled_messages.py`; no new scheduler.

## 4. Experience-driven step generation

`create_tracker` plans like the agent's "method before execution":
1. **RECALL** — `trajectory_memory` + `search_knowledge` for similar past
   projects (by the title/domain embedding).
2. If a similar project / a saved **step template** is found → propose those
   steps (e.g. "tooling from China" → design → approve drawings → pay → produce
   → ship → customs → deliver to factory).
3. If nothing is recalled → ask the user for the milestones, or lay down a
   minimal generic plan and refine via `ask_user`.

The agent never invents a confident plan from nothing when it has no
experience — it recalls or asks (ties to the calibration soul rules).

## 5. Archival → long-term memory (keep active context light)

When a project is marked `done` (all steps done or explicit completion):
1. **Digest** — summarize outcome + extract lessons and the *step template that
   worked* into the knowledge base (`save_knowledge`, category `projects`), and
   index the run as a trajectory.
2. **Archive** — set `status: "archived"`; the active tracker list
   (`list_trackers`) and the agent's working context only load `active`
   projects. Archived data is searchable (knowledge/trajectories) but never
   bloats the live prompt.
3. This closes the loop: an archived project's step template becomes the
   experience §4 recalls for the next similar project. The agent gets better at
   planning each time.

Reuses the existing `consolidation` / `save_knowledge` / `trajectory_memory`
machinery rather than a new archive store.

## 6. Agent tools (the interface to the tracker)

New tools in `builtin_tools.py` (owner/trusted gated):
- `create_tracker(title, domain, steps?)` — start a project; runs the §4 recall
  to propose steps when `steps` is omitted. Returns the tracker.
- `update_step(tracker_id, step_id, *, status?, note?, due_at?, title?)` —
  record progress; setting/clearing `due_at` (re)schedules the check-in.
- `add_step(tracker_id, title, due_at?, check_in_kind?)`.
- `list_trackers(status="active")` / `get_tracker(tracker_id)` — the agent's
  **unified view of all projects + their steps/check-ins**.
- `merge_tracker(source_id, into_id)` — fold one tracker's steps into another
  (and the loose-reminder→project case).
- `complete_tracker(tracker_id)` — triggers §5 archival.

`schedule_message` stays the simple universal tool; these are the "specialized
tool" for systematic, stateful work.

## 7. API + WebUI

- **API** — extend `backend/api/projects.py`: `GET /api/projects` returns
  trackers with steps/status; `GET/PUT /api/projects/{id}/steps`;
  `POST /api/projects/{id}/complete`. An SSE/poll endpoint
  (`GET /api/projects/stream` or reuse the chat SSE bus) pushes step updates so
  tables are **live**.
- **WebUI** — grow `ProjectsPanel.tsx` into a tracker board (merge with the
  existing Projects section):
  - **Live status table** per project: step · due · status · last check-in,
    updating in real time as the agent edits steps.
  - **Calendar** view: all steps' `due_at` + check-ins across projects on one
    month grid, so every deadline is visible at a glance.
  - Open questions (pending `ask_user`) surfaced on the project card.

## 8. Phasing

**MVP** (first plan):
- `tracker.json` model + atomic store layered on `project_mode.py`.
- Tools: `create_tracker` (with §4 recall), `update_step`, `add_step`,
  `list_trackers`, `get_tracker`.
- Check-ins: `kind: "check_in"` branch in the scheduled tick → wake agent.
- WebUI: live status table + calendar; extended `/api/projects`.

**Fast-follow** (later plans): `merge_tracker`, `complete_tracker` + full §5
archival/consolidation, domain templates (travel/research/personal), overdue
back-off tuning, rich table editing in the UI.

## 9. Testing

- **Unit:** tracker store round-trip (create/update/atomic write); standalone
  reminder ⇄ inbox-project equivalence; `update_step` (re)schedules/cancels the
  check-in row; `create_tracker` recall path proposes steps from a stubbed
  trajectory hit and falls back to ask when empty.
- **Check-in:** a `kind: "check_in"` due row routes to the agent-wake branch
  (mocked agent) and NOT to static `deliver`; an overdue step nudges once then
  honours `last_checked_at`.
- **Archival:** `complete_tracker` calls `save_knowledge` + flips status to
  `archived`; `list_trackers("active")` excludes it.
- **API:** trackers endpoint returns steps; complete endpoint archives.
- Full existing suite stays green (the scheduled-tick change is gated on `kind`).

## 10. Risks

- **Check-in spam.** Mitigation: `last_checked_at` + single-nudge-then-back-off;
  the `delivering`→`sent` guard prevents double-fire across restarts.
- **Proactive outreach is an external action.** Honour the soul's "external
  actions are careful": check-ins go only to the owner, are concise, and respect
  the user's timezone + quiet hours (reuse existing reminder delivery rules).
- **Scope.** The feature is large; the plan ships the MVP slice first (§8) and
  defers merge/archival-depth/UI-polish to fast-follows.
- **Migration.** Existing markdown-only projects keep working; `tracker.json` is
  additive (absent → treated as a zero-step project).
