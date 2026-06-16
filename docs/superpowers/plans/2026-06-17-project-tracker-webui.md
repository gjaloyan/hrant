# Project Tracker — WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the Projects tab into a tracker board — a live per-project status table and a month calendar of step due-dates / check-ins — consuming the already-live tracker API.

**Architecture:** Two new focused React components (`TrackerBoard.tsx`, `TrackerCalendar.tsx`) + tracker types/fetchers in `api.ts`. `ProjectsPanel.tsx` gains a small view toggle (Journal | Trackers | Calendar) and mounts them. "Live" = the existing panel polling pattern (`setInterval(load, 5000)`), matching `GoalsPanel.tsx` — there is no SSE for panels.

**Tech Stack:** React + TypeScript + Vite + Tailwind. **No frontend test framework exists** (scripts are only `dev`/`build`/`preview`), so each task's automated gate is `npm run build` (runs `tsc` typecheck + vite build) and every task includes a **manual verification** checklist. No automated component tests. No new npm dependencies (the calendar is a hand-rolled month grid).

**Backend (already live):** `GET /api/trackers?status=active` → `{trackers: Tracker[]}`; `GET /api/trackers/{id}`; `PUT /api/trackers/{id}/steps/{step_id}` (body `{status?,note?,due_at?,title?}`) → `{ok, step}`; `POST /api/trackers/{id}/complete` → `{ok, tracker}`.

**Out of scope:** creating trackers from the UI (creation is the agent's `create_tracker` tool); merge UI; domain templates; per-card open-questions (spec §7 "if feasible" — deferred: the backend MVP has no link from an `ask_user` question to a tracker yet).

---

### Task 1: Tracker types + API fetchers (`api.ts`)

**Files:** Modify `frontend/src/api.ts` (append near the other project funcs, ~line 562). No test file (no framework) — gate is the typecheck.

- [ ] **Step 1: Add types + fetchers**

Append to `frontend/src/api.ts`:

```typescript
// ---- Project trackers (living projects) ----
export interface TrackerStep {
  id: string;
  title: string;
  status: string; // pending | active | done | blocked
  due_at: string;
  check_in_kind: string; // ask_status | remind | none
  note: string;
  last_checked_at: string | null;
}

export interface Tracker {
  id: string;
  title: string;
  domain: string;
  status: string; // active | archived | done | cancelled
  created_at: string;
  steps: TrackerStep[];
  notes: string;
}

export const fetchTrackers = (status = "active") =>
  json_get<{ trackers: Tracker[] }>(
    `/api/trackers?status=${encodeURIComponent(status)}`,
  );

export const updateTrackerStep = (
  trackerId: string,
  stepId: string,
  patch: Partial<Pick<TrackerStep, "status" | "note" | "due_at" | "title">>,
) =>
  json_put<{ ok: boolean; step: TrackerStep }>(
    `/api/trackers/${encodeURIComponent(trackerId)}/steps/${encodeURIComponent(stepId)}`,
    patch,
  );

export const completeTracker = (trackerId: string) =>
  json_post<{ ok: boolean; tracker: Tracker }>(
    `/api/trackers/${encodeURIComponent(trackerId)}/complete`,
  );
```

- [ ] **Step 2: Typecheck**

Run (from `frontend/`): `npm run build`
Expected: build succeeds (no TS errors). If `tsc` reports an unused-symbol error because nothing imports these yet, that's fine — exported symbols are not "unused"; the build should pass. If it fails for another reason, fix it before continuing.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat(tracker-ui): Tracker types + API fetchers"
```

---

### Task 2: Live status board (`TrackerBoard.tsx`)

**Files:** Create `frontend/src/components/TrackerBoard.tsx`.

- [ ] **Step 1: Create the component**

```tsx
import { useCallback, useEffect, useState } from "react";
import {
  completeTracker,
  fetchTrackers,
  updateTrackerStep,
  type Tracker,
} from "../api";

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-slate-700 text-slate-200",
  active: "bg-sky-700 text-white",
  done: "bg-emerald-700 text-white",
  blocked: "bg-rose-700 text-white",
};
const STEP_STATUSES = ["pending", "active", "done", "blocked"];

function fmtDue(due: string): string {
  if (!due) return "—";
  const d = new Date(due);
  if (isNaN(d.getTime())) return due;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function TrackerBoard() {
  const [trackers, setTrackers] = useState<Tracker[]>([]);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    try {
      const r = await fetchTrackers("active");
      setTrackers(r.trackers || []);
      setErr("");
    } catch (e: any) {
      setErr(e.message || "failed to load trackers");
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  const onStepStatus = async (
    trackerId: string,
    stepId: string,
    status: string,
  ) => {
    await updateTrackerStep(trackerId, stepId, { status });
    load();
  };

  const onComplete = async (trackerId: string, title: string) => {
    if (!confirm(`Archive project "${title}"? It moves to long-term memory.`))
      return;
    await completeTracker(trackerId);
    load();
  };

  if (err)
    return <div className="p-4 text-rose-400 text-sm">Error: {err}</div>;
  if (trackers.length === 0)
    return (
      <div className="p-6 opacity-50 text-sm text-center">
        No active projects. The agent creates them with its create_tracker tool.
      </div>
    );

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-6">
      {trackers.map((t) => (
        <section
          key={t.id}
          className="bg-slate-900 rounded-lg border border-slate-800"
        >
          <header className="flex items-center justify-between px-4 py-2 border-b border-slate-800">
            <h3 className="font-bold">
              {t.title}
              <span className="ml-2 text-xs opacity-50">{t.domain}</span>
            </h3>
            <button
              onClick={() => onComplete(t.id, t.title)}
              className="text-xs bg-slate-800 hover:bg-emerald-700 rounded px-2 py-1"
            >
              complete
            </button>
          </header>
          <table className="w-full text-xs">
            <thead className="text-slate-400">
              <tr className="text-left">
                <th className="px-4 py-1 font-medium">Step</th>
                <th className="px-2 py-1 font-medium">Due</th>
                <th className="px-2 py-1 font-medium">Status</th>
                <th className="px-2 py-1 font-medium">Last check-in</th>
              </tr>
            </thead>
            <tbody>
              {t.steps.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-4 py-2 opacity-40">
                    (no steps yet)
                  </td>
                </tr>
              )}
              {t.steps.map((s) => (
                <tr key={s.id} className="border-t border-slate-800/60">
                  <td className="px-4 py-1.5">
                    {s.title}
                    {s.note && (
                      <span className="block opacity-50">{s.note}</span>
                    )}
                  </td>
                  <td className="px-2 py-1.5 whitespace-nowrap">
                    {fmtDue(s.due_at)}
                  </td>
                  <td className="px-2 py-1.5">
                    <select
                      value={s.status}
                      onChange={(e) =>
                        onStepStatus(t.id, s.id, e.target.value)
                      }
                      className={`rounded px-1 py-0.5 outline-none ${
                        STATUS_COLORS[s.status] || "bg-slate-700"
                      }`}
                    >
                      {STEP_STATUSES.map((st) => (
                        <option key={st} value={st}>
                          {st}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-2 py-1.5 opacity-60 whitespace-nowrap">
                    {s.last_checked_at ? fmtDue(s.last_checked_at) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run (from `frontend/`): `npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/TrackerBoard.tsx
git commit -m "feat(tracker-ui): live status board (polling 5s)"
```

---

### Task 3: Month calendar (`TrackerCalendar.tsx`)

**Files:** Create `frontend/src/components/TrackerCalendar.tsx`. Hand-rolled month grid, no calendar library.

- [ ] **Step 1: Create the component**

```tsx
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchTrackers, type Tracker } from "../api";

interface DayMark {
  tracker: string;
  step: string;
  kind: string; // check_in_kind
}

function ymd(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate(),
  ).padStart(2, "0")}`;
}

export default function TrackerCalendar() {
  const [trackers, setTrackers] = useState<Tracker[]>([]);
  const [month, setMonth] = useState(() => {
    const n = new Date();
    return new Date(n.getFullYear(), n.getMonth(), 1);
  });

  const load = useCallback(async () => {
    try {
      const r = await fetchTrackers("active");
      setTrackers(r.trackers || []);
    } catch {
      /* keep last good state */
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  // Map "YYYY-MM-DD" -> marks for every step that has a due_at.
  const marks = useMemo(() => {
    const m: Record<string, DayMark[]> = {};
    for (const t of trackers) {
      for (const s of t.steps) {
        if (!s.due_at) continue;
        const d = new Date(s.due_at);
        if (isNaN(d.getTime())) continue;
        const key = ymd(d);
        (m[key] = m[key] || []).push({
          tracker: t.title,
          step: s.title,
          kind: s.check_in_kind,
        });
      }
    }
    return m;
  }, [trackers]);

  // Build the 6x7 grid for the visible month (leading blanks for weekday offset).
  const cells = useMemo(() => {
    const first = new Date(month.getFullYear(), month.getMonth(), 1);
    const daysInMonth = new Date(
      month.getFullYear(),
      month.getMonth() + 1,
      0,
    ).getDate();
    const lead = first.getDay(); // 0=Sun
    const out: (Date | null)[] = [];
    for (let i = 0; i < lead; i++) out.push(null);
    for (let d = 1; d <= daysInMonth; d++)
      out.push(new Date(month.getFullYear(), month.getMonth(), d));
    while (out.length % 7 !== 0) out.push(null);
    return out;
  }, [month]);

  const shift = (delta: number) =>
    setMonth(new Date(month.getFullYear(), month.getMonth() + delta, 1));
  const todayKey = ymd(new Date());

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => shift(-1)}
          className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm"
        >
          ‹
        </button>
        <h3 className="font-bold">
          {month.toLocaleString(undefined, { month: "long", year: "numeric" })}
        </h3>
        <button
          onClick={() => shift(1)}
          className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm"
        >
          ›
        </button>
      </div>
      <div className="grid grid-cols-7 gap-1 text-xs">
        {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((d) => (
          <div key={d} className="text-center opacity-50 py-1">
            {d}
          </div>
        ))}
        {cells.map((d, i) => {
          if (!d) return <div key={i} className="min-h-16" />;
          const key = ymd(d);
          const dayMarks = marks[key] || [];
          return (
            <div
              key={i}
              className={`min-h-16 rounded p-1 border ${
                key === todayKey
                  ? "border-emerald-500 bg-slate-900"
                  : "border-slate-800 bg-slate-900/50"
              }`}
            >
              <div className="opacity-60">{d.getDate()}</div>
              {dayMarks.slice(0, 3).map((mk, j) => (
                <div
                  key={j}
                  title={`${mk.tracker}: ${mk.step}`}
                  className={`mt-0.5 truncate rounded px-1 ${
                    mk.kind === "remind"
                      ? "bg-amber-800/70"
                      : "bg-sky-800/70"
                  }`}
                >
                  {mk.step}
                </div>
              ))}
              {dayMarks.length > 3 && (
                <div className="opacity-50">+{dayMarks.length - 3}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run (from `frontend/`): `npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/TrackerCalendar.tsx
git commit -m "feat(tracker-ui): month calendar of step due-dates"
```

---

### Task 4: Wire the view toggle into `ProjectsPanel.tsx`

**Files:** Modify `frontend/src/components/ProjectsPanel.tsx`.

- [ ] **Step 1: Import the new components + add a view state**

At the top of `frontend/src/components/ProjectsPanel.tsx`, after the existing `import { ... } from "../api";` block, add:

```tsx
import TrackerBoard from "./TrackerBoard";
import TrackerCalendar from "./TrackerCalendar";
```

Inside the component, alongside the other `useState` hooks (after `const [selectedProject, ...]`), add:

```tsx
  const [view, setView] = useState<"journal" | "trackers" | "calendar">(
    "trackers",
  );
```

- [ ] **Step 2: Render the toggle + switch the right pane**

Replace the entire right-pane block — i.e. the `{/* Right: project overview */}` `<div className="flex-1 overflow-y-auto p-4"> ... </div>` (currently the last child before the closing `</div>` of the root) — with:

```tsx
      {/* Right: view switch (Trackers board / Calendar / Journal) */}
      <div className="flex flex-1 min-w-0 flex-col">
        <div className="flex gap-1 border-b border-slate-800 px-3 py-2 text-xs">
          {(["trackers", "calendar", "journal"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded px-3 py-1 ${
                view === v ? "bg-sky-700 text-white" : "bg-slate-800"
              }`}
            >
              {v === "trackers"
                ? "Trackers"
                : v === "calendar"
                ? "Calendar"
                : "Journal"}
            </button>
          ))}
        </div>
        {view === "trackers" && <TrackerBoard />}
        {view === "calendar" && <TrackerCalendar />}
        {view === "journal" && (
          <div className="flex-1 overflow-y-auto p-4">
            {selectedProject ? (
              <>
                <h2 className="text-lg font-bold mb-4">
                  Project: {selectedProject}
                  {selectedProject === current && (
                    <span className="ml-2 text-sm text-emerald-400">
                      (active)
                    </span>
                  )}
                </h2>
                <pre className="whitespace-pre-wrap text-sm bg-slate-900 rounded p-4 max-w-3xl">
                  {overview}
                </pre>
              </>
            ) : (
              <div className="opacity-50 text-sm text-center mt-8">
                Select a project or create a new one.
              </div>
            )}
          </div>
        )}
      </div>
```

(The left `<aside>` with the journal forms is unchanged — it still drives the Journal view.)

- [ ] **Step 3: Typecheck**

Run (from `frontend/`): `npm run build`
Expected: build succeeds with no TS errors.

- [ ] **Step 4: Manual verification (dev server)**

Run (from `frontend/`): `npm run dev`, open the printed localhost URL, go to the **Projects** tab. Verify:
1. The **Trackers** view is selected by default and shows active trackers, or the "No active projects" empty state if none exist (an empty state is correct when there are no active trackers; backend data flow is already proven by the store smoke).
2. Toggling **Calendar** shows the current month grid; months navigate with ‹ ›; a step with a due_at shows a chip on its day.
3. Toggling **Journal** shows the old overview pane unchanged.
4. Changing a step's status dropdown persists (the row's status stays after the 5 s poll refresh).

Note any rough edges as DONE_WITH_CONCERNS; they do not block the commit if the build passes and the views render.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ProjectsPanel.tsx
git commit -m "feat(tracker-ui): Trackers/Calendar/Journal view toggle in Projects"
```

---

## Notes for the implementer

- **No frontend tests exist** — the typecheck (`npm run build`) is the automated gate; manual browser checks cover behavior. Do NOT add a test framework in this plan.
- **"Live" = 5 s polling**, exactly like `GoalsPanel.tsx`/`AutonomicPanel.tsx`. Do not introduce SSE for the panel.
- **No new npm deps.** The calendar is a hand-rolled month grid.
- **Build from `frontend/`** (`cd frontend && npm run build`). The build runs `tsc && vite build` per `package.json`.
- Creating trackers is the agent's job (`create_tracker` tool) — the UI is read + step-status edits + complete, intentionally.
- After all tasks: `superpowers:finishing-a-development-branch`, then deploy is `git push` + `hrant update` (which rebuilds the frontend on the box).
