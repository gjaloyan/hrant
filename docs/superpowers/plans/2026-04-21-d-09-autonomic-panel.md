# D-09 — AutonomicPanel frontend (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `AutonomicPanel.tsx` React component as a new 11th tab, wire a compact autonomic indicator into `StatusBar.tsx`, and add 9 typed API functions to `frontend/src/api.ts` — completing Model X v0's observability.

**Architecture:** One React component (~450 lines) with a two-column layout matching `GoalsPanel.tsx` pattern. Fast 5s poll for status/pending/ticks; slow 30s poll for immune; lazy per-lever history on click. `App.tsx` pulls autonomic status alongside regular status and threads it into `StatusBar.tsx` props. No new test framework — quality gate is `npm run build` (tsc + vite) + manual smoke of kill-switch and approve/reject flows.

**Tech Stack:** React 18, TypeScript, Tailwind CSS, existing `json_get`/`json_post` helpers in `api.ts`, existing `Dot` helper in `StatusBar.tsx`.

**Parent spec:** [docs/superpowers/specs/2026-04-21-d-09-autonomic-panel-design.md](../specs/2026-04-21-d-09-autonomic-panel-design.md)

---

## File Structure

**New files (1):**

```
frontend/src/components/
└── AutonomicPanel.tsx          # ~450 lines
```

**Modified files (3):**

- `frontend/src/api.ts` — append autonomic section (~60 lines).
- `frontend/src/App.tsx` — 11th tab + autonomic state + StatusBar props wiring (~15 line delta).
- `frontend/src/components/StatusBar.tsx` — autonomic section + prop types (~15 line delta).

**Not modified:** backend, other panels, `Chat.tsx`, `GraphViewer.tsx`, `NoteViewer.tsx`, other components, Python tests.

**Test convention (existing project):** no vitest / jest. Quality gate = `cd frontend && npm run build` (TypeScript compile + Vite bundle) plus manual smoke check. Same bar every other frontend change passes through.

---

## Task 1: api.ts autonomic types + 9 functions

**Files:**
- Modify: `frontend/src/api.ts` (append at end of file)

- [ ] **Step 1: Append autonomic section to `frontend/src/api.ts`**

Find the end of the file (currently after `clearActiveModel`). Append:

```typescript


// ---------- Autonomic (Model X) ----------

export type AutonomicStatus = {
  enabled: boolean;
  enabled_path: string;
  scheduler_running: boolean;
  registered_levers: string[];
};

export type TickEntry = {
  ts: string;
  source: string;
  lever: string | null;
  params: Record<string, unknown>;
  reason: string;
  rule_name: string | null;
  executed: boolean;
  note: string;
};

export type LeverReport = {
  lever: string;
  params: Record<string, unknown>;
  started_at: string;
  finished_at: string;
  status: string;
  outcome: Record<string, unknown>;
  cost: {
    tokens_in: number;
    tokens_out: number;
    seconds: number;
    usd: number;
  };
  reason: string;
  follow_ups: string[];
};

export type PendingEntry = {
  id: string;
  lever: string;
  params: Record<string, unknown>;
  requested_at: string;
  status: string;
};

export type ImmuneSignature = {
  id: string;
  pattern: Record<string, unknown>;
  severity: string;
  fix_lever: string;
  fix_params: Record<string, unknown>;
  observed_count: number;
  success_rate: number | null;
};

export const fetchAutonomicStatus = () =>
  json_get<AutonomicStatus>("/api/autonomic/status");

export const fetchTicks = (limit = 50) =>
  json_get<{ ticks: TickEntry[] }>(`/api/autonomic/ticks?limit=${limit}`);

export const fetchLeverHistory = (name: string, limit = 10) =>
  json_get<{ lever: string; reports: LeverReport[] }>(
    `/api/autonomic/levers/${encodeURIComponent(name)}?limit=${limit}`,
  );

export const fetchPending = () =>
  json_get<{ pending: PendingEntry[] }>("/api/autonomic/pending");

export const enqueuePending = (lever: string, params: Record<string, unknown>) =>
  json_post<{ id: string; status: string }>("/api/autonomic/pending", { lever, params });

export const approvePending = (id: string) =>
  json_post<LeverReport>(`/api/autonomic/pending/${id}/approve`);

export const rejectPending = (id: string) =>
  json_post<{ ok: boolean; rejected_id: string }>(`/api/autonomic/pending/${id}/reject`);

export const fetchImmune = () =>
  json_get<{ signatures: ImmuneSignature[] }>("/api/autonomic/immune");

export const toggleKillSwitch = (enabled: boolean) =>
  json_post<{ enabled: boolean }>("/api/autonomic/kill-switch", { enabled });
```

- [ ] **Step 2: Verify TypeScript compile + Vite build**

Run: `cd frontend && npm run build`

Expected: build succeeds with no errors. The new exports are not yet consumed anywhere; that's fine.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat(frontend): add 9 typed autonomic API functions + 5 types"
```

---

## Task 2: AutonomicPanel scaffold — header + status + kill switch

Ship a minimal panel that renders the header block (kill-switch toggle, scheduler state, lever count, refresh button) and nothing else. Wire it into `App.tsx` so the new tab is clickable.

**Files:**
- Create: `frontend/src/components/AutonomicPanel.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create `frontend/src/components/AutonomicPanel.tsx` with header-only content**

```tsx
import { useCallback, useEffect, useState } from "react";
import {
  fetchAutonomicStatus,
  toggleKillSwitch,
  AutonomicStatus,
} from "../api";

export default function AutonomicPanel() {
  const [status, setStatus] = useState<AutonomicStatus | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 4000);
  };

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchAutonomicStatus());
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const handleKillSwitch = async () => {
    if (!status) return;
    if (status.enabled && !confirm("Disable autonomic scheduler?")) return;
    setBusy(true);
    try {
      const result = await toggleKillSwitch(!status.enabled);
      flash(result.enabled ? "Scheduler enabled" : "Scheduler disabled");
      refresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <div className="flex-1 flex flex-col overflow-hidden bg-slate-950/60">
        {/* Header */}
        <div className="p-3 border-b border-slate-800 flex items-center gap-3 flex-wrap">
          <button
            onClick={handleKillSwitch}
            disabled={busy || !status}
            className={`px-3 py-1.5 rounded text-xs font-semibold ${
              status?.enabled
                ? "bg-emerald-700 hover:bg-emerald-600"
                : "bg-rose-700 hover:bg-rose-600"
            } disabled:opacity-50`}
          >
            {status?.enabled ? "● ENABLED" : "● DISABLED"}
          </button>
          <span className="text-xs opacity-70">
            scheduler:{" "}
            <span
              className={
                status?.scheduler_running ? "text-emerald-400" : "text-slate-400"
              }
            >
              {status?.scheduler_running ? "running" : "stopped"}
            </span>
          </span>
          <span className="text-xs opacity-70">
            {status ? `${status.registered_levers.length} levers registered` : "…"}
          </span>
          <button
            onClick={refresh}
            className="ml-auto px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-xs"
          >
            Refresh
          </button>
        </div>

        {/* Sections (added in later tasks) */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4 text-xs">
          <div className="opacity-40">Pending, ticks, levers, immune sections arrive in later tasks.</div>
        </div>

        {msg && (
          <div className="p-2 text-xs text-sky-400 border-t border-slate-800">{msg}</div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add tab entry to `frontend/src/App.tsx`**

Edit the `Tab` union type to include `"autonomic"`:

```tsx
type Tab = "chat" | "goals" | "knowledge" | "graph" | "sessions" | "intelligence" | "autonomic" | "usage" | "projects" | "finetune" | "settings";
```

Edit the `TABS` constant to insert the autonomic entry between `intelligence` and `usage`:

```tsx
const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "💬" },
  { id: "goals", label: "Goals", icon: "🎯" },
  { id: "sessions", label: "Sessions", icon: "📊" },
  { id: "knowledge", label: "Knowledge", icon: "📚" },
  { id: "graph", label: "Graph", icon: "🔗" },
  { id: "intelligence", label: "Intelligence", icon: "🧠" },
  { id: "autonomic", label: "Autonomic", icon: "🦾" },
  { id: "usage", label: "Usage", icon: "📈" },
  { id: "projects", label: "Projects", icon: "📁" },
  { id: "finetune", label: "Fine-Tune", icon: "🎓" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];
```

Add the import at the top of `App.tsx` after the other panel imports:

```tsx
import AutonomicPanel from "./components/AutonomicPanel";
```

Add the render branch in the tab switch area (around where other panels are rendered, e.g. after `{tab === "intelligence" && <IntelligencePanel />}`):

```tsx
        {tab === "autonomic" && <AutonomicPanel />}
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build`

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/AutonomicPanel.tsx frontend/src/App.tsx
git commit -m "feat(frontend): AutonomicPanel scaffold with kill-switch header + tab registration"
```

---

## Task 3: Pending approvals section + approve/reject

Add the "Pending Approvals" block. This is the first actionable user-facing feature of the panel — the only block that mutates system state.

**Files:**
- Modify: `frontend/src/components/AutonomicPanel.tsx`

- [ ] **Step 1: Add pending state + fetch + handlers**

Open `frontend/src/components/AutonomicPanel.tsx`.

Extend imports:

```tsx
import {
  fetchAutonomicStatus,
  fetchPending,
  approvePending,
  rejectPending,
  toggleKillSwitch,
  AutonomicStatus,
  PendingEntry,
} from "../api";
```

Add state:

```tsx
const [pending, setPending] = useState<PendingEntry[]>([]);
```

Update `refresh` to fetch pending alongside status:

```tsx
const refresh = useCallback(async () => {
  try {
    const [s, p] = await Promise.all([fetchAutonomicStatus(), fetchPending()]);
    setStatus(s);
    setPending(p.pending);
  } catch (e: any) {
    flash("Error: " + e.message);
  }
}, []);
```

Add handlers:

```tsx
const handleApprove = async (id: string) => {
  setBusy(true);
  try {
    const report = await approvePending(id);
    flash(`Approved — ${report.lever}: ${report.status} (${report.reason})`);
    refresh();
  } catch (e: any) {
    flash("Error: " + e.message);
  } finally {
    setBusy(false);
  }
};

const handleReject = async (id: string) => {
  if (!confirm("Reject this pending action?")) return;
  setBusy(true);
  try {
    await rejectPending(id);
    flash("Rejected");
    refresh();
  } catch (e: any) {
    flash("Error: " + e.message);
  } finally {
    setBusy(false);
  }
};
```

- [ ] **Step 2: Render the Pending Approvals section**

Inside the `<div className="flex-1 overflow-y-auto p-3 space-y-4 text-xs">` block, replace the placeholder `<div className="opacity-40">...</div>` with:

```tsx
          {/* Pending Approvals */}
          <section>
            <h2 className="font-bold text-sm mb-2">
              ⚠ Pending Approvals ({pending.length})
            </h2>
            {pending.length === 0 ? (
              <div className="opacity-40">No pending actions</div>
            ) : (
              <div className="space-y-2">
                {pending.map((p) => (
                  <div
                    key={p.id}
                    className="p-2 rounded bg-amber-900/20 border border-amber-800/40 space-y-1"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-violet-400 font-semibold">{p.lever}</span>
                      <span className="text-slate-400 text-[10px]">
                        {p.requested_at}
                      </span>
                      <span className="ml-auto flex gap-1">
                        <button
                          onClick={() => handleApprove(p.id)}
                          disabled={busy}
                          className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-0.5 text-[10px] disabled:opacity-50"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => handleReject(p.id)}
                          disabled={busy}
                          className="bg-rose-700 hover:bg-rose-600 rounded px-2 py-0.5 text-[10px] disabled:opacity-50"
                        >
                          Reject
                        </button>
                      </span>
                    </div>
                    <div className="opacity-70 font-mono text-[10px]">
                      {JSON.stringify(p.params)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build`

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/AutonomicPanel.tsx
git commit -m "feat(frontend): AutonomicPanel pending approvals section with approve/reject"
```

---

## Task 4: Recent ticks + levers grid + detail drawer

Add the three observability blocks: recent ticks list (with limit selector), levers grid (19 tiles), and the right-side detail drawer that shows per-lever history when a tile is clicked.

**Files:**
- Modify: `frontend/src/components/AutonomicPanel.tsx`

- [ ] **Step 1: Extend imports, state, and fetch**

Update imports:

```tsx
import {
  fetchAutonomicStatus,
  fetchPending,
  approvePending,
  rejectPending,
  fetchTicks,
  fetchLeverHistory,
  toggleKillSwitch,
  AutonomicStatus,
  PendingEntry,
  TickEntry,
  LeverReport,
} from "../api";
```

Add state declarations (near the existing `pending` state):

```tsx
const [ticks, setTicks] = useState<TickEntry[]>([]);
const [tickLimit, setTickLimit] = useState<number>(50);
const [selectedLever, setSelectedLever] = useState<string | null>(null);
const [leverHistory, setLeverHistory] = useState<LeverReport[]>([]);
const [loadingHistory, setLoadingHistory] = useState(false);
```

Update `refresh` to also fetch ticks (but not every lever's history — that's lazy on click):

```tsx
const refresh = useCallback(async () => {
  try {
    const [s, p, t] = await Promise.all([
      fetchAutonomicStatus(),
      fetchPending(),
      fetchTicks(tickLimit),
    ]);
    setStatus(s);
    setPending(p.pending);
    setTicks(t.ticks);
  } catch (e: any) {
    flash("Error: " + e.message);
  }
}, [tickLimit]);
```

Add `openLever` handler:

```tsx
const openLever = async (name: string) => {
  setSelectedLever(name);
  setLoadingHistory(true);
  try {
    const result = await fetchLeverHistory(name, 10);
    setLeverHistory(result.reports);
  } catch (e: any) {
    flash("Error: " + e.message);
    setLeverHistory([]);
  } finally {
    setLoadingHistory(false);
  }
};

const closeLever = () => {
  setSelectedLever(null);
  setLeverHistory([]);
};
```

Add a small `categoryColor` helper function at module scope (outside the component, below the imports):

```tsx
const CATEGORY_COLORS: Record<string, string> = {
  FIRE_SERVER_HEALTH: "bg-rose-900/40",
  FIRE_ERROR_TRIAGE: "bg-rose-900/40",
  FIRE_SELF_HEAL: "bg-rose-900/40",
  FIRE_SERVICE_REPAIR: "bg-rose-900/40",
  FIRE_TOOL_INSTALL: "bg-amber-900/40",
};

function leverBgClass(name: string): string {
  return CATEGORY_COLORS[name] ?? "bg-sky-900/30";
}

function statusDotClass(status: string | undefined): string {
  if (!status) return "bg-slate-500";
  if (status === "success") return "bg-emerald-400";
  if (status === "failure") return "bg-rose-500";
  if (status === "skipped") return "bg-slate-400";
  if (status === "blocked_by_safety") return "bg-amber-400";
  if (status === "escalated") return "bg-violet-400";
  return "bg-slate-500";
}
```

- [ ] **Step 2: Render Recent Ticks section after Pending Approvals**

Inside the scrollable `<div>`, below the `</section>` of Pending Approvals, add:

```tsx
          {/* Recent Ticks */}
          <section>
            <div className="flex items-center gap-2 mb-2">
              <h2 className="font-bold text-sm">📊 Recent Ticks</h2>
              <select
                value={tickLimit}
                onChange={(e) => setTickLimit(parseInt(e.target.value, 10))}
                className="bg-slate-900 rounded px-1 py-0.5 text-[10px]"
              >
                <option value={10}>10</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
            {ticks.length === 0 ? (
              <div className="opacity-40">No ticks yet</div>
            ) : (
              <div className="space-y-0.5 font-mono text-[10px] max-h-72 overflow-y-auto">
                {ticks.map((t, i) => (
                  <div
                    key={i}
                    className={`flex gap-2 px-1 py-0.5 rounded ${
                      t.executed ? "bg-sky-900/20" : "bg-slate-800/20 opacity-60"
                    }`}
                  >
                    <span className="text-slate-500 w-32 shrink-0">{t.ts}</span>
                    <span className="text-violet-400 w-44 shrink-0 truncate">
                      {t.lever ?? "—"}
                    </span>
                    <span className="text-slate-300 truncate">{t.reason}</span>
                  </div>
                ))}
              </div>
            )}
          </section>
```

- [ ] **Step 3: Render Levers grid after Recent Ticks**

Below the ticks `</section>`, add:

```tsx
          {/* Levers grid */}
          <section>
            <h2 className="font-bold text-sm mb-2">
              🦾 Levers ({status?.registered_levers.length ?? 0})
            </h2>
            <div
              className="grid gap-1.5"
              style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}
            >
              {(status?.registered_levers ?? []).sort().map((name) => (
                <button
                  key={name}
                  onClick={() => openLever(name)}
                  className={`text-left p-2 rounded ${leverBgClass(
                    name,
                  )} hover:brightness-125 flex items-center gap-2`}
                >
                  <span
                    className={`inline-block w-2 h-2 rounded-full bg-slate-500`}
                  />
                  <span className="text-[10px] truncate">{name}</span>
                </button>
              ))}
            </div>
          </section>
```

- [ ] **Step 4: Add the right-side detail drawer**

Replace the outer wrapper's `<div className="flex flex-1 min-h-0 overflow-hidden">` content to include both the left column and a conditional right drawer.

Find the top-level JSX root:

```tsx
  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <div className="flex-1 flex flex-col overflow-hidden bg-slate-950/60">
```

Leave that unchanged. At the end of the root (before the closing `</div>` of the outermost wrapper), add the drawer:

```tsx
      {selectedLever && (
        <div className="w-[420px] border-l border-slate-800 bg-slate-900/80 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-slate-800 flex items-center gap-2">
            <h2 className="font-bold text-sm text-violet-400">{selectedLever}</h2>
            <button
              onClick={closeLever}
              className="ml-auto text-slate-400 hover:text-slate-200 text-sm"
            >
              ✕
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2 text-xs">
            {loadingHistory ? (
              <div className="opacity-50">Loading…</div>
            ) : leverHistory.length === 0 ? (
              <div className="opacity-40">No history yet</div>
            ) : (
              leverHistory.map((r, i) => (
                <div key={i} className="p-2 rounded bg-slate-800/60 space-y-1">
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-block w-2 h-2 rounded-full ${statusDotClass(
                        r.status,
                      )}`}
                    />
                    <span className="font-semibold">{r.status}</span>
                    <span className="text-slate-500 text-[10px] ml-auto">
                      {r.started_at}
                    </span>
                  </div>
                  <div className="opacity-70">{r.reason}</div>
                  {Object.keys(r.outcome).length > 0 && (
                    <details>
                      <summary className="cursor-pointer text-[10px] text-slate-400">
                        outcome
                      </summary>
                      <pre className="mt-1 font-mono text-[10px] whitespace-pre-wrap opacity-80">
                        {JSON.stringify(r.outcome, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}
```

- [ ] **Step 5: Verify build**

Run: `cd frontend && npm run build`

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/AutonomicPanel.tsx
git commit -m "feat(frontend): AutonomicPanel recent ticks + levers grid + detail drawer"
```

---

## Task 5: Immune signatures section + slow-poll wiring

Add the read-only immune signatures table and set up the second (slow) poll timer for it.

**Files:**
- Modify: `frontend/src/components/AutonomicPanel.tsx`

- [ ] **Step 1: Extend imports and state**

Update imports:

```tsx
import {
  fetchAutonomicStatus,
  fetchPending,
  approvePending,
  rejectPending,
  fetchTicks,
  fetchLeverHistory,
  fetchImmune,
  toggleKillSwitch,
  AutonomicStatus,
  PendingEntry,
  TickEntry,
  LeverReport,
  ImmuneSignature,
} from "../api";
```

Add state declaration:

```tsx
const [immune, setImmune] = useState<ImmuneSignature[]>([]);
```

Add a separate `refreshSlow` callback and a second `useEffect` for the 30-second timer:

```tsx
const refreshSlow = useCallback(async () => {
  try {
    const result = await fetchImmune();
    setImmune(result.signatures);
  } catch (e: any) {
    flash("Error: " + e.message);
  }
}, []);

useEffect(() => {
  refreshSlow();
  const t = setInterval(refreshSlow, 30000);
  return () => clearInterval(t);
}, [refreshSlow]);
```

- [ ] **Step 2: Render Immune section after the Levers grid**

Below the Levers `</section>`, add:

```tsx
          {/* Immune signatures */}
          <section>
            <h2 className="font-bold text-sm mb-2">
              🧬 Immune Signatures ({immune.length})
            </h2>
            {immune.length === 0 ? (
              <div className="opacity-40">No signatures loaded</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-800">
                      <th className="text-left py-1 pr-2">id</th>
                      <th className="text-left py-1 pr-2">severity</th>
                      <th className="text-left py-1 pr-2">fix_lever</th>
                      <th className="text-right py-1 pr-2">observed</th>
                      <th className="text-right py-1">success</th>
                    </tr>
                  </thead>
                  <tbody>
                    {immune.map((s) => (
                      <tr key={s.id} className="border-b border-slate-800/50">
                        <td className="py-1 pr-2 font-mono text-violet-400">{s.id}</td>
                        <td className="py-1 pr-2">{s.severity}</td>
                        <td className="py-1 pr-2 text-sky-400">{s.fix_lever}</td>
                        <td className="py-1 pr-2 text-right">{s.observed_count}</td>
                        <td className="py-1 text-right">
                          {s.success_rate !== null
                            ? `${Math.round(s.success_rate * 100)}%`
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build`

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/AutonomicPanel.tsx
git commit -m "feat(frontend): AutonomicPanel immune signatures table + slow-poll timer"
```

---

## Task 6: StatusBar autonomic indicator + App.tsx wiring

Wire autonomic state at `App.tsx` level so the bottom `StatusBar` can show a compact indicator + pending badge across all tabs.

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/StatusBar.tsx`

- [ ] **Step 1: Extend `App.tsx` state and refresh**

Edit `frontend/src/App.tsx`. Update the import list at the top:

```tsx
import { fetchStatus, newSession, fetchAutonomicStatus, fetchPending, StatusPayload, AutonomicStatus } from "./api";
```

Add two new state declarations inside the component, next to the existing `status` / `selectedTopic` / `tab` ones:

```tsx
const [autonomic, setAutonomic] = useState<AutonomicStatus | null>(null);
const [pendingCount, setPendingCount] = useState<number>(0);
```

Update the `refresh` function to fetch all three in parallel:

```tsx
const refresh = async () => {
  const [statusResult, autonomicResult, pendingResult] = await Promise.allSettled([
    fetchStatus(),
    fetchAutonomicStatus(),
    fetchPending(),
  ]);
  if (statusResult.status === "fulfilled") setStatus(statusResult.value);
  if (autonomicResult.status === "fulfilled") setAutonomic(autonomicResult.value);
  if (pendingResult.status === "fulfilled") setPendingCount(pendingResult.value.pending.length);
};
```

Update the StatusBar invocation to pass the new props:

```tsx
      <StatusBar status={status} autonomic={autonomic} pendingCount={pendingCount} />
```

- [ ] **Step 2: Extend `StatusBar.tsx` props and render**

Replace the content of `frontend/src/components/StatusBar.tsx` with:

```tsx
import { StatusPayload, AutonomicStatus } from "../api";

function Dot({ ok, title }: { ok: boolean | undefined; title: string }) {
  const color = ok === undefined ? "bg-slate-500" : ok ? "bg-emerald-400" : "bg-rose-500";
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${color}`}
      title={title + (ok === false ? " — unavailable" : "")}
    />
  );
}

export default function StatusBar({
  status,
  autonomic,
  pendingCount,
}: {
  status: StatusPayload | null;
  autonomic?: AutonomicStatus | null;
  pendingCount?: number;
}) {
  if (!status) return null;
  const r = status.router as any;
  const hasRouter = r && !("error" in r);

  const modeColors: Record<string, string> = {
    local_full: "bg-emerald-700",
    cloud_finetune: "bg-sky-700",
    local_cpu: "bg-amber-700",
    claude_only: "bg-violet-700",
  };

  return (
    <div className="flex items-center gap-4 px-4 py-2 text-xs border-t border-slate-800 bg-slate-900/60 flex-wrap">
      <span
        className={`px-2 py-0.5 rounded ${modeColors[status.mode] || "bg-slate-700"}`}
        title={`training: ${status.training_location}`}
      >
        {status.mode}
      </span>
      <span>{status.topics_total} topics</span>
      <span>
        core: {status.core_tokens}/{status.core_max}
      </span>
      <span>finetune: {status.finetune_count}</span>
      <span>project: {status.current_project || "—"}</span>

      {hasRouter && (
        <>
          <span className="border-l border-slate-700 pl-4 flex items-center gap-1">
            <Dot ok={r.model_a_available} title="Model A" />
            A: <span className="text-sky-400">{status.model_a}</span>
          </span>
          <span className="flex items-center gap-1">
            <Dot ok={r.model_b_available} title="Model B" />
            B: <span className="text-emerald-400">{status.model_b}</span>
            {status.model_version ? ` (${status.model_version})` : ""}
          </span>
          <span className="text-slate-400">
            today A:{r.api_calls_today} / B:{r.model_b_calls_today} · ${r.api_cost_today?.toFixed(3)}
            {r.budget_usd ? `/${r.budget_usd}` : ""}
          </span>
        </>
      )}

      {autonomic && (
        <span className="border-l border-slate-700 pl-4 flex items-center gap-1">
          <Dot
            ok={autonomic.enabled && autonomic.scheduler_running}
            title="Autonomic"
          />
          <span className="text-slate-400">
            autonomic: {autonomic.registered_levers.length} levers
          </span>
          {pendingCount !== undefined && pendingCount > 0 && (
            <span className="text-amber-400 font-bold">⚠ {pendingCount} pending</span>
          )}
        </span>
      )}

      {!hasRouter && <span className="ml-auto text-rose-400">router: {r?.error || "—"}</span>}
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build`

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/StatusBar.tsx
git commit -m "feat(frontend): StatusBar autonomic indicator + App.tsx state wiring"
```

---

## Task 7: README update + manual smoke verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append AutonomicPanel mention to README**

Find the existing `_Body + yellow lever (D-08):_` block in `README.md`. Add below the **HTTP endpoints** subsection (which was added in D-08):

```markdown
**Frontend (D-09):**
- `🦾 Autonomic` tab in the web UI (`frontend/src/components/AutonomicPanel.tsx`) shows kill switch, pending approvals with Approve/Reject, recent ticks, 19-lever grid, immune signatures, and a per-lever history drawer.
- `StatusBar` at the bottom of every tab shows `autonomic: N levers` with a health dot and `⚠ N pending` badge when yellow actions are queued.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README documents AutonomicPanel and StatusBar indicator"
```

- [ ] **Step 3: Manual smoke test (no commit — checklist only)**

Record the results in the PR body when finishing the branch.

1. Backend: `.venv/Scripts/python.exe -m uvicorn backend.main:app --reload`. Wait for `Autonomic scheduler started` in logs.
2. Frontend: `cd frontend && npm run dev`. Open the shown URL in browser.
3. Click **🦾 Autonomic** tab.
   - Header shows a green "● ENABLED" pill, `scheduler: running` (emerald text), `19 levers registered`.
   - StatusBar at the bottom shows `autonomic: 19 levers` with a green dot; no pending badge.
4. Enqueue a pending action from a separate terminal:
   ```bash
   curl -X POST http://localhost:8000/api/autonomic/pending \
     -H "Content-Type: application/json" \
     -d "{\"lever\":\"FIRE_TOOL_INSTALL\",\"params\":{\"command\":\"pip_install\",\"package\":\"httpx\"}}"
   ```
   Expected: response `{"id":"...","status":"queued"}`.
5. Within ~5s the Autonomic tab shows the pending entry (amber card with Approve / Reject buttons), StatusBar shows `⚠ 1 pending` in amber.
6. Click **Reject** → confirm → card disappears, badge goes away. Flash bar shows "Rejected".
7. Click the "● ENABLED" pill → confirm → pill turns rose, text now "● DISABLED". StatusBar dot turns rose. Reload tab — state persists. Click again to re-enable.
8. Click any lever tile (e.g. `FIRE_INTEGRITY_HEARTBEAT`) → right drawer opens. If the lever has no history yet, shows "No history yet". Start uvicorn long enough for `FIRE_INTEGRITY_HEARTBEAT` to fire (5 minutes via its cooldown) — the drawer then shows 1+ reports with outcome JSON expandable.
9. Immune Signatures table lists the 5 seed signatures with severity / fix_lever / observed_count / success_rate columns.
10. `cd frontend && npm run build` — build still succeeds after all changes.

If any step fails, fix before merging. Report issues in the PR body.

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] `cd frontend && npm run build` — succeeds with no errors.
- [ ] Backend tests still green: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q` — 257 passed (no backend changes in D-09, so this is a sanity check).
- [ ] Manual smoke test from Task 7 Step 3 — all 10 steps behave as described.
- [ ] Model X v0 complete: 19/19 levers + first-class UI observability.

If all pass, D-09 is done. Model X v0 is shipped.

---

## Out of scope for D-09

Explicitly NOT in this plan:

- **Unit tests** for the React component (vitest is not in the project; introducing it is a separate decision).
- **Enqueue-action form** in the UI (users enqueue via `curl` or future CLI; if needed often, build a small form in a micro-project).
- **Batched `GET /api/autonomic/levers`** endpoint for eager tile coloring (lazy per-tile fetch on click is the v0 trade-off; revisit if UX feels weak).
- **Immune signature editing** (read-only in v0; would need new backend endpoints and safety review).
- **Log streaming / SSE** for live tick updates (5-second poll is good enough at current rates).
- **Mobile layout** (dev tool, desktop-only).
- **i18n** (all labels English, matches existing panels).
- **Chart visualisations** (tick-frequency timeline, cost curve, regression graph) — future polish.
- **Test browser via Playwright/Cypress** — manual smoke is the bar today.
