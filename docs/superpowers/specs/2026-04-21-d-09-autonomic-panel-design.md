# D-09 — AutonomicPanel frontend (design)

**Status:** Design (no implementation)
**Date:** 2026-04-21
**Parent:** [Model X spec, Section 11 — phased delivery](./2026-04-16-model-x-autonomic-design.md)
**Delivers:** `AutonomicPanel.tsx` (new 11th tab) + `StatusBar` autonomic indicator + 9 typed API functions in `api.ts`. Final sub-project of Model X — brings first-class UI observability to the 19 levers shipped through D-01..D-08.

---

## 0. Context

D-01 through D-08 are merged. Backend is complete: 19 levers, 257 tests, 8 HTTP endpoints under `/api/autonomic/*`, yellow-approval flow with per-entry ids. D-09 is the React/TypeScript side that consumes all of that.

Existing frontend has 10 tab-based panels (`App.tsx` → `Tab` union) and a bottom `StatusBar` with mode + router indicators. Styling is Tailwind with a slate/sky/emerald palette. No unit-test framework for React today — quality gate is `tsc && vite build` plus manual smoke check. D-09 matches these conventions; no new infra.

**Goal:** one new tab (`🦾 Autonomic`) that lets the user observe the scheduler, approve/reject pending yellow actions, see recent tick decisions and lever history, list immune signatures, and toggle the kill switch. Bottom `StatusBar` gains a compact autonomic indicator with pending-count badge.

**Non-goals:** unit tests for the React component (no vitest in the project — out of scope); cortex-triggered TOOL_INSTALL UI (no flow for that in v0); pretty charts / timelines / log streaming (poll-and-render is enough for the data sizes at this stage); mobile layout (dev tool, desktop-only); i18n (all labels in English like other panels).

---

## 1. `api.ts` — 9 new functions + 5 types

Append to `frontend/src/api.ts`:

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
  cost: { tokens_in: number; tokens_out: number; seconds: number; usd: number; };
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

**Error handling:** existing `json_get` / `json_post` throw on non-2xx. Component catches and flashes errors via the same pattern used by `GoalsPanel.tsx` (`flash("Error: " + e.message)`).

---

## 2. `AutonomicPanel.tsx` — layout and behavior

Two-column layout, matching `GoalsPanel.tsx` pattern: main list on the left, detail drawer on the right that opens when a lever is clicked.

### 2.1 Left column — scrollable sections top-to-bottom

**Header (pinned top):**
- Kill-switch toggle — large pill button. Green "●  ENABLED" when `status.enabled === true`, rose "●  DISABLED" otherwise. Click flips; if disabling, pass through `confirm("Disable autonomic scheduler?")`.
- Scheduler status dot — small emerald/slate indicator.
- Lever count label — "19 levers registered" (read from `status.registered_levers.length`).
- "Refresh" button — hard-reloads all sections (otherwise `setInterval`).

**Pending approvals section** (top priority — the only actionable block):
- Section title "⚠ Pending Approvals (N)" where N = `pending.length`.
- Empty state: "No pending actions" (slate text).
- Each entry renders a card: lever name (violet-500), params summary (e.g. `pip_install httpx`), `requested_at` relative time, **Approve** (emerald) and **Reject** (rose) buttons.
- On Approve: call `approvePending(id)`, flash "Approved — rc=0" or "Approved — failed: {reason}", refresh pending list.
- On Reject: `confirm("Reject this pending action?")` → `rejectPending(id)` → refresh.

**Recent ticks section:**
- Section title "📊 Recent Ticks" + `limit` selector (10 / 50 / 100, default 50).
- Scrollable list (max-height ~400px, `overflow-y-auto`). Each row: `ts`, `lever or reason`, `executed` dot.
- Row color: executed=sky background, cooldown=slate background (40% opacity), idle=slate bg + muted text.

**Levers grid:**
- Section title "🦾 Levers (19)".
- 19 tiles in a responsive grid (CSS grid, auto-fit minmax(180px, 1fr)). Each tile: lever name, category badge (immune=rose, autonomic=sky, body=amber), last-report status dot (success=emerald, failure=rose, skipped=slate, blocked_by_safety=amber, unknown=slate).
- Click on tile → sets `selectedLever` state → right drawer opens with `fetchLeverHistory(name, 10)`.

**Immune signatures section:**
- Section title "🧬 Immune Signatures".
- Read-only table: id, severity, fix_lever, `observed_count`, `success_rate` (formatted as "N/M (X%)" or "—" if null).
- Empty state: "No signatures loaded".

### 2.2 Right column — per-lever detail drawer

- Hidden when `selectedLever === null`. Width 420px when open.
- Header: lever name + close button.
- List of last 10 `LeverReport` dicts: `started_at`, `status`, `reason`, expandable `<details>` with `outcome` JSON pretty-printed.
- Does NOT refresh on poll — opens on click, refreshes on close+reopen (keeps the drawer stable while user reads).

### 2.3 Data fetching + poll intervals

Two `setInterval` timers:

| Timer | Period | Fetches |
|---|---|---|
| Fast | 5 seconds | `fetchAutonomicStatus`, `fetchPending`, `fetchTicks(50)` |
| Slow | 30 seconds | `fetchImmune`; lever last-report map (one `fetchLeverHistory(name, 1)` per lever on a rotating schedule so we don't hammer 19 endpoints at once) |

Actually simpler for v0: slow timer pulls lever map from a single derived field — read the last ~200 entries from `lever_log.jsonl` via a single GET and bucket by lever client-side. That requires an extra endpoint `GET /api/autonomic/levers` (without a name) that returns the last N combined. **Out of scope for D-09** — cost vs benefit. Instead, slow timer fetches `fetchLeverHistory` **only for the currently selected lever**; unselected lever tiles show a "·" gray dot until clicked. Accepts the limitation that tile colors are lazy. Pragmatic; add batched endpoint later if the UX feels off.

`useEffect` cleanup clears both intervals on unmount.

### 2.4 Message bar

Same pattern as `GoalsPanel.tsx` — tiny `msg` state cleared via `setTimeout`, displayed in slate bar at bottom of left column. `flash(text)` helper.

### 2.5 No enqueue-from-UI in v0

Parent spec (D-08) envisioned `POST /api/autonomic/pending` being callable from the UI to enqueue a TOOL_INSTALL manually. For D-09 we **don't add an enqueue form** — keeping scope tight. Users today enqueue via `curl` or a future CLI command. When the user finds themselves needing this often, add a small form in a later micro-project. Section 5 out-of-scope captures this.

---

## 3. `App.tsx` and `StatusBar.tsx` integration

### 3.1 `App.tsx` changes

- Add `"autonomic"` to `Tab` union type.
- Add TABS entry: `{ id: "autonomic", label: "Autonomic", icon: "🦾" }`. Placement: between `"intelligence"` and `"usage"` so the "brain" metaphors cluster (intelligence → autonomic → usage).
- Add render branch: `{tab === "autonomic" && <AutonomicPanel />}`.
- Import `AutonomicPanel` from `./components/AutonomicPanel`.
- Pull autonomic state at App level so StatusBar can show the indicator:

```tsx
const [autonomicStatus, setAutonomicStatus] = useState<AutonomicStatus | null>(null);
const [pendingCount, setPendingCount] = useState<number>(0);
```

- In `refresh()`: call `fetchAutonomicStatus` and `fetchPending` alongside `fetchStatus`, update state. Use `Promise.allSettled` so one failure doesn't block others.
- Pass to StatusBar: `<StatusBar status={status} autonomic={autonomicStatus} pendingCount={pendingCount} />`.

### 3.2 `StatusBar.tsx` changes

- Add optional props `autonomic?: AutonomicStatus | null` and `pendingCount?: number`.
- Append new section after router block (before the final `{!hasRouter && ...}` line):

```tsx
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
```

The `Dot` helper already exists in this file; reuse it.

---

## 4. File layout

**New files (1):**

```
frontend/src/components/
└── AutonomicPanel.tsx          # ~450 lines
```

**Modified files (3):**

- `frontend/src/api.ts` — append autonomic section (~60 lines).
- `frontend/src/App.tsx` — 11th tab + state + StatusBar props (~15 line delta).
- `frontend/src/components/StatusBar.tsx` — autonomic section (~15 line delta).

**Not modified:** backend, other panels, Chat, GraphViewer, NoteViewer, SessionsPanel, FinetunePanel, SettingsPanel.

---

## 5. Testing strategy

**Quality gate:** `cd frontend && npm run build` must succeed (`tsc && vite build`). This catches type errors, import mistakes, and bundling issues — the same bar the rest of the frontend uses.

**Manual smoke test** (documented in the plan's post-checklist):

1. Start uvicorn backend: `.venv/Scripts/python.exe -m uvicorn backend.main:app --reload`.
2. Start frontend: `cd frontend && npm run dev`.
3. Open browser to the dev URL, click **🦾 Autonomic** tab.
4. Verify header shows "19 levers registered" + kill-switch ENABLED.
5. Verify StatusBar shows "autonomic: 19 levers" with green dot.
6. Click **Disable** on kill switch → page confirms, StatusBar dot turns grey, "DISABLED" pill in header.
7. Re-enable. Toggle works both ways.
8. Enqueue a pending from a separate terminal:
   ```bash
   curl -X POST http://localhost:8000/api/autonomic/pending \
     -H "Content-Type: application/json" \
     -d '{"lever":"FIRE_TOOL_INSTALL","params":{"command":"pip_install","package":"httpx"}}'
   ```
   Verify pending section shows the entry within ~5s, StatusBar shows `⚠ 1 pending` in amber.
9. Click **Reject** on the entry → confirm → pending list empties, StatusBar badge disappears.
10. Click any lever tile → right drawer opens, shows last 10 reports (or "No history yet" if the lever never fired in this run).
11. Immune section shows the 5 seed signatures.
12. `npm run build` succeeds with no TypeScript errors.

**No vitest / jest.** Frontend unit tests are not established in this codebase and introducing them is a separate decision. If flaky behavior shows up in production use, we add tests for the specific failing path.

---

## 6. Open questions (not blocking)

1. **Lever tile color for untouched levers** — the "lazy" approach (gray dot until clicked) is accepted for v0. If UX feels weak, add a `GET /api/autonomic/levers` batch endpoint later (backend change — D-10 or a micro-project).
2. **Pending-action enqueue from UI** — deliberately out of scope for D-09. Workaround: `curl` or the future CLI. When the user reaches for it often, build a small form.
3. **Tick log pagination** — v0 shows last 50 with a selector for 10/50/100. Scrolling past 100 lines is fine as-is; no infinite scroll or calendar picker.
4. **Immune signature editing** — read-only in v0. Edit flow would need new backend endpoints and safety review; deferred.

---

## 7. What comes after D-09

**Model X complete.** 19/19 levers live, scheduler running, UI observable, yellow actions gated by user approval. The full design from [Model X spec](./2026-04-16-model-x-autonomic-design.md) is delivered.

Candidate follow-ups (micro-projects, not a new D-NN):
- Linux OS inventory extras for `FIRE_CAPABILITY_SCAN` when the user deploys to Linux.
- Cortex-triggered TOOL_INSTALL (L1/L2 hook that detects missing package and enqueues pip_install automatically).
- Batched `GET /api/autonomic/levers` endpoint + eager lever-tile status coloring in the panel.
- Enqueue-action form in the AutonomicPanel.
- React unit tests via vitest (separate decision).
- Fail-count + expiry for stale pending approvals.
- Legacy `finetune_queue.jsonl` migration lever (yellow).
- L1 router (embedding classifier) and L2 diagnoser (stock Qwen-Coder-7B) — the "autonomic gets its own brain" transition described in parent spec section 2 (v1 milestone).

At that point v0 of Model X is closed; v1 work starts with real usage data collected from D-09 onward.
