import { useCallback, useEffect, useState } from "react";
import {
  completeTracker,
  createTodo,
  fetchTrackers,
  updateTrackerStep,
  type Tracker,
  type TrackerStep,
} from "../api";

// One family, so the set reads as a scale rather than five loose colours.
const STATUS_COLORS: Record<string, string> = {
  pending: "bg-surface-hover text-ink-dim border border-edge-strong",
  active: "bg-accent-soft text-accent border border-accent/30",
  done: "bg-ok/15 text-ok border border-ok/30",
  blocked: "bg-danger/15 text-danger border border-danger/30",
  stalled: "bg-warn/15 text-warn border border-warn/30",
};
const STEP_STATUSES = ["pending", "active", "done", "blocked", "stalled"];
const OPEN = (s: string) => s === "pending" || s === "active";

function fmtDue(due: string): string {
  if (!due) return "—";
  const d = new Date(due);
  if (isNaN(d.getTime())) return due;
  return d.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// The agent schedules in UTC; the user thinks in their own zone, so every
// stamp on this board is rendered by the browser's locale.
function fmtShort(when: string): string {
  const d = new Date(when);
  if (isNaN(d.getTime())) return when;
  const sameDay = d.toDateString() === new Date().toDateString();
  return d.toLocaleString(undefined, {
    ...(sameDay ? {} : { weekday: "short" }),
    hour: "2-digit",
    minute: "2-digit",
  });
}

// The follow-up state, in words. This is what the old board was blind to: a
// task with a date raises itself again on a growing gap until it is closed,
// and the user should watch that happen rather than be surprised by it.
function FollowUp({ step }: { step: TrackerStep }) {
  const sent = step.nudges || 0;
  // A step normally stalls having spent its whole budget, but it can be
  // parked for other reasons — "gave up after 0 reminders" reads as a bug.
  if (step.status === "stalled")
    return (
      <span className="text-warn">
        {sent > 0
          ? `no answer after ${sent} reminder${sent === 1 ? "" : "s"}`
          : "parked, not asked"}
      </span>
    );
  if (!OPEN(step.status)) return <span className="opacity-30">—</span>;
  if (!step.next_check_at)
    return (
      <span className="opacity-30">{step.due_at ? "armed" : "no date"}</span>
    );
  return (
    <span className="opacity-70">
      {sent > 0 && <>asked {sent}× · </>}
      next {fmtShort(step.next_check_at)}
    </span>
  );
}

export default function TrackerBoard() {
  const [trackers, setTrackers] = useState<Tracker[]>([]);
  const [err, setErr] = useState("");
  const [draft, setDraft] = useState("");
  const [draftDue, setDraftDue] = useState("");
  const [busy, setBusy] = useState(false);

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

  const setStatus = async (tId: string, sId: string, status: string) => {
    await updateTrackerStep(tId, sId, { status });
    load();
  };

  const onComplete = async (trackerId: string, title: string) => {
    if (!confirm(`Archive project "${title}"? It moves to long-term memory.`))
      return;
    await completeTracker(trackerId);
    load();
  };

  const onAdd = async () => {
    const title = draft.trim();
    if (!title || busy) return;
    setBusy(true);
    try {
      // datetime-local yields wall-clock with no zone; the Date ctor reads
      // it as local, which is the time the user actually meant.
      const due = draftDue
        ? new Date(draftDue).toISOString().slice(0, 19) + "Z"
        : "";
      await createTodo(title, due);
      setDraft("");
      setDraftDue("");
      load();
    } catch (e: any) {
      setErr(e.message || "could not add");
    } finally {
      setBusy(false);
    }
  };

  if (err) return <div className="p-4 text-sm text-danger">Error: {err}</div>;

  // A one-step inbox entry is a task; anything else is work with structure.
  // Same store, two shapes — rendering "buy medicine" as a project table is
  // what made the simple case feel wrong.
  const todos = trackers.filter((t) => t.domain === "inbox");
  const projects = trackers.filter((t) => t.domain !== "inbox");
  const isOpen = (t: Tracker) => t.steps.some((s) => OPEN(s.status));
  const ordered = [...todos.filter(isOpen), ...todos.filter((t) => !isOpen(t))];

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-8">
      {/* ---- Task list ---- */}
      <section>
        <h2 className="mb-2 text-micro font-semibold uppercase text-ink-dim">
          Task list
        </h2>

        <div className="flex flex-wrap gap-2 mb-3">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onAdd()}
            placeholder="Add a task…"
            className="min-w-[12rem] flex-1 text-sm"
          />
          <input
            type="datetime-local"
            value={draftDue}
            onChange={(e) => setDraftDue(e.target.value)}
            title="Optional. With a time, the task follows up until you close it."
            className="text-sm text-ink-dim"
          />
          <button
            onClick={onAdd}
            disabled={busy || !draft.trim()}
            className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
          >
            Add
          </button>
        </div>

        {todos.length === 0 ? (
          <p className="py-3 text-sm text-ink-faint">
            Nothing on the list. Add one above, or tell the agent “remind me
            to …”.
          </p>
        ) : (
          <ul className="divide-y divide-edge overflow-hidden rounded-xl2 border border-edge">
            {ordered.map((t) => {
              const s = t.steps[0];
              if (!s) return null;
              const done = s.status === "done";
              return (
                <li
                  key={t.id}
                  className={`flex items-center gap-3 bg-surface px-3 py-2 ${
                    s.status === "stalled" ? "border-l-2 border-warn" : ""
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={done}
                    onChange={() =>
                      setStatus(t.id, s.id, done ? "pending" : "done")
                    }
                    className="w-4 h-4 accent-emerald-600 shrink-0"
                  />
                  <span
                    className={`flex-1 min-w-0 truncate text-sm ${
                      done ? "line-through opacity-40" : ""
                    }`}
                    title={s.title}
                  >
                    {s.title}
                  </span>
                  <span className="text-xs opacity-60 whitespace-nowrap hidden sm:inline">
                    {s.due_at ? fmtShort(s.due_at) : ""}
                  </span>
                  <span className="text-xs whitespace-nowrap hidden md:inline">
                    <FollowUp step={s} />
                  </span>
                  {s.status === "stalled" && (
                    <button
                      onClick={() => setStatus(t.id, s.id, "pending")}
                      className="text-xs border border-edge-strong text-warn hover:bg-warn hover:text-white rounded px-2 py-0.5 whitespace-nowrap"
                      title="Restarts the reminders from the beginning"
                    >
                      still relevant
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* ---- Projects ---- */}
      <section>
        <h2 className="mb-2 text-micro font-semibold uppercase text-ink-dim">
          Projects
        </h2>
        {projects.length === 0 ? (
          <p className="py-3 text-sm text-ink-faint">
            No active projects. The agent opens one with create_tracker when
            the work has real internal structure.
          </p>
        ) : (
          <div className="space-y-6">
            {projects.map((t) => (
              <div
                key={t.id}
                className="rounded-xl2 border border-edge bg-surface"
              >
                <header className="flex items-center justify-between px-4 py-2 border-b border-edge">
                  <h3 className="font-bold">
                    {t.title}
                    <span className="ml-2 text-xs opacity-50">{t.domain}</span>
                    <span className="ml-2 text-xs opacity-40">
                      {t.steps.filter((s) => s.status === "done").length}/
                      {t.steps.length}
                    </span>
                  </h3>
                  <button
                    onClick={() => onComplete(t.id, t.title)}
                    className="text-xs border border-edge-strong text-ink-dim hover:bg-ok hover:text-white rounded px-2 py-1"
                  >
                    complete
                  </button>
                </header>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="text-ink-dim">
                      <tr className="text-left">
                        <th className="px-4 py-1 font-medium">Step</th>
                        <th className="px-2 py-1 font-medium">Due</th>
                        <th className="px-2 py-1 font-medium">Status</th>
                        <th className="px-2 py-1 font-medium">Follow-up</th>
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
                        <tr
                          key={s.id}
                          className={`border-t border-edge/60 ${
                            s.status === "stalled" ? "bg-warn/5" : ""
                          }`}
                        >
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
                                setStatus(t.id, s.id, e.target.value)
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
                          <td className="px-2 py-1.5 whitespace-nowrap">
                            <FollowUp step={s} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
