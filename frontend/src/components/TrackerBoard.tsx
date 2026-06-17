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
