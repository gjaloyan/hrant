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

  const cells = useMemo(() => {
    const first = new Date(month.getFullYear(), month.getMonth(), 1);
    const daysInMonth = new Date(
      month.getFullYear(),
      month.getMonth() + 1,
      0,
    ).getDate();
    const lead = first.getDay();
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
                    mk.kind === "remind" ? "bg-amber-800/70" : "bg-sky-800/70"
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
