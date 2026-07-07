import { useEffect, useState } from "react";
import { fetchFrames, type Frame } from "../api";

function fmtCreated(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function FramesView() {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetchFrames()
      .then((r) => setFrames(r.frames || []))
      .catch((e: any) => setErr(e.message || "failed to load frames"));
  }, []);

  if (err) return <div className="text-rose-400 text-sm p-3">{err}</div>;
  if (!frames.length)
    return (
      <div className="text-slate-400 text-sm p-3">
        No problem frames yet. The agent creates one (via{" "}
        <code>frame_problem</code>) whenever it interrogates a big build into
        its real components before scoping it with you.
      </div>
    );

  return (
    <div className="space-y-4">
      {frames.map((f) => {
        const cov = f.coverage || {
          mvp_components: f.components.filter((c) => c.mvp).length,
          total_listed: f.components.length,
          mvp_pct_of_listed: 0,
        };
        return (
          <div
            key={f.id}
            className="rounded-lg border border-slate-700 bg-slate-800/60 p-4"
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <span className="font-semibold">{f.title || "(untitled)"}</span>
                <span className="ml-2 text-xs text-slate-400">
                  {f.domain} · {fmtCreated(f.created_at)}
                </span>
              </div>
              <span
                className="text-xs px-2 py-1 rounded bg-slate-700 text-slate-200"
                title="How much of the full component map the scoped MVP covers"
              >
                MVP {cov.mvp_components}/{cov.total_listed} (~
                {cov.mvp_pct_of_listed}%)
              </span>
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {f.components.map((c) => (
                <span
                  key={c.name}
                  title={`${c.role || ""}${c.source ? ` · src: ${c.source}` : ""} · confidence: ${c.confidence}`}
                  className={
                    "text-xs px-2 py-0.5 rounded-full border " +
                    (c.mvp
                      ? "bg-emerald-800/70 border-emerald-600 text-emerald-100"
                      : "bg-transparent border-slate-600 text-slate-400")
                  }
                >
                  {c.name}
                </span>
              ))}
            </div>
            {f.proposed_scope && (
              <p className="mt-3 text-sm text-slate-300">
                <span className="text-slate-500">scope:</span>{" "}
                {f.proposed_scope}
              </p>
            )}
            {f.open_questions?.length > 0 && (
              <ul className="mt-2 text-xs text-amber-300/90 list-disc ml-5">
                {f.open_questions.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </div>
  );
}
