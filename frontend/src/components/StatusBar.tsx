import { StatusPayload } from "../api";

function Dot({ ok, title }: { ok: boolean | undefined; title: string }) {
  const color = ok === undefined ? "bg-slate-500" : ok ? "bg-emerald-400" : "bg-rose-500";
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${color}`}
      title={title + (ok === false ? " — unavailable" : "")}
    />
  );
}

export default function StatusBar({ status }: { status: StatusPayload | null }) {
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
      {!hasRouter && <span className="ml-auto text-rose-400">router: {r?.error || "—"}</span>}
    </div>
  );
}
