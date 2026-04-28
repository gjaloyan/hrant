import { StatusPayload } from "../../api";

type Props = {
  status: StatusPayload;
  onCompare: () => void;
  onRefresh: () => void;
};

export default function StatusTab({ status, onCompare, onRefresh }: Props) {
  const r = status.router as any;
  const hasRouter = r && !("error" in r);

  return (
    <div className="space-y-4">
      <h3 className="font-bold">System Status</h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs opacity-60 mb-1">Mode</div>
          <div className="font-bold">{status.mode}</div>
          <div className="text-xs opacity-60 mt-1">training: {status.training_location}</div>
        </div>
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs opacity-60 mb-1">Topics</div>
          <div className="font-bold">{status.topics_total}</div>
          <div className="text-xs opacity-60 mt-1">
            {Object.entries(status.by_category)
              .map(([k, v]) => `${k}: ${v}`)
              .join(", ")}
          </div>
        </div>
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs opacity-60 mb-1">Core Memory</div>
          <div className="font-bold">
            {status.core_tokens}/{status.core_max} tokens
          </div>
        </div>
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs opacity-60 mb-1">Finetune Queue</div>
          <div className="font-bold">{status.finetune_count}</div>
          <div className="text-xs opacity-60 mt-1">
            enabled: {status.finetune_enabled ? "yes" : "no"}
          </div>
        </div>
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs opacity-60 mb-1">Model A</div>
          <div className="font-bold text-sky-400">{status.model_a || "—"}</div>
          {hasRouter && (
            <div className="text-xs opacity-60 mt-1">
              available: {r.model_a_available ? "yes" : "no"}
            </div>
          )}
        </div>
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs opacity-60 mb-1">Model B</div>
          <div className="font-bold text-emerald-400">{status.model_b || "—"}</div>
          {status.model_version && (
            <div className="text-xs opacity-60 mt-1">version: {status.model_version}</div>
          )}
          {hasRouter && (
            <div className="text-xs opacity-60">
              available: {r.model_b_available ? "yes" : "no"}
            </div>
          )}
        </div>
      </div>

      {hasRouter && (
        <div className="bg-slate-800 rounded p-3 text-sm">
          <div className="font-semibold mb-2">Router Stats</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div>
              A calls today: <b>{r.api_calls_today}</b>
            </div>
            <div>
              B calls today: <b>{r.model_b_calls_today}</b>
            </div>
            <div>
              Cost today: <b>${r.api_cost_today?.toFixed(3)}</b>
              {r.budget_usd ? `/${r.budget_usd}` : ""}
            </div>
            <div>
              Total: A={r.total_a_calls} / B={r.total_b_calls}
            </div>
          </div>
          {r.last_reason && (
            <div className="text-xs opacity-60 mt-1">last routing: {r.last_reason}</div>
          )}
        </div>
      )}

      <div className="bg-slate-800 rounded p-3 text-sm">
        <div className="font-semibold mb-2">Project</div>
        <div className="text-xs">Active: {status.current_project || "(none)"}</div>
      </div>

      <div className="flex gap-2">
        <button
          onClick={onCompare}
          className="bg-violet-700 hover:bg-violet-600 rounded px-3 py-1 text-sm"
        >
          Compare Models
        </button>
        <button
          onClick={onRefresh}
          className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-sm"
        >
          Refresh
        </button>
      </div>
    </div>
  );
}
