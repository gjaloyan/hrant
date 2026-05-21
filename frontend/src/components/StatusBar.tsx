import { useEffect, useState } from "react";
import { StatusPayload, AutonomicStatus, EmbeddingsStatusResponse } from "../api";

// Audit follow-up — token-first daily counters. The cost-first
// view (`api_cost_today`) was based on a $0.01/call estimate that
// drifted significantly from real cost; tokens are now the source
// of truth, with cost as a derived secondary.
type TokensToday = {
  date: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_output_ratio: number;
  cost_usd: number;
  llm_calls: number;
};

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

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
  embeddings,
}: {
  status: StatusPayload | null;
  autonomic?: AutonomicStatus | null;
  pendingCount?: number;
  embeddings?: EmbeddingsStatusResponse | null;
}) {
  // Daily token counters from /api/tokens/today — token-first
  // display per audit follow-up.
  const [tokensToday, setTokensToday] = useState<TokensToday | null>(null);
  useEffect(() => {
    let cancelled = false;
    const fetchTokens = () => {
      fetch("/api/tokens/today")
        .then((r) => r.ok ? r.json() : null)
        .then((d) => { if (!cancelled && d) setTokensToday(d as TokensToday); })
        .catch(() => {/* leave previous value */});
    };
    fetchTokens();
    const id = setInterval(fetchTokens, 30_000);  // 30s refresh
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (!status) return null;
  const r = status.router as any;
  const hasRouter = r && !("error" in r);

  const modeColors: Record<string, string> = {
    local_full: "bg-emerald-700",
    cloud_finetune: "bg-sky-700",
    local_cpu: "bg-amber-700",
    cloud_only: "bg-violet-700",
    // Back-compat: old config.yaml may still report `claude_only`.
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
          <span
            className="text-slate-400"
            title={
              tokensToday
                ? `today input ${tokensToday.input_tokens.toLocaleString()} | ` +
                  `output ${tokensToday.output_tokens.toLocaleString()} | ` +
                  `ratio ${tokensToday.input_output_ratio.toFixed(1)}:1 | ` +
                  `LLM calls ${tokensToday.llm_calls} | cost $${tokensToday.cost_usd.toFixed(4)}`
                : "tokens not yet loaded"
            }
          >
            today {tokensToday ? (
              <>
                <span className="text-emerald-300">{fmtTokens(tokensToday.total_tokens)}t</span>
                {" · "}
                <span className={tokensToday.input_output_ratio > 15 ? "text-amber-400" : "text-slate-400"}>
                  ratio {tokensToday.input_output_ratio.toFixed(1)}:1
                </span>
                {" · "}
                A:{r.api_calls_today} / B:{r.model_b_calls_today}
              </>
            ) : (
              <>A:{r.api_calls_today} / B:{r.model_b_calls_today}</>
            )}
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

      {embeddings && (
        <span
          className="border-l border-slate-700 pl-4 flex items-center gap-1"
          title={
            embeddings.embedder.last_error
              ? `embedder: ${embeddings.embedder.last_error}`
              : `${embeddings.coverage.embedded}/${embeddings.coverage.total_notes} notes embedded`
          }
        >
          <Dot
            ok={
              embeddings.embedder.backend === "disabled"
                ? false
                : embeddings.embedder.backend
                ? true
                : undefined
            }
            title="Embeddings"
          />
          <span className="text-slate-400">
            memory: {embeddings.embedder.backend === "disabled" || !embeddings.embedder.backend
              ? <span className="text-amber-400">text-only</span>
              : <span className="font-mono">{embeddings.embedder.model}</span>
            }
          </span>
          {embeddings.coverage.missing > 0 && embeddings.embedder.backend !== "disabled" && (
            <span className="text-amber-400 font-bold">
              ⚠ {embeddings.coverage.missing} unembedded
            </span>
          )}
        </span>
      )}

      {!hasRouter && <span className="ml-auto text-rose-400">router: {r?.error || "—"}</span>}
    </div>
  );
}
