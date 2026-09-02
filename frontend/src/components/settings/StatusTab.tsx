import { useEffect, useState } from "react";
import { StatusPayload } from "../../api";

// Audit follow-up: Settings StatusTab was showing "Cost today" in
// USD as the headline daily metric. USD is a derived number based on
// per-model pricing that drifts; tokens are the source of truth.
// Token-first display matches the StatusBar refactor.
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

type Props = {
  status: StatusPayload;
  onCompare: () => void;
  onRefresh: () => void;
};

/** What is actually wrong, if anything.
 *
 * The screen is called System Status and its subtitle promises the health
 * of every moving part, but it rendered facts at uniform weight — "Model
 * B: available: no" beside "Topics: 28" — with nothing to say which of
 * those is a problem. In `cloud_only` mode an unavailable Model B is the
 * expected state, not a fault, and the screen could not tell you that.
 *
 * Values do not become status until something judges them.
 */
function assess(status: any, r: any, hasRouter: boolean) {
  const out: { tone: "danger" | "warn"; text: string }[] = [];
  const cloudOnly =
    status.mode === "cloud_only" || status.mode === "claude_only";

  if (!hasRouter) {
    out.push({ tone: "danger", text: `Router unavailable: ${r?.error || "unknown"}` });
  } else {
    if (!r.model_a_available)
      out.push({
        tone: "danger",
        text: `The main model (${status.model_a}) is not answering — turns will fail or fall back.`,
      });
    // Model B being down only matters where something would use it.
    if (!r.model_b_available && !cloudOnly)
      out.push({
        tone: "warn",
        text: "The fallback model is unavailable, so a failure of the main model has nowhere to go.",
      });
  }

  const corePct = (status.core_tokens / Math.max(1, status.core_max)) * 100;
  if (corePct >= 90)
    out.push({
      tone: "warn",
      text: `Core memory is ${Math.round(corePct)}% full, and it is re-sent on every turn.`,
    });

  return out;
}

export default function StatusTab({ status, onCompare, onRefresh }: Props) {
  const r = status.router as any;
  const hasRouter = r && !("error" in r);
  const problems = assess(status, r, hasRouter);

  const [tokensToday, setTokensToday] = useState<TokensToday | null>(null);
  useEffect(() => {
    let cancelled = false;
    const fetchTokens = () => {
      fetch("/api/tokens/today")
        .then((res) => (res.ok ? res.json() : null))
        .then((d) => {
          if (!cancelled && d) setTokensToday(d as TokensToday);
        })
        .catch(() => {
          /* leave previous value */
        });
    };
    fetchTokens();
    const id = setInterval(fetchTokens, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="space-y-4">
      {/* The verdict first. A screen that only lists values leaves the
          reader to work out which of them is bad. */}
      {problems.length === 0 ? (
        <div className="flex items-center gap-2 rounded-xl2 border border-ok/30 bg-ok/10 px-4 py-3 text-sm">
          <span className="h-2 w-2 rounded-full bg-ok" />
          <span>Everything reachable and within its limits.</span>
        </div>
      ) : (
        <div className="space-y-2">
          {problems.map((p, i) => (
            <div
              key={i}
              className={
                "flex items-start gap-2 rounded-xl2 border px-4 py-3 text-sm " +
                (p.tone === "danger"
                  ? "border-danger/30 bg-danger/10"
                  : "border-warn/30 bg-warn/10")
              }
            >
              <span
                className={
                  "mt-1.5 h-2 w-2 shrink-0 rounded-full " +
                  (p.tone === "danger" ? "bg-danger" : "bg-warn")
                }
              />
              <span>{p.text}</span>
            </div>
          ))}
        </div>
      )}

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
              {r.model_b_available
                ? "available"
                : status.mode === "cloud_only" || status.mode === "claude_only"
                ? "not used in this mode"
                : "unavailable"}
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
            <div title={tokensToday ? `input ${tokensToday.input_tokens.toLocaleString()} / output ${tokensToday.output_tokens.toLocaleString()} · ratio ${tokensToday.input_output_ratio.toFixed(1)}:1` : ""}>
              Tokens today: <b className="text-emerald-300">{tokensToday ? fmtTokens(tokensToday.total_tokens) : "—"}</b>
              {tokensToday && (
                <span className="opacity-60 ml-1">
                  ({tokensToday.llm_calls} calls
                  {tokensToday.cost_usd > 0 ? ` · $${tokensToday.cost_usd.toFixed(3)}` : ""})
                </span>
              )}
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
