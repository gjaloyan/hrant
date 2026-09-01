/** How the agent thought, step by step.
 *
 * This was the one thing the separate "Usage" section had that
 * Intelligence did not — the rest of that page duplicated panels already
 * here, and its headline figures read zero because they came from the
 * in-memory counter that empties on every restart. Moved rather than
 * copied, and the duplicate section retired.
 */
import { useState } from "react";
import type { RequestTrace } from "../../api";
import { Button, EmptyState } from "../../ui";

function StepBadge({ event }: { event: string }) {
  const colors: Record<string, string> = {
    classify: "bg-violet-900/60 text-violet-300",
    recall: "bg-emerald-900/60 text-emerald-300",
    think: "bg-sky-900/60 text-sky-300",
    decompose: "bg-amber-900/60 text-amber-300",
    subtask: "bg-orange-900/60 text-orange-300",
    verify: "bg-cyan-900/60 text-cyan-300",
    tool: "bg-slate-600 text-slate-300",
    tool_error: "bg-rose-900/60 text-rose-300",
    answer: "bg-green-900/60 text-green-300",
    critic: "bg-pink-900/60 text-pink-300",
    retry: "bg-yellow-900/60 text-yellow-300",
    learn: "bg-indigo-900/60 text-indigo-300",
  };
  const base = event.split(":")[0];
  const cls = colors[base] || "bg-slate-700 text-slate-400";
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] ${cls}`}>
      {event}
    </span>
  );
}

export default function ThinkingTraces({
  traces,
  onRefresh,
}: {
  traces: RequestTrace[];
  onRefresh: () => void;
}) {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-micro font-semibold uppercase text-ink-dim">
          Thinking traces
        </h3>
        <Button kind="ghost" size="sm" onClick={onRefresh}>
          Refresh
        </Button>
      </div>

      {traces.length === 0 && (
        <EmptyState title="No traces yet">
          A trace is recorded for each turn: how the agent classified the
          question, what it recalled, which tools it ran, and how it checked
          itself.
        </EmptyState>
      )}

      <div className="space-y-2">
        {traces.map((tr, i) => {
          const isOpen = expanded === i;
          const stepCount = tr.trace.length;
          const toolSteps = tr.trace.filter((s) => s.event.startsWith("tool")).length;
          const thinkSteps = stepCount - toolSteps;
          const lastTs = stepCount > 0 ? tr.trace[stepCount - 1].ts : 0;
          const lastTokens = stepCount > 0 ? tr.trace[stepCount - 1].tokens_so_far : 0;

          return (
            <div key={i} className="bg-slate-800/50 rounded overflow-hidden">
              <div
                className="p-3 cursor-pointer hover:bg-slate-800/80 transition-colors"
                onClick={() => setExpanded(isOpen ? null : i)}
              >
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-slate-500 w-32 shrink-0">{tr.ts.slice(0, 19).replace("T", " ")}</span>
                  <span className="text-slate-200 flex-1 truncate" title={tr.question}>
                    {tr.question}
                  </span>
                  <span className="text-slate-500 shrink-0">{thinkSteps} steps</span>
                  {toolSteps > 0 && <span className="text-slate-600 shrink-0">+{toolSteps} tools</span>}
                  <span className="text-slate-400 shrink-0">{lastTs.toFixed(1)}s</span>
                  {lastTokens > 0 && (
                    <span className="text-slate-500 shrink-0">{lastTokens.toLocaleString()} tok</span>
                  )}
                  {tr.usage.cost_usd != null && (
                    <span className="text-green-400 w-16 text-right shrink-0">
                      ${tr.usage.cost_usd.toFixed(4)}
                    </span>
                  )}
                  <span className="text-slate-600 text-[10px]">{isOpen ? "▲" : "▼"}</span>
                </div>
              </div>

              {isOpen && (
                <div className="border-t border-slate-700/50 p-3">
                  {/* Usage summary */}
                  {tr.usage && (
                    <div className="flex gap-4 text-[10px] text-slate-500 mb-3 pb-2 border-b border-slate-700/30">
                      {tr.usage.input_tokens != null && (
                        <span>Input: {tr.usage.input_tokens.toLocaleString()}</span>
                      )}
                      {tr.usage.output_tokens != null && (
                        <span>Output: {tr.usage.output_tokens.toLocaleString()}</span>
                      )}
                      {tr.usage.llm_calls != null && <span>LLM calls: {tr.usage.llm_calls}</span>}
                      {tr.usage.total_tokens != null && (
                        <span>Total: {tr.usage.total_tokens.toLocaleString()}</span>
                      )}
                    </div>
                  )}

                  {/* Full trace timeline */}
                  <div className="space-y-1 max-h-[600px] overflow-y-auto">
                    {tr.trace.map((step, j) => {
                      const tokenDelta =
                        j > 0 ? step.tokens_so_far - tr.trace[j - 1].tokens_so_far : step.tokens_so_far;
                      return (
                        <div
                          key={j}
                          className={`flex items-start gap-2 text-[11px] py-1 ${
                            step.event.startsWith("tool") ? "opacity-60" : ""
                          }`}
                        >
                          <span className="text-slate-600 w-12 text-right shrink-0 tabular-nums">
                            {step.ts.toFixed(1)}s
                          </span>
                          <StepBadge event={step.event} />
                          <span className="text-slate-300 flex-1 break-words">{step.message}</span>
                          {tokenDelta > 0 && (
                            <span className="text-slate-600 text-[10px] shrink-0 tabular-nums">
                              +{tokenDelta.toLocaleString()}
                            </span>
                          )}
                          <span className="text-slate-700 text-[10px] w-16 text-right shrink-0 tabular-nums">
                            {step.tokens_so_far.toLocaleString()}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- Main Usage Page ----
