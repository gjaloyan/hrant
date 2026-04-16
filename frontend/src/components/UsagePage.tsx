import { useCallback, useEffect, useState } from "react";
import {
  fetchUsageStats,
  fetchUsageCalls,
  fetchUsageTraces,
  fetchMemoryStats,
  fetchMemoryFacts,
  recallMemory,
  type UsageStats,
  type UsageCallRecord,
  type RequestTrace,
  type MemoryStats,
  type MemoryFact,
  type RecalledFact,
} from "../api";

function Card({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-800/60 rounded p-3">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
      <div className="text-lg font-bold text-slate-100 mt-0.5">{value}</div>
    </div>
  );
}

function MiniBar({ value, max, color = "bg-sky-500" }: { value: number; max: number; color?: string }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="h-4 bg-slate-700 rounded overflow-hidden flex-1">
      <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
    </div>
  );
}

// ---- Usage Section ----
function UsageSection({
  stats,
  calls,
  onRefresh,
}: {
  stats: UsageStats | null;
  calls: UsageCallRecord[];
  onRefresh: () => void;
}) {
  const [expandedCall, setExpandedCall] = useState<number | null>(null);

  if (!stats) return <div className="text-slate-500 p-4">Loading usage...</div>;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-200">Token Usage</h3>
        <button onClick={onRefresh} className="text-xs bg-slate-700 hover:bg-slate-600 rounded px-2 py-1">
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <Card label="Total Calls" value={stats.total_calls} />
        <Card label="Input Tokens" value={stats.total_input_tokens.toLocaleString()} />
        <Card label="Output Tokens" value={stats.total_output_tokens.toLocaleString()} />
        <Card label="Total Cost" value={`$${stats.total_cost_usd.toFixed(4)}`} />
      </div>

      {/* By task type */}
      {Object.keys(stats.by_task_type).length > 0 && (
        <div className="bg-slate-800/50 rounded p-3">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">By Task Type</h4>
          <div className="space-y-1">
            {Object.entries(stats.by_task_type)
              .sort((a, b) => b[1].cost - a[1].cost)
              .map(([task, d]) => (
                <div key={task} className="flex items-center gap-2 text-xs">
                  <span className="w-44 text-slate-400 truncate" title={task}>{task}</span>
                  <span className="w-12 text-right">{d.calls}x</span>
                  <MiniBar value={d.input + d.output} max={stats.total_input_tokens + stats.total_output_tokens} color="bg-sky-500" />
                  <span className="w-24 text-right text-slate-300">{(d.input + d.output).toLocaleString()}</span>
                  <span className="w-20 text-right text-green-400">${d.cost.toFixed(4)}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* By model */}
      {Object.keys(stats.by_model).length > 0 && (
        <div className="bg-slate-800/50 rounded p-3">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">By Model</h4>
          {Object.entries(stats.by_model).map(([model, d]) => (
            <div key={model} className="flex items-center gap-2 text-xs mb-1">
              <span className="w-48 text-sky-400 truncate" title={model}>{model}</span>
              <span className="w-12 text-right">{d.calls}x</span>
              <span className="w-24 text-right">{d.input.toLocaleString()} in</span>
              <span className="w-24 text-right">{d.output.toLocaleString()} out</span>
              <span className="w-20 text-right text-green-400">${d.cost.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Recent calls */}
      <div className="bg-slate-800/50 rounded p-3">
        <h4 className="text-sm font-semibold text-slate-300 mb-2">Recent API Calls</h4>
        <div className="space-y-1 max-h-[500px] overflow-y-auto">
          {calls.map((c, i) => (
            <div
              key={i}
              className="p-2 bg-slate-700/50 rounded cursor-pointer hover:bg-slate-700/80"
              onClick={() => setExpandedCall(expandedCall === i ? null : i)}
            >
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-500 w-16">{c.ts.slice(11, 19)}</span>
                <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-600">{c.task_type}</span>
                <span className="text-sky-400 text-[10px]">{c.model}</span>
                <span className="ml-auto text-slate-300">
                  {c.input_tokens.toLocaleString()} → {c.output_tokens.toLocaleString()}
                </span>
                <span className="text-green-400 w-16 text-right">${c.cost_usd.toFixed(4)}</span>
                <span className="text-slate-500 w-14 text-right">{c.duration_ms}ms</span>
              </div>
              {expandedCall === i && (
                <div className="mt-2 space-y-1">
                  <div className="grid grid-cols-4 gap-2 text-[10px]">
                    <div><span className="text-slate-500">Input:</span> {c.input_tokens.toLocaleString()}</div>
                    <div><span className="text-slate-500">Output:</span> {c.output_tokens.toLocaleString()}</div>
                    <div><span className="text-slate-500">Cache read:</span> {c.cache_read_tokens.toLocaleString()}</div>
                    <div><span className="text-slate-500">Cache create:</span> {c.cache_creation_tokens.toLocaleString()}</div>
                  </div>
                  {c.prompt_preview && (
                    <div className="mt-1">
                      <div className="text-[10px] text-slate-500 mb-0.5">Request body preview:</div>
                      <pre className="text-[10px] bg-slate-900 p-2 rounded overflow-x-auto whitespace-pre-wrap text-slate-300 max-h-40 overflow-y-auto">
                        {c.prompt_preview}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {calls.length === 0 && <div className="text-xs text-slate-500 py-2">No API calls yet</div>}
        </div>
      </div>
    </div>
  );
}

// ---- Memory Section ----
function MemorySection({
  stats,
  facts,
  onRefresh,
}: {
  stats: MemoryStats | null;
  facts: MemoryFact[];
  onRefresh: () => void;
}) {
  const [recallQuery, setRecallQuery] = useState("");
  const [recalled, setRecalled] = useState<RecalledFact[]>([]);
  const [recallBlock, setRecallBlock] = useState("");

  const doRecall = async () => {
    if (!recallQuery.trim()) return;
    try {
      const r = await recallMemory(recallQuery.trim());
      setRecalled(r.facts || []);
      setRecallBlock(r.block || "");
    } catch {}
  };

  if (!stats) return <div className="text-slate-500 p-4">Loading memory...</div>;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-200">Agent Memory</h3>
        <button onClick={onRefresh} className="text-xs bg-slate-700 hover:bg-slate-600 rounded px-2 py-1">
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <Card label="Facts Logged" value={stats.total_facts_logged} />
        <Card label="Memory Edges" value={stats.memory_edges_in_graph} />
        <Card label="Memory Entities" value={stats.memory_entities} />
        <Card label="Graph Total" value={`${stats.graph_total_entities} / ${stats.graph_total_edges}`} />
      </div>

      {/* Recall search */}
      <div className="bg-slate-800/50 rounded p-3">
        <h4 className="text-sm font-semibold text-slate-300 mb-2">Recall Memory</h4>
        <div className="flex gap-2 mb-2">
          <input
            className="flex-1 bg-slate-900 rounded px-2 py-1 text-xs outline-none focus:ring-1 focus:ring-sky-600"
            placeholder="Search memory (e.g. 'Armenia prices')..."
            value={recallQuery}
            onChange={(e) => setRecallQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doRecall()}
          />
          <button onClick={doRecall} className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs">
            Recall
          </button>
        </div>
        {recalled.length > 0 && (
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {recalled.map((f, i) => (
              <div key={i} className="text-xs bg-slate-700/50 rounded p-1.5 flex gap-2">
                <span className="text-emerald-400">{f.entity}</span>
                <span className="text-slate-500">{f.relation}</span>
                <span className="text-sky-300">{f.target}</span>
              </div>
            ))}
          </div>
        )}
        {recallBlock && (
          <pre className="mt-2 text-[10px] bg-slate-900 p-2 rounded text-slate-400 whitespace-pre-wrap max-h-32 overflow-y-auto">
            {recallBlock}
          </pre>
        )}
      </div>

      {/* Recent facts */}
      <div className="bg-slate-800/50 rounded p-3">
        <h4 className="text-sm font-semibold text-slate-300 mb-2">Recent Facts</h4>
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {facts.map((f, i) => (
            <div key={i} className="bg-slate-700/50 rounded p-2 text-xs">
              <div className="flex items-center gap-2 mb-1">
                <span className="px-1.5 py-0.5 rounded text-[10px] bg-violet-900/60 text-violet-300">
                  {f.category}
                </span>
                <span className="text-slate-500">{f.ts.slice(0, 16)}</span>
                <span className="text-slate-500 ml-auto">conf: {(f.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="text-slate-200 mb-1">{f.summary}</div>
              {f.triples.length > 0 && (
                <div className="space-y-0.5">
                  {f.triples.map((t, j) => (
                    <div key={j} className="text-[10px] flex gap-1">
                      <span className="text-emerald-400">{t[0]}</span>
                      <span className="text-slate-500">{t[1]}</span>
                      <span className="text-sky-300">{t[2]}</span>
                    </div>
                  ))}
                </div>
              )}
              {f.tags.length > 0 && (
                <div className="flex gap-1 mt-1 flex-wrap">
                  {f.tags.map((tag, j) => (
                    <span key={j} className="text-[10px] bg-slate-600/50 rounded px-1 py-0.5 text-slate-400">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {facts.length === 0 && <div className="text-xs text-slate-500 py-2">No facts extracted yet</div>}
        </div>
      </div>
    </div>
  );
}

// ---- Thinking Traces Section ----
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

function TracesSection({
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
        <h3 className="text-sm font-bold text-slate-200">Thinking Traces</h3>
        <button onClick={onRefresh} className="text-xs bg-slate-700 hover:bg-slate-600 rounded px-2 py-1">
          Refresh
        </button>
      </div>

      {traces.length === 0 && (
        <div className="text-xs text-slate-500 py-4 text-center">No thinking traces recorded yet</div>
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
export default function UsagePage() {
  const [subTab, setSubTab] = useState<"usage" | "memory" | "thinking">("usage");
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null);
  const [usageCalls, setUsageCalls] = useState<UsageCallRecord[]>([]);
  const [memStats, setMemStats] = useState<MemoryStats | null>(null);
  const [memFacts, setMemFacts] = useState<MemoryFact[]>([]);
  const [traces, setTraces] = useState<RequestTrace[]>([]);

  const loadUsage = useCallback(async () => {
    try {
      setUsageStats(await fetchUsageStats());
      const c = await fetchUsageCalls(100);
      setUsageCalls(c.calls || []);
    } catch {}
  }, []);

  const loadMemory = useCallback(async () => {
    try {
      setMemStats(await fetchMemoryStats());
      const f = await fetchMemoryFacts(50);
      setMemFacts(f.facts || []);
    } catch {}
  }, []);

  const loadTraces = useCallback(async () => {
    try {
      const r = await fetchUsageTraces(30);
      setTraces(r.traces || []);
    } catch {}
  }, []);

  useEffect(() => {
    loadUsage();
    loadMemory();
    loadTraces();
  }, [loadUsage, loadMemory, loadTraces]);

  const TABS = [
    { id: "usage" as const, label: "Token Usage" },
    { id: "thinking" as const, label: "Thinking" },
    { id: "memory" as const, label: "Memory" },
  ];

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
      <div className="px-4 py-2 border-b border-slate-700 flex gap-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setSubTab(t.id)}
            className={`px-3 py-1 rounded text-xs transition-colors ${
              subTab === t.id
                ? "bg-sky-700 text-white"
                : "bg-slate-800 hover:bg-slate-700 text-slate-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {subTab === "usage" && <UsageSection stats={usageStats} calls={usageCalls} onRefresh={loadUsage} />}
        {subTab === "thinking" && <TracesSection traces={traces} onRefresh={loadTraces} />}
        {subTab === "memory" && <MemorySection stats={memStats} facts={memFacts} onRefresh={loadMemory} />}
      </div>
    </div>
  );
}
