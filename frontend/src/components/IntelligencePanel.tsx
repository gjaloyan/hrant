import { useCallback, useEffect, useState } from "react";
import {
  fetchEvalStats,
  fetchMetaLearner,
  fetchFailures,
  extractPatterns,
  fetchAnalogies,
  fetchSelfModifier,
  fetchProposals,
  analyzeModule,
  approveProposal,
  rejectProposal,
  applyProposal,
  deleteProposal,
  fetchMemoryStats,
  fetchMemoryFacts,
  recallMemory,
  fetchUsageStats,
  fetchUsageCalls,
  type EvalStats,
  type MetaLearnerStats,
  type FailureEntry,
  type AnalogyPattern,
  type SelfModifierStats,
  type ModProposal,
  type MemoryStats,
  type MemoryFact,
  type RecalledFact,
  type UsageStats,
  type UsageCallRecord,
} from "../api";

// ---- Mini bar chart ----
function MiniBar({ value, max, color = "bg-sky-500" }: { value: number; max: number; color?: string }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="h-4 bg-slate-700 rounded overflow-hidden flex-1">
      <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function ConfBadge({ value }: { value: number }) {
  const color = value >= 85 ? "text-green-400" : value >= 60 ? "text-yellow-400" : "text-red-400";
  return <span className={`font-mono font-bold ${color}`}>{value}%</span>;
}

// ---- Evaluator Sub-Panel ----
function EvalPanel({ stats }: { stats: EvalStats | null }) {
  if (!stats) return <div className="text-slate-500 p-4">Loading evaluator...</div>;

  const today = stats.today;
  const trend = stats.weekly_trend || [];
  const maxInteractions = Math.max(1, ...trend.map((d) => d.total_interactions));

  return (
    <div className="space-y-4">
      {/* Overview cards */}
      <div className="grid grid-cols-4 gap-3">
        <Card label="Total Logged" value={stats.total_logged} />
        <Card label="Tasks" value={stats.total_tasks} />
        <Card label="Chats" value={stats.total_chats} />
        <Card label="Avg Confidence" value={<ConfBadge value={stats.overall_avg_confidence} />} />
      </div>

      {/* Today */}
      <div className="bg-slate-800/50 rounded p-3">
        <h4 className="text-sm font-semibold text-slate-300 mb-2">Today ({today.date})</h4>
        <div className="grid grid-cols-5 gap-2 text-xs">
          <Stat label="Interactions" val={today.total_interactions} />
          <Stat label="Tasks" val={today.tasks} />
          <Stat label="Chats" val={today.chats} />
          <Stat label="Contradictions" val={today.total_contradictions} warn />
          <Stat label="Unverified" val={today.total_unverified} warn />
        </div>
      </div>

      {/* Weekly trend */}
      <div className="bg-slate-800/50 rounded p-3">
        <h4 className="text-sm font-semibold text-slate-300 mb-2">Weekly Trend</h4>
        <div className="space-y-1">
          {trend.map((d) => (
            <div key={d.date} className="flex items-center gap-2 text-xs">
              <span className="w-20 text-slate-400">{d.date.slice(5)}</span>
              <MiniBar value={d.total_interactions} max={maxInteractions} />
              <span className="w-8 text-right">{d.total_interactions}</span>
              <ConfBadge value={d.avg_confidence} />
            </div>
          ))}
        </div>
      </div>

      {/* Regressions */}
      {stats.regressions.length > 0 && (
        <div className="bg-red-900/30 border border-red-800 rounded p-3">
          <h4 className="text-sm font-semibold text-red-400 mb-2">⚠ Regressions Detected</h4>
          {stats.regressions.map((r, i) => (
            <div key={i} className="text-xs text-red-300 mb-1">
              <strong>{r.domain}</strong>: {r.last_week_avg}% → {r.this_week_avg}%
              (drop: {r.drop}%, samples: {r.sample_size})
            </div>
          ))}
        </div>
      )}

      {/* Suggestions */}
      {stats.suggestions.length > 0 && (
        <div className="bg-slate-800/50 rounded p-3">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">Improvement Suggestions</h4>
          {stats.suggestions.map((s, i) => (
            <div key={i} className="flex items-start gap-2 text-xs mb-1">
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                s.priority >= 8 ? "bg-red-700" : s.priority >= 6 ? "bg-yellow-700" : "bg-slate-600"
              }`}>
                P{s.priority}
              </span>
              <span className="text-slate-300">{s.suggestion}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Meta-Learner Sub-Panel ----
function MetaPanel({
  stats,
  failures,
  onExtract,
}: {
  stats: MetaLearnerStats | null;
  failures: FailureEntry[];
  onExtract: () => void;
}) {
  const [showFailures, setShowFailures] = useState(false);

  if (!stats) return <div className="text-slate-500 p-4">Loading meta-learner...</div>;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Card label="Total Failures" value={stats.total_failures} />
        <Card label="Avg Severity" value={stats.avg_severity} />
        <Card label="Patterns Found" value={stats.patterns_count} />
      </div>

      {/* Root causes */}
      {Object.keys(stats.by_root_cause).length > 0 && (
        <div className="bg-slate-800/50 rounded p-3">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">By Root Cause</h4>
          {Object.entries(stats.by_root_cause).map(([cause, count]) => (
            <div key={cause} className="flex items-center gap-2 text-xs mb-1">
              <span className="w-32 text-slate-400">{cause}</span>
              <MiniBar value={count} max={stats.total_failures} color="bg-orange-500" />
              <span className="w-6 text-right">{count}</span>
            </div>
          ))}
        </div>
      )}

      {/* Error patterns */}
      {stats.patterns.length > 0 && (
        <div className="bg-slate-800/50 rounded p-3">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">Error Patterns</h4>
          {stats.patterns.map((p, i) => (
            <div key={i} className="text-xs mb-2 p-2 bg-slate-700/50 rounded">
              <div className="text-slate-200 font-medium">{p.pattern}</div>
              <div className="text-slate-400 mt-1">
                freq: {p.frequency}x | domains: {p.domains.join(", ")} | fix: {p.suggested_fix}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <button onClick={onExtract} className="px-3 py-1 bg-orange-700 hover:bg-orange-600 rounded text-xs">
          Extract Patterns
        </button>
        <button
          onClick={() => setShowFailures(!showFailures)}
          className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs"
        >
          {showFailures ? "Hide" : "Show"} Recent Failures ({failures.length})
        </button>
      </div>

      {showFailures && failures.length > 0 && (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {failures.map((f, i) => (
            <div key={i} className="bg-slate-800/50 rounded p-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">{f.ts}</span>
                <ConfBadge value={f.confidence} />
              </div>
              <div className="text-slate-200 mt-1 truncate">{f.question}</div>
              {f.analysis && (
                <div className="text-slate-400 mt-1">
                  root: {f.analysis.root_cause} | domain: {f.analysis.domain} |
                  fix: {f.analysis.fix_action}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Analogy Sub-Panel ----
function AnalogyPanel({ patterns }: { patterns: AnalogyPattern[] }) {
  if (patterns.length === 0)
    return <div className="text-slate-500 p-4">No patterns extracted yet. Patterns are auto-extracted from high-confidence answers.</div>;

  return (
    <div className="space-y-2">
      <div className="text-xs text-slate-400 mb-2">{patterns.length} patterns across domains</div>
      {patterns.map((p) => (
        <div key={p.pattern_id} className="bg-slate-800/50 rounded p-3">
          <div className="text-sm text-slate-200 font-medium">{p.pattern}</div>
          <div className="text-xs text-slate-400 mt-1">
            <span className="text-sky-400">{p.domain}</span> | {p.abstract_form}
          </div>
          {p.mechanism && (
            <div className="text-xs text-slate-500 mt-1">Mechanism: {p.mechanism}</div>
          )}
          {p.applicable_when && (
            <div className="text-xs text-slate-500">When: {p.applicable_when}</div>
          )}
          {p.examples.length > 0 && (
            <div className="text-xs text-slate-500 mt-1">
              Examples: {p.examples.map((e) => e.question).join("; ")}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ---- Self-Modifier Sub-Panel ----
function ModifierPanel({
  stats,
  proposals,
  onAnalyze,
  onApprove,
  onReject,
  onApply,
  onDelete,
  onRefresh,
}: {
  stats: SelfModifierStats | null;
  proposals: ModProposal[];
  onAnalyze: (module: string) => void;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
  onApply: (id: string) => void;
  onDelete: (id: string) => void;
  onRefresh: () => void;
}) {
  const [selectedModule, setSelectedModule] = useState("");
  const [analyzing, setAnalyzing] = useState(false);

  if (!stats) return <div className="text-slate-500 p-4">Loading self-modifier...</div>;

  const handleAnalyze = async () => {
    if (!selectedModule) return;
    setAnalyzing(true);
    await onAnalyze(selectedModule);
    setAnalyzing(false);
    onRefresh();
  };

  const statusColor: Record<string, string> = {
    pending: "bg-yellow-700",
    approved: "bg-green-700",
    rejected: "bg-red-700",
    applied: "bg-sky-700",
    failed: "bg-red-900",
  };

  const riskColor: Record<string, string> = {
    low: "text-green-400",
    medium: "text-yellow-400",
    high: "text-red-400",
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Card label="Total Proposals" value={stats.total} />
        <Card label="Pending" value={stats.by_status?.pending || 0} />
        <Card label="Applied" value={stats.by_status?.applied || 0} />
      </div>

      {/* Analyze module */}
      <div className="flex gap-2 items-center">
        <select
          value={selectedModule}
          onChange={(e) => setSelectedModule(e.target.value)}
          className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-sm flex-1"
        >
          <option value="">Select module...</option>
          {(stats.modules || []).map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <button
          onClick={handleAnalyze}
          disabled={!selectedModule || analyzing}
          className="px-3 py-1 bg-purple-700 hover:bg-purple-600 disabled:bg-slate-700 rounded text-xs"
        >
          {analyzing ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {/* Proposals */}
      {proposals.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-300">Proposals</h4>
          {proposals.map((p) => (
            <div key={p.id} className="bg-slate-800/50 rounded p-3">
              <div className="flex items-center gap-2 mb-1">
                <span className={`px-1.5 py-0.5 rounded text-[10px] ${statusColor[p.status] || "bg-slate-600"}`}>
                  {p.status}
                </span>
                <span className={`text-[10px] ${riskColor[p.risk] || ""}`}>risk: {p.risk}</span>
                <span className="text-[10px] text-slate-500">{p.impact}</span>
                <span className="text-[10px] text-slate-500">{p.module}</span>
              </div>
              <div className="text-sm text-slate-200 font-medium">{p.title}</div>
              <div className="text-xs text-slate-400 mt-1">{p.description}</div>
              {p.reasoning && (
                <div className="text-xs text-slate-500 mt-1 italic">{p.reasoning}</div>
              )}
              {(p.old_code || p.new_code) && (
                <details className="mt-2">
                  <summary className="text-xs text-sky-400 cursor-pointer">Show diff</summary>
                  <div className="mt-1 text-xs font-mono">
                    {p.old_code && (
                      <pre className="bg-red-900/30 p-2 rounded overflow-x-auto mb-1 whitespace-pre-wrap">
                        - {p.old_code}
                      </pre>
                    )}
                    {p.new_code && (
                      <pre className="bg-green-900/30 p-2 rounded overflow-x-auto whitespace-pre-wrap">
                        + {p.new_code}
                      </pre>
                    )}
                  </div>
                </details>
              )}
              <div className="flex gap-1 mt-2">
                {p.status === "pending" && (
                  <>
                    <button onClick={() => onApprove(p.id)} className="px-2 py-0.5 bg-green-700 hover:bg-green-600 rounded text-[10px]">
                      Approve
                    </button>
                    <button onClick={() => onReject(p.id)} className="px-2 py-0.5 bg-red-700 hover:bg-red-600 rounded text-[10px]">
                      Reject
                    </button>
                  </>
                )}
                {p.status === "approved" && (
                  <button onClick={() => onApply(p.id)} className="px-2 py-0.5 bg-sky-700 hover:bg-sky-600 rounded text-[10px]">
                    Apply
                  </button>
                )}
                <button onClick={() => onDelete(p.id)} className="px-2 py-0.5 bg-slate-700 hover:bg-slate-600 rounded text-[10px]">
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Usage Sub-Panel ----
function UsagePanel({
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
                  <span className="w-40 text-slate-400 truncate" title={task}>{task}</span>
                  <span className="w-12 text-right">{d.calls}x</span>
                  <MiniBar value={d.input + d.output} max={stats.total_input_tokens + stats.total_output_tokens} color="bg-sky-500" />
                  <span className="w-20 text-right text-slate-300">{(d.input + d.output).toLocaleString()}</span>
                  <span className="w-16 text-right text-green-400">${d.cost.toFixed(4)}</span>
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
              <span className="w-20 text-right">{d.input.toLocaleString()} in</span>
              <span className="w-20 text-right">{d.output.toLocaleString()} out</span>
              <span className="w-16 text-right text-green-400">${d.cost.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Recent calls */}
      <div className="bg-slate-800/50 rounded p-3">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-semibold text-slate-300">Recent API Calls</h4>
          <button onClick={onRefresh} className="text-xs text-slate-400 hover:text-slate-200">Refresh</button>
        </div>
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
        </div>
      </div>
    </div>
  );
}

// ---- Memory Sub-Panel ----
function MemoryPanel({
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

  const handleRecall = async () => {
    if (!recallQuery.trim()) return;
    try {
      const data = await recallMemory(recallQuery.trim());
      setRecalled(data.facts || []);
      setRecallBlock(data.block || "");
    } catch {}
  };

  if (!stats) return <div className="text-slate-500 p-4">Loading memory...</div>;

  const categoryColors: Record<string, string> = {
    price: "bg-green-700",
    personal: "bg-purple-700",
    technical: "bg-sky-700",
    event: "bg-yellow-700",
    location: "bg-orange-700",
    preference: "bg-pink-700",
    relationship: "bg-indigo-700",
    rule: "bg-red-700",
    general: "bg-slate-600",
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-3">
        <Card label="Facts Logged" value={stats.total_facts_logged} />
        <Card label="Memory Edges" value={stats.memory_edges_in_graph} />
        <Card label="Memory Entities" value={stats.memory_entities} />
        <Card label="Graph Total" value={`${stats.graph_total_entities} ent / ${stats.graph_total_edges} edges`} />
      </div>

      {/* Recall search */}
      <div className="bg-slate-800/50 rounded p-3">
        <h4 className="text-sm font-semibold text-slate-300 mb-2">Recall Memory</h4>
        <div className="flex gap-2">
          <input
            value={recallQuery}
            onChange={(e) => setRecallQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleRecall()}
            placeholder="Search memory... (e.g. 'Armenian prices', 'project deadline')"
            className="flex-1 bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-sm"
          />
          <button onClick={handleRecall} className="px-3 py-1.5 bg-sky-700 hover:bg-sky-600 rounded text-xs">
            Recall
          </button>
        </div>
        {recalled.length > 0 && (
          <div className="mt-2 space-y-1">
            {recalled.map((f, i) => (
              <div key={i} className="text-xs flex gap-2 items-center p-1.5 bg-slate-700/50 rounded">
                <span className="text-sky-400 font-medium">{f.entity}</span>
                <span className="text-slate-500">{f.relation.replace(/_/g, " ")}</span>
                <span className="text-green-400 font-medium">{f.target}</span>
                <span className="text-slate-600 ml-auto">{f.weight.toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}
        {recallBlock && (
          <details className="mt-2">
            <summary className="text-xs text-slate-400 cursor-pointer">Raw context block</summary>
            <pre className="mt-1 text-xs bg-slate-900 p-2 rounded overflow-x-auto whitespace-pre-wrap text-slate-300">
              {recallBlock}
            </pre>
          </details>
        )}
      </div>

      {/* Recent facts */}
      <div className="bg-slate-800/50 rounded p-3">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-semibold text-slate-300">Recent Facts</h4>
          <button onClick={onRefresh} className="text-xs text-slate-400 hover:text-slate-200">Refresh</button>
        </div>
        {facts.length === 0 ? (
          <div className="text-xs text-slate-500">
            No facts extracted yet. Facts are automatically extracted from conversations.
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {facts.map((f, i) => (
              <div key={i} className="p-2 bg-slate-700/50 rounded">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] ${categoryColors[f.category] || "bg-slate-600"}`}>
                    {f.category}
                  </span>
                  <span className="text-[10px] text-slate-500">{f.ts}</span>
                  <span className="text-[10px] text-slate-500 ml-auto">conf: {(f.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="text-xs text-slate-200">{f.summary}</div>
                <div className="mt-1 space-y-0.5">
                  {f.triples.map((t, j) => (
                    <div key={j} className="text-[11px] flex gap-1.5">
                      <span className="text-sky-400">{t[0]}</span>
                      <span className="text-slate-500">{t[1]}</span>
                      <span className="text-green-400">{t[2]}</span>
                    </div>
                  ))}
                </div>
                {f.tags.length > 0 && (
                  <div className="mt-1 flex gap-1 flex-wrap">
                    {f.tags.map((tag) => (
                      <span key={tag} className="text-[10px] px-1 py-0.5 bg-slate-600 rounded">{tag}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- Helpers ----
function Card({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-slate-800/50 rounded p-3 text-center">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-lg font-bold text-slate-100 mt-1">{value}</div>
    </div>
  );
}

function Stat({ label, val, warn }: { label: string; val: number; warn?: boolean }) {
  return (
    <div className="text-center">
      <div className="text-slate-400">{label}</div>
      <div className={`font-bold ${warn && val > 0 ? "text-red-400" : "text-slate-100"}`}>{val}</div>
    </div>
  );
}

// ---- Main Panel ----
type SubTab = "usage" | "memory" | "evaluator" | "meta" | "analogies" | "modifier";

export default function IntelligencePanel() {
  const [subTab, setSubTab] = useState<SubTab>("usage");
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null);
  const [usageCalls, setUsageCalls] = useState<UsageCallRecord[]>([]);
  const [memStats, setMemStats] = useState<MemoryStats | null>(null);
  const [memFacts, setMemFacts] = useState<MemoryFact[]>([]);
  const [evalStats, setEvalStats] = useState<EvalStats | null>(null);
  const [metaStats, setMetaStats] = useState<MetaLearnerStats | null>(null);
  const [failures, setFailures] = useState<FailureEntry[]>([]);
  const [analogies, setAnalogies] = useState<AnalogyPattern[]>([]);
  const [modStats, setModStats] = useState<SelfModifierStats | null>(null);
  const [proposals, setProposals] = useState<ModProposal[]>([]);

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

  const loadEval = useCallback(async () => {
    try { setEvalStats(await fetchEvalStats()); } catch {}
  }, []);

  const loadMeta = useCallback(async () => {
    try {
      setMetaStats(await fetchMetaLearner());
      const f = await fetchFailures();
      setFailures(f.failures || []);
    } catch {}
  }, []);

  const loadAnalogies = useCallback(async () => {
    try {
      const data = await fetchAnalogies();
      setAnalogies(data.patterns || []);
    } catch {}
  }, []);

  const loadModifier = useCallback(async () => {
    try {
      setModStats(await fetchSelfModifier());
      const p = await fetchProposals();
      setProposals(p.proposals || []);
    } catch {}
  }, []);

  useEffect(() => {
    loadUsage();
    loadMemory();
    loadEval();
    loadMeta();
    loadAnalogies();
    loadModifier();
  }, [loadUsage, loadMemory, loadEval, loadMeta, loadAnalogies, loadModifier]);

  const handleExtract = async () => {
    try { await extractPatterns(); await loadMeta(); } catch {}
  };

  const handleAnalyze = async (module: string) => {
    try { await analyzeModule(module); } catch {}
  };

  const handleApprove = async (id: string) => {
    try { await approveProposal(id); await loadModifier(); } catch {}
  };

  const handleReject = async (id: string) => {
    try { await rejectProposal(id); await loadModifier(); } catch {}
  };

  const handleApply = async (id: string) => {
    try { await applyProposal(id); await loadModifier(); } catch {}
  };

  const handleDelete = async (id: string) => {
    try { await deleteProposal(id); await loadModifier(); } catch {}
  };

  const SUB_TABS: { id: SubTab; label: string }[] = [
    { id: "usage", label: "Usage" },
    { id: "memory", label: "Memory" },
    { id: "evaluator", label: "Evaluator" },
    { id: "meta", label: "Meta-Learner" },
    { id: "analogies", label: "Analogies" },
    { id: "modifier", label: "Self-Modifier" },
  ];

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
      {/* Sub-tab bar */}
      <div className="px-4 py-2 border-b border-slate-700 flex gap-1">
        {SUB_TABS.map((t) => (
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

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {subTab === "usage" && <UsagePanel stats={usageStats} calls={usageCalls} onRefresh={loadUsage} />}
        {subTab === "memory" && <MemoryPanel stats={memStats} facts={memFacts} onRefresh={loadMemory} />}
        {subTab === "evaluator" && <EvalPanel stats={evalStats} />}
        {subTab === "meta" && <MetaPanel stats={metaStats} failures={failures} onExtract={handleExtract} />}
        {subTab === "analogies" && <AnalogyPanel patterns={analogies} />}
        {subTab === "modifier" && (
          <ModifierPanel
            stats={modStats}
            proposals={proposals}
            onAnalyze={handleAnalyze}
            onApprove={handleApprove}
            onReject={handleReject}
            onApply={handleApply}
            onDelete={handleDelete}
            onRefresh={loadModifier}
          />
        )}
      </div>
    </div>
  );
}
