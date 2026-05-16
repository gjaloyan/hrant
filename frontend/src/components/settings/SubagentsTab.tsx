import { useEffect, useMemo, useRef, useState } from "react";
import {
  SubagentSession,
  SubagentStats,
  SubagentStatus,
  fetchActiveSubagents,
  fetchSubagent,
  fetchSubagentRoles,
  fetchSubagentStats,
  fetchSubagents,
} from "../../api";

type Props = { flash: (msg: string) => void };

// Two views: live "Active" panel that polls every 3s while there's
// at least one running session, and a "History" panel paged through
// persisted sessions. Both use the same row renderer so the layout
// stays consistent.
type ViewMode = "active" | "history";

const STATUS_STYLE: Record<SubagentStatus, string> = {
  running:   "bg-amber-600/80 text-amber-50",
  completed: "bg-emerald-700 text-emerald-100",
  failed:    "bg-red-700 text-red-100",
};

const ROLE_STYLE: Record<string, string> = {
  researcher: "bg-cyan-700 text-cyan-50",
  coder:      "bg-violet-700 text-violet-50",
  reviewer:   "bg-rose-700 text-rose-50",
};

function fmtAge(ts: number): string {
  if (!ts) return "—";
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

function fmtElapsed(s: SubagentSession): string {
  if (s.elapsed_ms > 0) {
    if (s.elapsed_ms < 1000) return `${s.elapsed_ms}ms`;
    return `${(s.elapsed_ms / 1000).toFixed(1)}s`;
  }
  if (s.status === "running" && s.started_at) {
    const live = Date.now() / 1000 - s.started_at;
    return `${live.toFixed(1)}s (live)`;
  }
  return "—";
}

function fmtToolSummary(summary: Record<string, number>): string {
  const entries = Object.entries(summary || {});
  if (entries.length === 0) return "no tools";
  return entries.map(([n, c]) => `${n}(${c})`).join(", ");
}

function fmtTask(task: string, max = 80): string {
  if (!task) return "(no task)";
  if (task.length <= max) return task;
  return task.slice(0, max - 1) + "…";
}

function RolePill({ role }: { role: string }) {
  const cls = ROLE_STYLE[role] || "bg-slate-700 text-slate-200";
  return (
    <span className={`px-2 py-0.5 text-xs rounded ${cls}`}>{role}</span>
  );
}

function StatusPill({ status }: { status: SubagentStatus }) {
  const cls = STATUS_STYLE[status] || "bg-slate-700 text-slate-200";
  return (
    <span className={`px-2 py-0.5 text-xs rounded ${cls}`}>{status}</span>
  );
}

function SessionRow({
  session,
  onClick,
  isSelected,
}: {
  session: SubagentSession;
  onClick: () => void;
  isSelected: boolean;
}) {
  return (
    <div
      onClick={onClick}
      className={`px-3 py-2 border-b border-slate-700 cursor-pointer hover:bg-slate-800 ${
        isSelected ? "bg-slate-800" : ""
      }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <RolePill role={session.role} />
        <StatusPill status={session.status} />
        <span className="text-xs text-slate-400">
          {fmtAge(session.created_at)}
        </span>
        <span className="text-xs text-slate-500 ml-auto">
          {fmtElapsed(session)} · {fmtToolSummary(session.tool_summary)}
        </span>
      </div>
      <div className="text-sm text-slate-200 truncate">
        {fmtTask(session.task, 120)}
      </div>
      {session.parent_job_id && (
        <div className="text-xs text-slate-500 mt-1">
          parent job: <code className="font-mono">{session.parent_job_id}</code>
        </div>
      )}
    </div>
  );
}

function SessionDetail({ session }: { session: SubagentSession }) {
  return (
    <div className="p-4 space-y-3 overflow-y-auto h-full">
      <div className="flex items-center gap-2">
        <RolePill role={session.role} />
        <StatusPill status={session.status} />
        <span className="text-xs text-slate-400">
          {fmtAge(session.created_at)} · {fmtElapsed(session)}
        </span>
      </div>

      <div>
        <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
          Task
        </div>
        <div className="text-sm text-slate-100 whitespace-pre-wrap font-mono bg-slate-950/50 p-2 rounded">
          {session.task}
        </div>
      </div>

      {session.parent_job_id && (
        <div className="text-xs text-slate-400">
          Parent job:{" "}
          <code className="font-mono text-slate-300">{session.parent_job_id}</code>
          {session.parent_speaker && (
            <>
              {" · speaker "}
              <code className="font-mono text-slate-300">
                {session.parent_speaker}
              </code>
            </>
          )}
        </div>
      )}

      {session.status === "completed" && session.answer && (
        <div>
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
            Answer ({session.answer.length.toLocaleString()} chars)
          </div>
          <div className="text-sm text-slate-100 whitespace-pre-wrap bg-emerald-900/20 border border-emerald-800/40 p-3 rounded">
            {session.answer}
          </div>
        </div>
      )}

      {session.status === "failed" && session.error && (
        <div>
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
            Error
          </div>
          <div className="text-sm text-red-300 whitespace-pre-wrap bg-red-900/20 border border-red-800/40 p-3 rounded">
            {session.error}
          </div>
        </div>
      )}

      {session.tool_calls && session.tool_calls.length > 0 && (
        <div>
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
            Tool calls ({session.iterations})
          </div>
          <div className="space-y-2">
            {session.tool_calls.map((tc, i) => (
              <div
                key={i}
                className={`text-xs bg-slate-950/50 border ${
                  tc.is_error ? "border-red-800/50" : "border-slate-700"
                } p-2 rounded`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <code className="font-mono text-slate-200">{tc.name}</code>
                  {tc.is_error && (
                    <span className="text-red-400 text-[10px]">error</span>
                  )}
                </div>
                {tc.args && Object.keys(tc.args).length > 0 && (
                  <div className="text-slate-400 font-mono text-[11px] truncate">
                    args: {JSON.stringify(tc.args)}
                  </div>
                )}
                {tc.result_preview && (
                  <div className="text-slate-300 font-mono text-[11px] mt-1 whitespace-pre-wrap break-all">
                    {tc.result_preview}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="text-xs text-slate-500 pt-2 border-t border-slate-800">
        Session id: <code className="font-mono">{session.id}</code>
      </div>
    </div>
  );
}

export default function SubagentsTab({ flash }: Props) {
  const [view, setView] = useState<ViewMode>("active");
  const [active, setActive] = useState<SubagentSession[]>([]);
  const [history, setHistory] = useState<SubagentSession[]>([]);
  const [stats, setStats] = useState<SubagentStats | null>(null);
  const [roles, setRoles] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<SubagentSession | null>(null);
  const [statusFilter, setStatusFilter] = useState<SubagentStatus | "">("");
  const [roleFilter, setRoleFilter] = useState<string>("");
  const [loading, setLoading] = useState(false);

  // Reuse the polling pattern from JobsTab: a ref tracks "should we
  // still be polling?" so the interval doesn't tear down + rebuild
  // every render.
  const shouldPollRef = useRef(false);

  const refresh = async () => {
    setLoading(true);
    try {
      if (view === "active") {
        const r = await fetchActiveSubagents();
        setActive(r.subagents);
      } else {
        const r = await fetchSubagents({
          status: (statusFilter || undefined) as SubagentStatus | undefined,
          role: roleFilter || undefined,
          limit: 100,
        });
        setHistory(r.subagents);
      }
      const s = await fetchSubagentStats();
      setStats(s);
    } catch (e: any) {
      flash("Subagents load failed: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // Also load the roles registry once for the filter dropdown.
    fetchSubagentRoles().then((r) => setRoles(r.roles)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, statusFilter, roleFilter]);

  useEffect(() => {
    // Active panel: poll every 3s while anything is running.
    // History panel: no polling — only auto-refresh on filter
    // change (handled above by the deps array).
    shouldPollRef.current = view === "active" && active.length > 0;
  }, [view, active]);

  useEffect(() => {
    const id = setInterval(() => {
      if (shouldPollRef.current) refresh();
    }, 3000);
    return () => clearInterval(id);
    // refresh closes over `view` etc. — deliberately ignored so the
    // interval stays stable across renders; the ref above gates it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onSelect = async (s: SubagentSession) => {
    // Re-fetch on click so we get the freshest tool_calls list
    // (the list endpoint returns the same full record, but
    // re-fetching also covers the case where a running session
    // finished between list and click).
    try {
      const full = await fetchSubagent(s.id);
      setSelected(full);
    } catch (e: any) {
      flash("Could not load session: " + e.message);
      setSelected(s);
    }
  };

  const rows = view === "active" ? active : history;

  const headerStats = useMemo(() => {
    if (!stats) return null;
    return (
      <div className="flex items-center gap-3 text-xs text-slate-300">
        <span>
          <span className="text-amber-400">{stats.running}</span> running
        </span>
        <span>
          <span className="text-emerald-400">{stats.completed}</span> completed
        </span>
        <span>
          <span className="text-red-400">{stats.failed}</span> failed
        </span>
        <span className="text-slate-500">
          ({stats.total_persisted} persisted)
        </span>
      </div>
    );
  }, [stats]);

  return (
    <div className="flex flex-col h-full">
      {/* Header: title + view toggle + stats */}
      <div className="px-3 py-2 border-b border-slate-700 flex items-center gap-3">
        <h2 className="text-base font-semibold text-slate-100">Subagents</h2>
        <div className="flex bg-slate-800 rounded overflow-hidden text-sm">
          <button
            onClick={() => {
              setView("active");
              setSelected(null);
            }}
            className={`px-3 py-1 ${
              view === "active"
                ? "bg-amber-700 text-white"
                : "text-slate-300 hover:bg-slate-700"
            }`}
          >
            Active ({stats?.running ?? 0})
          </button>
          <button
            onClick={() => {
              setView("history");
              setSelected(null);
            }}
            className={`px-3 py-1 ${
              view === "history"
                ? "bg-amber-700 text-white"
                : "text-slate-300 hover:bg-slate-700"
            }`}
          >
            History
          </button>
        </div>
        <div className="ml-auto">{headerStats}</div>
        <button
          onClick={refresh}
          disabled={loading}
          className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded text-slate-200"
        >
          {loading ? "…" : "↻"}
        </button>
      </div>

      {/* History filter row */}
      {view === "history" && (
        <div className="px-3 py-2 border-b border-slate-700 flex gap-2 text-sm">
          <select
            value={statusFilter}
            onChange={(e) =>
              setStatusFilter(e.target.value as SubagentStatus | "")
            }
            className="bg-slate-800 text-slate-200 px-2 py-1 rounded"
          >
            <option value="">all statuses</option>
            <option value="completed">completed</option>
            <option value="failed">failed</option>
            <option value="running">running (orphans)</option>
          </select>
          <select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
            className="bg-slate-800 text-slate-200 px-2 py-1 rounded"
          >
            <option value="">all roles</option>
            {Object.keys(roles).map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Two-pane layout: list left, detail right */}
      <div className="flex-1 flex overflow-hidden">
        <div className="w-1/2 overflow-y-auto border-r border-slate-700">
          {rows.length === 0 ? (
            <div className="p-6 text-center text-slate-500 text-sm">
              {view === "active"
                ? "No subagents running right now."
                : "No sessions match these filters."}
              {view === "active" && (
                <div className="mt-2 text-xs">
                  Subagents show up here while the main agent is delegating
                  to <code>researcher</code> / <code>coder</code> / <code>
                  reviewer</code>. Active state is process-local — restarting
                  the agent clears this list.
                </div>
              )}
            </div>
          ) : (
            rows.map((s) => (
              <SessionRow
                key={s.id}
                session={s}
                isSelected={selected?.id === s.id}
                onClick={() => onSelect(s)}
              />
            ))
          )}
        </div>
        <div className="w-1/2 overflow-hidden">
          {selected ? (
            <SessionDetail session={selected} />
          ) : (
            <div className="p-6 text-center text-slate-500 text-sm">
              Pick a session on the left to see its task, answer, and tool
              calls.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
