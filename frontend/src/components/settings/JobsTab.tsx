import { useEffect, useMemo, useState } from "react";
import {
  Job,
  JobStatus,
  cancelJob,
  deleteJob,
  fetchJob,
  fetchJobStats,
  fetchJobs,
  retryJob,
} from "../../api";

type Props = { flash: (msg: string) => void };

// Status → Tailwind class for the badge pill. Matches the warm
// palette used by `hrant config` in the CLI (orange accents, green
// for completed, red for failed, amber for interrupted).
const STATUS_STYLE: Record<JobStatus, string> = {
  queued:      "bg-slate-700 text-slate-200",
  running:     "bg-amber-600/80 text-amber-50",
  completed:   "bg-emerald-700 text-emerald-100",
  failed:      "bg-red-700 text-red-100",
  interrupted: "bg-yellow-600 text-yellow-50",
  cancelled:   "bg-slate-600 text-slate-300",
};

const STATUS_LABEL: Record<JobStatus, string> = {
  queued:      "queued",
  running:     "running",
  completed:   "completed",
  failed:      "failed",
  interrupted: "interrupted",
  cancelled:   "cancelled",
};

function fmtAge(ts: number): string {
  // Compact "Nm ago" / "Nh ago" / "Nd ago" — easier to scan than
  // a full datetime when the user has dozens of jobs.
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

function fmtElapsed(job: Job): string {
  if (job.started_at && job.completed_at) {
    const s = job.completed_at - job.started_at;
    if (s < 1) return `${Math.round(s * 1000)}ms`;
    return `${s.toFixed(1)}s`;
  }
  return "—";
}

export default function JobsTab({ flash }: Props) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [statusFilter, setStatusFilter] = useState<JobStatus | "">("");
  const [stats, setStats] = useState<Record<JobStatus, number> | null>(null);
  const [selected, setSelected] = useState<Job | null>(null);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const refresh = async () => {
    setLoading(true);
    try {
      const r = await fetchJobs({
        status: (statusFilter || undefined) as JobStatus | undefined,
        limit: 100,
      });
      setJobs(r.jobs);
      setTotal(r.total);
      const s = await fetchJobStats();
      setStats(s);
    } catch (e: any) {
      flash("Load failed: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Initial load + on filter change.
  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter]);

  // Auto-refresh every 5s while there are running jobs — keeps the
  // tab feeling live without polling forever once everything's idle.
  useEffect(() => {
    if (!autoRefresh) return;
    const hasActive = jobs.some((j) => j.status === "running" || j.status === "queued");
    if (!hasActive && jobs.length > 0) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, jobs]);

  const handleRetry = async (id: string) => {
    try {
      const r = await retryJob(id);
      flash(`Retry queued (new id: ${r.new_job_id})`);
      refresh();
    } catch (e: any) {
      flash("Retry failed: " + e.message);
    }
  };

  const handleCancel = async (id: string) => {
    if (!confirm("Cancel this job?")) return;
    try {
      await cancelJob(id);
      flash("Cancelled.");
      refresh();
    } catch (e: any) {
      flash("Cancel failed: " + e.message);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this job record permanently?")) return;
    try {
      await deleteJob(id);
      flash("Deleted.");
      if (selected?.id === id) setSelected(null);
      refresh();
    } catch (e: any) {
      flash("Delete failed: " + e.message);
    }
  };

  const handleOpen = async (id: string) => {
    try {
      const full = await fetchJob(id);
      setSelected(full);
    } catch (e: any) {
      flash("Open failed: " + e.message);
    }
  };

  // Quick filter chips: status → count, click to filter the list.
  const filterChips = useMemo(() => {
    const order: JobStatus[] = [
      "running", "interrupted", "failed", "queued", "completed", "cancelled",
    ];
    return order.map((s) => ({
      status: s,
      count: stats?.[s] ?? 0,
      active: statusFilter === s,
    }));
  }, [stats, statusFilter]);

  return (
    <div className="flex gap-4 h-full min-h-0">
      {/* Left: list + filters */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Header + filter chips */}
        <div className="mb-3">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold text-slate-100">
              Jobs <span className="text-slate-500 text-sm">({total} total)</span>
            </h2>
            <div className="flex items-center gap-3">
              <label className="text-xs text-slate-400 flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
                auto-refresh
              </label>
              <button
                onClick={refresh}
                className="text-xs bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded"
                disabled={loading}
              >
                {loading ? "Loading…" : "Refresh"}
              </button>
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5">
            <button
              onClick={() => setStatusFilter("")}
              className={`text-xs px-2 py-1 rounded ${
                statusFilter === ""
                  ? "bg-sky-700 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              All
            </button>
            {filterChips.map(({ status, count, active }) => (
              <button
                key={status}
                onClick={() => setStatusFilter(active ? "" : status)}
                className={`text-xs px-2 py-1 rounded transition ${
                  active
                    ? STATUS_STYLE[status]
                    : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                }`}
              >
                {STATUS_LABEL[status]} {count > 0 && <span className="opacity-70">({count})</span>}
              </button>
            ))}
          </div>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto space-y-1.5">
          {jobs.length === 0 ? (
            <div className="text-slate-500 text-sm italic p-4">
              {loading ? "Loading…" : "No jobs match the filter."}
            </div>
          ) : (
            jobs.map((job) => (
              <div
                key={job.id}
                onClick={() => handleOpen(job.id)}
                className={`bg-slate-800 hover:bg-slate-700/80 rounded p-2.5 cursor-pointer transition border ${
                  selected?.id === job.id
                    ? "border-sky-500"
                    : "border-transparent"
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-xs px-1.5 py-0.5 rounded ${STATUS_STYLE[job.status]}`}>
                    {STATUS_LABEL[job.status]}
                  </span>
                  <span className="text-xs text-slate-500">{job.channel}</span>
                  <span className="text-xs text-slate-500">{fmtAge(job.created_at)}</span>
                  <span className="text-xs text-slate-500 font-mono">{job.id}</span>
                  {job.retry_count > 0 && (
                    <span className="text-xs text-amber-400">retry #{job.retry_count}</span>
                  )}
                  {job.interrupted_count > 0 && (
                    <span className="text-xs text-yellow-400">
                      interrupted {job.interrupted_count}×
                    </span>
                  )}
                </div>
                <div className="text-sm text-slate-200 line-clamp-2">
                  {job.prompt || <span className="italic text-slate-500">(empty prompt)</span>}
                </div>
                {job.response && (
                  <div className="text-xs text-slate-400 mt-1 line-clamp-1">
                    → {job.response}
                  </div>
                )}
                {job.error && (
                  <div className="text-xs text-red-400 mt-1 line-clamp-1">
                    {job.error}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right: details pane */}
      <div className="w-[460px] shrink-0 border-l border-slate-800 pl-4 flex flex-col min-h-0">
        {selected ? (
          <>
            <div className="mb-3 flex items-center justify-between">
              <div className="font-mono text-xs text-slate-500">{selected.id}</div>
              <button
                onClick={() => setSelected(null)}
                className="text-slate-500 hover:text-slate-300 text-sm"
                title="close"
              >
                ✕
              </button>
            </div>

            <div className="flex items-center gap-2 mb-3">
              <span className={`text-xs px-2 py-0.5 rounded ${STATUS_STYLE[selected.status]}`}>
                {STATUS_LABEL[selected.status]}
              </span>
              <span className="text-xs text-slate-500">{selected.channel}</span>
              <span className="text-xs text-slate-500">{selected.speaker_id}</span>
              <span className="text-xs text-slate-500 ml-auto">
                elapsed: {fmtElapsed(selected)}
              </span>
            </div>

            <div className="flex gap-2 mb-3">
              {selected.status !== "running" && selected.status !== "queued" && (
                <button
                  onClick={() => handleRetry(selected.id)}
                  className="text-xs bg-sky-700 hover:bg-sky-600 px-2.5 py-1 rounded"
                >
                  Retry
                </button>
              )}
              {(selected.status === "running" || selected.status === "queued") && (
                <button
                  onClick={() => handleCancel(selected.id)}
                  className="text-xs bg-amber-700 hover:bg-amber-600 px-2.5 py-1 rounded"
                >
                  Cancel
                </button>
              )}
              <button
                onClick={() => handleDelete(selected.id)}
                className="text-xs bg-red-800 hover:bg-red-700 px-2.5 py-1 rounded ml-auto"
              >
                Delete
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 text-sm">
              <section>
                <div className="text-xs text-slate-400 mb-1">PROMPT</div>
                <div className="bg-slate-900/60 rounded p-2 whitespace-pre-wrap text-slate-200">
                  {selected.prompt}
                </div>
              </section>

              {selected.response && (
                <section>
                  <div className="text-xs text-slate-400 mb-1">RESPONSE</div>
                  <div className="bg-slate-900/60 rounded p-2 whitespace-pre-wrap text-slate-300">
                    {selected.response}
                  </div>
                </section>
              )}

              {selected.error && (
                <section>
                  <div className="text-xs text-red-400 mb-1">ERROR</div>
                  <div className="bg-red-950/40 rounded p-2 whitespace-pre-wrap text-red-200 text-xs">
                    {selected.error}
                  </div>
                </section>
              )}

              {selected.tool_calls.length > 0 && (
                <section>
                  <div className="text-xs text-slate-400 mb-1">
                    TOOL CALLS ({selected.tool_calls.length})
                  </div>
                  <div className="space-y-1">
                    {selected.tool_calls.map((tc, i) => (
                      <div key={i} className="bg-slate-900/60 rounded p-2 text-xs">
                        <span className={tc.ok ? "text-emerald-400" : "text-red-400"}>
                          {tc.ok ? "✓" : "✗"}
                        </span>{" "}
                        <span className="font-mono text-slate-200">{tc.name}</span>
                        {tc.args_summary && (
                          <span className="text-slate-500"> {tc.args_summary}</span>
                        )}
                        {tc.error && (
                          <div className="text-red-400 mt-0.5">{tc.error}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {selected.attempts.length > 0 && (
                <section>
                  <div className="text-xs text-slate-400 mb-1">
                    PROVIDER ATTEMPTS ({selected.attempts.length})
                  </div>
                  <div className="space-y-1">
                    {selected.attempts.map((a, i) => (
                      <div key={i} className="bg-slate-900/60 rounded p-2 text-xs">
                        <span className={a.ok ? "text-emerald-400" : "text-red-400"}>
                          {a.ok ? "✓" : "✗"}
                        </span>{" "}
                        <span className="font-mono text-slate-200">
                          {a.provider_id} / {a.model}
                        </span>
                        {a.elapsed_ms !== undefined && (
                          <span className="text-slate-500"> {a.elapsed_ms}ms</span>
                        )}
                        {a.error && (
                          <div className="text-red-400 mt-0.5">{a.error}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <section>
                <div className="text-xs text-slate-400 mb-1">TIMELINE</div>
                <div className="bg-slate-900/60 rounded p-2 text-xs text-slate-400 space-y-0.5">
                  <div>created: {new Date(selected.created_at * 1000).toLocaleString()}</div>
                  {selected.started_at && (
                    <div>started: {new Date(selected.started_at * 1000).toLocaleString()}</div>
                  )}
                  {selected.completed_at && (
                    <div>ended: {new Date(selected.completed_at * 1000).toLocaleString()}</div>
                  )}
                </div>
              </section>
            </div>
          </>
        ) : (
          <div className="text-slate-500 text-sm italic">
            Pick a job from the left to see its full record.
          </div>
        )}
      </div>
    </div>
  );
}
