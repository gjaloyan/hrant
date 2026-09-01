import { useEffect, useState } from "react";
import {
  ConsolidationStatus,
  Digest,
  DigestSummary,
  fetchConsolidationStatus,
  fetchDigest,
  fetchDigests,
  runConsolidation,
} from "../../api";

type Props = { flash: (msg: string) => void };

const STATUS_STYLE: Record<string, string> = {
  success:     "bg-emerald-700 text-emerald-100",
  partial:     "bg-amber-700 text-amber-50",
  failed:      "bg-red-700 text-red-100",
  skipped:     "bg-slate-600 text-slate-300",
  in_progress: "bg-sky-700 text-sky-100",
};

function fmtAge(ts: number): string {
  if (!ts) return "—";
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

/** "17h 33m" rather than "63198s". Raw seconds are how the API speaks;
 *  nobody reads a five-digit number as a duration. */
function fmtDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  if (h < 24) return m ? `${h}h ${m}m` : `${h}h`;
  const d = Math.floor(h / 24);
  return `${d}d ${h % 24}h`;
}

export default function MemoryDigestsTab({ flash }: Props) {
  const [status, setStatus] = useState<ConsolidationStatus | null>(null);
  const [digests, setDigests] = useState<DigestSummary[]>([]);
  const [selected, setSelected] = useState<Digest | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const [s, d] = await Promise.all([
        fetchConsolidationStatus(),
        fetchDigests(60),
      ]);
      setStatus(s);
      setDigests(d.digests);
    } catch (e: any) {
      flash("Load failed: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleRun = async (dry_run: boolean) => {
    if (running) return;
    if (
      !dry_run &&
      !confirm(
        "Run a full consolidation now? This calls the active LLM " +
        "and may write to user.md / memory_facts.jsonl. Takes ~30-60s.",
      )
    )
      return;
    setRunning(true);
    try {
      const d = await runConsolidation(dry_run);
      flash(
        dry_run
          ? `Dry-run done: ${d.new_facts.filter((f) => f.promoted).length} facts would be added`
          : `Consolidation ${d.status}: ${d.new_facts.filter((f) => f.promoted).length} facts added`,
      );
      await refresh();
      setSelected(d);
    } catch (e: any) {
      flash("Run failed: " + e.message);
    } finally {
      setRunning(false);
    }
  };

  const handleOpen = async (date_str: string) => {
    try {
      const d = await fetchDigest(date_str);
      setSelected(d);
    } catch (e: any) {
      flash("Open failed: " + e.message);
    }
  };

  return (
    <div className="flex gap-4 h-full min-h-0">
      {/* Left: status + list */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Status banner */}
        {status && (
          <div className="mb-3 bg-slate-800 rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-base font-semibold text-slate-100">
                Daily memory consolidation
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={() => handleRun(true)}
                  disabled={running}
                  className="text-xs bg-slate-700 hover:bg-slate-600 px-2.5 py-1 rounded disabled:opacity-40"
                  title="Run the pipeline but skip memory_facts/profile writes"
                >
                  {running ? "Running…" : "Dry run"}
                </button>
                <button
                  onClick={() => handleRun(false)}
                  disabled={running}
                  className="text-xs bg-sky-700 hover:bg-sky-600 px-2.5 py-1 rounded disabled:opacity-40"
                >
                  {running ? "Running…" : "Run now"}
                </button>
                <button
                  onClick={refresh}
                  disabled={loading}
                  className="text-xs bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded"
                >
                  Refresh
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
              <div>
                <span className="text-slate-500">Last run:</span>{" "}
                {status.state.last_run_at ? (
                  <>
                    {fmtAge(status.state.last_run_at)}{" "}
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] ${
                        STATUS_STYLE[status.state.last_run_status] || ""
                      }`}
                    >
                      {status.state.last_run_status}
                    </span>
                  </>
                ) : (
                  <span className="italic">never</span>
                )}
              </div>
              <div>
                <span className="text-slate-500">Total runs:</span>{" "}
                {status.state.total_runs}
              </div>
              <div>
                <span className="text-slate-500">Next gate:</span>{" "}
                {status.would_fire_now ? (
                  <span className="text-amber-400">ready</span>
                ) : (
                  <span className="text-slate-400">{status.gate_reason}</span>
                )}
              </div>
              {status.idle_for_seconds !== null && (
                <div>
                  <span className="text-slate-500">Idle for:</span>{" "}
                  {fmtDuration(status.idle_for_seconds)} of{" "}
                  {fmtDuration(status.config.idle_threshold_seconds)} needed
                </div>
              )}
              {status.cooldown_remaining_seconds > 0 && (
                <div>
                  <span className="text-slate-500">Cooldown:</span>{" "}
                  {fmtDuration(status.cooldown_remaining_seconds)} remaining
                </div>
              )}
              {status.state.last_run_jobs_analyzed > 0 && (
                <div>
                  <span className="text-slate-500">Last analyzed:</span>{" "}
                  {status.state.last_run_jobs_analyzed} turns,{" "}
                  {status.state.last_run_facts_added} facts added
                </div>
              )}
            </div>
            {status.state.last_run_error && (
              <div className="mt-2 text-xs text-red-400">
                Last error: {status.state.last_run_error}
              </div>
            )}
          </div>
        )}

        {/* Digest list */}
        <div className="flex-1 overflow-y-auto space-y-1.5">
          {digests.length === 0 ? (
            <div className="text-slate-500 text-sm italic p-4">
              {loading
                ? "Loading…"
                : "No digests yet. Click \"Run now\" or wait for the scheduler."}
            </div>
          ) : (
            digests.map((d) => (
              <div
                key={d.date}
                onClick={() => handleOpen(d.date)}
                className={`bg-slate-800 hover:bg-slate-700/80 rounded p-2.5 cursor-pointer border ${
                  selected?.date === d.date
                    ? "border-sky-500"
                    : "border-transparent"
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-mono text-slate-200">{d.date}</span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      STATUS_STYLE[d.status] || ""
                    }`}
                  >
                    {d.status}
                  </span>
                  <span className="text-xs text-slate-500">{d.turns_analyzed} turns</span>
                  <span className="text-xs text-emerald-400">{d.new_facts_count} facts</span>
                  {d.open_threads_count > 0 && (
                    <span className="text-xs text-amber-400">{d.open_threads_count} open</span>
                  )}
                </div>
                <div className="text-xs text-slate-400 line-clamp-2">
                  {d.narrative_preview || (
                    <span className="italic text-slate-500">(empty narrative)</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right: details */}
      <div className="flex w-[min(30rem,50%)] min-w-0 shrink-0 flex-col border-l border-edge pl-4 min-h-0">
        {selected ? (
          <>
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-sm font-mono text-slate-200">{selected.date}</div>
                <div className="text-xs text-slate-500">
                  {selected.turns_analyzed} turns · {selected.speakers_active.length} speaker(s)
                </div>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="text-slate-500 hover:text-slate-300 text-sm"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 text-sm">
              <section>
                <div className="text-xs text-slate-400 mb-1">NARRATIVE</div>
                <div className="bg-slate-900/60 rounded p-2 whitespace-pre-wrap text-slate-200">
                  {selected.narrative || (
                    <span className="italic text-slate-500">empty</span>
                  )}
                </div>
              </section>

              {selected.new_facts.length > 0 && (
                <section>
                  <div className="text-xs text-slate-400 mb-1">
                    FACTS ({selected.new_facts.filter((f) => f.promoted).length} promoted,{" "}
                    {selected.new_facts.filter((f) => !f.promoted).length} skipped)
                  </div>
                  <div className="space-y-1">
                    {selected.new_facts.map((f, i) => (
                      <div key={i} className="bg-slate-900/60 rounded p-2 text-xs">
                        <span className={f.promoted ? "text-emerald-400" : "text-slate-500"}>
                          {f.promoted ? "✓" : "·"}
                        </span>{" "}
                        <span className="text-slate-300">[{f.category}]</span>{" "}
                        <span className="text-slate-200">{f.text}</span>
                        {f.reason_if_skipped && (
                          <span className="text-slate-500"> ({f.reason_if_skipped})</span>
                        )}
                        {f.related_topics.length > 0 && (
                          <div className="text-slate-500 mt-0.5">
                            topics: {f.related_topics.join(", ")}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {selected.open_threads.length > 0 && (
                <section>
                  <div className="text-xs text-slate-400 mb-1">
                    OPEN THREADS ({selected.open_threads.length})
                  </div>
                  <ul className="bg-slate-900/60 rounded p-2 text-xs text-amber-200 space-y-0.5 list-disc list-inside">
                    {selected.open_threads.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </section>
              )}

              {selected.profile_updates.length > 0 && (
                <section>
                  <div className="text-xs text-slate-400 mb-1">
                    PROFILE UPDATES ({selected.profile_updates.length})
                  </div>
                  <div className="space-y-1">
                    {selected.profile_updates.map((up, i) => (
                      <div key={i} className="bg-slate-900/60 rounded p-2 text-xs">
                        <div className="font-mono text-slate-300">{up.speaker_id}</div>
                        <div className="text-slate-200 mt-0.5 whitespace-pre-wrap">
                          {up.appended_text}
                        </div>
                      </div>
                    ))}
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

              <section>
                <div className="text-xs text-slate-400 mb-1">TIMELINE</div>
                <div className="bg-slate-900/60 rounded p-2 text-xs text-slate-400 space-y-0.5">
                  <div>started: {new Date(selected.started_at * 1000).toLocaleString()}</div>
                  {selected.completed_at && (
                    <div>completed: {new Date(selected.completed_at * 1000).toLocaleString()}</div>
                  )}
                  {selected.completed_at && selected.started_at && (
                    <div>
                      duration: {fmtDuration(selected.completed_at - selected.started_at)}
                    </div>
                  )}
                </div>
              </section>
            </div>
          </>
        ) : (
          <div className="text-slate-500 text-sm italic">
            Pick a digest from the left, or click "Run now" to create today's.
          </div>
        )}
      </div>
    </div>
  );
}
