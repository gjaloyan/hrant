import { useCallback, useEffect, useState } from "react";
import {
  fetchAutonomicStatus,
  fetchPending,
  approvePending,
  rejectPending,
  toggleKillSwitch,
  AutonomicStatus,
  PendingEntry,
} from "../api";

export default function AutonomicPanel() {
  const [status, setStatus] = useState<AutonomicStatus | null>(null);
  const [pending, setPending] = useState<PendingEntry[]>([]);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 4000);
  };

  const refresh = useCallback(async () => {
    try {
      const [s, p] = await Promise.all([fetchAutonomicStatus(), fetchPending()]);
      setStatus(s);
      setPending(p.pending);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const handleApprove = async (id: string) => {
    setBusy(true);
    try {
      const report = await approvePending(id);
      flash(`Approved — ${report.lever}: ${report.status} (${report.reason})`);
      refresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleReject = async (id: string) => {
    if (!confirm("Reject this pending action?")) return;
    setBusy(true);
    try {
      await rejectPending(id);
      flash("Rejected");
      refresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleKillSwitch = async () => {
    if (!status) return;
    if (status.enabled && !confirm("Disable autonomic scheduler?")) return;
    setBusy(true);
    try {
      const result = await toggleKillSwitch(!status.enabled);
      flash(result.enabled ? "Scheduler enabled" : "Scheduler disabled");
      refresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <div className="flex-1 flex flex-col overflow-hidden bg-slate-950/60">
        {/* Header */}
        <div className="p-3 border-b border-slate-800 flex items-center gap-3 flex-wrap">
          <button
            onClick={handleKillSwitch}
            disabled={busy || !status}
            className={`px-3 py-1.5 rounded text-xs font-semibold ${
              status?.enabled
                ? "bg-emerald-700 hover:bg-emerald-600"
                : "bg-rose-700 hover:bg-rose-600"
            } disabled:opacity-50`}
          >
            {status?.enabled ? "● ENABLED" : "● DISABLED"}
          </button>
          <span className="text-xs opacity-70">
            scheduler:{" "}
            <span
              className={
                status?.scheduler_running ? "text-emerald-400" : "text-slate-400"
              }
            >
              {status?.scheduler_running ? "running" : "stopped"}
            </span>
          </span>
          <span className="text-xs opacity-70">
            {status ? `${status.registered_levers.length} levers registered` : "…"}
          </span>
          <button
            onClick={refresh}
            className="ml-auto px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-xs"
          >
            Refresh
          </button>
        </div>

        {/* Sections */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4 text-xs">
          {/* Pending Approvals */}
          <section>
            <h2 className="font-bold text-sm mb-2">
              ⚠ Pending Approvals ({pending.length})
            </h2>
            {pending.length === 0 ? (
              <div className="opacity-40">No pending actions</div>
            ) : (
              <div className="space-y-2">
                {pending.map((p) => (
                  <div
                    key={p.id}
                    className="p-2 rounded bg-amber-900/20 border border-amber-800/40 space-y-1"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-violet-400 font-semibold">{p.lever}</span>
                      <span className="text-slate-400 text-[10px]">
                        {p.requested_at}
                      </span>
                      <span className="ml-auto flex gap-1">
                        <button
                          onClick={() => handleApprove(p.id)}
                          disabled={busy}
                          className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-0.5 text-[10px] disabled:opacity-50"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => handleReject(p.id)}
                          disabled={busy}
                          className="bg-rose-700 hover:bg-rose-600 rounded px-2 py-0.5 text-[10px] disabled:opacity-50"
                        >
                          Reject
                        </button>
                      </span>
                    </div>
                    <div className="opacity-70 font-mono text-[10px]">
                      {JSON.stringify(p.params)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>

        {msg && (
          <div className="p-2 text-xs text-sky-400 border-t border-slate-800">{msg}</div>
        )}
      </div>
    </div>
  );
}
