import { useCallback, useEffect, useState } from "react";
import {
  fetchAutonomicStatus,
  fetchPending,
  approvePending,
  rejectPending,
  fetchTicks,
  fetchLeverHistory,
  toggleKillSwitch,
  AutonomicStatus,
  PendingEntry,
  TickEntry,
  LeverReport,
} from "../api";

const CATEGORY_COLORS: Record<string, string> = {
  FIRE_SERVER_HEALTH: "bg-rose-900/40",
  FIRE_ERROR_TRIAGE: "bg-rose-900/40",
  FIRE_SELF_HEAL: "bg-rose-900/40",
  FIRE_SERVICE_REPAIR: "bg-rose-900/40",
  FIRE_TOOL_INSTALL: "bg-amber-900/40",
};

function leverBgClass(name: string): string {
  return CATEGORY_COLORS[name] ?? "bg-sky-900/30";
}

function statusDotClass(status: string | undefined): string {
  if (!status) return "bg-slate-500";
  if (status === "success") return "bg-emerald-400";
  if (status === "failure") return "bg-rose-500";
  if (status === "skipped") return "bg-slate-400";
  if (status === "blocked_by_safety") return "bg-amber-400";
  if (status === "escalated") return "bg-violet-400";
  return "bg-slate-500";
}

export default function AutonomicPanel() {
  const [status, setStatus] = useState<AutonomicStatus | null>(null);
  const [pending, setPending] = useState<PendingEntry[]>([]);
  const [ticks, setTicks] = useState<TickEntry[]>([]);
  const [tickLimit, setTickLimit] = useState<number>(50);
  const [selectedLever, setSelectedLever] = useState<string | null>(null);
  const [leverHistory, setLeverHistory] = useState<LeverReport[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 4000);
  };

  const refresh = useCallback(async () => {
    try {
      const [s, p, t] = await Promise.all([
        fetchAutonomicStatus(),
        fetchPending(),
        fetchTicks(tickLimit),
      ]);
      setStatus(s);
      setPending(p.pending);
      setTicks(t.ticks);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  }, [tickLimit]);

  const openLever = async (name: string) => {
    setSelectedLever(name);
    setLoadingHistory(true);
    try {
      const result = await fetchLeverHistory(name, 10);
      setLeverHistory(result.reports);
    } catch (e: any) {
      flash("Error: " + e.message);
      setLeverHistory([]);
    } finally {
      setLoadingHistory(false);
    }
  };

  const closeLever = () => {
    setSelectedLever(null);
    setLeverHistory([]);
  };

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

          {/* Recent Ticks */}
          <section>
            <div className="flex items-center gap-2 mb-2">
              <h2 className="font-bold text-sm">📊 Recent Ticks</h2>
              <select
                value={tickLimit}
                onChange={(e) => setTickLimit(parseInt(e.target.value, 10))}
                className="bg-slate-900 rounded px-1 py-0.5 text-[10px]"
              >
                <option value={10}>10</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
            {ticks.length === 0 ? (
              <div className="opacity-40">No ticks yet</div>
            ) : (
              <div className="space-y-0.5 font-mono text-[10px] max-h-72 overflow-y-auto">
                {ticks.map((t, i) => (
                  <div
                    key={i}
                    className={`flex gap-2 px-1 py-0.5 rounded ${
                      t.executed ? "bg-sky-900/20" : "bg-slate-800/20 opacity-60"
                    }`}
                  >
                    <span className="text-slate-500 w-32 shrink-0">{t.ts}</span>
                    <span className="text-violet-400 w-44 shrink-0 truncate">
                      {t.lever ?? "—"}
                    </span>
                    <span className="text-slate-300 truncate">{t.reason}</span>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* Levers grid */}
          <section>
            <h2 className="font-bold text-sm mb-2">
              🦾 Levers ({status?.registered_levers.length ?? 0})
            </h2>
            <div
              className="grid gap-1.5"
              style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}
            >
              {(status?.registered_levers ?? []).slice().sort().map((name) => (
                <button
                  key={name}
                  onClick={() => openLever(name)}
                  className={`text-left p-2 rounded ${leverBgClass(
                    name,
                  )} hover:brightness-125 flex items-center gap-2`}
                >
                  <span
                    className={`inline-block w-2 h-2 rounded-full bg-slate-500`}
                  />
                  <span className="text-[10px] truncate">{name}</span>
                </button>
              ))}
            </div>
          </section>
        </div>

        {msg && (
          <div className="p-2 text-xs text-sky-400 border-t border-slate-800">{msg}</div>
        )}
      </div>

      {selectedLever && (
        <div className="w-[420px] border-l border-slate-800 bg-slate-900/80 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-slate-800 flex items-center gap-2">
            <h2 className="font-bold text-sm text-violet-400">{selectedLever}</h2>
            <button
              onClick={closeLever}
              className="ml-auto text-slate-400 hover:text-slate-200 text-sm"
            >
              ✕
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2 text-xs">
            {loadingHistory ? (
              <div className="opacity-50">Loading…</div>
            ) : leverHistory.length === 0 ? (
              <div className="opacity-40">No history yet</div>
            ) : (
              leverHistory.map((r, i) => (
                <div key={i} className="p-2 rounded bg-slate-800/60 space-y-1">
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-block w-2 h-2 rounded-full ${statusDotClass(
                        r.status,
                      )}`}
                    />
                    <span className="font-semibold">{r.status}</span>
                    <span className="text-slate-500 text-[10px] ml-auto">
                      {r.started_at}
                    </span>
                  </div>
                  <div className="opacity-70">{r.reason}</div>
                  {Object.keys(r.outcome).length > 0 && (
                    <details>
                      <summary className="cursor-pointer text-[10px] text-slate-400">
                        outcome
                      </summary>
                      <pre className="mt-1 font-mono text-[10px] whitespace-pre-wrap opacity-80">
                        {JSON.stringify(r.outcome, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
