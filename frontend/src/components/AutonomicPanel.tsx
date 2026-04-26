import { useCallback, useEffect, useState } from "react";
import {
  fetchAutonomicStatus,
  toggleKillSwitch,
  AutonomicStatus,
} from "../api";

export default function AutonomicPanel() {
  const [status, setStatus] = useState<AutonomicStatus | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 4000);
  };

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchAutonomicStatus());
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

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

        {/* Sections (added in later tasks) */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4 text-xs">
          <div className="opacity-40">Pending, ticks, levers, immune sections arrive in later tasks.</div>
        </div>

        {msg && (
          <div className="p-2 text-xs text-sky-400 border-t border-slate-800">{msg}</div>
        )}
      </div>
    </div>
  );
}
