/** The background loop: what it can do, what it did, and the stop button.
 *
 * Everything needed was already here — the levers, the kill switch, the
 * immune signatures — but the order buried it. The screen opened with a
 * scheduler tick-interval field and a wall of raw tick log
 * (`2026-09-01T15:12:11.355778+00:00  FIRE_CAPABILITY_SCAN
 * rule_matched:capability_scan_tick`, wrapping onto two lines), and the
 * thirty levers sat below the fold. A log is evidence; it is not the
 * point of a control panel.
 *
 * Reordered around what the reader wants: is it running and how do I stop
 * it, what CAN it do, what needs me, what did it just do. The tick
 * interval is a rarely-touched knob and now lives behind Advanced.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  approvePending,
  fetchAutonomicSettings,
  fetchAutonomicStatus,
  fetchImmune,
  fetchLeverHistory,
  fetchPending,
  fetchTicks,
  putAutonomicSettings,
  rejectPending,
  toggleKillSwitch,
  type AutonomicStatus,
  type ImmuneSignature,
  type LeverReport,
  type TickEntry,
} from "../api";
import { Badge, Button, Card, EmptyState, Flash, Spinner, cx } from "../ui";

/** FIRE_EMBEDDING_BACKFILL -> "Embedding backfill". The prefix and the
 *  shouting are an internal naming convention, not information. */
function leverLabel(name: string): string {
  const s = (name || "").replace(/^FIRE_/, "").replace(/_/g, " ").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function when(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (mins < 1440) return `${Math.round(mins / 60)}h ago`;
  return d.toLocaleDateString();
}

const exact = (iso: string) => {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
};

const STATUS_TONE: Record<string, "ok" | "warn" | "danger" | "neutral"> = {
  SUCCESS: "ok",
  SKIPPED: "neutral",
  FAILURE: "danger",
  ESCALATED: "warn",
  BLOCKED_BY_SAFETY: "warn",
  NOT_EXECUTED: "neutral",
};

export default function AutonomicPanel() {
  const [status, setStatus] = useState<AutonomicStatus | null>(null);
  const [ticks, setTicks] = useState<TickEntry[]>([]);
  const [pending, setPending] = useState<any[]>([]);
  const [immune, setImmune] = useState<ImmuneSignature[]>([]);
  const [interval, setIntervalSec] = useState<number>(30);
  const [selected, setSelected] = useState<string | null>(null);
  const [history, setHistory] = useState<LeverReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [advanced, setAdvanced] = useState(false);
  const [msg, setMsg] = useState("");

  const flash = (t: string) => {
    setMsg(t);
    setTimeout(() => setMsg(""), 3000);
  };

  const load = useCallback(async () => {
    try {
      const [st, pd, tk, im, cfg] = await Promise.allSettled([
        fetchAutonomicStatus(),
        fetchPending(),
        fetchTicks(40),
        fetchImmune(),
        fetchAutonomicSettings(),
      ]);
      if (st.status === "fulfilled") setStatus(st.value);
      if (pd.status === "fulfilled") setPending((pd.value as any).pending || []);
      if (tk.status === "fulfilled") setTicks((tk.value as any).ticks || []);
      if (im.status === "fulfilled")
        setImmune((im.value as any).signatures || []);
      if (cfg.status === "fulfilled")
        setIntervalSec((cfg.value as any).tick_interval_seconds ?? 30);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);

  const openLever = async (name: string) => {
    setSelected(name);
    setHistory([]);
    try {
      const r = await fetchLeverHistory(name, 10);
      setHistory((r as any).reports || []);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const flipKill = async () => {
    if (!status) return;
    const turningOff = status.enabled;
    if (
      turningOff &&
      !confirm(
        "Stop the background loop?\n\n" +
          "Reminders stop being delivered, nothing gets consolidated, and " +
          "the agent stops noticing its own failures until you turn it back on.",
      )
    )
      return;
    try {
      await toggleKillSwitch(!status.enabled);
      flash(turningOff ? "Background loop stopped" : "Background loop running");
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  /** Last time each lever actually ran, from the tick log we already have. */
  const lastRun = useMemo(() => {
    const m: Record<string, TickEntry> = {};
    for (const t of ticks) if (t.lever && !m[t.lever]) m[t.lever] = t;
    return m;
  }, [ticks]);

  const levers = useMemo(
    () => (status?.registered_levers ?? []).slice().sort(),
    [status],
  );

  if (loading)
    return (
      <div className="flex flex-1 items-center gap-2 p-6 text-sm text-ink-dim">
        <Spinner /> Loading…
      </div>
    );

  const running = !!status?.enabled && !!status?.scheduler_running;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-5 px-4 py-5 sm:px-6">
        {/* Is it running, and how do I stop it. Nothing above this. */}
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl2 border border-edge bg-surface px-4 py-3">
          <div className="flex items-center gap-3">
            <span
              className={cx(
                "h-2.5 w-2.5 rounded-full",
                running ? "bg-ok" : "bg-danger",
              )}
            />
            <div>
              <p className="font-semibold">
                {running ? "Running" : status?.enabled ? "Enabled, not ticking" : "Stopped"}
              </p>
              <p className="text-xs text-ink-dim">
                {levers.length} background jobs available · checks every{" "}
                {interval}s
              </p>
            </div>
          </div>
          <Button kind={status?.enabled ? "danger" : "ok"} onClick={flipKill}>
            {status?.enabled ? "Stop background work" : "Start background work"}
          </Button>
        </div>

        {/* Anything waiting on a human comes before anything else. */}
        {pending.length > 0 && (
          <Card
            title={`${pending.length} action${pending.length === 1 ? "" : "s"} waiting for you`}
            subtitle="The loop stopped short of doing these on its own."
          >
            <ul className="divide-y divide-edge">
              {pending.map((p: any) => (
                <li key={p.id} className="flex items-center gap-3 py-2">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">
                      {leverLabel(p.lever)}
                    </p>
                    <p className="truncate text-xs text-ink-dim">
                      {p.reason || JSON.stringify(p.params || {})}
                    </p>
                  </div>
                  <Button
                    kind="ok"
                    size="sm"
                    onClick={async () => {
                      await approvePending(p.id);
                      flash("Approved");
                      load();
                    }}
                  >
                    Approve
                  </Button>
                  <Button
                    kind="danger"
                    size="sm"
                    onClick={async () => {
                      await rejectPending(p.id);
                      flash("Rejected");
                      load();
                    }}
                  >
                    Reject
                  </Button>
                </li>
              ))}
            </ul>
          </Card>
        )}

        {/* The levers ARE the panel. They used to sit below the log. */}
        <Card
          title="Background jobs"
          subtitle="Each one runs on its own schedule. Click for what it did and when."
        >
          <div className="grid gap-1.5 sm:grid-cols-2">
            {levers.map((name) => {
              const t = lastRun[name];
              return (
                <button
                  key={name}
                  onClick={() => openLever(name)}
                  className={cx(
                    "flex items-center gap-2 rounded-lg border px-2.5 py-2 text-left",
                    selected === name
                      ? "border-accent/40 bg-accent-soft"
                      : "border-edge hover:bg-surface-hover",
                  )}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">
                      {leverLabel(name)}
                    </span>
                    <span className="block truncate text-[11px] text-ink-faint">
                      {t ? `ran ${when(t.ts)}` : "not seen recently"}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </Card>

        {selected && (
          <Card
            title={leverLabel(selected)}
            subtitle="Last ten runs"
            actions={
              <Button kind="ghost" size="sm" onClick={() => setSelected(null)}>
                Close
              </Button>
            }
          >
            {history.length === 0 && (
              <EmptyState title="No runs recorded">
                This job has not reported since the service last started.
              </EmptyState>
            )}
            <ul className="divide-y divide-edge">
              {history.map((r, i) => (
                <li key={i} className="flex flex-wrap items-center gap-2 py-2 text-sm">
                  <Badge tone={STATUS_TONE[r.status] || "neutral"}>
                    {r.status.toLowerCase()}
                  </Badge>
                  <span className="text-xs text-ink-dim" title={exact(r.started_at)}>
                    {when(r.started_at)}
                  </span>
                  {r.reason && (
                    <span className="text-xs text-ink-faint">{r.reason}</span>
                  )}
                  {r.cost?.seconds > 0 && (
                    <span className="ml-auto text-[11px] text-ink-faint">
                      {r.cost.seconds.toFixed(1)}s
                      {r.cost.usd > 0 && ` · $${r.cost.usd.toFixed(4)}`}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </Card>
        )}

        {/* Evidence, last. Local time, readable names, one line each. */}
        <Card title="Recent activity" subtitle="What the loop picked, most recent first.">
          <ul className="divide-y divide-edge text-sm">
            {ticks.slice(0, 20).map((t, i) => (
              <li key={i} className="flex items-center gap-3 py-1.5">
                <span
                  className="w-16 shrink-0 text-[11px] text-ink-faint"
                  title={exact(t.ts)}
                >
                  {when(t.ts)}
                </span>
                <span className="min-w-0 flex-1 truncate">
                  {t.lever ? leverLabel(t.lever) : (
                    <span className="text-ink-faint">nothing to do</span>
                  )}
                </span>
                {t.executed === false && t.lever && <Badge>skipped</Badge>}
              </li>
            ))}
          </ul>
        </Card>

        <div>
          <Button kind="ghost" size="sm" onClick={() => setAdvanced((v) => !v)}>
            {advanced ? "Hide advanced" : "Advanced"}
          </Button>
        </div>

        {advanced && (
          <>
            <Card
              title="Tick interval"
              subtitle="How often the loop decides what to do next. Lower reacts faster and costs more CPU."
            >
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={1}
                  max={3600}
                  value={interval}
                  onChange={(e) => setIntervalSec(Number(e.target.value))}
                  className="w-28 text-sm"
                />
                <span className="text-sm text-ink-dim">seconds</span>
                <Button
                  kind="primary"
                  size="sm"
                  onClick={async () => {
                    try {
                      await putAutonomicSettings(interval);
                      flash("Applied live");
                      load();
                    } catch (e: any) {
                      flash("Error: " + e.message);
                    }
                  }}
                >
                  Save
                </Button>
              </div>
            </Card>

            <Card
              title="Immune signatures"
              subtitle="Failures the agent learned to recognise and recover from."
            >
              {immune.length === 0 ? (
                <EmptyState title="Nothing learned yet">
                  A signature is written when the agent diagnoses a failure it
                  could recover from unaided.
                </EmptyState>
              ) : (
                <ul className="divide-y divide-edge">
                  {immune.map((s: any, i) => (
                    <li key={i} className="py-2">
                      <p className="text-sm font-medium">{s.name || s.id}</p>
                      <p className="text-xs text-ink-dim">
                        {s.description || s.pattern}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </>
        )}
      </div>
      <Flash text={msg} />
    </div>
  );
}
