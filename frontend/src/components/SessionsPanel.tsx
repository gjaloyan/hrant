import { useCallback, useEffect, useState } from "react";
import {
  fetchSessions,
  fetchSession,
  fetchSessionStats,
  archiveSessions,
  deleteSession,
  newSession,
  SessionSummary,
  SessionDetail,
  SessionStats,
} from "../api";

/** Simple bar chart drawn with CSS — no external chart library needed. */
function MiniBarChart({
  data,
  color = "#38bdf8",
  height = 120,
}: {
  data: { label: string; value: number }[];
  color?: string;
  height?: number;
}) {
  const max = Math.max(...data.map((d) => d.value), 1);
  return (
    <div className="flex items-end gap-1" style={{ height }}>
      {data.map((d, i) => (
        <div key={i} className="flex flex-col items-center flex-1 min-w-0">
          <div
            className="w-full rounded-t"
            style={{
              height: `${(d.value / max) * (height - 20)}px`,
              backgroundColor: color,
              minHeight: d.value > 0 ? 2 : 0,
            }}
            title={`${d.label}: ${d.value}`}
          />
          <div className="text-[8px] opacity-50 truncate w-full text-center mt-0.5">
            {d.label}
          </div>
        </div>
      ))}
    </div>
  );
}

function ConfidenceTimeline({
  data,
}: {
  data: { session_id: string; date: string; avg_confidence: number; turns: number }[];
}) {
  if (data.length === 0) return <div className="text-xs opacity-40">No data yet</div>;
  const maxTurns = Math.max(...data.map((d) => d.turns), 1);
  return (
    <div className="flex items-end gap-1 h-24">
      {data.slice(-30).map((d, i) => {
        const conf = d.avg_confidence;
        const color =
          conf >= 90 ? "#34d399" : conf >= 70 ? "#fbbf24" : conf >= 50 ? "#fb923c" : "#f87171";
        return (
          <div key={i} className="flex flex-col items-center flex-1 min-w-0">
            <div
              className="w-full rounded-t"
              style={{
                height: `${(d.turns / maxTurns) * 80}px`,
                backgroundColor: color,
                minHeight: 2,
              }}
              title={`${d.date.slice(0, 10)}: ${conf}% avg, ${d.turns} turns`}
            />
            <div className="text-[7px] opacity-40 truncate w-full text-center">
              {d.date.slice(5, 10)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) return "active";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

export default function SessionsPanel() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [selected, setSelected] = useState<SessionDetail | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [archiveDays, setArchiveDays] = useState(90);
  const [msg, setMsg] = useState("");
  const [view, setView] = useState<"list" | "stats">("list");

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 4000);
  };

  const load = useCallback(async () => {
    try {
      const [sessData, statsData] = await Promise.all([
        fetchSessions(showArchived),
        fetchSessionStats(),
      ]);
      setSessions(sessData.sessions);
      setCurrentId(sessData.current_id);
      setStats(statsData);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  }, [showArchived]);

  useEffect(() => {
    load();
  }, [load]);

  const handleSelect = async (id: string) => {
    try {
      const data = await fetchSession(id);
      setSelected(data.session);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this session?")) return;
    try {
      await deleteSession(id);
      if (selected?.id === id) setSelected(null);
      load();
      flash("Session deleted");
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleArchive = async () => {
    try {
      const res = await archiveSessions(archiveDays);
      flash(`Archived ${res.archived} sessions`);
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleNewSession = async () => {
    try {
      await newSession();
      load();
      setSelected(null);
      flash("New session started");
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  // Prepare daily chart data
  const dailyData = stats
    ? Object.entries(stats.daily_counts)
        .sort(([a], [b]) => a.localeCompare(b))
        .slice(-14)
        .map(([label, value]) => ({ label: label.slice(5), value }))
    : [];

  const intentData = stats
    ? Object.entries(stats.intents).map(([label, value]) => ({ label, value }))
    : [];

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      {/* Left: session list */}
      <div className="w-80 shrink-0 border-r border-slate-800 bg-slate-950/60 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-3 border-b border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="font-bold text-sm">Sessions</h2>
            <div className="flex gap-1">
              <button
                onClick={() => setView(view === "list" ? "stats" : "list")}
                className="bg-slate-800 hover:bg-slate-700 rounded px-2 py-0.5 text-xs"
              >
                {view === "list" ? "Stats" : "List"}
              </button>
              <button
                onClick={handleNewSession}
                className="bg-sky-800 hover:bg-sky-700 rounded px-2 py-0.5 text-xs"
              >
                + New
              </button>
            </div>
          </div>

          {stats && (
            <div className="flex gap-2 text-xs">
              <span className="text-emerald-400">{stats.total_sessions} sessions</span>
              <span className="text-sky-400">{stats.total_turns} turns</span>
              {stats.archived_count > 0 && (
                <span className="text-amber-400">{stats.archived_count} archived</span>
              )}
            </div>
          )}

          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1 text-xs cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showArchived}
                onChange={(e) => setShowArchived(e.target.checked)}
              />
              Show archived
            </label>
          </div>
        </div>

        {/* Session list or stats */}
        <div className="flex-1 overflow-y-auto">
          {view === "stats" && stats ? (
            <div className="p-3 space-y-4">
              {/* Daily sessions chart */}
              <div>
                <div className="text-[10px] font-semibold opacity-50 mb-1 uppercase tracking-wider">
                  Sessions per day (last 14d)
                </div>
                {dailyData.length > 0 ? (
                  <MiniBarChart data={dailyData} color="#38bdf8" height={100} />
                ) : (
                  <div className="text-xs opacity-40">No data</div>
                )}
              </div>

              {/* Confidence timeline */}
              <div>
                <div className="text-[10px] font-semibold opacity-50 mb-1 uppercase tracking-wider">
                  Confidence over time (last 30 sessions)
                </div>
                <ConfidenceTimeline data={stats.confidence_over_time} />
                <div className="flex gap-2 mt-1 text-[8px] opacity-40">
                  <span className="text-emerald-400">90%+</span>
                  <span className="text-amber-400">70-89%</span>
                  <span className="text-orange-400">50-69%</span>
                  <span className="text-red-400">&lt;50%</span>
                </div>
              </div>

              {/* Intent breakdown */}
              <div>
                <div className="text-[10px] font-semibold opacity-50 mb-1 uppercase tracking-wider">
                  Intent distribution
                </div>
                {intentData.length > 0 ? (
                  <MiniBarChart data={intentData} color="#a78bfa" height={80} />
                ) : (
                  <div className="text-xs opacity-40">No data</div>
                )}
              </div>

              {/* Archive controls */}
              <div className="border-t border-slate-700 pt-3">
                <div className="text-[10px] font-semibold opacity-50 mb-2 uppercase tracking-wider">
                  Archive
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span>Older than</span>
                  <input
                    type="number"
                    value={archiveDays}
                    onChange={(e) => setArchiveDays(Number(e.target.value))}
                    className="bg-slate-900 rounded px-2 py-0.5 w-16 text-center outline-none"
                    min={1}
                  />
                  <span>days</span>
                  <button
                    onClick={handleArchive}
                    className="bg-amber-800 hover:bg-amber-700 rounded px-2 py-0.5"
                  >
                    Archive
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-2 space-y-1">
              {sessions.length === 0 && (
                <div className="text-xs opacity-40 p-2">No sessions yet</div>
              )}
              {sessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleSelect(s.id)}
                  className={`w-full text-left rounded p-2 transition-colors text-xs ${
                    selected?.id === s.id
                      ? "bg-sky-800"
                      : s.id === currentId
                      ? "bg-emerald-900/40 hover:bg-emerald-900/60"
                      : s.archived
                      ? "bg-slate-800/40 hover:bg-slate-800/60 opacity-60"
                      : "bg-slate-800/60 hover:bg-slate-700"
                  }`}
                >
                  <div className="flex justify-between items-start gap-1">
                    <span className="font-medium truncate">
                      {s.title || "(untitled)"}
                    </span>
                    <div className="flex items-center gap-1 shrink-0">
                      {s.id === currentId && (
                        <span className="text-[9px] bg-emerald-700 rounded px-1">active</span>
                      )}
                      {s.archived && (
                        <span className="text-[9px] bg-slate-600 rounded px-1">archived</span>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-2 mt-0.5 opacity-60 text-[10px]">
                    <span>{s.started.slice(0, 16)}</span>
                    <span>{s.turn_count} turns</span>
                    {s.avg_confidence > 0 && <span>{s.avg_confidence}%</span>}
                    <span>{formatDuration(s.duration_seconds)}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {msg && (
          <div className="p-2 text-xs text-sky-400 border-t border-slate-800">{msg}</div>
        )}
      </div>

      {/* Right: session detail */}
      <div className="flex-1 overflow-y-auto p-4">
        {!selected ? (
          <div className="flex items-center justify-center h-full opacity-40 text-sm">
            Select a session to view its conversation log
          </div>
        ) : (
          <div className="space-y-4">
            {/* Header */}
            <div className="flex justify-between items-start gap-4">
              <div>
                <h3 className="font-bold text-lg">{selected.title || "(untitled)"}</h3>
                <div className="flex gap-3 text-xs opacity-60 mt-1">
                  <span>Started: {selected.started}</span>
                  {selected.ended && <span>Ended: {selected.ended}</span>}
                  <span>Duration: {formatDuration(selected.duration_seconds)}</span>
                </div>
              </div>
              <button
                onClick={() => handleDelete(selected.id)}
                className="bg-rose-800 hover:bg-rose-700 rounded px-2 py-1 text-xs shrink-0"
              >
                Delete
              </button>
            </div>

            {/* Stats bar */}
            <div className="flex gap-3 text-xs">
              <div className="bg-slate-800 rounded px-3 py-1.5">
                <span className="opacity-50">Turns: </span>
                <span className="font-bold">{selected.turn_count}</span>
              </div>
              <div className="bg-slate-800 rounded px-3 py-1.5">
                <span className="opacity-50">Avg confidence: </span>
                <span className={`font-bold ${
                  selected.avg_confidence >= 90
                    ? "text-emerald-400"
                    : selected.avg_confidence >= 70
                    ? "text-amber-400"
                    : "text-rose-400"
                }`}>
                  {selected.avg_confidence}%
                </span>
              </div>
              {Object.entries(selected.intents).map(([intent, count]) => (
                <div key={intent} className="bg-slate-800 rounded px-3 py-1.5">
                  <span className="opacity-50">{intent}: </span>
                  <span className="font-bold">{count}</span>
                </div>
              ))}
              {selected.topics_used.length > 0 && (
                <div className="bg-slate-800 rounded px-3 py-1.5">
                  <span className="opacity-50">Topics: </span>
                  <span>{selected.topics_used.join(", ")}</span>
                </div>
              )}
            </div>

            {/* Conversation log */}
            <div className="space-y-3">
              <div className="text-[10px] font-semibold opacity-50 uppercase tracking-wider">
                Conversation Log
              </div>
              {selected.turns.length === 0 && (
                <div className="text-xs opacity-40">(empty session)</div>
              )}
              {selected.turns.map((turn: any, i: number) => (
                <div key={i} className="bg-slate-800/60 rounded p-3 space-y-2 text-sm">
                  {/* Turn header */}
                  <div className="flex gap-2 text-[10px] opacity-50">
                    <span>{turn.ts}</span>
                    <span className="bg-slate-700 px-1 rounded">{turn.intent}</span>
                    {turn.confidence > 0 && (
                      <span className={
                        turn.confidence >= 90
                          ? "text-emerald-400"
                          : turn.confidence >= 70
                          ? "text-amber-400"
                          : "text-rose-400"
                      }>
                        {turn.confidence}%
                      </span>
                    )}
                    {turn.topics?.length > 0 && (
                      <span>topics: {turn.topics.join(", ")}</span>
                    )}
                  </div>
                  {/* User message */}
                  <div>
                    <span className="text-sky-400 font-semibold text-xs">User: </span>
                    <span className="whitespace-pre-wrap">{turn.user}</span>
                  </div>
                  {/* Agent response */}
                  <div>
                    <span className="text-emerald-400 font-semibold text-xs">Agent: </span>
                    <span className="whitespace-pre-wrap">{turn.answer}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
