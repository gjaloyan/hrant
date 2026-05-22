import { useEffect, useMemo, useRef, useState } from "react";
import {
  downloadLogs,
  fetchLogs,
  fetchLogSources,
  LogEvent,
  LogLevel,
  LogSource,
  LogSourcesPayload,
} from "../../api";

type Props = { flash: (msg: string) => void };

const LEVEL_STYLES: Record<LogLevel, string> = {
  debug: "text-slate-500",
  info: "text-slate-200",
  warning: "text-amber-300 bg-amber-950/30",
  error: "text-rose-300 bg-rose-950/30",
  critical: "text-rose-100 bg-rose-900/60",
};

const SOURCE_STYLES: Record<LogSource, string> = {
  python: "bg-slate-700/40 text-slate-200",
  tool: "bg-violet-700/40 text-violet-200",
  job: "bg-emerald-700/40 text-emerald-200",
  supervisor: "bg-indigo-700/40 text-indigo-200",
  agent: "bg-sky-700/40 text-sky-200",
};

function fmtTs(ts: number): string {
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms}`;
}

export default function LogsTab({ flash }: Props) {
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [streaming, setStreaming] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [level, setLevel] = useState<LogLevel | "">("");
  const [source, setSource] = useState<LogSource | "">("");
  const [search, setSearch] = useState("");
  const [sources, setSources] = useState<LogSourcesPayload | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const pausedBuffer = useRef<LogEvent[]>([]);
  const streamingRef = useRef(streaming);

  // Keep ref in sync so the EventSource onmessage closure reads the
  // latest pause state (it captures `streaming` at mount time
  // otherwise).
  useEffect(() => {
    streamingRef.current = streaming;
  }, [streaming]);

  // Backfill on mount.
  useEffect(() => {
    fetchLogSources().then(setSources).catch(() => {});
    fetchLogs({ limit: 500 })
      .then((r) => setEvents(r.events))
      .catch((e: any) => flash("logs backfill: " + (e?.message || e)));
  }, [flash]);

  // SSE subscription — created once on mount, lives until unmount.
  useEffect(() => {
    const es = new EventSource("/api/logs/stream");
    es.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data) as LogEvent;
        if (streamingRef.current) {
          setEvents((prev) => [...prev, ev].slice(-5000));
        } else {
          pausedBuffer.current.push(ev);
        }
      } catch {
        /* ignore */
      }
    };
    es.onerror = () => {
      // EventSource auto-reconnects on transient drops.
    };
    return () => es.close();
  }, []);

  // Resume from paused buffer.
  useEffect(() => {
    if (streaming && pausedBuffer.current.length > 0) {
      setEvents((prev) => [...prev, ...pausedBuffer.current].slice(-5000));
      pausedBuffer.current = [];
    }
  }, [streaming]);

  // Auto-scroll.
  useEffect(() => {
    if (!autoScroll || !listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [events, autoScroll]);

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase();
    return events.filter((e) => {
      if (level && e.level !== level) return false;
      if (source && e.source !== source) return false;
      if (s) {
        const hay = (
          e.message + " " + e.logger + " " + JSON.stringify(e.meta || {})
        ).toLowerCase();
        if (!hay.includes(s)) return false;
      }
      return true;
    });
  }, [events, level, source, search]);

  const visible = useMemo(() => filtered.slice(-500), [filtered]);

  const onCopy = async () => {
    const text = filtered
      .map(
        (e) =>
          `${fmtTs(e.ts)} ${e.level.toUpperCase().padEnd(8)} ${e.source.padEnd(10)} ${e.logger}  ${e.message}`,
      )
      .join("\n");
    try {
      await navigator.clipboard.writeText(text);
      flash(`copied ${filtered.length} log lines`);
    } catch (err: any) {
      flash("copy failed: " + (err?.message || err));
    }
  };

  const onScroll = () => {
    if (!listRef.current) return;
    const el = listRef.current;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
    setAutoScroll(atBottom);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-wrap gap-2 items-center px-3 py-2 border-b border-slate-700/40 bg-slate-900/40">
        <button
          onClick={() => setStreaming((v) => !v)}
          className={`px-2 py-1 text-xs rounded ${
            streaming
              ? "bg-emerald-700/40 text-emerald-200"
              : "bg-amber-700/40 text-amber-200"
          }`}
          title={streaming ? "Pause stream" : "Resume stream"}
        >
          {streaming ? "⏸ live" : "▶ paused"}
        </button>
        <label className="flex items-center gap-1 text-xs text-slate-300">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          auto-scroll
        </label>
        <select
          value={level}
          onChange={(e) => setLevel(e.target.value as LogLevel | "")}
          className="text-xs bg-slate-800 text-slate-200 rounded px-1 py-0.5"
        >
          <option value="">all levels</option>
          {(sources?.levels || []).map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
        <select
          value={source}
          onChange={(e) => setSource(e.target.value as LogSource | "")}
          className="text-xs bg-slate-800 text-slate-200 rounded px-1 py-0.5"
        >
          <option value="">all sources</option>
          {(sources?.sources || []).map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="search…"
          className="text-xs bg-slate-800 text-slate-200 rounded px-2 py-0.5 flex-1 min-w-[10rem]"
        />
        {search && (
          <button
            onClick={() => setSearch("")}
            className="text-xs text-slate-400 hover:text-slate-200"
          >
            clear
          </button>
        )}
        <button
          onClick={() => downloadLogs("jsonl")}
          className="px-2 py-1 text-xs bg-slate-800 text-slate-200 rounded"
          title="Download as JSONL"
        >
          ⬇ jsonl
        </button>
        <button
          onClick={() => downloadLogs("txt")}
          className="px-2 py-1 text-xs bg-slate-800 text-slate-200 rounded"
          title="Download as plain text"
        >
          ⬇ txt
        </button>
        <button
          onClick={onCopy}
          className="px-2 py-1 text-xs bg-slate-800 text-slate-200 rounded"
          title="Copy filtered view to clipboard"
        >
          ⧉ copy
        </button>
        <span className="text-xs text-slate-500 ml-auto">
          {filtered.length} / {events.length}
        </span>
      </div>
      <div
        ref={listRef}
        onScroll={onScroll}
        className="flex-1 overflow-auto font-mono text-[11px] leading-[1.45] bg-slate-950/60"
        style={{ minHeight: "400px" }}
      >
        {visible.length === 0 ? (
          <div className="p-4 text-slate-500 text-center">
            no log events match current filters
          </div>
        ) : (
          visible.map((e, idx) => (
            <div
              key={`${e.ts}-${idx}`}
              className={`flex gap-2 px-3 py-0.5 hover:bg-slate-800/40 ${LEVEL_STYLES[e.level] || ""}`}
            >
              <span className="text-slate-500 shrink-0">{fmtTs(e.ts)}</span>
              <span
                className={`px-1 rounded text-[10px] uppercase tracking-wider shrink-0 ${SOURCE_STYLES[e.source] || ""}`}
              >
                {e.source}
              </span>
              <span className="text-slate-400 shrink-0">{e.level.toUpperCase()}</span>
              <span className="text-slate-400 shrink-0 max-w-[14rem] truncate">
                {e.logger}
              </span>
              <span className="break-all">{e.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
