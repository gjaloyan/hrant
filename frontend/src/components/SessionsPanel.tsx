/** Past conversations.
 *
 * The old panel put a 320px list beside a 950px pane that said "Select a
 * session", stacked two full-width filter rows above the list so the list
 * itself overflowed sideways, and printed `telegram:1358056500` where a
 * name belongs. Every card said "active" twice — once as a badge, once as
 * text — and showed a bare "30%" that named nothing.
 *
 * Rebuilt around the two questions this screen actually answers: which
 * conversation was that, and what was said in it.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  archiveSessions,
  deleteSession,
  fetchSession,
  fetchSessions,
  fetchSessionStats,
  fetchSpeakers,
  fetchThreads,
  type SessionDetail,
  type SessionStats,
  type SessionSummary,
  type SpeakerSummary,
  type ThreadSummary,
} from "../api";
import { Badge, Button, EmptyState, Flash, Spinner, cx } from "../ui";
import { Speaker, nameOf, useSpeakers } from "../ui/speakers";

/** "2h ago" beats a UTC stamp for scanning; the exact time is one hover
 *  away, and the browser renders it in the reader's own zone. */
function ago(iso: string): string {
  const d = new Date((iso || "").replace(" ", "T"));
  if (isNaN(d.getTime())) return iso || "—";
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (mins < 60 * 24) return `${Math.round(mins / 60)}h ago`;
  const days = Math.round(mins / 1440);
  if (days < 30) return `${days}d ago`;
  return d.toLocaleDateString();
}

const exact = (iso: string) => {
  const d = new Date((iso || "").replace(" ", "T"));
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
};

function ConfidenceTag({ value }: { value: number }) {
  if (!value) return null;
  return (
    <Badge
      tone={value >= 90 ? "ok" : value >= 70 ? "warn" : "danger"}
      title="How sure the agent was that it had verified its own answers in this conversation."
    >
      self-check {Math.round(value)}%
    </Badge>
  );
}

export default function SessionsPanel() {
  const speakerMap = useSpeakers();

  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [selected, setSelected] = useState<SessionDetail | null>(null);

  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [msg, setMsg] = useState("");
  const [q, setQ] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [thread, setThread] = useState("");
  const [showArchived, setShowArchived] = useState(false);

  const flash = (t: string) => {
    setMsg(t);
    setTimeout(() => setMsg(""), 3000);
  };

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, sp, th, st] = await Promise.all([
        fetchSessions(showArchived, speaker || undefined, thread || undefined),
        fetchSpeakers(),
        fetchThreads(),
        fetchSessionStats(),
      ]);
      setSessions((s as any).sessions || []);
      setSpeakers((sp as any).speakers || []);
      setThreads((th as any).threads || []);
      setStats(st);
    } catch (e: any) {
      flash("Error: " + (e.message || "could not load sessions"));
    } finally {
      setLoading(false);
    }
  }, [speaker, thread, showArchived]);

  useEffect(() => {
    load();
  }, [load]);

  const open = async (id: string) => {
    setLoadingDetail(true);
    try {
      const r = await fetchSession(id);
      setSelected((r as any).session || (r as any));
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setLoadingDetail(false);
    }
  };

  const remove = async (s: SessionSummary) => {
    // Name what is about to go. "Delete this session?" tells the reader
    // nothing about which one they clicked.
    const label = s.title || `${nameOf(s.speaker_id, speakerMap)}, ${ago(s.started)}`;
    if (!confirm(`Delete "${label}"?\n\nThe transcript is removed permanently.`))
      return;
    try {
      await deleteSession(s.id);
      if (selected?.id === s.id) setSelected(null);
      flash("Session deleted");
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const cleanUp = async () => {
    const days = 30;
    if (
      !confirm(
        `Archive every session older than ${days} days?\n\n` +
          "Archived sessions stay readable — they are hidden from this list " +
          "unless you tick “Include archived”.",
      )
    )
      return;
    try {
      const r = await archiveSessions(days);
      flash(`Archived ${(r as any).archived} session(s)`);
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const visible = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return sessions;
    return sessions.filter((s) =>
      `${s.title} ${nameOf(s.speaker_id, speakerMap)} ${s.thread_label}`
        .toLowerCase()
        .includes(needle),
    );
  }, [sessions, q, speakerMap]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* One toolbar instead of two stacked filter rows inside the list. */}
      <div className="flex flex-wrap items-center gap-2 border-b border-edge px-3 py-2">
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search conversations…"
          className="min-w-[10rem] flex-1 text-sm"
          aria-label="Search conversations"
        />
        <select
          value={speaker}
          onChange={(e) => setSpeaker(e.target.value)}
          className="text-sm"
          aria-label="Filter by person"
        >
          <option value="">Everyone</option>
          {speakers.map((s) => (
            <option key={s.speaker_id} value={s.speaker_id}>
              {nameOf(s.speaker_id, speakerMap)} ({s.session_count})
            </option>
          ))}
        </select>
        <select
          value={thread}
          onChange={(e) => setThread(e.target.value)}
          className="text-sm"
          aria-label="Filter by chat"
        >
          <option value="">All chats</option>
          {threads.map((t) => (
            <option key={t.session_key} value={t.session_key}>
              {t.thread_label} ({t.session_count})
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1.5 text-xs text-ink-dim">
          <input
            type="checkbox"
            checked={showArchived}
            onChange={(e) => setShowArchived(e.target.checked)}
            className="h-3.5 w-3.5 accent-accent"
          />
          Include archived
        </label>
        <span className="ml-auto flex items-center gap-2 text-xs text-ink-faint">
          {stats && (
            <span title="Total across every speaker">
              {stats.total_sessions} conversations · {stats.total_turns} turns
            </span>
          )}
          <Button
            kind="ghost"
            size="sm"
            onClick={cleanUp}
            title="Hide conversations older than 30 days from this list"
          >
            Clean up
          </Button>
        </span>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* List — wider than before, and it no longer scrolls sideways. */}
        <div className="w-[22rem] shrink-0 overflow-y-auto border-r border-edge">
          {loading && (
            <div className="flex items-center gap-2 p-4 text-sm text-ink-dim">
              <Spinner /> Loading…
            </div>
          )}
          {!loading && visible.length === 0 && (
            <EmptyState title="Nothing here">
              {q || speaker || thread
                ? "No conversation matches these filters."
                : "Conversations appear here once you have talked to the agent."}
            </EmptyState>
          )}
          <ul className="divide-y divide-edge">
            {visible.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => open(s.id)}
                  className={cx(
                    "w-full px-3 py-2.5 text-left",
                    selected?.id === s.id
                      ? "bg-accent-soft"
                      : "hover:bg-surface-hover",
                  )}
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="truncate text-sm font-medium">
                      {s.title || "(untitled)"}
                    </span>
                    <span
                      className="shrink-0 text-[11px] text-ink-faint"
                      title={exact(s.started)}
                    >
                      {ago(s.started)}
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink-dim">
                    <Speaker id={s.speaker_id} />
                    <span className="text-ink-faint">·</span>
                    <span className="truncate">{s.thread_label}</span>
                    <span className="text-ink-faint">·</span>
                    <span>
                      {s.turn_count} turn{s.turn_count === 1 ? "" : "s"}
                    </span>
                    {/* One status, not two. Archived is the exception worth
                        marking; "active" on every row is noise. */}
                    {s.archived && <Badge>archived</Badge>}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Detail */}
        <div className="min-w-0 flex-1 overflow-y-auto">
          {loadingDetail && (
            <div className="flex items-center gap-2 p-6 text-sm text-ink-dim">
              <Spinner /> Opening…
            </div>
          )}
          {!loadingDetail && !selected && (
            <EmptyState
              icon="🗂"
              title="Pick a conversation"
            >
              Every turn is kept: what was asked, what the agent answered, and
              how sure it was.
            </EmptyState>
          )}
          {!loadingDetail && selected && (
            <div className="mx-auto max-w-3xl px-4 py-5 sm:px-6">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h2 className="text-lg font-semibold">
                    {selected.title || "(untitled)"}
                  </h2>
                  <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-dim">
                    <Speaker id={selected.speaker_id} showRole />
                    <span className="text-ink-faint">·</span>
                    <span title={`session_key: ${selected.session_key}`}>
                      {selected.thread_label}
                    </span>
                    <span className="text-ink-faint">·</span>
                    <span title={exact(selected.started)}>
                      {ago(selected.started)}
                    </span>
                    <ConfidenceTag value={selected.avg_confidence} />
                  </div>
                </div>
                <Button
                  kind="danger"
                  size="sm"
                  onClick={() =>
                    remove(selected as unknown as SessionSummary)
                  }
                >
                  Delete
                </Button>
              </div>

              {selected.topics_used?.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {selected.topics_used.map((t) => (
                    <Badge key={t} tone="neutral" title="Knowledge topic used">
                      {t}
                    </Badge>
                  ))}
                </div>
              )}

              <div className="mt-5 space-y-4">
                {selected.turns.length === 0 && (
                  <p className="text-sm text-ink-faint">
                    This conversation has no turns recorded.
                  </p>
                )}
                {selected.turns.map((turn: any, i: number) => (
                  <div key={i} className="space-y-2">
                    <div className="flex justify-end">
                      <div className="max-w-[85%] rounded-xl2 rounded-br-sm bg-accent-soft px-3 py-2 text-sm whitespace-pre-wrap">
                        {turn.user}
                      </div>
                    </div>
                    <div className="flex justify-start">
                      <div className="max-w-[90%] rounded-xl2 rounded-bl-sm border border-edge bg-surface px-3 py-2 text-sm whitespace-pre-wrap">
                        {turn.answer}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 pl-1 text-[11px] text-ink-faint">
                      <span title={exact(turn.ts)}>{ago(turn.ts)}</span>
                      {turn.intent && <Badge>{turn.intent}</Badge>}
                      <ConfidenceTag value={turn.confidence} />
                      {turn.topics?.length > 0 && (
                        <span>used: {turn.topics.join(", ")}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <Flash text={msg} />
    </div>
  );
}
