// Reminders tab (2026-06-12). The agent could already schedule
// messages from chat ("напомни через 2 дня проверить заказ") via the
// schedule_message tool; this tab gives the owner a Settings surface
// to see the ledger, create reminders with an explicit picker, and
// cancel pending ones.
import { useEffect, useMemo, useState } from "react";
import {
  ScheduledMessage,
  cancelScheduledMessage,
  createScheduledMessage,
  fetchScheduledMessages,
} from "../../api";

const STATUS_STYLE: Record<string, string> = {
  pending: "bg-sky-900/60 text-sky-300",
  delivering: "bg-amber-900/60 text-amber-300",
  sent: "bg-emerald-900/60 text-emerald-300",
  failed: "bg-rose-900/60 text-rose-300",
  cancelled: "bg-slate-700 text-slate-400",
};

function localTime(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

// datetime-local gives local wall-clock; the API wants ISO UTC Z.
function toUtcZ(local: string): string {
  const d = new Date(local);
  return d.toISOString().replace(/\.\d{3}Z$/, "Z");
}

export default function RemindersTab({ flash }: { flash: (m: string) => void }) {
  const [rows, setRows] = useState<ScheduledMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>("");

  const [text, setText] = useState("");
  const [when, setWhen] = useState<string>("custom");
  const [customDt, setCustomDt] = useState("");
  const [creating, setCreating] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetchScheduledMessages(statusFilter || undefined);
      setRows(r.messages || []);
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [statusFilter]);

  const quickPicks: { id: string; label: string; minutes?: number }[] = [
    { id: "60", label: "in 1 hour", minutes: 60 },
    { id: "1440", label: "tomorrow (24h)", minutes: 1440 },
    { id: "2880", label: "in 2 days", minutes: 2880 },
    { id: "custom", label: "pick date & time" },
  ];

  const create = async () => {
    const t = text.trim();
    if (!t) { flash("Reminder text is required"); return; }
    setCreating(true);
    try {
      if (when === "custom") {
        if (!customDt) { flash("Pick a date & time"); return; }
        await createScheduledMessage({ text: t, due_at: toUtcZ(customDt) });
      } else {
        await createScheduledMessage({
          text: t, delay_minutes: parseInt(when, 10),
        });
      }
      setText("");
      flash("Reminder scheduled");
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setCreating(false);
    }
  };

  const pendingFirst = useMemo(() => {
    const order: Record<string, number> = {
      pending: 0, delivering: 1, failed: 2, sent: 3, cancelled: 4,
    };
    return [...rows].sort((a, b) => {
      const so = (order[a.status] ?? 9) - (order[b.status] ?? 9);
      if (so !== 0) return so;
      return (b.due_at || "").localeCompare(a.due_at || "");
    });
  }, [rows]);

  return (
    <div className="space-y-4">
      {/* Create */}
      <div className="bg-slate-800 rounded p-4 space-y-3">
        <div className="font-semibold text-sm">New reminder</div>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder='e.g. "Check the order status"'
          className="w-full bg-slate-900 rounded px-3 py-2 text-sm"
        />
        <div className="flex gap-2 flex-wrap items-center text-xs">
          {quickPicks.map((q) => (
            <button
              key={q.id}
              onClick={() => setWhen(q.id)}
              className={`rounded px-3 py-1 ${
                when === q.id
                  ? "bg-sky-700 text-white"
                  : "bg-slate-700 hover:bg-slate-600"
              }`}
            >
              {q.label}
            </button>
          ))}
          {when === "custom" && (
            <input
              type="datetime-local"
              value={customDt}
              onChange={(e) => setCustomDt(e.target.value)}
              className="bg-slate-900 rounded px-2 py-1"
            />
          )}
          <button
            onClick={create}
            disabled={creating}
            className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 rounded px-4 py-1 ml-auto"
          >
            {creating ? "Scheduling…" : "Schedule"}
          </button>
        </div>
        <div className="text-xs text-slate-500">
          Delivered to this WebUI chat (and Telegram if the channel is
          linked). You can also just ask the agent in chat: “напомни
          через 2 дня проверить заказ”.
        </div>
      </div>

      {/* Ledger */}
      <div className="bg-slate-800 rounded p-4 space-y-2">
        <div className="flex items-center justify-between">
          <div className="font-semibold text-sm">Scheduled ({rows.length})</div>
          <div className="flex gap-2 text-xs items-center">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-900 rounded px-2 py-1"
            >
              <option value="">all statuses</option>
              <option value="pending">pending</option>
              <option value="sent">sent</option>
              <option value="failed">failed</option>
              <option value="cancelled">cancelled</option>
            </select>
            <button
              onClick={load}
              className="bg-slate-700 hover:bg-slate-600 rounded px-2 py-1"
            >
              ↻
            </button>
          </div>
        </div>
        {loading && <div className="text-xs text-slate-500">loading…</div>}
        {!loading && pendingFirst.length === 0 && (
          <div className="text-xs text-slate-500">No reminders yet.</div>
        )}
        <div className="space-y-1">
          {pendingFirst.map((m) => (
            <div
              key={m.id}
              className="flex items-center gap-3 bg-slate-900 rounded px-3 py-2 text-xs"
            >
              <span
                className={`rounded px-1.5 py-0.5 shrink-0 ${
                  STATUS_STYLE[m.status] || "bg-slate-700"
                }`}
              >
                {m.status}
              </span>
              <div className="flex-1 min-w-0">
                <div className="truncate text-slate-200">{m.text}</div>
                <div className="text-slate-500">
                  due {localTime(m.due_at)}
                  {m.status === "sent" && m.delivered_at && (
                    <> · delivered {localTime(m.delivered_at)}</>
                  )}
                  {m.status === "failed" && m.last_error && (
                    <> · <span className="text-rose-400">{m.last_error.slice(0, 80)}</span></>
                  )}
                  {" "}· to {m.target_speaker}
                </div>
              </div>
              {m.status === "pending" && (
                <button
                  onClick={async () => {
                    try {
                      await cancelScheduledMessage(m.id);
                      flash("Cancelled");
                      load();
                    } catch (e: any) {
                      flash("Error: " + e.message);
                    }
                  }}
                  className="bg-rose-900 hover:bg-rose-800 rounded px-2 py-1 shrink-0"
                >
                  Cancel
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
