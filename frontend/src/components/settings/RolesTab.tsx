import { useEffect, useState } from "react";
import {
  Role,
  RolesState,
  ScheduledMessage,
  cancelScheduledMessage,
  fetchRoles,
  forgetSpeaker,
  fetchRelationships,
  fetchScheduledMessages,
  fetchTelegramContacts,
  setRelationships,
  setRole,
} from "../../api";

type Props = { flash: (msg: string) => void };

const ROLE_COLOUR: Record<Role, string> = {
  owner: "bg-emerald-700/60 text-emerald-100",
  trusted: "bg-sky-700/60 text-sky-100",
  guest: "bg-slate-700/60 text-slate-200",
};

const CONFIRM_TAIL =
  "\n\nThey keep no special access. If they message again they come back as a guest, which is what anyone unknown gets.";

const ROLE_ORDER: Role[] = ["owner", "trusted", "guest"];

export default function RolesTab({ flash }: Props) {
  const [roles, setRoles] = useState<RolesState | null>(null);
  const [relationships, setRels] = useState<Record<string, string>>({});
  const [scheduled, setScheduled] = useState<ScheduledMessage[]>([]);
  const [tgContacts, setTgContacts] = useState<
    Record<string, { chat_id: number; username?: string; label?: string; last_seen?: string }>
  >({});
  const [busy, setBusy] = useState(false);

  // New-alias form state
  const [newAlias, setNewAlias] = useState("");
  const [newTarget, setNewTarget] = useState("");

  const refresh = async () => {
    try {
      const [r, rel, sch, tg] = await Promise.all([
        fetchRoles(),
        fetchRelationships(),
        fetchScheduledMessages(),
        fetchTelegramContacts(),
      ]);
      setRoles(r);
      setRels(rel.relationships);
      setScheduled(sch.messages);
      setTgContacts(tg.contacts);
    } catch (e: any) {
      flash("Load failed: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleForget = async (sid: string, label: string) => {
    const who = label ? `${label} (${sid})` : sid;
    if (
      !confirm(
        `Remove ${who} from the list?` + CONFIRM_TAIL
      )
    )
      return;
    setBusy(true);
    try {
      await forgetSpeaker(sid);
      flash("Removed " + who);
      await refresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleRoleChange = async (speaker_id: string, role: Role, label?: string) => {
    setBusy(true);
    try {
      await setRole(speaker_id, role, label);
      flash(`Set ${speaker_id} → ${role}`);
      await refresh();
    } catch (e: any) {
      flash("Set role failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleAddRelationship = async () => {
    const alias = newAlias.trim().toLowerCase();
    const target = newTarget.trim();
    if (!alias || !target) {
      flash("Alias and target speaker_id required");
      return;
    }
    setBusy(true);
    try {
      const next = { ...relationships, [alias]: target };
      await setRelationships(next);
      setRels(next);
      setNewAlias("");
      setNewTarget("");
      flash(`Added alias "${alias}" → ${target}`);
    } catch (e: any) {
      flash("Save failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteRelationship = async (alias: string) => {
    if (!confirm(`Remove alias "${alias}"?`)) return;
    setBusy(true);
    try {
      const next = { ...relationships };
      delete next[alias];
      await setRelationships(next);
      setRels(next);
      flash(`Removed "${alias}"`);
    } catch (e: any) {
      flash("Save failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleCancelScheduled = async (id: string) => {
    if (!confirm("Cancel this scheduled message?")) return;
    setBusy(true);
    try {
      await cancelScheduledMessage(id);
      flash("Cancelled");
      await refresh();
    } catch (e: any) {
      flash("Cancel failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  // Build a unified speaker list:  every speaker in roles.speakers
  // (explicitly classified), every seen_speakers entry, plus all
  // owner_speaker_ids. De-duplicated; rendered with current role.
  const allSpeakers: string[] = (() => {
    const set = new Set<string>();
    if (roles) {
      roles.owner_speaker_ids.forEach((s) => set.add(s));
      Object.keys(roles.speakers).forEach((s) => set.add(s));
      roles.seen_speakers.forEach((s) => set.add(s.speaker_id));
    }
    return Array.from(set);
  })();

  const roleOf = (sid: string): Role => {
    if (!roles) return "guest";
    if (roles.owner_speaker_ids.includes(sid)) return "owner";
    const entry = roles.speakers[sid];
    if (entry?.role && ROLE_ORDER.includes(entry.role)) return entry.role;
    return "guest";
  };

  const labelOf = (sid: string): string =>
    (roles?.speakers[sid]?.label) || "";

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs text-slate-400">
          Each speaker (you on the WebUI, each Telegram user, future channels)
          gets a role. Owner can do anything (self-modification, code, configs,
          messages). Trusted can chat freely and schedule messages to the
          owner. Guest is chat-only. The agent's system prompt + every
          dangerous tool enforce this server-side.
        </p>
      </div>

      {/* --- Speakers + Roles --- */}
      <div>
        <h4 className="font-semibold mb-2 text-sm">Speakers</h4>
        <div className="bg-slate-800 rounded overflow-hidden">
          <div className="grid grid-cols-[1fr_120px_2fr_auto] gap-3 px-3 py-2 text-[10px] uppercase tracking-wide text-slate-400 bg-slate-900/50">
            <div>speaker_id</div>
            <div>role</div>
            <div>label</div>
            <div>actions</div>
          </div>
          {allSpeakers.length === 0 && (
            <div className="px-3 py-4 text-xs text-slate-500">
              No speakers yet — once someone messages the bot or you chat
              from the WebUI, they appear here.
            </div>
          )}
          {allSpeakers.map((sid) => {
            const role = roleOf(sid);
            const label = labelOf(sid);
            return (
              <div
                key={sid}
                className="grid grid-cols-[1fr_120px_2fr_auto] gap-3 px-3 py-2 text-xs items-center border-t border-slate-700/50"
              >
                <div className="font-mono truncate">{sid}</div>
                <div>
                  <span className={`text-[10px] uppercase rounded px-1.5 py-0.5 ${ROLE_COLOUR[role]}`}>
                    {role}
                  </span>
                </div>
                <div>
                  <input
                    defaultValue={label}
                    placeholder="(label — e.g. Wife, Gor, …)"
                    onBlur={(e) => {
                      const v = e.target.value.trim();
                      if (v !== label) {
                        handleRoleChange(sid, role, v);
                      }
                    }}
                    className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none focus:ring-1 focus:ring-sky-600"
                  />
                </div>
                <div className="flex items-center gap-1">
                  {ROLE_ORDER.map((r) => (
                    <button
                      key={r}
                      onClick={() => handleRoleChange(sid, r, label || undefined)}
                      disabled={busy || r === role}
                      className={`text-[10px] rounded px-2 py-0.5 ${
                        r === role
                          ? "bg-slate-700 opacity-40"
                          : "bg-slate-700 hover:bg-slate-600"
                      }`}
                    >
                      {r}
                    </button>
                  ))}
                  {/* Demoting to guest left the row on the list forever,
                      which is right for someone you still talk to and
                      wrong for a stale entry. Removing is not a ban: an
                      unknown speaker is a guest, so they return as one. */}
                  <button
                    onClick={() => handleForget(sid, label)}
                    disabled={busy || sid === "webui:default"}
                    title={
                      sid === "webui:default"
                        ? "This is you — the local owner cannot be removed"
                        : "Remove from the list"
                    }
                    className="ml-1 rounded px-1.5 py-0.5 text-[10px] text-ink-faint hover:bg-danger hover:text-white disabled:opacity-25 disabled:hover:bg-transparent disabled:hover:text-ink-faint"
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* --- Relationships --- */}
      <div>
        <h4 className="font-semibold mb-2 text-sm">Relationships (aliases)</h4>
        <p className="text-[11px] text-slate-400 mb-2">
          Map a friendly name to a speaker_id so you can say "remind my wife
          to call me" instead of "remind telegram:222". Owner-curated.
        </p>
        <div className="bg-slate-800 rounded overflow-hidden">
          <div className="grid grid-cols-[1fr_2fr_auto] gap-3 px-3 py-2 text-[10px] uppercase tracking-wide text-slate-400 bg-slate-900/50">
            <div>alias</div>
            <div>speaker_id</div>
            <div></div>
          </div>
          {Object.entries(relationships).length === 0 && (
            <div className="px-3 py-3 text-xs text-slate-500">
              No aliases yet. Add one below.
            </div>
          )}
          {Object.entries(relationships).map(([alias, target]) => (
            <div
              key={alias}
              className="grid grid-cols-[1fr_2fr_auto] gap-3 px-3 py-2 text-xs items-center border-t border-slate-700/50"
            >
              <div className="font-semibold">{alias}</div>
              <div className="font-mono text-slate-300 truncate">{target}</div>
              <button
                onClick={() => handleDeleteRelationship(alias)}
                disabled={busy}
                className="bg-rose-800/70 hover:bg-rose-700 rounded px-2 py-0.5 text-[10px]"
              >
                Remove
              </button>
            </div>
          ))}
          <div className="grid grid-cols-[1fr_2fr_auto] gap-3 px-3 py-2 border-t border-slate-700/50">
            <input
              value={newAlias}
              onChange={(e) => setNewAlias(e.target.value)}
              placeholder="alias (e.g. wife)"
              className="bg-slate-900 rounded px-2 py-1 text-xs outline-none focus:ring-1 focus:ring-sky-600"
            />
            <input
              value={newTarget}
              onChange={(e) => setNewTarget(e.target.value)}
              placeholder="speaker_id (e.g. telegram:222)"
              className="bg-slate-900 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
            />
            <button
              onClick={handleAddRelationship}
              disabled={busy}
              className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1 text-xs"
            >
              Add
            </button>
          </div>
        </div>
      </div>

      {/* --- Telegram contacts (read-only) --- */}
      <div>
        <h4 className="font-semibold mb-2 text-sm">Telegram contacts</h4>
        <p className="text-[11px] text-slate-400 mb-2">
          Captured automatically each time a Telegram user messages the bot.
          Used by scheduled messages to deliver back to the right chat_id.
        </p>
        <div className="bg-slate-800 rounded overflow-hidden">
          {Object.entries(tgContacts).length === 0 ? (
            <div className="px-3 py-3 text-xs text-slate-500">
              No Telegram contacts yet. They appear here automatically.
            </div>
          ) : (
            Object.entries(tgContacts).map(([uid, c]) => (
              <div
                key={uid}
                className="grid grid-cols-[1fr_1fr_1fr_1fr] gap-3 px-3 py-2 text-xs border-t border-slate-700/50"
              >
                <div className="font-mono">telegram:{uid}</div>
                <div className="text-slate-300">{c.label || c.username || "—"}</div>
                <div className="font-mono text-slate-400">chat_id={c.chat_id}</div>
                <div className="text-slate-500 text-[10px]">{c.last_seen || ""}</div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* --- Scheduled messages --- */}
      <div>
        <h4 className="font-semibold mb-2 text-sm">Scheduled messages</h4>
        <div className="bg-slate-800 rounded overflow-hidden">
          {scheduled.length === 0 ? (
            <div className="px-3 py-3 text-xs text-slate-500">
              No scheduled messages yet. Ask the agent: "remind my wife to
              call me at 10am tomorrow".
            </div>
          ) : (
            scheduled.map((m) => (
              <div
                key={m.id}
                className="px-3 py-2 text-xs border-t border-slate-700/50 space-y-1"
              >
                <div className="flex items-center gap-2 flex-wrap">
                  <span
                    className={`text-[10px] uppercase rounded px-1.5 py-0.5 ${
                      m.status === "pending"
                        ? "bg-amber-700/60"
                        : m.status === "sent"
                        ? "bg-emerald-700/60"
                        : m.status === "failed"
                        ? "bg-rose-700/60"
                        : "bg-slate-700/60"
                    }`}
                  >
                    {m.status}
                  </span>
                  <span className="text-slate-400">due {m.due_at}</span>
                  <span className="font-mono text-slate-300 ml-auto">{m.id}</span>
                  {m.status === "pending" && (
                    <button
                      onClick={() => handleCancelScheduled(m.id)}
                      disabled={busy}
                      className="bg-rose-800/70 hover:bg-rose-700 rounded px-2 py-0.5 text-[10px]"
                    >
                      Cancel
                    </button>
                  )}
                </div>
                <div className="text-slate-300 break-words">
                  → <span className="font-mono">{m.target_speaker}</span>: {m.text}
                </div>
                {m.last_error && (
                  <div className="text-rose-300 text-[10px]">⚠ {m.last_error}</div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
