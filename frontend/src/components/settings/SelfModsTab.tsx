import { useEffect, useState } from "react";
import {
  SelfModArchive,
  SelfModPatch,
  fetchSelfModHistory,
  fetchSelfMods,
  restoreSelfModArchive,
  restoreSelfModFromHistory,
  revertAllSelfMods,
  revertSelfMod,
} from "../../api";

type Props = { flash: (msg: string) => void };

const STATUS_COLOURS: Record<SelfModPatch["status"], string> = {
  applied: "bg-emerald-700/50 text-emerald-200",
  needs_review: "bg-amber-700/60 text-amber-100",
  reverted: "bg-slate-700/50 text-slate-300",
};

export default function SelfModsTab({ flash }: Props) {
  const [patches, setPatches] = useState<SelfModPatch[] | null>(null);
  const [archives, setArchives] = useState<SelfModArchive[] | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const [a, h] = await Promise.all([fetchSelfMods(), fetchSelfModHistory()]);
      setPatches(a.patches);
      setArchives(h.archives);
    } catch (e: any) {
      flash("Self-mods load failed: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleRestoreOne = async (archive_id: string, patch_filename: string, title: string) => {
    if (!confirm(`Re-apply "${title}" as a fresh active modification? The archive entry stays intact.`)) {
      return;
    }
    setBusy(true);
    try {
      await restoreSelfModFromHistory(archive_id, patch_filename);
      flash(`Re-applied: ${title}`);
      await refresh();
    } catch (e: any) {
      flash("Restore failed: " + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const handleRestoreBatch = async (a: SelfModArchive) => {
    if (
      !confirm(
        `Re-apply ALL ${a.patch_count} patch(es) from archive ${a.archived_at}? ` +
          "Stops at the first conflict (engine stays safe; later patches stay archived).",
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      const r = await restoreSelfModArchive(a.archive_id);
      if (r.failed_at) {
        flash(
          `Restored ${r.restored.length} patch(es); stopped at ${r.failed_at}: ${r.error}`,
        );
      } else {
        flash(`Restored all ${r.restored.length} patch(es) from ${a.archived_at}`);
      }
      await refresh();
    } catch (e: any) {
      flash("Batch restore failed: " + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const handleRevert = async (p: SelfModPatch) => {
    const isTip = patches && patches[patches.length - 1].id === p.id;
    const warning = isTip
      ? `Revert "${p.title}"? This reverse-applies the patch and removes it from the manifest.`
      : `Revert "${p.title}"? This patch is NOT the most recent — later patches were stacked on top and may conflict. The engine could end up in an inconsistent state. Proceed?`;
    if (!confirm(warning)) return;
    setBusy(true);
    try {
      await revertSelfMod(p.id);
      flash(`Reverted: ${p.title}`);
      await refresh();
    } catch (e: any) {
      flash("Revert failed: " + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const handleRevertAll = async () => {
    if (
      !confirm(
        "Revert ALL self-modifications and reset the engine to origin/master? " +
          "This is destructive: every local patch will be deleted. " +
          "Your user data (knowledge, workspace, settings) is unaffected.",
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      await revertAllSelfMods();
      flash("All self-mods reverted; engine reset to origin/master.");
      await refresh();
    } catch (e: any) {
      flash("Revert all failed: " + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const active = (patches || []).filter((p) => p.status !== "reverted");

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          <button
            onClick={refresh}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Refresh
          </button>
          <button
            onClick={handleRevertAll}
            disabled={busy || active.length === 0}
            className="bg-rose-800/70 hover:bg-rose-700 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Revert all → official
          </button>
        </div>
      </div>

      <div className="bg-slate-800/60 border border-slate-700 rounded p-3 text-[11px] text-slate-400 space-y-1">
        <div>
          Local modifications to your installed copy of Hrant. Each entry is
          a unified diff in{" "}
          <span className="font-mono">~/.hrant/data/self_mods/</span>; nothing
          ever leaves your machine.
        </div>
        <div>
          ⚠ On <span className="font-mono">hrant update</span>, every active
          self-mod is <b>archived</b> into{" "}
          <span className="font-mono">~/.hrant/data/self_mods/history/&lt;ts&gt;/</span>{" "}
          and the engine is reset to the official version. The archive
          history is kept so you can re-apply patches afterwards from the
          <b> History</b> section below.
        </div>
      </div>

      {patches === null && <div className="text-slate-500 text-sm">Loading…</div>}
      {patches && patches.length === 0 && (
        <div className="bg-slate-800 rounded p-6 text-center text-slate-400">
          No self-modifications yet. When you ask the agent to modify its own
          code, the change will be recorded here.
        </div>
      )}

      {patches && patches.length > 0 && (
        <div className="space-y-2">
          {patches.map((p) => (
            <div
              key={p.id}
              className="bg-slate-800 rounded p-3 space-y-1 text-sm"
            >
              <div className="flex items-center gap-2 flex-wrap">
                <span
                  className={`text-[10px] uppercase tracking-wide rounded px-1.5 py-0.5 ${STATUS_COLOURS[p.status]}`}
                >
                  {p.status}
                </span>
                <span className="font-mono text-xs text-slate-400">{p.id}</span>
                <span className="font-semibold">{p.title || p.slug}</span>
                <span className="text-xs text-slate-400 ml-auto">{p.created}</span>
              </div>
              <div className="text-xs text-slate-400 font-mono">
                file: {p.file} · patch: {p.patch_filename}
              </div>
              {p.last_error && (
                <div className="text-xs text-rose-300 font-mono break-all">
                  last error: {p.last_error}
                </div>
              )}
              {p.status !== "reverted" && (
                <div className="flex gap-2 pt-1">
                  <button
                    onClick={() => handleRevert(p)}
                    disabled={busy}
                    className="bg-rose-800/70 hover:bg-rose-700 disabled:opacity-40 rounded px-3 py-1 text-xs"
                  >
                    Revert
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* History — archived patches from past `hrant update` events */}
      <div className="pt-4 border-t border-slate-800">
        <h4 className="font-semibold mb-2">History (archived self-mods)</h4>
        <div className="bg-slate-800/40 border border-slate-700 rounded p-3 text-[11px] text-slate-400 mb-3">
          Each <span className="font-mono">hrant update</span> archives the
          active self-mods into a timestamped bundle. Use{" "}
          <b>Re-apply</b> to apply one patch as a fresh active modification,
          or <b>Re-apply all</b> to attempt the whole bundle in order
          (stops at the first conflict).
        </div>
        {archives === null && (
          <div className="text-slate-500 text-sm">Loading…</div>
        )}
        {archives && archives.length === 0 && (
          <div className="bg-slate-800 rounded p-6 text-center text-slate-400">
            No archives yet. Archives appear here after every{" "}
            <span className="font-mono">hrant update</span> that had active
            self-mods.
          </div>
        )}
        {archives && archives.length > 0 && (
          <div className="space-y-3">
            {archives.map((a) => (
              <div
                key={a.archive_id}
                className="bg-slate-800 rounded p-3 space-y-2 text-sm"
              >
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-mono text-xs text-slate-400">
                    {a.archived_at}
                  </span>
                  <span className="text-xs text-slate-400">
                    · {a.patch_count} patch(es)
                  </span>
                  <button
                    onClick={() => handleRestoreBatch(a)}
                    disabled={busy || a.patch_count === 0}
                    className="ml-auto bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1 text-xs"
                  >
                    Re-apply all ({a.patch_count})
                  </button>
                </div>
                <div className="space-y-1">
                  {a.entries.map((e) => (
                    <div
                      key={e.patch_filename}
                      className="flex items-center gap-2 text-xs bg-slate-900/50 rounded px-2 py-1"
                    >
                      <span className="font-semibold flex-1 truncate">
                        {e.title || e.slug}
                      </span>
                      <span className="text-slate-500 font-mono truncate">
                        {e.file}
                      </span>
                      <button
                        onClick={() =>
                          handleRestoreOne(
                            a.archive_id,
                            e.patch_filename,
                            e.title || e.slug,
                          )
                        }
                        disabled={busy}
                        className="bg-sky-800/70 hover:bg-sky-700 disabled:opacity-40 rounded px-2 py-0.5 text-[10px]"
                      >
                        Re-apply
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
