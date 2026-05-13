import { useEffect, useState } from "react";
import {
  SelfModPatch,
  fetchSelfMods,
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
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const r = await fetchSelfMods();
      setPatches(r.patches);
    } catch (e: any) {
      flash("Self-mods load failed: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

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
  const needsReview = (patches || []).filter((p) => p.status === "needs_review");

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-bold">Self-Modifications</h3>
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
          Local modifications to the engine. Each entry is a unified diff in{" "}
          <span className="font-mono">~/.hrant/data/self_mods/</span>; the official
          GitHub remote never sees them.
        </div>
        <div>
          On <span className="font-mono">hrant update</span> the engine is reset
          to <span className="font-mono">origin/master</span>, then the patches
          are re-applied in order (best-effort 3-way merge). A patch that
          conflicts with the new engine flips to{" "}
          <span className="text-amber-300">needs_review</span> and is left
          un-applied; the engine stays stable.
        </div>
      </div>

      {needsReview.length > 0 && (
        <div className="bg-amber-900/30 border border-amber-700/60 rounded p-3 text-xs space-y-1">
          <div className="font-semibold">
            ⚠ {needsReview.length} patch(es) need review
          </div>
          <div>
            These conflicted with the latest engine update. The engine is at
            the official version for the affected files; revert them or fix
            them manually in <span className="font-mono">data_dir/self_mods/</span>.
          </div>
        </div>
      )}

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
    </div>
  );
}
