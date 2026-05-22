import { useEffect, useState } from "react";
import {
  createPipelineProfile,
  deletePipelineProfile,
  fetchActivePipelineProfile,
  fetchPipelineProfile,
  fetchPipelineProfiles,
  PipelineProfile,
  PipelineProfileSummary,
  setActivePipelineProfile,
  updatePipelineProfile,
} from "../../api";

type Props = { flash: (msg: string) => void };

type SubTab = "engine" | "reasoning" | "prompt" | "logging" | "history";

function emptyProfile(id: string, name: string): PipelineProfile {
  const now = Date.now() / 1000;
  return {
    id,
    name,
    description: "",
    created_at: now,
    updated_at: now,
    engine_overrides: {},
    reasoning_overrides: { routing: {}, fallback: "" },
    prompt_overrides: { sections: {} },
    logging_overrides: { root: "", modules: {} },
  };
}

export default function PipelineTab({ flash }: Props) {
  const [summaries, setSummaries] = useState<PipelineProfileSummary[]>([]);
  const [activeId, setActiveId] = useState<string>("default");
  const [editingId, setEditingId] = useState<string>("default");
  const [editing, setEditing] = useState<PipelineProfile | null>(null);
  const [subtab, setSubtab] = useState<SubTab>("engine");
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const [list, act] = await Promise.all([
        fetchPipelineProfiles(),
        fetchActivePipelineProfile(),
      ]);
      setSummaries(list.profiles);
      setActiveId(act.active_id);
    } catch (e: any) {
      flash("Pipeline list failed: " + (e?.message || e));
    }
  };

  const loadEditing = async (id: string) => {
    try {
      const p = await fetchPipelineProfile(id);
      setEditing(p);
      setEditingId(id);
    } catch (e: any) {
      flash("Load profile failed: " + (e?.message || e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);
  useEffect(() => {
    if (summaries.length && !editing) {
      loadEditing(activeId);
    }
  }, [summaries, activeId]);

  const onActivate = async () => {
    setBusy(true);
    try {
      await setActivePipelineProfile(editingId);
      setActiveId(editingId);
      flash(`Activated: ${editingId}`);
    } catch (e: any) {
      flash("Activate failed: " + (e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  const onSave = async () => {
    if (!editing) return;
    setBusy(true);
    try {
      await updatePipelineProfile(editing);
      flash("Saved");
      await refresh();
    } catch (e: any) {
      flash("Save failed: " + (e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  const onNew = async () => {
    const id = prompt("New profile id (a-z, 0-9, _-):", "");
    if (!id) return;
    const name = prompt("Display name:", id) || id;
    try {
      await createPipelineProfile(emptyProfile(id, name));
      flash("Created: " + id);
      await refresh();
      await loadEditing(id);
    } catch (e: any) {
      flash("Create failed: " + (e?.message || e));
    }
  };

  const onDelete = async () => {
    if (!editing) return;
    if (editing.id === "default" || editing.id === activeId) {
      flash("Cannot delete default or active profile");
      return;
    }
    if (!confirm(`Delete profile "${editing.id}"?`)) return;
    try {
      await deletePipelineProfile(editing.id);
      flash("Deleted: " + editing.id);
      setEditing(null);
      await refresh();
      await loadEditing("default");
    } catch (e: any) {
      flash("Delete failed: " + (e?.message || e));
    }
  };

  if (!editing) {
    return <div className="p-4 text-slate-400">Loading…</div>;
  }

  return (
    <div className="flex flex-col h-full">
      {/* Top selector strip */}
      <div className="flex flex-wrap gap-2 items-center px-3 py-2 border-b border-slate-700/40 bg-slate-900/40">
        <span className="text-xs text-slate-400">Active:</span>
        <select
          value={editingId}
          onChange={(e) => loadEditing(e.target.value)}
          className="text-xs bg-slate-800 text-slate-200 rounded px-2 py-1"
        >
          {summaries.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id === activeId ? "★ " : ""}{s.name} ({s.id})
            </option>
          ))}
        </select>
        <button
          disabled={busy || editingId === activeId}
          onClick={onActivate}
          className="text-xs px-2 py-1 bg-emerald-700/40 text-emerald-200 rounded disabled:opacity-40"
        >
          Activate
        </button>
        <button
          disabled={busy}
          onClick={onSave}
          className="text-xs px-2 py-1 bg-sky-700/40 text-sky-200 rounded disabled:opacity-40"
        >
          Save
        </button>
        <button
          disabled={busy}
          onClick={onNew}
          className="text-xs px-2 py-1 bg-slate-700 text-slate-200 rounded"
        >
          + New
        </button>
        <button
          disabled={busy || editing.id === "default" || editing.id === activeId}
          onClick={onDelete}
          className="text-xs px-2 py-1 bg-rose-800/40 text-rose-200 rounded disabled:opacity-40"
        >
          Delete
        </button>
      </div>
      {/* Editor metadata */}
      <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/30 space-y-2">
        <input
          value={editing.name}
          onChange={(e) => setEditing({ ...editing, name: e.target.value })}
          placeholder="Display name"
          className="w-full text-sm bg-slate-800 text-slate-100 rounded px-2 py-1"
        />
        <input
          value={editing.description}
          onChange={(e) => setEditing({ ...editing, description: e.target.value })}
          placeholder="Description"
          className="w-full text-xs bg-slate-800 text-slate-300 rounded px-2 py-1"
        />
      </div>
      {/* Sub-tab strip */}
      <div className="flex gap-1 px-3 py-1 border-b border-slate-800 bg-slate-950/40">
        {(["engine", "reasoning", "prompt", "logging", "history"] as SubTab[]).map((t) => (
          <button
            key={t}
            onClick={() => setSubtab(t)}
            className={`text-xs px-2 py-1 rounded ${
              subtab === t
                ? "bg-slate-700 text-slate-100"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      {/* Sub-tab body */}
      <div className="flex-1 overflow-auto p-3">
        {subtab === "engine" && (
          <EngineEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab !== "engine" && (
          <div className="text-slate-500 text-sm">
            ({subtab} editor implemented in Tasks 10-11)
          </div>
        )}
      </div>
    </div>
  );
}

function EngineEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const overrides = editing.engine_overrides || {};
  const setSection = (section: string, fields: Record<string, unknown>) => {
    setEditing({
      ...editing,
      engine_overrides: { ...overrides, [section]: fields },
    });
  };
  const routerSec = (overrides.router as Record<string, unknown>) || {};
  const verifySec = (overrides.verification as Record<string, unknown>) || {};
  return (
    <div className="space-y-4 text-sm text-slate-200">
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Router</div>
        <label className="flex items-center gap-2">
          <span className="w-56">tool_loop_input_budget</span>
          <input
            type="number"
            value={String(routerSec.tool_loop_input_budget ?? "")}
            placeholder="(default 0 — disabled)"
            onChange={(e) => {
              const v = e.target.value;
              const next = { ...routerSec };
              if (v === "") delete next.tool_loop_input_budget;
              else next.tool_loop_input_budget = parseInt(v, 10);
              setSection("router", next);
            }}
            className="bg-slate-800 text-slate-100 rounded px-2 py-1 w-32"
          />
        </label>
      </div>
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
          Verification
        </div>
        <label className="flex items-center gap-2">
          <span className="w-56">min_confidence (0-100)</span>
          <input
            type="number"
            value={String(verifySec.min_confidence ?? "")}
            placeholder="(use default)"
            onChange={(e) => {
              const v = e.target.value;
              const next = { ...verifySec };
              if (v === "") delete next.min_confidence;
              else next.min_confidence = parseInt(v, 10);
              setSection("verification", next);
            }}
            className="bg-slate-800 text-slate-100 rounded px-2 py-1 w-32"
          />
        </label>
      </div>
    </div>
  );
}
