import { FIELD_NAMES } from "./fieldNames";
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
      {/* The most conceptually opaque screen in Settings: it opened on a
          profile selector with no statement of what a profile IS. */}
      <p className="border-b border-edge px-3 py-2 text-xs text-ink-dim">
        A <b className="text-ink">profile</b> is a named set of overrides
        applied on top of the defaults for a whole turn — a way to run the
        agent cheaply, or exhaustively, without editing settings each time.
        Only the <b className="text-ink">active</b> one has any effect;
        blank fields fall through to the default.
      </p>
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
        {subtab === "reasoning" && (
          <ReasoningEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab === "prompt" && (
          <PromptEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab === "logging" && (
          <LoggingEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab === "history" && (
          <HistoryViewer
            editingId={editing.id}
            onRestored={() => loadEditing(editing.id)}
            flash={flash}
          />
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
          <span className="w-56">
            <span className="block">{FIELD_NAMES.tool_loop_input_budget}</span>
            <span className="block font-mono text-[10px] text-ink-faint">
              tool_loop_input_budget
            </span>
          </span>
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
          <span className="w-56">
            <span className="block">{FIELD_NAMES.min_confidence}</span>
            <span className="block font-mono text-[10px] text-ink-faint">
              min_confidence · 0–100
            </span>
          </span>
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

function ReasoningEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const r = editing.reasoning_overrides || {};
  const routing = (r.routing as Record<string, string>) || {};
  const levels = ["", "none", "low", "medium", "high"];
  const tasks = [
    "chat", "quick_answer", "classification", "keyword_extraction",
    "task", "task_analysis", "note_creation", "verification",
    "simple_lookup", "note_search", "learning",
    "complex_solving", "supervisor", "self_critic", "skill_reflection",
  ];
  const setRouting = (task: string, level: string) => {
    const next = { ...routing };
    if (!level) delete next[task];
    else next[task] = level;
    setEditing({
      ...editing,
      reasoning_overrides: { ...r, routing: next },
    });
  };
  return (
    <div className="space-y-4 text-sm text-slate-200">
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
          Routing — leave blank to use default
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {tasks.map((t) => (
            <label key={t} className="flex items-center gap-2 text-xs">
              <span className="w-44 truncate">{t}</span>
              <select
                value={routing[t] || ""}
                onChange={(e) => setRouting(t, e.target.value)}
                className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
              >
                {levels.map((l) => (
                  <option key={l} value={l}>
                    {l || "(default)"}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </div>
      <div>
        <label className="flex items-center gap-2 text-sm">
          <span className="w-44">fallback</span>
          <select
            value={(r.fallback as string) || ""}
            onChange={(e) => {
              const next = { ...r };
              if (!e.target.value) delete next.fallback;
              else next.fallback = e.target.value;
              setEditing({ ...editing, reasoning_overrides: next });
            }}
            className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
          >
            {levels.map((l) => (
              <option key={l} value={l}>
                {l || "(default)"}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}

function PromptEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const [defaults, setDefaults] = useState<{
    order: string[]; sections: Record<string, string>;
  } | null>(null);
  const [section, setSection] = useState<string>("");
  useEffect(() => {
    import("../../api").then(({ fetchSystemPromptSections }) =>
      fetchSystemPromptSections().then((d) => {
        setDefaults(d);
        if (!section && d.order.length > 0) setSection(d.order[0]);
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (!defaults) return <div className="text-slate-500">Loading…</div>;
  const sections = (editing.prompt_overrides?.sections || {}) as Record<
    string, string | null
  >;
  const override = sections[section];
  const mode: "default" | "override" | "skip" =
    !(section in sections) ? "default"
      : override === null ? "skip" : "override";
  const setMode = (m: "default" | "override" | "skip") => {
    const next = { ...sections };
    if (m === "default") delete next[section];
    else if (m === "skip") next[section] = null;
    else next[section] = defaults.sections[section];
    setEditing({
      ...editing,
      prompt_overrides: { sections: next },
    });
  };
  const setBody = (body: string) => {
    setEditing({
      ...editing,
      prompt_overrides: {
        sections: { ...sections, [section]: body },
      },
    });
  };
  return (
    <div className="space-y-3 text-sm text-slate-200">
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-400">Section:</span>
        <select
          value={section}
          onChange={(e) => setSection(e.target.value)}
          className="bg-slate-800 text-slate-100 rounded px-2 py-1 text-xs"
        >
          {defaults.order.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        {(["default", "override", "skip"] as const).map((m) => (
          <label key={m} className="flex items-center gap-1 text-xs">
            <input
              type="radio"
              name="mode"
              checked={mode === m}
              onChange={() => setMode(m)}
            />
            {m}
          </label>
        ))}
      </div>
      {mode === "override" && (
        <textarea
          value={(override as string) || ""}
          onChange={(e) => setBody(e.target.value)}
          rows={20}
          className="w-full font-mono text-[11px] bg-slate-900 text-slate-100 rounded p-2"
        />
      )}
      {mode !== "override" && (
        <pre className="font-mono text-[11px] bg-slate-950/40 text-slate-400 rounded p-2 max-h-96 overflow-auto whitespace-pre-wrap">
{defaults.sections[section]}
        </pre>
      )}
    </div>
  );
}

function LoggingEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const l = editing.logging_overrides || {};
  const modules = (l.modules || {}) as Record<string, string>;
  const levels = ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
  const setRoot = (v: string) => {
    const next = { ...l };
    if (!v) delete next.root;
    else next.root = v;
    setEditing({ ...editing, logging_overrides: next });
  };
  const setModule = (mod: string, level: string) => {
    const next = { ...modules };
    if (!level) delete next[mod];
    else next[mod] = level;
    setEditing({
      ...editing,
      logging_overrides: { ...l, modules: next },
    });
  };
  return (
    <div className="space-y-4 text-sm text-slate-200">
      <label className="flex items-center gap-2">
        <span className="w-32">root</span>
        <select
          value={(l.root as string) || ""}
          onChange={(e) => setRoot(e.target.value)}
          className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
        >
          {levels.map((lv) => (
            <option key={lv} value={lv}>{lv || "(default)"}</option>
          ))}
        </select>
      </label>
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
          Per-module overrides
        </div>
        <table className="text-xs w-full">
          <tbody>
            {Object.entries(modules).map(([mod, level]) => (
              <tr key={mod}>
                <td className="pr-2 py-0.5 font-mono">{mod}</td>
                <td>
                  <select
                    value={level}
                    onChange={(e) => setModule(mod, e.target.value)}
                    className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
                  >
                    {levels.filter((x) => x).map((lv) => (
                      <option key={lv} value={lv}>{lv}</option>
                    ))}
                  </select>
                </td>
                <td>
                  <button
                    onClick={() => setModule(mod, "")}
                    className="text-rose-300 text-xs px-1"
                  >
                    delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button
          onClick={() => {
            const mod = prompt("Module name (e.g. backend.unified_agent):", "");
            if (mod) setModule(mod, "INFO");
          }}
          className="text-xs px-2 py-1 bg-slate-700 text-slate-200 rounded mt-2"
        >
          + add module
        </button>
      </div>
    </div>
  );
}

function HistoryViewer({
  editingId,
  onRestored,
  flash,
}: {
  editingId: string;
  onRestored: () => void;
  flash: (m: string) => void;
}) {
  const [rows, setRows] = useState<
    Array<{ timestamp: number; name: string; updated_at: number; description: string }>
  >([]);
  const refresh = async () => {
    try {
      const { fetchPipelineProfileHistory } = await import("../../api");
      const r = await fetchPipelineProfileHistory(editingId);
      setRows(r.history);
    } catch (e: any) {
      flash("History load failed: " + (e?.message || e));
    }
  };
  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editingId]);
  const restore = async (ts: number) => {
    if (!confirm("Restore this version? Current version will be saved to history first.")) return;
    try {
      const { restorePipelineProfile } = await import("../../api");
      await restorePipelineProfile(editingId, ts);
      flash("Restored.");
      onRestored();
      await refresh();
    } catch (e: any) {
      flash("Restore failed: " + (e?.message || e));
    }
  };
  if (rows.length === 0) {
    return <div className="text-slate-500 text-sm">No history yet.</div>;
  }
  return (
    <table className="text-xs w-full">
      <thead>
        <tr className="text-slate-400">
          <th className="text-left py-1">When</th>
          <th className="text-left">Name</th>
          <th className="text-left">Description</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.timestamp} className="border-t border-slate-800">
            <td className="py-1 text-slate-400">
              {new Date(r.updated_at * 1000).toLocaleString()}
            </td>
            <td className="text-slate-200">{r.name}</td>
            <td className="text-slate-400">{r.description}</td>
            <td>
              <button
                onClick={() => restore(r.timestamp)}
                className="text-xs px-2 py-0.5 bg-amber-700/40 text-amber-200 rounded"
              >
                Restore
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
