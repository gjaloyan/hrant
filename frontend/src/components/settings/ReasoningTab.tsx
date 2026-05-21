import { useEffect, useState } from "react";
import {
  fetchReasoningRouting,
  saveReasoningRouting,
  setReasoningOverride,
  type ReasoningRoutingConfig,
} from "../../api";

// Reasoning routing matrix. Each task_type maps to a level
// (`low` / `medium` / `high`); supervisor / complex_solving get
// `high` by default. The per-turn override at the top is a
// quick boost — it overrides every routing entry for the next
// turn (or until cleared).
//
// The configured map merges over the backend defaults; users add
// rows for custom task_types they want to bias differently.

const LEVEL_COLORS: Record<string, string> = {
  none: "bg-slate-700",
  low: "bg-sky-700",
  medium: "bg-amber-700",
  high: "bg-rose-700",
};

const LEVEL_LABELS: Record<string, string> = {
  none: "Off",
  low: "Low",
  medium: "Medium",
  high: "High",
};

type FlashFn = (text: string, kind?: "ok" | "warn" | "error") => void;

export default function ReasoningTab({ flash }: { flash: FlashFn }) {
  const [cfg, setCfg] = useState<ReasoningRoutingConfig | null>(null);
  const [editedRouting, setEditedRouting] = useState<Record<string, string>>({});
  const [editedFallback, setEditedFallback] = useState<string>("medium");
  const [saving, setSaving] = useState(false);
  const [newKey, setNewKey] = useState("");

  const load = async () => {
    try {
      const c = await fetchReasoningRouting();
      setCfg(c);
      setEditedRouting({ ...c.routing });
      setEditedFallback(c.fallback);
    } catch (e: any) {
      flash(`Failed to load reasoning routing: ${e?.message || e}`, "error");
    }
  };
  useEffect(() => { void load(); /* eslint-disable-next-line */ }, []);

  const updateLevel = (taskType: string, level: string) => {
    setEditedRouting((prev) => ({ ...prev, [taskType]: level }));
  };

  const removeRow = (taskType: string) => {
    setEditedRouting((prev) => {
      const next = { ...prev };
      delete next[taskType];
      return next;
    });
  };

  const addRow = () => {
    const key = newKey.trim();
    if (!key) return;
    if (key in editedRouting) {
      flash(`'${key}' already in the routing`, "warn");
      return;
    }
    setEditedRouting((prev) => ({ ...prev, [key]: editedFallback || "medium" }));
    setNewKey("");
  };

  const save = async () => {
    setSaving(true);
    try {
      await saveReasoningRouting(editedRouting, editedFallback);
      flash("Reasoning routing saved", "ok");
      await load();
    } catch (e: any) {
      flash(`Save failed: ${e?.message || e}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const applyOverride = async (level: string) => {
    try {
      await setReasoningOverride(level);
      flash(level ? `Override: ${level}` : "Override cleared", "ok");
      await load();
    } catch (e: any) {
      flash(`Override failed: ${e?.message || e}`, "error");
    }
  };

  const resetToDefaults = () => {
    if (!cfg) return;
    setEditedRouting({ ...cfg.defaults });
    setEditedFallback("medium");
    flash("Reset to defaults (not yet saved)", "warn");
  };

  if (!cfg) {
    return (
      <div className="text-sm text-slate-400 p-4">
        loading reasoning routing…
      </div>
    );
  }

  const levels = cfg.valid_levels.filter((l) => l !== "none");

  return (
    <div className="p-4 space-y-6 max-w-3xl">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">Reasoning Routing</h2>
        <p className="text-xs text-slate-400 mt-1 leading-relaxed">
          GPT-5.x via OpenAI Codex (ChatGPT subscription) accepts a
          <code className="mx-1 bg-slate-800 rounded px-1">reasoning.effort</code>
          parameter — <strong>low</strong> is fastest + cheapest,
          <strong> high</strong> spends more tokens for deeper planning.
          Map each task type below. Tasks not listed fall back to the
          fallback level. The override at the top temporarily bumps
          everything to one level for the next turn.
        </p>
      </div>

      {/* Per-turn override */}
      <div className="rounded-lg border border-violet-700/30 bg-violet-950/30 p-3">
        <div className="text-xs uppercase tracking-wide text-violet-300 mb-2">
          Per-turn override
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {["", ...cfg.valid_levels].map((lv) => {
            const active = cfg.override === lv;
            const label = lv === "" ? "Off (use routing)" : LEVEL_LABELS[lv] || lv;
            return (
              <button
                key={lv || "off"}
                onClick={() => applyOverride(lv)}
                className={`text-xs px-3 py-1.5 rounded transition-colors ${
                  active
                    ? `${LEVEL_COLORS[lv] || "bg-violet-700"} text-white`
                    : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>
        <div className="text-[11px] text-slate-400 mt-2">
          Current override:{" "}
          <code className="bg-slate-800 rounded px-1">
            {cfg.override || "(none — using routing)"}
          </code>
        </div>
      </div>

      {/* Routing matrix */}
      <div className="rounded-lg border border-slate-700/40 bg-slate-900/40 p-3">
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs uppercase tracking-wide text-slate-400">
            Task-type → level
          </div>
          <button
            onClick={resetToDefaults}
            className="text-[11px] text-slate-400 hover:text-slate-200 underline"
          >
            reset to defaults
          </button>
        </div>
        <div className="space-y-1.5 text-sm">
          {Object.keys(editedRouting).sort().map((taskType) => (
            <div key={taskType} className="flex items-center gap-2">
              <code className="text-xs font-mono text-slate-300 flex-1 truncate" title={taskType}>
                {taskType}
              </code>
              <div className="flex gap-1">
                {levels.map((lv) => {
                  const active = editedRouting[taskType] === lv;
                  return (
                    <button
                      key={lv}
                      onClick={() => updateLevel(taskType, lv)}
                      className={`text-[11px] px-2.5 py-1 rounded transition-colors ${
                        active
                          ? `${LEVEL_COLORS[lv]} text-white`
                          : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                      }`}
                    >
                      {LEVEL_LABELS[lv]}
                    </button>
                  );
                })}
                <button
                  onClick={() => removeRow(taskType)}
                  className="text-[11px] px-2 py-1 rounded text-slate-500 hover:text-rose-300 hover:bg-rose-950/30"
                  title="remove from routing"
                >
                  ×
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Add row */}
        <div className="mt-3 pt-3 border-t border-slate-700/40 flex items-center gap-2">
          <input
            type="text"
            placeholder="add task_type…"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addRow()}
            className="flex-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs"
          />
          <button
            onClick={addRow}
            disabled={!newKey.trim()}
            className="text-xs px-3 py-1 rounded bg-emerald-700 hover:bg-emerald-600 disabled:bg-slate-700 disabled:text-slate-500"
          >
            add
          </button>
        </div>
      </div>

      {/* Fallback */}
      <div className="rounded-lg border border-slate-700/40 bg-slate-900/40 p-3">
        <div className="text-xs uppercase tracking-wide text-slate-400 mb-2">
          Fallback level (used for task_types not listed above)
        </div>
        <div className="flex gap-2">
          {levels.map((lv) => {
            const active = editedFallback === lv;
            return (
              <button
                key={lv}
                onClick={() => setEditedFallback(lv)}
                className={`text-xs px-3 py-1.5 rounded transition-colors ${
                  active
                    ? `${LEVEL_COLORS[lv]} text-white`
                    : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                }`}
              >
                {LEVEL_LABELS[lv]}
              </button>
            );
          })}
        </div>
      </div>

      {/* Save bar */}
      <div className="flex items-center gap-2">
        <button
          onClick={save}
          disabled={saving}
          className="px-4 py-1.5 rounded bg-emerald-700 hover:bg-emerald-600 text-sm disabled:bg-slate-700"
        >
          {saving ? "saving…" : "save routing"}
        </button>
        <button
          onClick={load}
          className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-xs"
        >
          reload
        </button>
        {cfg.updated_at > 0 && (
          <span className="text-[11px] text-slate-500 ml-auto">
            last saved: {new Date(cfg.updated_at * 1000).toLocaleString()}
          </span>
        )}
      </div>
    </div>
  );
}
