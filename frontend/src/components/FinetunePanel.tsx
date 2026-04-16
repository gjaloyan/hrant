import { useEffect, useState } from "react";
import {
  boostFinetuneExample,
  deleteFinetuneExample,
  editFinetuneExample,
  exportForCloud,
  fetchFinetuneExamples,
  fetchFinetuneStatus,
  fetchModelVersions,
  fetchStatus,
  FinetuneExample,
  FinetuneStatusPayload,
  importGguf,
  ModelVersionsState,
  rollbackModel,
  StatusPayload,
  startFinetune,
  switchModelVersion,
} from "../api";

function CategoryBadge({ cat }: { cat: string }) {
  const colors: Record<string, string> = {
    correction: "bg-rose-700",
    troubleshooting: "bg-amber-700",
    factual_qa: "bg-sky-700",
    procedure: "bg-emerald-700",
    decision: "bg-violet-700",
    other: "bg-slate-700",
  };
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded ${colors[cat] || "bg-slate-700"}`}>
      {cat}
    </span>
  );
}

export default function FinetunePanel() {
  const [status, setStatus] = useState<FinetuneStatusPayload | null>(null);
  const [appStatus, setAppStatus] = useState<StatusPayload | null>(null);
  const [items, setItems] = useState<FinetuneExample[]>([]);
  const [versions, setVersions] = useState<ModelVersionsState | null>(null);
  const [progress, setProgress] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [ggufPath, setGgufPath] = useState("");
  const [ggufTag, setGgufTag] = useState("");

  const refresh = async () => {
    const [s, e, v, a] = await Promise.all([
      fetchFinetuneStatus(),
      fetchFinetuneExamples(),
      fetchModelVersions(),
      fetchStatus(),
    ]);
    setStatus(s);
    setItems(e.items || []);
    setVersions(v);
    setAppStatus(a);
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleStart = async () => {
    setRunning(true);
    setProgress([]);
    try {
      await startFinetune((ev: any) => {
        if (ev.type === "progress") {
          setProgress((p) => [...p, `${ev.stage}: ${ev.message}`]);
        } else if (ev.type === "done") {
          setProgress((p) => [...p, `Done: ${ev.model}`]);
        } else if (ev.type === "error") {
          setProgress((p) => [...p, `Error: ${ev.message}`]);
        }
      });
    } finally {
      setRunning(false);
      refresh();
    }
  };

  const handleExportCloud = async () => {
    setProgress([]);
    try {
      const res = await exportForCloud();
      if (res.package_dir) {
        setProgress([
          `Package ready: ${res.package_dir}`,
          `Files: ${(res.files || []).join(", ")}`,
          res.instructions,
        ]);
      } else {
        setProgress([`Error: ${res.detail || JSON.stringify(res)}`]);
      }
    } catch (e: any) {
      setProgress([`Error: ${e.message || e}`]);
    }
    refresh();
  };

  const handleImportGguf = async () => {
    if (!ggufPath.trim()) return;
    setProgress([]);
    try {
      const res = await importGguf(ggufPath.trim(), ggufTag.trim() || undefined);
      if (res.ok) {
        setProgress([`Model ${res.model} registered in Ollama`]);
        setGgufPath("");
        setGgufTag("");
      } else {
        setProgress([`Error: ${res.detail || JSON.stringify(res)}`]);
      }
    } catch (e: any) {
      setProgress([`Error: ${e.message || e}`]);
    }
    refresh();
  };

  const handleDelete = async (id: string) => {
    await deleteFinetuneExample(id);
    refresh();
  };
  const handleBoost = async (id: string) => {
    await boostFinetuneExample(id);
    refresh();
  };

  const startEdit = (it: FinetuneExample) => {
    setEditingId(it.id);
    setEditText(it.assistant);
  };
  const saveEdit = async () => {
    if (!editingId) return;
    await editFinetuneExample(editingId, { assistant: editText });
    setEditingId(null);
    refresh();
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
      <div className="p-4 border-b border-slate-800 bg-slate-900/40">
        <h2 className="text-xl font-bold mb-2">Fine-Tune Data Manager</h2>
        {status && (
          <div className="text-sm space-y-1">
            <div>
              Total examples: <b>{status.total}</b> · Curated:{" "}
              <b>{status.curated}</b> · Min: <b>{status.min_required}</b>
            </div>
            <div>
              Ready for training:{" "}
              {status.ready ? (
                <span className="text-emerald-400">YES</span>
              ) : (
                <span className="text-amber-400">need more data</span>
              )}
            </div>
            <div className="flex gap-2 flex-wrap text-xs opacity-80">
              {Object.entries(status.by_category).map(([k, v]) => (
                <span key={k} className="bg-slate-800 px-1.5 rounded">
                  {k}: {v}
                </span>
              ))}
            </div>
          </div>
        )}
        {appStatus && (
          <div className="mt-2 text-xs opacity-80">
            mode: <b>{appStatus.mode}</b> · training:{" "}
            <b>{appStatus.training_location}</b> · finetune:{" "}
            {appStatus.finetune_enabled ? "yes" : "no"}
          </div>
        )}

        <div className="flex gap-2 mt-3 flex-wrap">
          {appStatus?.training_location === "local" ||
          appStatus?.training_location === "local_cpu" ? (
            <button
              onClick={handleStart}
              disabled={running || !status?.ready}
              className="bg-rose-700 hover:bg-rose-600 disabled:opacity-40 rounded px-3 py-1 text-sm"
              title="Local LoRA fine-tune (Unsloth)"
            >
              {running ? "training..." : "Start Fine-Tune (local)"}
            </button>
          ) : null}

          {appStatus?.training_location === "cloud" && (
            <button
              onClick={handleExportCloud}
              disabled={!status?.ready}
              className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1 text-sm"
              title="Prepare package for cloud GPU training"
            >
              Export for Cloud GPU
            </button>
          )}

          <button
            onClick={refresh}
            className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm"
          >
            Refresh
          </button>
          <a
            href="/api/finetune/export"
            className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm inline-block"
          >
            Export JSONL
          </a>
        </div>

        {appStatus?.training_location === "cloud" && (
          <div className="mt-3 p-2 bg-slate-900 rounded border border-slate-800">
            <div className="text-xs opacity-70 mb-1">
              Import GGUF (after cloud GPU training):
            </div>
            <div className="flex gap-1">
              <input
                className="flex-1 bg-slate-950 rounded px-2 py-1 text-xs outline-none"
                placeholder="path to .gguf file"
                value={ggufPath}
                onChange={(e) => setGgufPath(e.target.value)}
              />
              <input
                className="w-20 bg-slate-950 rounded px-2 py-1 text-xs outline-none"
                placeholder="tag (v2)"
                value={ggufTag}
                onChange={(e) => setGgufTag(e.target.value)}
              />
              <button
                onClick={handleImportGguf}
                className="bg-emerald-700 hover:bg-emerald-600 rounded px-3 text-xs"
              >
                Import
              </button>
            </div>
          </div>
        )}

        {appStatus?.mode === "claude_only" && (
          <div className="mt-2 text-xs text-violet-300 bg-violet-900/30 rounded p-2">
            mode: claude_only — local model disabled, fine-tune unavailable.
            Data is still collected in finetune_queue for future use.
          </div>
        )}
        {progress.length > 0 && (
          <pre className="mt-2 bg-slate-950 rounded p-2 text-xs max-h-40 overflow-y-auto">
            {progress.join("\n")}
          </pre>
        )}
      </div>

      {versions && (
        <div className="px-4 py-2 border-b border-slate-800 bg-slate-900/30 text-xs">
          <div className="font-semibold mb-1">Model Versions</div>
          <div className="flex gap-2 flex-wrap items-center">
            {versions.versions.map((v) => (
              <button
                key={v.tag}
                onClick={() => switchModelVersion(v.tag).then(refresh)}
                className={`px-2 py-0.5 rounded ${
                  v.tag === versions.current
                    ? "bg-emerald-700"
                    : "bg-slate-800 hover:bg-slate-700"
                }`}
                title={`${v.model_id} (${v.examples_count} ex)`}
              >
                {v.tag}
              </button>
            ))}
            <button
              onClick={() => rollbackModel().then(refresh)}
              disabled={!versions.rollback_enabled}
              className="bg-slate-800 hover:bg-slate-700 px-2 py-0.5 rounded disabled:opacity-40"
            >
              rollback
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {items.length === 0 && (
          <div className="opacity-50 text-sm">Queue is empty</div>
        )}
        {items.map((it) => (
          <div key={it.id} className="bg-slate-900 rounded p-3 text-sm">
            <div className="flex items-center gap-2 mb-1">
              <CategoryBadge cat={it.metadata.category} />
              <span className="text-xs opacity-60">
                conf {it.metadata.confidence}% · score {it.score.toFixed(2)} ·{" "}
                sources: {it.metadata.source_notes.join(", ") || "—"}
              </span>
              {it.metadata.boosted && (
                <span className="text-xs text-amber-400">boosted</span>
              )}
            </div>
            <div className="opacity-80 mb-1">
              <b>Q:</b> {it.user}
            </div>
            {editingId === it.id ? (
              <div className="space-y-1">
                <textarea
                  className="w-full bg-slate-950 rounded p-2 text-xs outline-none"
                  rows={5}
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                />
                <div className="flex gap-1">
                  <button
                    onClick={saveEdit}
                    className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-0.5 text-xs"
                  >
                    save
                  </button>
                  <button
                    onClick={() => setEditingId(null)}
                    className="bg-slate-700 rounded px-2 py-0.5 text-xs"
                  >
                    cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="opacity-70">
                <b>A:</b> {it.assistant}
              </div>
            )}
            <div className="flex gap-2 mt-2 text-xs">
              <button
                onClick={() => startEdit(it)}
                className="bg-slate-800 hover:bg-slate-700 rounded px-2 py-0.5"
              >
                edit
              </button>
              <button
                onClick={() => handleBoost(it.id)}
                className="bg-amber-800 hover:bg-amber-700 rounded px-2 py-0.5"
              >
                boost
              </button>
              <button
                onClick={() => handleDelete(it.id)}
                className="bg-rose-800 hover:bg-rose-700 rounded px-2 py-0.5"
              >
                remove
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
