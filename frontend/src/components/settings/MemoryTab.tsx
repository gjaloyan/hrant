import { useEffect, useState } from "react";
import {
  EmbedderConfig,
  EmbeddingsStatusResponse,
  backfillEmbeddings,
  fetchEmbeddingsStatus,
  resetEmbedder,
  updateEmbeddingsConfig,
} from "../../api";

type Props = { flash: (msg: string) => void };

const BACKENDS: { id: NonNullable<EmbedderConfig["backend"]>; label: string }[] = [
  { id: "auto", label: "Auto (try llama.cpp → ollama → providers)" },
  { id: "llama_cpp", label: "llama.cpp server" },
  { id: "ollama", label: "Ollama" },
  { id: "openai", label: "OpenAI / OpenAI-compatible provider" },
  { id: "cohere", label: "Cohere" },
  { id: "disabled", label: "Disabled (text + graph only)" },
];

export default function MemoryTab({ flash }: Props) {
  const [status, setStatus] = useState<EmbeddingsStatusResponse | null>(null);
  const [busy, setBusy] = useState(false);
  // Form state — initialized from server config on first load
  const [backend, setBackend] = useState<NonNullable<EmbedderConfig["backend"]>>("auto");
  const [llamaUrl, setLlamaUrl] = useState("");
  const [llamaModel, setLlamaModel] = useState("bge-m3");
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [ollamaModel, setOllamaModel] = useState("nomic-embed-text");
  const [openaiModel, setOpenaiModel] = useState("text-embedding-3-small");
  const [cohereModel, setCohereModel] = useState("embed-english-v3.0");

  const refresh = async () => {
    try {
      const s = await fetchEmbeddingsStatus();
      setStatus(s);
      const cfg = s.embedder.config || {};
      setBackend(cfg.backend ?? "auto");
      if (cfg.llama_cpp?.url) setLlamaUrl(cfg.llama_cpp.url);
      if (cfg.llama_cpp?.model) setLlamaModel(cfg.llama_cpp.model);
      if (cfg.ollama?.url) setOllamaUrl(cfg.ollama.url);
      if (cfg.ollama?.model) setOllamaModel(cfg.ollama.model);
      if (cfg.openai?.model) setOpenaiModel(cfg.openai.model);
      if (cfg.cohere?.model) setCohereModel(cfg.cohere.model);
    } catch (e: any) {
      flash("Error loading embeddings status: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const buildCfgFromForm = (): EmbedderConfig => {
    const cfg: EmbedderConfig = { backend };
    if (backend === "llama_cpp" || backend === "auto") {
      cfg.llama_cpp = { url: llamaUrl.trim(), model: llamaModel.trim() };
    }
    if (backend === "ollama" || backend === "auto") {
      cfg.ollama = { url: ollamaUrl.trim(), model: ollamaModel.trim() };
    }
    if (backend === "openai" || backend === "auto") {
      cfg.openai = { model: openaiModel.trim() };
    }
    if (backend === "cohere" || backend === "auto") {
      cfg.cohere = { model: cohereModel.trim() };
    }
    return cfg;
  };

  const handleSave = async () => {
    setBusy(true);
    try {
      const result = await updateEmbeddingsConfig(buildCfgFromForm());
      setStatus(result);
      const live = result.embedder;
      if (live.backend === "disabled") {
        flash(`Saved, but no backend is reachable: ${live.last_error || "see config"}`);
      } else {
        flash(`Embedder online: ${live.backend} / ${live.model} (${live.dim}-dim)`);
      }
    } catch (e: any) {
      flash("Save failed: " + (e.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    setBusy(true);
    try {
      const s = await resetEmbedder();
      setStatus(s);
      flash(s.embedder.backend === "disabled" ? "Re-probe: no backend available" : `Re-probed: ${s.embedder.backend}`);
    } catch (e: any) {
      flash("Reset failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleBackfill = async (force: boolean) => {
    setBusy(true);
    try {
      const r = await backfillEmbeddings(force);
      flash(
        r.ok
          ? `Backfilled ${r.embedded ?? 0} note(s), skipped ${r.skipped ?? 0}, errors ${r.errors ?? 0}`
          : `Backfill failed: ${r.reason || "unknown"}`,
      );
      await refresh();
    } catch (e: any) {
      flash("Backfill failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const liveBackend = status?.embedder.backend || "—";
  const liveModel = status?.embedder.model || "—";
  const liveDim = status?.embedder.dim ?? "—";
  const cov = status?.coverage;
  const vstore = status?.vector_store;

  return (
    <div className="space-y-4">

      {/* Live status banner */}
      <div className="bg-slate-800 rounded p-3 text-sm space-y-1">
        <div className="flex items-center gap-3 flex-wrap">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              liveBackend === "disabled"
                ? "bg-amber-400"
                : liveBackend === "—"
                ? "bg-slate-500"
                : "bg-emerald-400"
            }`}
          />
          <span className="font-semibold">
            backend: <span className="font-mono">{liveBackend}</span>
          </span>
          <span className="text-slate-400">
            model: <span className="font-mono">{liveModel}</span>
          </span>
          <span className="text-slate-400">
            dim: <span className="font-mono">{liveDim}</span>
          </span>
          {vstore && (
            <span className="text-slate-400">
              vectors stored: <b>{vstore.count}</b>
            </span>
          )}
          <button
            onClick={refresh}
            disabled={busy}
            className="ml-auto bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-2 py-0.5 text-xs"
          >
            Refresh
          </button>
        </div>

        {status?.embedder.last_error && (
          <div className="text-xs text-rose-300 font-mono mt-1 break-all">
            last error: {status.embedder.last_error}
          </div>
        )}

        {liveBackend === "disabled" && (
          <div className="bg-amber-900/30 border border-amber-800/50 rounded px-2 py-1.5 text-xs mt-2">
            ⚠ Embeddings disabled — search falls back to fuzzy + graph signals only.
            New notes save without vectors. Configure a backend below or fix the
            target server, then click <b>Reset / re-probe</b> to retry.
          </div>
        )}

        {cov && cov.missing > 0 && liveBackend !== "disabled" && (
          <div className="bg-amber-900/30 border border-amber-800/50 rounded px-2 py-1.5 text-xs mt-2">
            ⚠ {cov.missing} of {cov.total_notes} notes have no vector
            ({cov.stale_store ? "store stamp mismatch — needs rebuild" : "missing in store"}).
            Click <b>Backfill missing</b> to embed them.
          </div>
        )}

        {cov && cov.missing === 0 && liveBackend !== "disabled" && cov.total_notes > 0 && (
          <div className="text-xs text-emerald-300 mt-2">
            ✓ All {cov.total_notes} notes embedded.
          </div>
        )}
      </div>

      {/* Backend selection */}
      <div className="bg-slate-800 rounded p-3 space-y-3 text-sm">
        <div>
          <label className="text-xs text-slate-400 block mb-1">Backend</label>
          <select
            className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
            value={backend}
            onChange={(e) => setBackend(e.target.value as any)}
          >
            {BACKENDS.map((b) => (
              <option key={b.id} value={b.id}>
                {b.label}
              </option>
            ))}
          </select>
        </div>

        {(backend === "llama_cpp" || backend === "auto") && (
          <fieldset className="border border-slate-700 rounded p-2 space-y-2">
            <legend className="text-xs text-slate-400 px-1">llama.cpp</legend>
            <div>
              <label className="text-[10px] text-slate-500 block mb-0.5">Server URL</label>
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                placeholder="http://host:8080  (or http://host:8080/v1)"
                value={llamaUrl}
                onChange={(e) => setLlamaUrl(e.target.value)}
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-500 block mb-0.5">Model name (echoed in request body)</label>
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                placeholder="bge-m3"
                value={llamaModel}
                onChange={(e) => setLlamaModel(e.target.value)}
              />
            </div>
          </fieldset>
        )}

        {(backend === "ollama" || backend === "auto") && (
          <fieldset className="border border-slate-700 rounded p-2 space-y-2">
            <legend className="text-xs text-slate-400 px-1">Ollama</legend>
            <div>
              <label className="text-[10px] text-slate-500 block mb-0.5">Base URL</label>
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                placeholder="http://localhost:11434"
                value={ollamaUrl}
                onChange={(e) => setOllamaUrl(e.target.value)}
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-500 block mb-0.5">Model</label>
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                placeholder="nomic-embed-text"
                value={ollamaModel}
                onChange={(e) => setOllamaModel(e.target.value)}
              />
            </div>
          </fieldset>
        )}

        {(backend === "openai" || backend === "auto") && (
          <fieldset className="border border-slate-700 rounded p-2 space-y-2">
            <legend className="text-xs text-slate-400 px-1">OpenAI / OpenAI-compatible</legend>
            <div>
              <label className="text-[10px] text-slate-500 block mb-0.5">Embedding model</label>
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                placeholder="text-embedding-3-small"
                value={openaiModel}
                onChange={(e) => setOpenaiModel(e.target.value)}
              />
            </div>
            <div className="text-[10px] text-slate-500">
              Uses the API key + base_url of the first <b>openai</b>-type provider in your provider list (or any <b>openai_compatible</b>/<b>together</b>/<b>openrouter</b>/<b>groq</b> as fallback).
            </div>
          </fieldset>
        )}

        {(backend === "cohere" || backend === "auto") && (
          <fieldset className="border border-slate-700 rounded p-2 space-y-2">
            <legend className="text-xs text-slate-400 px-1">Cohere</legend>
            <div>
              <label className="text-[10px] text-slate-500 block mb-0.5">Embedding model</label>
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                placeholder="embed-english-v3.0"
                value={cohereModel}
                onChange={(e) => setCohereModel(e.target.value)}
              />
            </div>
            <div className="text-[10px] text-slate-500">
              Uses the API key from your <b>cohere</b>-type provider.
            </div>
          </fieldset>
        )}

        <div className="flex gap-2 pt-1">
          <button
            onClick={handleSave}
            disabled={busy}
            className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs font-semibold"
          >
            Save & probe
          </button>
          <button
            onClick={handleReset}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Reset / re-probe
          </button>
          <span className="ml-auto flex gap-2">
            <button
              onClick={() => handleBackfill(false)}
              disabled={busy || liveBackend === "disabled"}
              className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
            >
              Backfill missing
            </button>
            <button
              onClick={() => {
                if (!confirm("Force-rebuild all vectors? This re-embeds every note.")) return;
                handleBackfill(true);
              }}
              disabled={busy || liveBackend === "disabled"}
              className="bg-amber-700 hover:bg-amber-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
            >
              Force rebuild
            </button>
          </span>
        </div>
      </div>
    </div>
  );
}
