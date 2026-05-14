import { useEffect, useState } from "react";
import {
  FailoverConfig,
  FailoverInfo,
  fetchFailover,
  putFailover,
  reorderFailover,
  toggleFailover,
  type AvailableModel,
  type ProviderConfig,
} from "../../api";

type Props = {
  flash: (msg: string) => void;
  providers: ProviderConfig[];
  availableModels: AvailableModel[];
};

const CATEGORY_LABEL: Record<string, string> = {
  rate_limit:     "Rate limit (429)",
  server_error:   "Server error (5xx)",
  timeout:        "Timeout",
  auth_error:     "Auth error (401/403)",
  connection:     "Connection",
  bad_request:    "Bad request (400)",
  content_policy: "Content policy",
  context_length: "Context length",
  unknown:        "Unknown",
};

export default function FailoverPanel({ flash, providers, availableModels }: Props) {
  const [info, setInfo] = useState<FailoverInfo | null>(null);
  const [busy, setBusy] = useState(false);

  // New-entry form state
  const [newProviderId, setNewProviderId] = useState<string>("");
  const [newModel, setNewModel] = useState<string>("");

  const refresh = async () => {
    try {
      const r = await fetchFailover();
      setInfo(r);
    } catch (e: any) {
      flash("Failover load failed: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  if (!info) {
    return (
      <div className="bg-slate-800 rounded p-4 text-sm text-slate-400">
        Loading failover config…
      </div>
    );
  }

  const cfg = info.config;

  const save = async (next: FailoverConfig) => {
    setBusy(true);
    try {
      const r = await putFailover(next);
      setInfo({ ...info, config: r.config });
    } catch (e: any) {
      flash("Save failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleToggle = async () => {
    setBusy(true);
    try {
      const r = await toggleFailover(!cfg.enabled);
      setInfo({ ...info, config: r.config });
      flash(r.config.enabled ? "Failover enabled" : "Failover disabled");
    } catch (e: any) {
      flash("Toggle failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleMove = async (idx: number, delta: number) => {
    const to = idx + delta;
    if (to < 0 || to >= cfg.chain.length) return;
    setBusy(true);
    try {
      const r = await reorderFailover(idx, to);
      setInfo({ ...info, config: r.config });
    } catch (e: any) {
      flash("Reorder failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleRemove = async (idx: number) => {
    const newChain = [...cfg.chain];
    newChain.splice(idx, 1);
    await save({ ...cfg, chain: newChain });
  };

  const handleAdd = async () => {
    if (!newProviderId || !newModel) {
      flash("Pick a provider and a model first.");
      return;
    }
    // De-dupe — same backend rule, but giving the UI an early hint.
    if (cfg.chain.some((e) => e.provider_id === newProviderId && e.model === newModel)) {
      flash("That provider/model is already in the chain.");
      return;
    }
    await save({
      ...cfg,
      chain: [...cfg.chain, { provider_id: newProviderId, model: newModel }],
    });
    setNewProviderId("");
    setNewModel("");
  };

  const toggleCategory = async (cat: string) => {
    const set = new Set(cfg.retry_on);
    if (set.has(cat)) set.delete(cat);
    else set.add(cat);
    await save({ ...cfg, retry_on: Array.from(set) });
  };

  const handleMaxAttempts = async (n: number) => {
    await save({ ...cfg, max_attempts: Math.max(1, Math.min(10, n)) });
  };

  // For the new-entry form: filter availableModels to the selected provider.
  const modelsForNewProvider = availableModels.filter(
    (m) => m.provider_id === newProviderId,
  );

  return (
    <div className="bg-slate-800 rounded p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-semibold">Failover chain</h4>
          <p className="text-xs text-slate-400 mt-0.5">
            When the active model returns a retryable error, try these in order.
            Logged into each Job's <span className="font-mono">attempts[]</span> so you can
            see what was tried in <span className="font-mono">Settings → Jobs</span>.
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={cfg.enabled}
            onChange={handleToggle}
            disabled={busy}
            className="accent-amber-500"
          />
          <span className={cfg.enabled ? "text-amber-400" : "text-slate-500"}>
            {cfg.enabled ? "Enabled" : "Disabled"}
          </span>
        </label>
      </div>

      {/* Chain — ordered list, drag-by-button. */}
      <div>
        <div className="text-xs text-slate-400 mb-1">CHAIN ORDER</div>
        {cfg.chain.length === 0 ? (
          <div className="text-sm text-slate-500 italic py-2">
            (empty — add one or more fallback providers below)
          </div>
        ) : (
          <div className="space-y-1.5">
            {cfg.chain.map((entry, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 bg-slate-900/60 rounded px-2 py-1.5"
              >
                <span className="text-xs text-slate-500 font-mono w-5">{idx + 1}.</span>
                <span className="text-sm font-mono text-slate-200 truncate">
                  {entry.provider_id}
                </span>
                <span className="text-slate-500">/</span>
                <span className="text-sm text-emerald-300 truncate">{entry.model}</span>
                <div className="ml-auto flex items-center gap-1">
                  <button
                    onClick={() => handleMove(idx, -1)}
                    disabled={idx === 0 || busy}
                    className="text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-30 disabled:hover:bg-slate-700 px-1.5 py-0.5 rounded"
                    title="Move up"
                  >
                    ↑
                  </button>
                  <button
                    onClick={() => handleMove(idx, 1)}
                    disabled={idx === cfg.chain.length - 1 || busy}
                    className="text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-30 disabled:hover:bg-slate-700 px-1.5 py-0.5 rounded"
                    title="Move down"
                  >
                    ↓
                  </button>
                  <button
                    onClick={() => handleRemove(idx)}
                    disabled={busy}
                    className="text-xs bg-red-800/70 hover:bg-red-700 px-1.5 py-0.5 rounded"
                    title="Remove"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add new entry */}
      <div>
        <div className="text-xs text-slate-400 mb-1">ADD FALLBACK</div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={newProviderId}
            onChange={(e) => {
              setNewProviderId(e.target.value);
              setNewModel("");
            }}
            className="bg-slate-900 rounded px-2 py-1 text-sm min-w-[180px]"
          >
            <option value="">Pick provider…</option>
            {providers.filter((p) => p.enabled !== false).map((p) => (
              <option key={p.id} value={p.id}>
                {p.name || p.id}
              </option>
            ))}
          </select>
          <select
            value={newModel}
            onChange={(e) => setNewModel(e.target.value)}
            className="bg-slate-900 rounded px-2 py-1 text-sm min-w-[220px]"
            disabled={!newProviderId}
          >
            <option value="">Pick model…</option>
            {modelsForNewProvider.map((m) => (
              <option key={m.model} value={m.model}>
                {m.model}
              </option>
            ))}
          </select>
          <button
            onClick={handleAdd}
            disabled={busy || !newProviderId || !newModel}
            className="text-sm bg-sky-700 hover:bg-sky-600 disabled:opacity-40 px-3 py-1 rounded"
          >
            Add
          </button>
        </div>
      </div>

      {/* Retry policy */}
      <div>
        <div className="text-xs text-slate-400 mb-1">RETRY ON</div>
        <div className="flex flex-wrap gap-1.5">
          {info.categories.map((cat) => {
            const on = cfg.retry_on.includes(cat);
            return (
              <button
                key={cat}
                onClick={() => toggleCategory(cat)}
                disabled={busy}
                className={`text-xs px-2 py-1 rounded transition ${
                  on
                    ? "bg-emerald-700 text-emerald-50"
                    : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                }`}
              >
                {CATEGORY_LABEL[cat] || cat}
              </button>
            );
          })}
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Tip: <span className="font-mono">bad_request</span> and{" "}
          <span className="font-mono">content_policy</span> usually mean the prompt
          itself is the problem — trying another provider will hit the same wall.
        </p>
      </div>

      {/* Max attempts */}
      <div className="flex items-center gap-3">
        <label className="text-xs text-slate-400">MAX ATTEMPTS</label>
        <input
          type="number"
          min={1}
          max={10}
          value={cfg.max_attempts}
          onChange={(e) => handleMaxAttempts(parseInt(e.target.value, 10) || 1)}
          className="bg-slate-900 rounded px-2 py-1 text-sm w-16"
        />
        <span className="text-xs text-slate-500">
          incl. the primary call
        </span>
      </div>
    </div>
  );
}
