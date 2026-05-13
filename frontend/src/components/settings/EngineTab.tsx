import { useEffect, useState } from "react";
import {
  EngineConfigEnvelope,
  EngineRouterCfg,
  EngineVerificationCfg,
  fetchEngineConfig,
  putEngineConfig,
  resetEngineConfig,
} from "../../api";

type Props = { flash: (msg: string) => void };

/** Format helper — show how a current value differs from the default. */
const overrideBadge = (
  field: string,
  overrides: Record<string, unknown> | undefined,
) =>
  overrides && Object.prototype.hasOwnProperty.call(overrides, field) ? (
    <span className="text-[10px] bg-amber-700/40 text-amber-200 rounded px-1.5 py-0.5">
      overridden
    </span>
  ) : null;

export default function EngineTab({ flash }: Props) {
  const [envelope, setEnvelope] = useState<EngineConfigEnvelope | null>(null);
  const [busy, setBusy] = useState(false);

  // Form state (mirrors envelope.effective)
  const [router, setRouter] = useState<EngineRouterCfg>({});
  const [verification, setVerification] = useState<EngineVerificationCfg>({});

  const refresh = async () => {
    try {
      const r = await fetchEngineConfig();
      setEnvelope(r);
      setRouter(r.effective.router || {});
      setVerification(r.effective.verification || {});
    } catch (e: any) {
      flash("Engine config load failed: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleSave = async () => {
    setBusy(true);
    try {
      const r = await putEngineConfig({ router, verification });
      setEnvelope(r);
      const rejN = r.rejected?.length || 0;
      if (rejN > 0) {
        flash(`Saved — but rejected: ${r.rejected.join(", ")}`);
      } else {
        flash("Engine config saved (live).");
      }
    } catch (e: any) {
      flash("Save failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    if (!confirm("Reset router + verification to defaults? This wipes runtime_overrides.json.")) {
      return;
    }
    setBusy(true);
    try {
      const r = await resetEngineConfig();
      setEnvelope(r);
      setRouter(r.effective.router || {});
      setVerification(r.effective.verification || {});
      flash("Engine config reset to defaults.");
    } catch (e: any) {
      flash("Reset failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const updateRouter = <K extends keyof EngineRouterCfg>(k: K, v: EngineRouterCfg[K]) =>
    setRouter((prev) => ({ ...prev, [k]: v }));

  const updateVerif = <K extends keyof EngineVerificationCfg>(
    k: K,
    v: EngineVerificationCfg[K],
  ) => setVerification((prev) => ({ ...prev, [k]: v }));

  const numInput = (
    value: number | undefined,
    onChange: (n: number) => void,
    step = 1,
    min?: number,
    max?: number,
  ) => (
    <input
      type="number"
      step={step}
      min={min}
      max={max}
      value={value ?? ""}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-32 bg-slate-900 rounded px-2 py-1 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
    />
  );

  const boolInput = (value: boolean | undefined, onChange: (b: boolean) => void) => (
    <input
      type="checkbox"
      checked={!!value}
      onChange={(e) => onChange(e.target.checked)}
      className="w-4 h-4 accent-sky-600"
    />
  );

  const Row = ({
    label,
    field,
    section,
    children,
    hint,
  }: {
    label: string;
    field: string;
    section: "router" | "verification";
    children: React.ReactNode;
    hint?: string;
  }) => (
    <div className="flex items-center gap-3 py-1.5">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-slate-300">{label}</span>
          {overrideBadge(field, envelope?.overrides[section] as any)}
        </div>
        {hint && <div className="text-[11px] text-slate-500">{hint}</div>}
      </div>
      {children}
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-bold">Engine — Router & Verification</h3>
        <div className="flex gap-2">
          <button
            onClick={refresh}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Refresh
          </button>
          <button
            onClick={handleReset}
            disabled={busy}
            className="bg-rose-800/70 hover:bg-rose-700 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Reset to defaults
          </button>
          <button
            onClick={handleSave}
            disabled={busy}
            className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Save (applies live)
          </button>
        </div>
      </div>

      <div className="bg-slate-800/60 border border-slate-700 rounded p-3 text-[11px] text-slate-400 space-y-1">
        <div>
          Changes take effect for the <b>next</b> request — no restart needed.
          In-flight requests keep their snapshotted values. Values shown here
          are the <i>effective</i> ones (defaults + your overrides).
        </div>
        <div>
          Saved to <span className="font-mono">knowledge/runtime_overrides.json</span>.
          Reset wipes that file and restores the mode preset.
        </div>
      </div>

      {/* Router section */}
      <div className="bg-slate-800 rounded p-3 space-y-1">
        <div className="text-sm font-semibold mb-2">Router (Model A ↔ Model B)</div>

        <Row
          label="daily_api_budget_usd"
          field="daily_api_budget_usd"
          section="router"
          hint="Hard daily ceiling for paid API calls (USD). Router falls back to Model B once exceeded."
        >
          {numInput(router.daily_api_budget_usd, (v) => updateRouter("daily_api_budget_usd", v), 0.5, 0, 1000)}
        </Row>

        <Row
          label="estimated_cost_per_call_usd"
          field="estimated_cost_per_call_usd"
          section="router"
          hint="Per-call cost estimate used to gate against the budget before a call fires."
        >
          {numInput(router.estimated_cost_per_call_usd, (v) => updateRouter("estimated_cost_per_call_usd", v), 0.001, 0, 10)}
        </Row>

        <Row
          label="api_ping_cache_seconds"
          field="api_ping_cache_seconds"
          section="router"
          hint="How long to trust a successful API liveness probe before re-pinging."
        >
          {numInput(router.api_ping_cache_seconds, (v) => updateRouter("api_ping_cache_seconds", v), 5, 0, 3600)}
        </Row>

        <Row
          label="fallback_to_local"
          field="fallback_to_local"
          section="router"
          hint="On Claude failure / budget exhaustion, retry with Model B (local Qwen)."
        >
          {boolInput(router.fallback_to_local, (v) => updateRouter("fallback_to_local", v))}
        </Row>

        <Row
          label="tool_synth_max_tokens"
          field="tool_synth_max_tokens"
          section="router"
          hint="Max tokens granted to the forced synthesis call when tool-loop hits max_iterations."
        >
          {numInput(router.tool_synth_max_tokens, (v) => updateRouter("tool_synth_max_tokens", v), 100, 256, 16000)}
        </Row>

        <Row
          label="tool_loop_input_budget"
          field="tool_loop_input_budget"
          section="router"
          hint="Hard ceiling on summed input tokens across one whole tool-loop (runaway guard)."
        >
          {numInput(router.tool_loop_input_budget, (v) => updateRouter("tool_loop_input_budget", v), 10000, 10000, 2000000)}
        </Row>
      </div>

      {/* Verification section */}
      <div className="bg-slate-800 rounded p-3 space-y-1">
        <div className="text-sm font-semibold mb-2">Verification & Self-Critic</div>

        <Row
          label="enabled"
          field="enabled"
          section="verification"
          hint="Turn the post-answer verification stage on/off entirely."
        >
          {boolInput(verification.enabled, (v) => updateVerif("enabled", v))}
        </Row>

        <Row
          label="min_confidence"
          field="min_confidence"
          section="verification"
          hint="Minimum confidence to ship an answer. Below this, the verifier flags it."
        >
          {numInput(verification.min_confidence, (v) => updateVerif("min_confidence", v), 1, 0, 100)}
        </Row>

        <Row
          label="require_sources"
          field="require_sources"
          section="verification"
          hint="Require the answer to cite evidence (Knowledge / Notes / Workspace)."
        >
          {boolInput(verification.require_sources, (v) => updateVerif("require_sources", v))}
        </Row>

        <Row
          label="critic_threshold"
          field="critic_threshold"
          section="verification"
          hint="Self-critic confidence below which a retry is triggered."
        >
          {numInput(verification.critic_threshold, (v) => updateVerif("critic_threshold", v), 1, 0, 100)}
        </Row>

        <Row
          label="critic_max_retries"
          field="critic_max_retries"
          section="verification"
          hint="How many times the self-critic can request a retry on a single turn."
        >
          {numInput(verification.critic_max_retries, (v) => updateVerif("critic_max_retries", v), 1, 0, 10)}
        </Row>

        <Row
          label="critic_retry_token_budget"
          field="critic_retry_token_budget"
          section="verification"
          hint="Stop retrying once the whole turn has consumed this many total tokens."
        >
          {numInput(
            verification.critic_retry_token_budget,
            (v) => updateVerif("critic_retry_token_budget", v),
            1000,
            1000,
            1000000,
          )}
        </Row>
      </div>
    </div>
  );
}
