import { useEffect, useState } from "react";
import {
  EngineConfigEnvelope,
  EngineKnowledgeCfg,
  EngineRouterCfg,
  EngineVerificationCfg,
  EngineWorkspaceCfg,
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

/** Human names for the raw config keys.
 *
 * Every row on this screen was labelled with its key —
 * `daily_api_budget_usd`, `api_ping_cache_seconds` — in monospace, so the
 * screen read like a JSON file. The keys still matter (they are what
 * lands in knowledge/runtime_overrides.json, which this panel names), so
 * they stay visible as the secondary line rather than disappearing.
 */
const FIELD_NAMES: Record<string, string> = {
  daily_api_budget_usd: "Daily spending limit",
  estimated_cost_per_call_usd: "Assumed cost per call",
  api_ping_cache_seconds: "Trust a liveness check for",
  fallback_to_local: "Fall back to the local model",
  tool_synth_max_tokens: "Tool-result summary length",
  tool_loop_input_budget: "Tool loop input budget",
  tool_loop_max_tool_calls: "Tool calls per turn",
  audit_loop_input_budget: "Audit loop input budget",
  audit_loop_max_iterations: "Audit loop iterations",
  audit_loop_max_tool_calls: "Audit loop tool calls",
  enabled: "Enabled",
  min_confidence: "Minimum confidence to answer",
  require_sources: "Require a source for every claim",
  critic_threshold: "Send for revision below",
  critic_max_retries: "Revision attempts",
  critic_retry_token_budget: "Revision token budget",
  core_memory_max_tokens: "Core memory size limit",
  note_max_tokens: "Note size limit",
  auto_promote_threshold: "Auto-promote a note at",
  finetune_min_examples: "Examples needed to train",
  inbox_retention_days: "Keep incoming files for",
  outbox_retention_days: "Keep outgoing files for",
  notes_retention_days: "Keep notes for",
  turns_retention_days: "Keep conversation turns for",
};

export default function EngineTab({ flash }: Props) {
  const [envelope, setEnvelope] = useState<EngineConfigEnvelope | null>(null);
  const [busy, setBusy] = useState(false);

  // Form state (mirrors envelope.effective)
  const [router, setRouter] = useState<EngineRouterCfg>({});
  const [verification, setVerification] = useState<EngineVerificationCfg>({});
  const [workspace, setWorkspace] = useState<EngineWorkspaceCfg>({});
  const [knowledge, setKnowledge] = useState<EngineKnowledgeCfg>({});

  const refresh = async () => {
    try {
      const r = await fetchEngineConfig();
      setEnvelope(r);
      setRouter(r.effective.router || {});
      setVerification(r.effective.verification || {});
      setWorkspace(r.effective.workspace || {});
      setKnowledge(r.effective.knowledge || {});
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
      const r = await putEngineConfig({ router, verification, workspace, knowledge });
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
    if (!confirm("Reset every setting on this page to its default?\n\nThis discards all your overrides and restores the values that come with the current mode.")) {
      return;
    }
    setBusy(true);
    try {
      const r = await resetEngineConfig();
      setEnvelope(r);
      setRouter(r.effective.router || {});
      setVerification(r.effective.verification || {});
      setWorkspace(r.effective.workspace || {});
      setKnowledge(r.effective.knowledge || {});
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

  const updateWs = <K extends keyof EngineWorkspaceCfg>(k: K, v: EngineWorkspaceCfg[K]) =>
    setWorkspace((prev) => ({ ...prev, [k]: v }));

  const updateKb = <K extends keyof EngineKnowledgeCfg>(k: K, v: EngineKnowledgeCfg[K]) =>
    setKnowledge((prev) => ({ ...prev, [k]: v }));

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
    section: "router" | "verification" | "workspace" | "knowledge";
    children: React.ReactNode;
    hint?: string;
  }) => (
    <div className="flex items-center gap-3 py-1.5">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm">{FIELD_NAMES[label] || label}</span>
          {overrideBadge(field, envelope?.overrides[section] as any)}
        </div>
        {hint && <div className="text-[11px] text-ink-dim">{hint}</div>}
        {FIELD_NAMES[label] && (
          <div className="font-mono text-[10px] text-ink-faint">{label}</div>
        )}
      </div>
      {children}
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          <button
            onClick={refresh}
            disabled={busy}
            className="rounded-lg border border-edge-strong px-3 py-1.5 text-xs text-ink-dim hover:bg-surface-hover hover:text-ink disabled:opacity-40"
          >
            Refresh
          </button>
          <button
            onClick={handleReset}
            disabled={busy}
            title="Discard every change on this page and go back to the mode preset"
            className="rounded-lg border border-edge-strong px-3 py-1.5 text-xs text-ink-dim hover:bg-danger hover:text-white disabled:opacity-40"
          >
            Reset all
          </button>
          <button
            onClick={handleSave}
            disabled={busy}
            className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40"
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
          hint="When the main model fails or the budget runs out, retry on the fallback model."
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

        <Row
          label="tool_loop_max_tool_calls"
          field="tool_loop_max_tool_calls"
          section="router"
          hint="Optional normal-mode tool-call ceiling. Zero keeps the generous unlimited default."
        >
          {numInput(router.tool_loop_max_tool_calls, (v) => updateRouter("tool_loop_max_tool_calls", v), 1, 0, 5000)}
        </Row>

        <div className="pt-3 pb-1 text-xs font-semibold text-slate-300">
          Read-only audit budget
        </div>

        <Row
          label="audit_loop_max_iterations"
          field="audit_loop_max_iterations"
          section="router"
          hint="Maximum LLM iterations in one explicit audit-mode turn."
        >
          {numInput(router.audit_loop_max_iterations, (v) => updateRouter("audit_loop_max_iterations", v), 1, 1, 200)}
        </Row>

        <Row
          label="audit_loop_max_tool_calls"
          field="audit_loop_max_tool_calls"
          section="router"
          hint="Maximum admitted read-only tool calls in one audit-mode turn."
        >
          {numInput(router.audit_loop_max_tool_calls, (v) => updateRouter("audit_loop_max_tool_calls", v), 1, 1, 500)}
        </Row>

        <Row
          label="audit_loop_input_budget"
          field="audit_loop_input_budget"
          section="router"
          hint="Maximum accumulated input tokens in one audit-mode turn."
        >
          {numInput(router.audit_loop_input_budget, (v) => updateRouter("audit_loop_input_budget", v), 10000, 10000, 500000)}
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

      {/* Workspace retention section */}
      <div className="bg-slate-800 rounded p-3 space-y-1">
        <div className="text-sm font-semibold mb-2">
          Workspace retention (auto-sweep, in days)
        </div>
        <div className="text-[11px] text-slate-500 mb-2">
          <b>0</b> means "never auto-delete this subtree". Sweep runs as part
          of the autonomic loop; changes apply on the next tick.
        </div>

        <Row
          label="inbox_retention_days"
          field="inbox_retention_days"
          section="workspace"
          hint="Uploaded files (workspace/inbox/). Default 90 — uploads accumulate fast."
        >
          {numInput(workspace.inbox_retention_days, (v) => updateWs("inbox_retention_days", v), 1, 0, 3650)}
        </Row>

        <Row
          label="outbox_retention_days"
          field="outbox_retention_days"
          section="workspace"
          hint="Agent outputs (workspace/outbox/). Default 0 — never sweep agent-produced work."
        >
          {numInput(workspace.outbox_retention_days, (v) => updateWs("outbox_retention_days", v), 1, 0, 3650)}
        </Row>

        <Row
          label="notes_retention_days"
          field="notes_retention_days"
          section="workspace"
          hint="Scratch notes (workspace/notes/). Default 0 — keep indefinitely."
        >
          {numInput(workspace.notes_retention_days, (v) => updateWs("notes_retention_days", v), 1, 0, 3650)}
        </Row>

        <Row
          label="turns_retention_days"
          field="turns_retention_days"
          section="workspace"
          hint="Per-turn artifact JSONs (workspace/turns/). Default 30 — accumulates fastest of all."
        >
          {numInput(workspace.turns_retention_days, (v) => updateWs("turns_retention_days", v), 1, 0, 3650)}
        </Row>
      </div>

      {/* Knowledge caps section */}
      <div className="bg-slate-800 rounded p-3 space-y-1">
        <div className="text-sm font-semibold mb-2">Knowledge capacity</div>

        <Row
          label="core_memory_max_tokens"
          field="core_memory_max_tokens"
          section="knowledge"
          hint="Cap on the ambient context bundle (system block) sent on every turn."
        >
          {numInput(
            knowledge.core_memory_max_tokens,
            (v) => updateKb("core_memory_max_tokens", v),
            100,
            256,
            32000,
          )}
        </Row>

        <Row
          label="auto_promote_threshold"
          field="auto_promote_threshold"
          section="knowledge"
          hint="A topic must be referenced this many times before auto-promotion to core memory."
        >
          {numInput(knowledge.auto_promote_threshold, (v) => updateKb("auto_promote_threshold", v), 1, 1, 1000)}
        </Row>

        <Row
          label="finetune_min_examples"
          field="finetune_min_examples"
          section="knowledge"
          hint="Minimum Q&A pairs in the finetune queue before training is allowed."
        >
          {numInput(knowledge.finetune_min_examples, (v) => updateKb("finetune_min_examples", v), 5, 1, 100000)}
        </Row>

        <Row
          label="note_max_tokens"
          field="note_max_tokens"
          section="knowledge"
          hint="Cap on a single note's body before it's split / refused."
        >
          {numInput(knowledge.note_max_tokens, (v) => updateKb("note_max_tokens", v), 50, 100, 16000)}
        </Row>
      </div>
    </div>
  );
}
