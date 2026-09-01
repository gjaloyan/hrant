/** Human names for the raw config keys.
 *
 * Every row on this screen was labelled with its key —
 * `daily_api_budget_usd`, `api_ping_cache_seconds` — in monospace, so the
 * screen read like a JSON file. The keys still matter (they are what
 * lands in knowledge/runtime_overrides.json, which this panel names), so
 * they stay visible as the secondary line rather than disappearing.
 */
export const FIELD_NAMES: Record<string, string> = {
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
