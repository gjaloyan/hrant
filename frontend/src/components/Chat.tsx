import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Badge } from "../ui";
import {
  AgentAnswer, chatStream, StreamEvent, addFromChat, addCorrection, fetchCurrentSession,
  fetchActiveModel, setActiveModel, clearActiveModel,
  fetchTurn, fetchConversation,
  type ActiveModelSelection, type AvailableModel,
  type AttachmentMeta, uploadAttachment, transcribeAudio, attachmentUrl,
  type ThinkingStep, type TurnArtifact, type TokenUsage,
} from "../api";
import { Markdown } from "../markdown";

type Msg =
  | { role: "user"; text: string; attachments?: AttachmentMeta[] }
  | {
      role: "agent";
      text: string;
      meta?: AgentAnswer;
      progress?: string[];
      // Phase 2: SSE-level error (LLM dropped, transport error,
      // unhandled exception in the runner). When present, an
      // ErrorCard replaces the answer body. Separate from
      // verification.contradictions which are content-level.
      error?: string;
      // Round A: when the message was restored from session/conversation
      // history without its full thinking_trace, turn_id points at the
      // on-disk TurnWorkspace artefact. The first time the user expands
      // a tool card on this message, we lazy-fetch via /api/turns/<id>
      // and populate `meta.thinking_trace` so subsequent expands are
      // instant. `lazy_loading` flips to true while the fetch is
      // in-flight so the card can render a "loading…" hint.
      turn_id?: string;
      lazy_loading?: boolean;
      lazy_error?: string;
      // Round F-pre: counts surfaced from the persisted session/conv
      // row so we can render "🔧 tools: N calls" + "🔬 LLM calls: N"
      // on restored messages WITHOUT lazy-fetching the full trace.
      // Live messages set these via the answer event.
      n_tool_calls?: number;
      n_llm_calls?: number;
    };

export type ChatHandle = {
  clearMessages: () => void;
};

function ConfidenceBadge({ c }: { c: number }) {
  const tone = c >= 90 ? "ok" : c >= 70 ? "warn" : "danger";
  return (
    <Badge
      tone={tone}
      title={
        "How sure the agent is that it verified its own answer. " +
        "Low means it could not check its claims, not that the answer is wrong."
      }
    >
      self-check {c}%
    </Badge>
  );
}

function LLMCallItem({ call }: { call: import("../api").LLMCallDetail }) {
  // Compact summary; expand for redacted system + user prompts.
  return (
    <details className="bg-slate-950/40 rounded px-2 py-1 text-[11px]">
      <summary className="cursor-pointer flex items-center gap-2 flex-wrap">
        <span className="text-slate-400 font-mono">{call.label}</span>
        <span className="text-slate-500">{call.task_type}</span>
        {call.model && <span className="text-slate-500 font-mono">{call.model}</span>}
        {call.duration_ms > 0 && (
          <span className="text-slate-500">{(call.duration_ms / 1000).toFixed(1)}s</span>
        )}
        {(call.input_tokens > 0 || call.output_tokens > 0) && (
          <span className="text-slate-500">
            {call.input_tokens.toLocaleString()} in / {call.output_tokens.toLocaleString()} out
          </span>
        )}
      </summary>
      <div className="mt-2 space-y-2 text-[10px]">
        <div>
          <div className="text-slate-500 mb-0.5">system (redacted, file blobs replaced):</div>
          <pre className="bg-slate-900 rounded p-2 max-h-64 overflow-auto whitespace-pre-wrap break-words">
            {call.system_redacted}
          </pre>
        </div>
        <div>
          <div className="text-slate-500 mb-0.5">user (redacted):</div>
          <pre className="bg-slate-900 rounded p-2 max-h-64 overflow-auto whitespace-pre-wrap break-words">
            {call.user_redacted}
          </pre>
        </div>
        {call.response_preview && (
          <div>
            <div className="text-slate-500 mb-0.5">response (first 600 chars):</div>
            <pre className="bg-slate-900 rounded p-2 max-h-32 overflow-auto whitespace-pre-wrap break-words">
              {call.response_preview}
            </pre>
          </div>
        )}
      </div>
    </details>
  );
}

// OpenClaw-style tool card. Round F: progressive reveal — the
// agent emits two events per tool, "tool_starting" (just args,
// before execution) and "tool" / "tool_error" (with result, after
// execution). Each shows as its own card in the trace order:
//   ▸ ⚡ Tool call · read_file        (pending …)
//   ▸ ⚡ Tool output · read_file      (got 4 KB ✓)
// matching OpenClaw screenshots. The `kind` is auto-derived from
// the step's event so the same component renders both.
function ToolCallCard({ step }: { step: ThinkingStep }) {
  const tc = step.tool_call;
  const [open, setOpen] = useState(false);
  if (!tc) return null;
  const isStarting = step.event === "tool_starting";
  const isError = tc.is_error || step.event === "tool_error";
  const hasResult = Boolean((tc.result || "").trim());
  const status = isError ? "error" : isStarting ? "pending" : hasResult ? "ok" : "pending";
  const argsKeys = Object.keys(tc.args || {});
  const hasArgs = argsKeys.length > 0;
  const summary = buildToolSummary(tc);
  const headerLabel = isError
    ? "tool error"
    : isStarting
      ? "tool call"
      : "tool output";

  return (
    <div
      className={`rounded-md overflow-hidden border ${
        status === "error"
          ? "bg-rose-950/30 border-rose-900/40"
          : "bg-neutral-900/60 border-white/[0.06]"
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-[11px] text-left hover:bg-white/[0.03] transition-colors"
      >
        <span className="text-neutral-600 select-none leading-none">·</span>
        <span
          className={`select-none leading-none ${
            status === "error" ? "text-rose-400" : "text-amber-400/90"
          }`}
          aria-hidden
        >
          ⚡
        </span>
        <span className="font-medium text-neutral-200 lowercase">
          {headerLabel}
        </span>
        <span className="font-mono text-[10px] text-amber-400/70 px-1.5 py-0.5 rounded bg-white/[0.04]">
          {tc.name}
        </span>
        {summary && !open && (
          <span className="hidden sm:inline truncate text-[10px] text-neutral-500 italic">
            {summary}
          </span>
        )}
        <span
          className={`ml-auto text-[10px] select-none ${
            status === "error"
              ? "text-rose-400"
              : status === "ok"
                ? "text-emerald-500/70"
                : "text-neutral-600"
          }`}
        >
          {status === "error" ? "✗" : status === "ok" ? "✓" : "…"}
        </span>
        <span className="text-[10px] text-neutral-600 select-none ml-1">
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="px-3 pb-2 pt-1 border-t border-white/[0.05] space-y-2 text-[11px]">
          {summary && (
            <div className="text-neutral-400 italic leading-snug">{summary}</div>
          )}
          {hasArgs && (
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em] text-neutral-500 mb-1">
                Tool input
              </div>
              <pre className="bg-black/40 rounded p-2 overflow-x-auto whitespace-pre-wrap break-words text-[10px] leading-snug max-h-64 text-neutral-200">
                {JSON.stringify(tc.args, null, 2)}
              </pre>
            </div>
          )}
          {hasResult && (
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em] text-neutral-500 mb-1 flex items-center gap-2">
                <span>Tool output{tc.is_error ? " · error" : ""}</span>
                {tc.result_truncated && tc.result_full_len ? (
                  <span className="text-amber-400/80 normal-case tracking-normal text-[9px]">
                    preview, {tc.result.length.toLocaleString()} of{" "}
                    {tc.result_full_len.toLocaleString()} chars
                  </span>
                ) : null}
              </div>
              <ToolResultBody text={tc.result || ""} />
            </div>
          )}
          <div className="flex items-center gap-3 text-[9px] text-neutral-600 pt-1 border-t border-white/[0.04]">
            <span>{step.ts.toFixed(1)}s</span>
            {tc.duration_ms ? (
              <span>{(tc.duration_ms / 1000).toFixed(1)}s exec</span>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}

// One-line italic detail under the header, OpenClaw-style. Picks the
// most identifying argument(s) for the tool. The pill is one
// tool-pair (call + output) per ThinkingStep, so the summary always
// describes the call's intent — `read_file:backend/llm.py:1026-1180`,
// `calc:2+2`, etc. Falls back to a comma-joined args list for tools
// we don't have a custom format for.
function buildToolSummary(
  tc: NonNullable<ThinkingStep["tool_call"]>,
): string {
  const args = tc.args || {};
  const a = (k: string): string => {
    const v = (args as Record<string, unknown>)[k];
    return typeof v === "string" ? v : v == null ? "" : JSON.stringify(v);
  };
  if (tc.name === "read_file" || tc.name === "view_file") {
    const path = a("path");
    const s = a("start_line");
    const e = a("end_line");
    if (path && s && e) return `${path}:${s}-${e}`;
    if (path) return path;
  }
  if (tc.name === "locate_symbol") {
    const sym = a("name");
    const path = a("path");
    if (sym && path) return `${sym} in ${path}`;
  }
  if (tc.name === "calc") {
    const expr = a("expression") || a("expr");
    if (expr) return expr;
  }
  if (tc.name === "web_search" || tc.name === "fetch_url") {
    const target = a("query") || a("url");
    if (target) return target;
  }
  if (tc.name === "save_to_workspace") {
    const fn = a("filename");
    const sd = a("subdir") || "outbox";
    if (fn) return `${sd}/${fn}`;
  }
  // Generic fallback: comma-separated key=value list, truncated.
  const argsKeys = Object.keys(args);
  if (argsKeys.length === 0) return "";
  return argsKeys
    .map((k) => {
      const s = a(k);
      return `${k}=${s.length > 40 ? s.slice(0, 40) + "…" : s}`;
    })
    .join(", ");
}

// Result body. If the text parses as JSON, render with indent;
// otherwise show as preformatted text. Capped height with overflow.
function ToolResultBody({ text }: { text: string }) {
  let formatted = text;
  let isJson = false;
  const trimmed = text.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      formatted = JSON.stringify(JSON.parse(trimmed), null, 2);
      isJson = true;
    } catch {
      /* leave as plain text */
    }
  }
  return (
    <pre
      className={`bg-slate-950/80 rounded p-2 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-snug max-h-64 ${
        isJson ? "text-slate-300" : "text-slate-300"
      }`}
    >
      {formatted}
    </pre>
  );
}

// ─── Phase 2 card primitives ──────────────────────────────────────
//
// Each agent-message renders a stack of cards: AnswerCard,
// optional ToolTrace details, optional AskUserCard (Phase 1),
// optional TaskStatusCard (live bg-job polling), optional
// ErrorCard. Shared visual conventions:
//   • rounded-lg border + dark-but-not-black background
//   • header strip with icon + uppercase tracking-wide title +
//     status pill on the right
//   • body padded, scrollable when content overflows
//   • color signal: emerald=ok, amber=running, rose=error,
//     violet=question, slate=neutral/info
//
// Cards are intentionally siblings (not nested in one
// AnswerCard) so each can be styled / collapsed independently
// and the existing thinking-trace / token-usage footers under
// the message stay where users expect them.


// AnswerCard — renders the agent's final answer text with the
// markdown renderer (code fences with copy buttons, inline code
// highlights, bold/italic, lists, headings). For chat answers
// (short, often single-line) the card chrome is intentionally
// minimal — just the markdown body so quick replies don't look
// over-decorated. For task-mode answers (longer, multi-paragraph)
// we wrap in a card with a subtle header so the user can tell
// "this is the final answer" apart from intermediate trace.
function AnswerCard({
  text,
  is_chat,
}: {
  text: string;
  is_chat: boolean;
}) {
  if (!text) return null;
  // Chat-fast-path: render inline, no card chrome.
  if (is_chat) {
    return (
      <div className="text-slate-100">
        <Markdown source={text} />
      </div>
    );
  }
  return (
    <div className="rounded-lg overflow-hidden border border-slate-700/40 bg-slate-900/40">
      <div className="px-3 py-1.5 border-b border-slate-700/40 bg-slate-900/30 flex items-center gap-2">
        <span className="text-emerald-400 leading-none" aria-hidden>💬</span>
        <span className="text-[10px] uppercase tracking-[0.08em] text-slate-400 font-medium">
          Answer
        </span>
      </div>
      <div className="px-3 py-2 text-[13px]">
        <Markdown source={text} />
      </div>
    </div>
  );
}


// TaskStatusCard — live status for a background job the agent
// spawned via `start_background_job`. Polls /api/background-jobs/<id>
// every 5s while the job is running; shows progress bar when the
// job set `total_units` + `progress_probe_cmd`. On terminal status
// the polling stops + the card flips colour (emerald=done,
// rose=error/killed/vanished, amber=interrupted/stale).
function TaskStatusCard({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<any | null>(null);
  const [progress, setProgress] = useState<{ done: number | null; total: number | null; pct: number | null } | null>(null);
  const [error, setError] = useState<string>("");
  const [open, setOpen] = useState(false);
  const [logTail, setLogTail] = useState<string>("");
  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
      try {
        const jr = await fetch(`/api/background-jobs/${encodeURIComponent(jobId)}`);
        if (!jr.ok) {
          if (!cancelled) setError(`${jr.status} ${jr.statusText}`);
          return;
        }
        const jdata = await jr.json();
        if (cancelled) return;
        setJob(jdata);
        if (jdata.progress_probe_cmd && jdata.total_units) {
          try {
            const pr = await fetch(`/api/background-jobs/${encodeURIComponent(jobId)}/progress`);
            if (pr.ok && !cancelled) {
              setProgress(await pr.json());
            }
          } catch { /* probe failures are non-fatal */ }
        }
      } catch (e: any) {
        if (!cancelled) setError(e?.message || String(e));
      }
    };
    fetchAll();
    // Audit follow-up (M-6): poll every 15s instead of 5s; STOP
    // polling entirely once the job reaches a terminal state. The
    // old 5s tick at terminal would run forever as a no-op,
    // burning a setInterval slot per visible job. For a 9-hour
    // benchmark this drops from ~6500 polls to ~2200, and to 0
    // once the job ends. Live updates remain responsive — 15s is
    // imperceptible against multi-hour runtimes.
    const terminalStatuses = new Set([
      "done", "error", "killed", "interrupted", "vanished",
    ]);
    const status = job?.status || "";
    if (terminalStatuses.has(status)) {
      return () => { cancelled = true; };
    }
    const iv = setInterval(() => {
      if (cancelled) return;
      const s = job?.status || "";
      if (s === "running" || s === "" || s === "stale" || !s) {
        fetchAll();
      } else {
        // Reached a terminal state during this poll cycle —
        // self-stop so we don't keep firing setInterval ticks.
        clearInterval(iv);
      }
    }, 15_000);
    return () => { cancelled = true; clearInterval(iv); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, job?.status]);

  const loadLog = async () => {
    try {
      const r = await fetch(
        `/api/background-jobs/${encodeURIComponent(jobId)}/log?stream=stdout&tail=4000`,
      );
      if (r.ok) {
        const d = await r.json();
        setLogTail(d.tail || "");
      }
    } catch { /* swallow */ }
  };

  if (!job) {
    return (
      <div className="rounded-lg border border-slate-700/40 bg-slate-900/30 px-3 py-2 text-[11px] text-slate-400">
        📊 loading background job <code className="font-mono">{jobId}</code>
        {error && <span className="text-rose-400 ml-2">· {error}</span>}
      </div>
    );
  }
  const status: string = job.status || "?";
  const isTerminal = !(status === "running" || status === "" || status === "stale");
  const accent = status === "done"
    ? "border-emerald-700/40 bg-emerald-950/20"
    : ["error", "killed", "vanished"].includes(status)
      ? "border-rose-700/40 bg-rose-950/20"
      : ["interrupted", "stale"].includes(status)
        ? "border-amber-700/40 bg-amber-950/20"
        : "border-sky-700/40 bg-sky-950/20";
  const statusIcon = status === "done"
    ? "✅"
    : ["error", "killed", "vanished"].includes(status)
      ? "❌"
      : ["interrupted", "stale"].includes(status)
        ? "⚠️"
        : "📊";
  const elapsed = (() => {
    const started = job.started_at || 0;
    const finished = job.finished_at || Math.floor(Date.now() / 1000);
    const total = Math.max(0, Math.floor(finished - started));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
  })();
  const pct = progress?.pct ?? null;
  const done = progress?.done ?? null;
  const total = progress?.total ?? job.total_units ?? null;
  const pctInt = pct !== null ? Math.round(pct * 100) : null;

  return (
    <div className={`rounded-lg overflow-hidden border ${accent}`}>
      <div className="px-3 py-1.5 flex items-center gap-2 border-b border-white/[0.04]">
        <span className="leading-none" aria-hidden>{statusIcon}</span>
        <span className="text-[10px] uppercase tracking-[0.08em] text-slate-300 font-medium">
          Background job
        </span>
        <span className="text-xs text-slate-200">{job.label || job.job_id}</span>
        <span className="ml-auto text-[10px] uppercase font-mono text-slate-400">
          {status}{job.exit_code !== null && job.exit_code !== undefined ? ` · exit ${job.exit_code}` : ""}
        </span>
      </div>
      <div className="px-3 py-2 text-[11px] text-slate-300 space-y-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span><span className="text-slate-500">id:</span> <code className="font-mono text-slate-300">{job.job_id}</code></span>
          <span><span className="text-slate-500">elapsed:</span> {elapsed}</span>
          {job.retry_count > 0 && <span><span className="text-slate-500">retry:</span> {job.retry_count}</span>}
          {job.pid && <span><span className="text-slate-500">pid:</span> {job.pid}</span>}
        </div>
        {pctInt !== null && total !== null && (
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-slate-400">progress</span>
              <span className="text-slate-200 tabular-nums">{pctInt}%</span>
              <span className="text-slate-500">({done}/{total})</span>
            </div>
            <div className="h-1.5 rounded bg-slate-800 overflow-hidden">
              <div
                className={`h-full ${isTerminal ? "bg-emerald-500" : "bg-sky-500"} transition-all`}
                style={{ width: `${Math.min(100, Math.max(0, pctInt))}%` }}
              />
            </div>
          </div>
        )}
        {job.command && (
          <div className="text-[10px] text-slate-500 truncate" title={job.command}>
            <span className="text-slate-600">$</span> <span className="font-mono">{job.command}</span>
          </div>
        )}
        <details
          open={open}
          onToggle={(e: React.SyntheticEvent<HTMLDetailsElement>) => {
            const o = e.currentTarget.open;
            setOpen(o);
            if (o && !logTail) void loadLog();
          }}
        >
          <summary className="cursor-pointer text-[10px] text-slate-500 hover:text-slate-300">
            stdout (tail)
          </summary>
          <pre className="mt-1 max-h-48 overflow-auto bg-black/40 rounded p-2 text-[10px] leading-snug font-mono whitespace-pre-wrap break-words">
            {logTail || (
              <span className="text-slate-600">loading…</span>
            )}
          </pre>
        </details>
        {isTerminal && job.stderr_tail && (
          <div>
            <div className="text-[10px] text-rose-400 mb-1">stderr (tail)</div>
            <pre className="max-h-32 overflow-auto bg-rose-950/30 rounded p-2 text-[10px] leading-snug font-mono whitespace-pre-wrap break-words text-rose-200">
              {job.stderr_tail}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}


// ErrorCard — surface SSE / tool-loop errors as a structured card
// rather than dropping them into the answer text. Pre-Phase-2 the
// chat would show "ERROR: { type: error, ... }" inline as if it
// were the agent's answer, which read like the agent was confused.
// Now it's a clean rose-themed card the user can spot at a glance.
// Quick reasoning-effort selector for the chat input bar. Reads
// the current override from /api/reasoning-routing and lets the
// user bump GPT-5.x reasoning to high / drop to low for the next
// turn without leaving the chat. "Off" clears the override and
// falls back to the configured per-task routing matrix.
function ReasoningQuickPick({ busy }: { busy: boolean }) {
  const [override, setOverride] = useState<string>("");
  const [open, setOpen] = useState(false);
  const [busyHere, setBusyHere] = useState(false);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { fetchReasoningRouting } = await import("../api");
        const c = await fetchReasoningRouting();
        if (!cancelled) setOverride(c.override || "");
      } catch { /* swallow — non-critical */ }
    })();
    return () => { cancelled = true; };
  }, []);
  const apply = async (level: string) => {
    setBusyHere(true);
    try {
      const { setReasoningOverride } = await import("../api");
      await setReasoningOverride(level);
      setOverride(level);
    } catch { /* swallow */ }
    finally {
      setBusyHere(false);
      setOpen(false);
    }
  };
  const colorFor = (lv: string) => (
    lv === "high" ? "bg-rose-700 text-white"
      : lv === "medium" ? "bg-amber-700 text-white"
        : lv === "low" ? "bg-accent-soft text-accent font-medium"
          : "bg-slate-700 text-slate-300"
  );
  const labelFor = (lv: string) => (
    lv === "high" ? "🧠 high"
      : lv === "medium" ? "🧠 med"
        : lv === "low" ? "🧠 low"
          : "🧠 auto"
  );
  return (
    <div className="relative self-start">
      <button
        onClick={() => setOpen(!open)}
        disabled={busy || busyHere}
        title="Reasoning effort for next turn"
        className={`rounded px-2 text-[11px] font-medium min-h-[44px] sm:min-h-0 sm:py-1 ${
          colorFor(override)
        } disabled:opacity-50 hover:opacity-90 transition-opacity`}
      >
        {labelFor(override)}
      </button>
      {open && (
        <div className="absolute bottom-full mb-1 left-0 bg-slate-900 border border-slate-700 rounded shadow-lg z-10 py-1 min-w-[120px]">
          {["", "low", "medium", "high"].map((lv) => (
            <button
              key={lv || "off"}
              onClick={() => apply(lv)}
              className={`block w-full text-left text-xs px-3 py-1.5 hover:bg-slate-800 ${
                override === lv ? "bg-slate-800 font-medium" : ""
              }`}
            >
              {lv === "" ? "Off (use routing)" : `${labelFor(lv)} (${lv})`}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg overflow-hidden border border-rose-700/40 bg-rose-950/30">
      <div className="px-3 py-1.5 border-b border-rose-800/40 bg-rose-900/30 flex items-center gap-2">
        <span className="text-rose-400 leading-none" aria-hidden>❌</span>
        <span className="text-[10px] uppercase tracking-[0.08em] text-rose-300 font-medium">
          Error
        </span>
      </div>
      <div className="px-3 py-2 text-[12px] text-rose-100 whitespace-pre-wrap break-words leading-snug">
        {message}
      </div>
    </div>
  );
}


// AskUserCard — structured question prompt analogous to Claude
// Code's AskUserQuestion. Rendered when an agent turn ends with a
// `meta.question` payload. Cleaner than free-form "should I X or
// Y?" prose: a tagged header, the question, a "why" subtitle, and
// clickable options. After the user picks, the card disables and
// shows "✓ You picked: …".
function AskUserCard({
  question,
  onAnswered,
}: {
  question: import("../api").PendingQuestion;
  onAnswered: (
    res: import("../api").AgentAnswer & {
      answered_question_id?: string;
      answer_choice?: string;
      answer_text?: string;
    },
  ) => void;
}) {
  const [picking, setPicking] = useState<string | null>(null);
  const [error, setError] = useState<string>("");
  // Track local picked-choice state in case the parent doesn't
  // refresh meta — gives instant feedback.
  const [localPicked, setLocalPicked] = useState<string>(
    question.answered ? (question.answer_choice || "") : "",
  );
  const isAnswered = question.answered || Boolean(localPicked);
  const pickedLabel =
    question.options.find((o) => o.id === (question.answer_choice || localPicked))
      ?.label || "";

  const handlePick = async (optionId: string) => {
    if (picking || isAnswered) return;
    setPicking(optionId);
    setError("");
    try {
      const { answerPendingQuestion } = await import("../api");
      const res = await answerPendingQuestion(question.question_id, optionId);
      setLocalPicked(optionId);
      onAnswered(res);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setPicking(null);
    }
  };

  return (
    <div className="rounded-lg overflow-hidden border border-violet-700/40 bg-violet-950/30">
      <div className="px-3 py-2 flex items-center gap-2 border-b border-violet-700/30 bg-violet-900/20">
        <span className="text-violet-300 leading-none" aria-hidden>
          ❓
        </span>
        <span className="text-[10px] uppercase tracking-[0.08em] text-violet-300 font-medium">
          {question.header || "Question"}
        </span>
        <span className="ml-auto text-[10px] text-violet-400/60 font-mono">
          {question.question_id}
        </span>
      </div>
      <div className="px-3 py-3 space-y-3">
        <div className="text-sm leading-snug text-slate-100">
          {question.question}
        </div>
        {question.why && (
          <div className="text-xs italic text-violet-300/80 leading-snug">
            {question.why}
          </div>
        )}
        <div className="space-y-1.5">
          {question.options.map((opt) => {
            const isPicked = picking === opt.id || localPicked === opt.id;
            const isDefault =
              opt.id === question.default_option_id && !isAnswered;
            return (
              <button
                key={opt.id}
                type="button"
                disabled={isAnswered || picking !== null}
                onClick={() => handlePick(opt.id)}
                className={`w-full text-left rounded border px-3 py-2 transition-colors text-sm ${
                  isPicked
                    ? "border-emerald-600/60 bg-emerald-900/30"
                    : isDefault
                      ? "border-violet-500/40 bg-violet-900/20 hover:bg-violet-900/40"
                      : "border-violet-700/30 bg-slate-900/40 hover:bg-violet-900/30"
                } ${isAnswered ? "opacity-60 cursor-default" : "cursor-pointer"}`}
              >
                <div className="flex items-center gap-2 flex-wrap">
                  {isDefault && (
                    <span className="text-amber-400" title="Recommended">
                      ⭐
                    </span>
                  )}
                  <span className="font-medium">{opt.label}</span>
                  {isPicked && <span className="ml-auto text-emerald-400">✓</span>}
                </div>
                {opt.description && (
                  <div className="text-xs text-slate-400 mt-1 leading-snug">
                    {opt.description}
                  </div>
                )}
              </button>
            );
          })}
        </div>
        {isAnswered && (
          <div className="text-xs text-emerald-400">
            ✓ You picked: <span className="font-medium">{pickedLabel || localPicked}</span>
            {" · "}
            <span className="text-slate-400">agent working…</span>
          </div>
        )}
        {error && (
          <div className="text-xs text-rose-400">⚠ {error}</div>
        )}
      </div>
    </div>
  );
}

// `onNewSession` used to live here as a button stacked under Send. It moved
// to the app header on 2026-09-01: starting a fresh conversation is one of
// the two things anyone does most, and it should not be a small grey square
// competing with the primary action.
const Chat = forwardRef<ChatHandle, {
  onRefreshStatus: () => void;
  project: string | null;
}>(function Chat({ onRefreshStatus, project }, ref) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [correcting, setCorrecting] = useState<number | null>(null);
  const [correctionText, setCorrectionText] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [activeModel, setActiveModelState] = useState<ActiveModelSelection | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [showModelPicker, setShowModelPicker] = useState(false);
  // Dev mode: when on, render a 🔬 LLM calls panel under each agent
  // turn that shows the redacted system + user prompt for every LLM
  // call we made. Persists across reloads via localStorage so the
  // operator doesn't have to re-enable it on every page load.
  const [devMode, setDevMode] = useState<boolean>(() => {
    try {
      return localStorage.getItem("agi.devMode") === "1";
    } catch {
      return false;
    }
  });
  // Round C: which channel's conversation history is currently
  // displayed. "webui" (default) shows the WebUI bucket; "telegram"
  // shows the Telegram bucket. Sending a message while on
  // channel=telegram tags it under the TG bucket so the agent's
  // memory of the TG thread stays continuous when the next real TG
  // message arrives. Persists in localStorage so the user's last
  // selection survives reload.
  const [channel, setChannelState] = useState<string>(() => {
    try {
      return localStorage.getItem("agi.channel") || "webui";
    } catch {
      return "webui";
    }
  });
  const setChannel = (c: string) => {
    try {
      localStorage.setItem("agi.channel", c);
    } catch {
      /* ignore */
    }
    setChannelState(c);
    // Force a re-load so history matches the new filter.
    setLoaded(false);
    setMsgs([]);
  };
  const endRef = useRef<HTMLDivElement>(null);

  // Multimodal: pending attachments + voice recording
  const [pending, setPending] = useState<AttachmentMeta[]>([]);
  const [uploading, setUploading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingStreamRef = useRef<MediaStream | null>(null);

  const handleAttachFiles = async (files: FileList | File[] | null) => {
    if (!files || (files as FileList).length === 0) return;
    setUploading(true);
    try {
      const arr = Array.from(files as FileList);
      const uploaded: AttachmentMeta[] = [];
      for (const f of arr) {
        try {
          const meta = await uploadAttachment(f);
          uploaded.push(meta);
        } catch (e: any) {
          // eslint-disable-next-line no-alert
          alert(`Failed to upload ${f.name}: ${e?.message || e}`);
        }
      }
      setPending((p) => [...p, ...uploaded]);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const removePending = (sha: string) =>
    setPending((p) => p.filter((a) => a.sha256 !== sha));

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files?.length) handleAttachFiles(e.dataTransfer.files);
  };
  const onPaste = (e: React.ClipboardEvent) => {
    if (!e.clipboardData?.files?.length) return;
    handleAttachFiles(e.clipboardData.files);
  };

  // Voice recording: MediaRecorder → blob → /api/transcribe → fill input
  const startRecording = async () => {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingStreamRef.current = stream;
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      recordedChunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size) recordedChunksRef.current.push(e.data);
      };
      mr.onstop = async () => {
        const blob = new Blob(recordedChunksRef.current, {
          type: mr.mimeType || "audio/webm",
        });
        recordingStreamRef.current?.getTracks().forEach((t) => t.stop());
        recordingStreamRef.current = null;
        if (blob.size === 0) return;
        setTranscribing(true);
        try {
          const ext = (blob.type.split("/")[1] || "webm").split(";")[0];
          const result = await transcribeAudio(blob, `voice.${ext}`);
          setInput((prev) => (prev ? prev + " " : "") + (result.text || ""));
        } catch (e: any) {
          // eslint-disable-next-line no-alert
          alert(`Transcription failed: ${e?.message || e}`);
        } finally {
          setTranscribing(false);
        }
      };
      mr.start();
      setRecording(true);
    } catch (e: any) {
      // eslint-disable-next-line no-alert
      alert(`Microphone access denied or unavailable: ${e?.message || e}`);
    }
  };
  const stopRecording = () => {
    if (!recording) return;
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  useImperativeHandle(ref, () => ({
    clearMessages: () => setMsgs([]),
  }));

  const scroll = () =>
    setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 50);

  const loadModels = async () => {
    try {
      const data = await fetchActiveModel();
      setAvailableModels(data.models || []);
      if (data.active && data.active.provider_id) {
        setActiveModelState(data.active as ActiveModelSelection);
      } else {
        setActiveModelState(null);
      }
    } catch { /* ignore */ }
  };

  // Round A: lazy-fetch the on-disk TurnWorkspace artefact for a
  // restored history message and merge its thinking_trace + llm_calls
  // + claims/evidence/token_usage into the message's `meta`. Triggered
  // the first time the user expands the "🔧 tools" section on a message
  // that came back from session history without a trace inline.
  const hydrateTurn = async (idx: number) => {
    setMsgs((cur) => {
      const m = cur[idx];
      if (!m || m.role !== "agent" || !m.turn_id || m.lazy_loading) return cur;
      const next = cur.slice();
      next[idx] = { ...m, lazy_loading: true, lazy_error: undefined };
      return next;
    });
    try {
      const cur = msgs[idx];
      if (!cur || cur.role !== "agent" || !cur.turn_id) return;
      const data: TurnArtifact = await fetchTurn(cur.turn_id);
      setMsgs((latest) => {
        const m = latest[idx];
        if (!m || m.role !== "agent") return latest;
        const next = latest.slice();
        next[idx] = {
          ...m,
          lazy_loading: false,
          meta: {
            ...(m.meta || {
              answer: m.text,
              verification: data.verification,
              learned_topics: [],
              used_topics: [],
            }),
            thinking_trace: data.thinking_trace || [],
            llm_calls: data.llm_calls || [],
            claims: data.claims || [],
            evidence: data.evidence || [],
            token_usage: data.token_usage,
            verification: data.verification ?? m.meta?.verification ?? {
              confidence: 0,
              verified_claims: [],
              unverified_claims: [],
              contradictions: [],
              notes_used: [],
            },
          },
        };
        return next;
      });
    } catch (e: any) {
      setMsgs((latest) => {
        const m = latest[idx];
        if (!m || m.role !== "agent") return latest;
        const next = latest.slice();
        next[idx] = {
          ...m,
          lazy_loading: false,
          lazy_error: e?.message || "load failed",
        };
        return next;
      });
    }
  };

  // Load current channel's turns on mount or when channel changes.
  // - "webui" → fetchCurrentSession (rich session view)
  // - other channels → fetchConversation(channel) which returns
  //   ConversationTurnRow rows from CONVERSATION (channel-tagged)
  useEffect(() => {
    if (loaded) return;
    loadModels();
    (async () => {
      try {
        let restored: Msg[] = [];
        if (channel === "webui") {
          const data = await fetchCurrentSession();
          if (data.session && data.session.turns && data.session.turns.length > 0) {
            for (const t of data.session.turns) {
              restored.push({ role: "user", text: t.user });
              restored.push({
                role: "agent",
                text: t.answer,
                turn_id: t.turn_id || "",
                n_tool_calls: t.n_tool_calls ?? 0,
                n_llm_calls: t.n_llm_calls ?? 0,
                meta: {
                  answer: t.answer,
                  verification: {
                    confidence: t.confidence ?? 0,
                    verified_claims: [],
                    unverified_claims: [],
                    contradictions: [],
                    notes_used: [],
                  },
                  learned_topics: [],
                  used_topics: t.topics ?? [],
                  is_chat: t.is_chat,
                  // Round F-pre: token bar comes from persisted row
                  // so a refresh keeps showing it without a lazy
                  // fetch. Cast through unknown because the
                  // ConversationTurn / SessionTurn types differ
                  // slightly but both carry this same shape.
                  token_usage: (t as { token_usage?: TokenUsage | null }).token_usage ?? undefined,
                },
              });
            }
          }
        } else {
          // Round C: cross-channel view (e.g. "telegram"). Use the
          // channel-aware conversation endpoint. Returns lighter
          // rows (no thinking_trace inline); turn_id powers lazy
          // load on expand exactly like the webui path.
          const data = await fetchConversation(channel, 50);
          for (const t of data.turns) {
            restored.push({ role: "user", text: t.user });
            restored.push({
              role: "agent",
              text: t.answer,
              turn_id: t.turn_id || "",
              meta: {
                answer: t.answer,
                verification: {
                  confidence: t.confidence ?? 0,
                  verified_claims: [],
                  unverified_claims: [],
                  contradictions: [],
                  notes_used: [],
                },
                learned_topics: [],
                used_topics: t.topics ?? [],
                is_chat: t.is_chat,
              },
            });
          }
        }
        if (restored.length > 0) {
          setMsgs(restored);
          setTimeout(() => endRef.current?.scrollIntoView({ behavior: "auto" }), 100);
        }
      } catch { /* ignore — first load with empty session */ }
      setLoaded(true);
    })();
  }, [loaded, channel]);

  const send = async () => {
    if ((!input.trim() && pending.length === 0) || busy) return;
    const text = input.trim();
    const attached = pending;
    setInput("");
    setPending([]);
    setMsgs((m) => [...m, { role: "user", text, attachments: attached }]);
    setBusy(true);

    const progress: string[] = [];
    setMsgs((m) => [...m, { role: "agent", text: "", progress }]);
    scroll();

    // try/catch/finally (2026-08-08 audit). `chatStream` REJECTS on any
    // transport failure — backend restarted (`hrant update` does exactly
    // that), Tailscale blip, laptop sleeps. The rejection propagated out of
    // send(), so setBusy(false) below never ran: `busy` stayed true forever,
    // every control is disabled={busy}, and the first line of send() bails on
    // busy — so the composer was dead until a manual page reload, with no
    // error shown. And because setInput("") runs BEFORE the call, the message
    // the owner had just typed was gone with no way to recover it.
    try {
      await chatStream(text, project, (ev: StreamEvent) => {
      if (ev.type === "progress") {
        progress.push(`${ev.event}: ${ev.message}`);
        setMsgs((m) => {
          if (m.length === 0) return m;
          const lastIdx = m.length - 1;
          const last = m[lastIdx];
          if (last.role !== "agent") return m;
          // Build a NEW Msg (no mutation) so React always sees a
          // fresh object reference and re-renders. This was the fix
          // for live pills not appearing — the previous version
          // mutated `last.meta = {...}` in place which sometimes
          // didn't trigger downstream renders.
          const updates: Partial<typeof last> = {
            progress: [...progress],
          };
          if (ev.tool_call) {
            const synthetic: ThinkingStep = {
              ts: 0,
              event: ev.event || "tool",
              message: ev.message,
              tokens_so_far: 0,
              tool_call: ev.tool_call,
            };
            const prevMeta = last.meta;
            const baseMeta = prevMeta || {
              answer: "",
              verification: {
                confidence: 0,
                verified_claims: [],
                unverified_claims: [],
                contradictions: [],
                notes_used: [],
              },
              learned_topics: [],
              used_topics: [],
              thinking_trace: [],
            };
            updates.meta = {
              ...baseMeta,
              thinking_trace: [
                ...(baseMeta.thinking_trace || []),
                synthetic,
              ],
            };
          }
          const copy = [...m];
          copy[lastIdx] = { ...last, ...updates };
          return copy;
        });
        scroll();
      } else if (ev.type === "answer") {
        setMsgs((m) => {
          if (m.length === 0) return m;
          const lastIdx = m.length - 1;
          const last = m[lastIdx];
          if (last.role !== "agent") return m;
          const copy = [...m];
          copy[lastIdx] = {
            ...last,
            text: ev.data.answer,
            meta: ev.data,
            // Round A: stamp turn_id so refreshing the page later
            // and re-expanding tool cards lazy-loads via /api/turns.
            turn_id: ev.data.turn_id || "",
          };
          return copy;
        });
        scroll();
      } else if (ev.type === "error") {
        setMsgs((m) => {
          if (m.length === 0) return m;
          const lastIdx = m.length - 1;
          const last = m[lastIdx];
          if (last.role !== "agent") return m;
          const copy = [...m];
          // Phase 2: surface via ErrorCard instead of stuffing into
          // the answer text. Keeping `text` empty so the AnswerCard
          // doesn't double-render.
          copy[lastIdx] = { ...last, text: "", error: ev.message };
          return copy;
        });
      }
      }, attached.map((a) => a.sha256), channel);
    } catch (e: any) {
      // Surface it where the user is already looking. The Msg type has an
      // `error` field and Chat already renders an ErrorCard for it, so this
      // reuses the existing path rather than inventing a second one.
      const detail = String(e?.message || e || "connection lost");
      setMsgs((m) => {
        if (m.length === 0) return m;
        const copy = [...m];
        const lastIdx = copy.length - 1;
        const last = copy[lastIdx];
        // Only the agent variant carries `error`; the placeholder pushed
        // just above always is one, but narrow rather than assume.
        if (last.role === "agent") {
          copy[lastIdx] = { ...last, error: detail };
        }
        return copy;
      });
      // Give the typed message back instead of eating it.
      setInput((cur) => (cur.trim() ? cur : text));
      setPending((cur) => (cur.length ? cur : attached));
    } finally {
      setBusy(false);
    }
    onRefreshStatus();
  };

  const handleAddToFinetune = async (msgIdx: number) => {
    const agentMsg = msgs[msgIdx];
    if (agentMsg.role !== "agent" || !agentMsg.meta) return;
    let userText = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        userText = msgs[i].text;
        break;
      }
    }
    if (!userText) return;
    try {
      await addFromChat({
        question: userText,
        answer: agentMsg.meta.answer,
        used_topics: agentMsg.meta.used_topics,
        confidence: agentMsg.meta.verification.confidence,
        project,
      });
      alert("Added to finetune queue");
    } catch (e: any) {
      alert("Error: " + e.message);
    }
  };

  const handleCorrection = async (msgIdx: number) => {
    const agentMsg = msgs[msgIdx];
    if (agentMsg.role !== "agent" || !correctionText.trim()) return;
    let userText = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        userText = msgs[i].text;
        break;
      }
    }
    if (!userText) return;
    try {
      await addCorrection({
        question: userText,
        wrong_answer: agentMsg.text,
        corrected_answer: correctionText.trim(),
        project,
      });
      setCorrecting(null);
      setCorrectionText("");
      alert("Correction saved");
    } catch (e: any) {
      alert("Error: " + e.message);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-w-0">
      <div className="flex-1 overflow-y-auto px-3 py-4 sm:px-4">
        <div className="mx-auto w-full max-w-3xl space-y-3 sm:space-y-4">
        {msgs.length === 0 && (
          <div className="opacity-50 text-sm text-center mt-8">
            Ask a question. The agent will analyze the task, load knowledge,
            answer, and verify itself.
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : ""}>
            <div
              className={`inline-block max-w-[92%] sm:max-w-[85%] rounded-lg p-2.5 sm:p-3 text-sm whitespace-pre-wrap break-words ${
                m.role === "user" ? "bg-sky-800" : "bg-slate-800"
              }`}
            >
              {/* Inline attachments for user turn (images thumbnail, audio link) */}
              {m.role === "user" && m.attachments && m.attachments.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-1.5 not-prose">
                  {m.attachments.map((a) => (
                    a.kind === "image" ? (
                      <a key={a.sha256} href={attachmentUrl(a.sha256)} target="_blank" rel="noreferrer">
                        <img
                          src={attachmentUrl(a.sha256)}
                          alt={a.filename}
                          className="max-h-32 max-w-32 rounded border border-slate-700 object-cover"
                        />
                      </a>
                    ) : a.kind === "audio" ? (
                      <div key={a.sha256} className="bg-slate-900 rounded px-2 py-1 text-xs">
                        🎤 <a href={attachmentUrl(a.sha256)} target="_blank" rel="noreferrer" className="underline">voice</a>
                        {a.transcript ? <span className="opacity-70 ml-1">— {a.transcript.slice(0, 80)}{a.transcript.length > 80 ? "…" : ""}</span> : null}
                      </div>
                    ) : (
                      <div key={a.sha256} className="bg-slate-900 rounded px-2 py-1 text-xs">
                        📎 <a href={attachmentUrl(a.sha256)} target="_blank" rel="noreferrer" className="underline">{a.filename || a.sha256.slice(0, 12)}</a>
                      </div>
                    )
                  ))}
                </div>
              )}
              {/* Progress indicators while loading — compact, no tool details */}
              {m.role === "agent" && m.progress && m.progress.length > 0 && !m.text && (
                <div className="text-xs opacity-60 space-y-0.5">
                  {m.progress
                    .filter((p) => !p.startsWith("tool:") && !p.startsWith("tool_error:"))
                    .slice(-6)
                    .map((p, j) => (
                      <div key={j}>· {p}</div>
                    ))}
                </div>
              )}
              {/* Tool calls — OpenClaw-style cards. Renders BOTH while
                  the agent is still working (live pills via SSE) and
                  on restored history (lazy load via /api/turns/<id>).
                  Round F-pre: a restored row can also carry a
                  persisted `n_tool_calls` summary number, so we show
                  the count immediately and the user clicks to expand
                  for the full trace. */}
              {m.role === "agent" && (() => {
                const turnId = m.role === "agent" ? m.turn_id : "";
                const lazyLoading = m.role === "agent" ? m.lazy_loading : false;
                const lazyError = m.role === "agent" ? m.lazy_error : "";
                const persistedCount = m.role === "agent" ? (m.n_tool_calls || 0) : 0;
                const traceLoaded = m.meta?.thinking_trace !== undefined;
                const toolSteps = (m.meta?.thinking_trace || []).filter(
                  (s) =>
                    s.tool_call &&
                    (s.event === "tool" ||
                      s.event === "tool_error" ||
                      s.event === "tool_starting")
                );
                const hasTools = toolSteps.length > 0;
                // Lazy-load is "available" when we know there's a turn
                // artefact AND we haven't loaded the trace yet. The
                // persisted count tells us whether it's worth offering.
                const lazyAvailable = Boolean(turnId) && !traceLoaded;
                // Show count from whichever source we have:
                //   live trace (preferred) → persisted summary → 0
                const displayCount = hasTools ? toolSteps.length : persistedCount;
                // Hide entirely only when: trace is loaded, persisted
                // count is also 0, no lazy fetch in flight. Otherwise
                // we have at least one number to show.
                if (!hasTools && !displayCount && !lazyLoading) return null;
                const onToggle = (e: React.SyntheticEvent<HTMLDetailsElement>) => {
                  if (e.currentTarget.open && lazyAvailable && !lazyLoading) {
                    void hydrateTurn(i);
                  }
                };
                return (
                  <details
                    className="mb-2"
                    open={hasTools}
                    onToggle={onToggle}
                  >
                    <summary className="text-[11px] opacity-60 cursor-pointer hover:opacity-90 mb-1">
                      🔧 tools:{" "}
                      {hasTools
                        ? `${toolSteps.length} call${toolSteps.length === 1 ? "" : "s"}`
                        : displayCount
                          ? `${displayCount} call${displayCount === 1 ? "" : "s"}${
                              lazyAvailable ? " · click to load" : ""
                            }`
                          : lazyLoading
                            ? "loading…"
                            : "none"}
                      {lazyError && (
                        <span className="text-rose-400 ml-2">· {lazyError}</span>
                      )}
                    </summary>
                    <div className="space-y-1 mt-1">
                      {toolSteps.map((s, j) => (
                        <ToolCallCard key={j} step={s} />
                      ))}
                    </div>
                  </details>
                );
              })()}
              {/* Dev mode: redacted system+user prompts per LLM call */}
              {devMode && m.role === "agent" && (() => {
                const liveCalls = m.meta?.llm_calls || [];
                const persistedCount = m.n_llm_calls || 0;
                const totalCount = liveCalls.length || persistedCount;
                if (totalCount === 0) return null;
                return (
                  <details className="mb-2">
                    <summary className="text-[11px] opacity-70 cursor-pointer hover:opacity-100 mb-1 text-amber-300">
                      🔬 LLM calls: {totalCount}
                      {liveCalls.length === 0 && persistedCount > 0
                        ? " · click to load"
                        : ""}{" "}
                      (dev mode)
                    </summary>
                    {liveCalls.length > 0 && (
                      <div className="space-y-1 mt-1">
                        {liveCalls.map((c, j) => (
                          <LLMCallItem key={j} call={c} />
                        ))}
                      </div>
                    )}
                  </details>
                );
              })()}
              {/* Compact thinking trace — also visible mid-stream
                  so live progress is observable. Auto-opens while
                  the answer is still streaming (text empty) so the
                  user sees what the agent is doing in real time;
                  collapses once the final answer arrives. */}
              {m.role === "agent" && m.meta?.thinking_trace && m.meta.thinking_trace.length > 0 && (
                <details className="mb-2" open={!m.text}>
                  <summary className="text-[10px] opacity-40 cursor-pointer hover:opacity-60">
                    thinking: {m.meta.thinking_trace.filter(s => !s.event.startsWith("tool")).length} steps
                    {m.meta.thinking_trace.length > 0 && ` · ${m.meta.thinking_trace[m.meta.thinking_trace.length - 1].ts.toFixed(1)}s`}
                  </summary>
                  <div className="text-[10px] opacity-50 mt-1 space-y-0.5 max-h-24 overflow-y-auto">
                    {m.meta.thinking_trace
                      .filter(s => !s.event.startsWith("tool"))
                      .map((s, j) => (
                        <div key={j}>
                          <span className="text-slate-500">{s.ts.toFixed(1)}s</span>{" "}
                          <span className="text-slate-400">{s.event}:</span> {s.message}
                        </div>
                      ))}
                  </div>
                </details>
              )}
              {/* AskUserQuestion card — when the agent turn ended
                  with a structured question, show clickable options
                  instead of (or alongside) the plain answer body.
                  Once the user picks, the resume turn's answer is
                  appended as a fresh agent message below. */}
              {m.role === "agent" && m.meta?.question && (
                <div className="mb-2">
                  <AskUserCard
                    question={m.meta.question}
                    onAnswered={(res) => {
                      // Append the resume turn's answer as a new
                      // agent message. The "user picked X" message
                      // doesn't get an explicit row — the AskUserCard
                      // itself shows "✓ You picked: X" so the trail
                      // is visible without extra clutter.
                      setMsgs((prev) => [
                        ...prev,
                        {
                          role: "agent",
                          text: res.answer || "",
                          meta: res,
                          turn_id: res.turn_id || undefined,
                        },
                      ]);
                    }}
                  />
                </div>
              )}
              {/* Task-status cards — live polling for background
                  jobs the agent spawned this turn. The check looks
                  through tool_starting / tool events for any
                  start_background_job invocation and surfaces the
                  resulting job_id. Each job_id renders once;
                  duplicates (e.g. supervisor retry chains) skip. */}
              {m.role === "agent" && (() => {
                if (!m.meta?.thinking_trace) return null;
                const jobIds: string[] = [];
                const seen = new Set<string>();
                for (const s of m.meta.thinking_trace) {
                  const tc = s.tool_call;
                  if (!tc || tc.name !== "start_background_job") continue;
                  // job_id lives in the JSON result body.
                  const resultText = tc.result || "";
                  const m1 = /"job_id"\s*:\s*"(bg-[a-f0-9]+)"/i.exec(resultText);
                  if (m1 && !seen.has(m1[1])) {
                    seen.add(m1[1]);
                    jobIds.push(m1[1]);
                  }
                }
                if (jobIds.length === 0) return null;
                return (
                  <div className="mb-2 space-y-2">
                    {jobIds.map((jid) => (
                      <TaskStatusCard key={jid} jobId={jid} />
                    ))}
                  </div>
                );
              })()}
              {/* When there's no question card, render the answer
                  text via the Markdown card. The "❓ {question}"
                  placeholder in meta.answer would be redundant with
                  the AskUserCard, so we skip it. User messages stay
                  plain (no markdown — user input is literal). */}
              {m.role === "user" && (
                <span className="whitespace-pre-wrap">{m.text}</span>
              )}
              {/* Honest model notice: shown ONLY when the turn silently fell
                  back off the selected model (e.g. Codex quota -> OpenRouter).
                  The agent must never claim a model it did not use. */}
              {m.role === "agent" && m.meta?.model_note && (
                <div className="mb-1 text-[11px] rounded bg-amber-900/50 border border-amber-700/50 px-2 py-1 text-amber-200">
                  ⚠️ {m.meta.model_note}
                </div>
              )}
              {m.role === "agent" && !m.meta?.question && m.text && (
                <AnswerCard text={m.text} is_chat={Boolean(m.meta?.is_chat)} />
              )}
              {m.role === "agent" && m.meta?.model_used && (
                <div className="mt-0.5 text-[10px] opacity-50 font-mono">
                  model: {m.meta.model_used}
                </div>
              )}
              {m.role === "agent" && m.error && (
                <ErrorCard message={m.error} />
              )}

              {/* Verification footer */}
              {m.role === "agent" && m.meta && !m.meta.is_chat && (
                <div className="mt-2 pt-2 border-t border-slate-700 text-xs opacity-80 space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <ConfidenceBadge c={m.meta.verification.confidence} />
                    {m.meta.used_topics.length > 0 && (
                      <span>topics: {m.meta.used_topics.join(", ")}</span>
                    )}
                    {m.meta.learned_topics.length > 0 && (
                      <span>new: {m.meta.learned_topics.join(", ")}</span>
                    )}
                  </div>
                  {m.meta.verification.unverified_claims.length > 0 && (
                    <details>
                      <summary className="cursor-pointer">
                        unverified ({m.meta.verification.unverified_claims.length})
                      </summary>
                      <ul className="mt-1 pl-4 list-disc">
                        {m.meta.verification.unverified_claims.map((c, j) => (
                          <li key={j}>{c}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                  {m.meta.verification.contradictions.length > 0 && (
                    <details>
                      <summary className="cursor-pointer text-rose-400">
                        contradictions ({m.meta.verification.contradictions.length})
                      </summary>
                      <ul className="mt-1 pl-4 list-disc">
                        {m.meta.verification.contradictions.map((c, j) => (
                          <li key={j}>{c}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              )}

              {/* Token usage — audit T4: tokens-first display.
                  USD removed from this per-message footer per owner
                  directive ("user need to see token usage only").
                  Cost is still in the API payload + UsagePage for
                  longer-term analytics. */}
              {m.role === "agent" && m.meta?.token_usage && m.meta.token_usage.total_tokens > 0 && (
                <div className="mt-1 text-[10px] opacity-50 flex gap-3 flex-wrap">
                  {/* `in` is FRESH (billed-at-full-price) input only;
                      cached reads bill ~10% and are shown separately
                      so the real spend is readable at a glance. */}
                  <span>tokens: {m.meta.token_usage.input_tokens.toLocaleString()} in (fresh) / {m.meta.token_usage.output_tokens.toLocaleString()} out</span>
                  {m.meta.token_usage.cache_read_tokens > 0 && (
                    <span>+ {m.meta.token_usage.cache_read_tokens.toLocaleString()} cached (~10% price)</span>
                  )}
                  <span>{m.meta.token_usage.total_tokens.toLocaleString()} total</span>
                  <span>{m.meta.token_usage.llm_calls} calls</span>
                </div>
              )}

              {/* Action buttons for agent messages */}
              {m.role === "agent" && m.meta && m.text && (
                <div className="mt-2 pt-2 border-t border-slate-700 flex gap-2 text-xs">
                  <button
                    onClick={() => handleAddToFinetune(i)}
                    className="rounded-md border border-edge-strong px-2 py-0.5 text-ink-dim hover:bg-surface-hover hover:text-ink"
                    title="Add this Q&A to the fine-tuning set"
                  >
                    Teach
                  </button>
                  <button
                    onClick={() => {
                      setCorrecting(correcting === i ? null : i);
                      setCorrectionText("");
                    }}
                    className="rounded-md border border-edge-strong px-2 py-0.5 text-ink-dim hover:bg-surface-hover hover:text-ink"
                    title="Tell the agent what it got wrong"
                  >
                    Correct
                  </button>
                </div>
              )}

              {/* Correction input */}
              {correcting === i && (
                <div className="mt-2 space-y-1">
                  <textarea
                    className="w-full bg-slate-950 rounded p-2 text-xs"
                    rows={3}
                    placeholder="Enter the correct answer..."
                    value={correctionText}
                    onChange={(e) => setCorrectionText(e.target.value)}
                  />
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleCorrection(i)}
                      className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-0.5 text-xs"
                    >
                      save correction
                    </button>
                    <button
                      onClick={() => setCorrecting(null)}
                      className="bg-slate-700 rounded px-2 py-0.5 text-xs"
                    >
                      cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-edge bg-surface/40 p-3">
        <div className="mx-auto w-full max-w-3xl space-y-2">
        {/* Model selector + dev-mode toggle */}
        <div className="flex items-center gap-2 relative">
          <button
            onClick={() => { setShowModelPicker(!showModelPicker); if (!showModelPicker) loadModels(); }}
            className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 rounded px-2.5 py-1 text-xs transition-colors"
            title="Select model"
          >
            <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            <span className="text-slate-300">
              {activeModel ? `${activeModel.provider_name} / ${activeModel.model}` : "Default model"}
            </span>
            <svg className="w-3 h-3 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          <button
            onClick={() => {
              const next = !devMode;
              setDevMode(next);
              try { localStorage.setItem("agi.devMode", next ? "1" : "0"); } catch {}
            }}
            title={devMode ? "Dev mode ON — click to disable" : "Dev mode OFF — click to show LLM prompts"}
            className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors ${
              devMode
                ? "bg-amber-700 hover:bg-amber-600 text-white"
                : "bg-slate-800 hover:bg-slate-700 text-slate-400"
            }`}
          >
            🔬 dev
          </button>

          {/* Round C: channel selector. Switching reloads history
              from the channel's bucket; sending tags the new turn
              with this channel so cross-channel composition keeps
              context continuous. */}
          <select
            value={channel}
            onChange={(e) => setChannel(e.target.value)}
            title="Which channel's conversation are you working with"
            className="bg-slate-800 hover:bg-slate-700 rounded px-2 py-1 text-xs text-slate-300 border-0 focus:ring-1 focus:ring-slate-600"
          >
            <option value="webui">💻 webui</option>
            <option value="telegram">📱 telegram</option>
          </select>

          {showModelPicker && (
            <div className="absolute bottom-full left-0 mb-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 min-w-[300px] max-h-[400px] overflow-y-auto">
              <div className="p-2 border-b border-slate-700">
                <button
                  onClick={async () => {
                    await clearActiveModel();
                    setActiveModelState(null);
                    setShowModelPicker(false);
                  }}
                  className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors ${
                    !activeModel ? "bg-accent-soft text-accent font-medium" : "hover:bg-slate-700 text-slate-300"
                  }`}
                >
                  Default (from config)
                </button>
              </div>
              <div className="p-2 space-y-1">
                {(() => {
                  const grouped: Record<string, AvailableModel[]> = {};
                  for (const m of availableModels) {
                    const key = m.provider_name;
                    if (!grouped[key]) grouped[key] = [];
                    grouped[key].push(m);
                  }
                  return Object.entries(grouped).map(([provName, models]) => (
                    <div key={provName}>
                      <div className="text-[10px] text-slate-500 uppercase tracking-wider px-3 py-1">{provName}</div>
                      {models.map((m) => {
                        const isActive = activeModel?.provider_id === m.provider_id && activeModel?.model === m.model;
                        return (
                          <button
                            key={`${m.provider_id}:${m.model}`}
                            onClick={async () => {
                              try {
                                const res = await setActiveModel(m.provider_id, m.model);
                                setActiveModelState(res.active);
                              } catch { /* ignore */ }
                              setShowModelPicker(false);
                            }}
                            className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors ${
                              isActive ? "bg-accent-soft text-accent font-medium" : "hover:bg-slate-700 text-slate-300"
                            }`}
                          >
                            {m.model}
                          </button>
                        );
                      })}
                    </div>
                  ));
                })()}
                {availableModels.length === 0 && (
                  <div className="text-xs text-slate-500 px-3 py-2">No providers configured</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Pending attachments preview row (above textarea) */}
        {pending.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-1.5">
            {pending.map((a) => (
              <div key={a.sha256} className="relative bg-slate-800 rounded p-1 flex items-center gap-1">
                {a.kind === "image" ? (
                  <img
                    src={attachmentUrl(a.sha256)}
                    alt={a.filename}
                    className="h-12 w-12 rounded object-cover"
                  />
                ) : a.kind === "audio" ? (
                  <span className="px-2 text-xs">🎤 {a.filename}</span>
                ) : (
                  <span className="px-2 text-xs">📎 {a.filename || a.sha256.slice(0, 8)}</span>
                )}
                <button
                  onClick={() => removePending(a.sha256)}
                  disabled={busy}
                  title="Remove"
                  className="text-slate-400 hover:text-rose-400 px-1 text-xs"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}

        <div
          className="flex gap-2"
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,audio/*,application/pdf,text/*"
            className="hidden"
            onChange={(e) => handleAttachFiles(e.target.files)}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={busy || uploading}
            title="Attach files (or drag & drop / paste)"
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded px-3 text-base self-start min-h-[44px] sm:min-h-0 sm:py-1 sm:px-2 sm:text-sm"
          >
            {uploading ? "…" : "📎"}
          </button>
          <button
            onClick={recording ? stopRecording : startRecording}
            disabled={busy || transcribing}
            title={recording ? "Stop recording" : "Record voice"}
            className={`rounded px-3 text-base self-start min-h-[44px] sm:min-h-0 sm:py-1 sm:px-2 sm:text-sm disabled:opacity-50 ${
              recording
                ? "bg-rose-700 hover:bg-rose-600 animate-pulse"
                : "bg-slate-700 hover:bg-slate-600"
            }`}
          >
            {transcribing ? "…" : recording ? "⏹" : "🎤"}
          </button>
          <ReasoningQuickPick busy={busy} />
          <textarea
            className="flex-1 resize-none rounded-lg border border-edge-strong bg-canvas p-2.5 text-sm outline-none focus:border-accent"
            rows={2}
            placeholder="Ask anything…  Enter sends, Shift+Enter for a new line"
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onPaste={onPaste}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button
            onClick={send}
            disabled={busy}
            title="Send  (Enter)"
            className="self-stretch rounded-lg bg-accent px-5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-50"
          >
            {busy ? "…" : "Send"}
          </button>
        </div>
        </div>
      </div>
    </div>
  );
});

export default Chat;
