/** The bottom bar.
 *
 * It carried the right facts in an unreadable shape: one run-on line of
 * bare values — "cloud_only 28 topics core: 13/4000 finetune: 92 project: —
 * A: gpt-5.6-luna B: (v0) today 543.5kt · ratio 20.7:1 · A:89 / B:0" — where
 * nothing said which number belonged to what, and a warning looked like
 * every other item.
 *
 * Same data, grouped and labelled: each figure now carries its own caption,
 * related figures sit in one block, and only a real problem is coloured.
 */
import { useEffect, useState } from "react";
import {
  StatusPayload,
  AutonomicStatus,
  EmbeddingsStatusResponse,
} from "../api";
import { Badge, cx } from "../ui";

type TokensToday = {
  date: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_output_ratio: number;
  cost_usd: number;
  llm_calls: number;
};

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function Dot({ ok, title }: { ok: boolean | undefined; title: string }) {
  return (
    <span
      title={title + (ok === false ? " — unavailable" : "")}
      className={cx(
        "inline-block h-1.5 w-1.5 shrink-0 rounded-full",
        ok === undefined ? "bg-ink-faint" : ok ? "bg-ok" : "bg-danger",
      )}
    />
  );
}

/** One figure with its caption. The caption is the fix: a number nobody
 *  can name is a number nobody can act on. */
function Stat({
  label,
  children,
  title,
}: {
  label: string;
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <div className="flex flex-col leading-tight" title={title}>
      <span className="text-[9px] uppercase tracking-wide text-ink-faint">
        {label}
      </span>
      <span className="text-xs text-ink">{children}</span>
    </div>
  );
}

const Divider = () => (
  <span className="mx-1 hidden h-6 w-px shrink-0 bg-edge sm:block" />
);

export default function StatusBar({
  status,
  autonomic,
  pendingCount,
  embeddings,
}: {
  status: StatusPayload | null;
  autonomic?: AutonomicStatus | null;
  pendingCount?: number;
  embeddings?: EmbeddingsStatusResponse | null;
}) {
  const [today, setToday] = useState<TokensToday | null>(null);
  useEffect(() => {
    let cancelled = false;
    const pull = () => {
      fetch("/api/tokens/today")
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (!cancelled && d) setToday(d as TokensToday);
        })
        .catch(() => {
          /* keep the previous value rather than blanking the bar */
        });
    };
    pull();
    const id = setInterval(pull, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!status) return null;
  const r = status.router as any;
  const hasRouter = r && !("error" in r);

  const MODE_TONE: Record<string, "ok" | "accent" | "warn" | "neutral"> = {
    local_full: "ok",
    cloud_finetune: "accent",
    local_cpu: "warn",
    cloud_only: "accent",
    claude_only: "accent",
  };

  const embOff =
    embeddings &&
    (embeddings.embedder.backend === "disabled" || !embeddings.embedder.backend);

  return (
    <footer className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-edge bg-surface/70 px-3 py-1.5 sm:px-4">
      <Badge
        tone={MODE_TONE[status.mode] || "neutral"}
        title={`training location: ${status.training_location}`}
      >
        {status.mode.replace(/_/g, " ")}
      </Badge>

      {hasRouter && (
        <>
          <Divider />
          <Stat label="model a" title="Primary model">
            <span className="inline-flex items-center gap-1">
              <Dot ok={r.model_a_available} title="Model A" />
              <span className="text-accent">{status.model_a}</span>
            </span>
          </Stat>
          <Stat label="model b" title="Fallback / local model">
            <span className="inline-flex items-center gap-1">
              <Dot ok={r.model_b_available} title="Model B" />
              <span className="text-ok">
                {status.model_b}
                {status.model_version ? ` (${status.model_version})` : ""}
              </span>
            </span>
          </Stat>
        </>
      )}

      <Divider />
      <Stat
        label="tokens today"
        title={
          today
            ? `in ${today.input_tokens.toLocaleString()} · out ${today.output_tokens.toLocaleString()} · $${today.cost_usd.toFixed(4)}`
            : "not loaded yet"
        }
      >
        {today ? `${fmtTokens(today.total_tokens)}t` : "—"}
      </Stat>
      {today && (
        <Stat
          label="in : out"
          title="Input-to-output ratio. A high number means the agent is re-reading much more than it writes."
        >
          <span className={today.input_output_ratio > 15 ? "text-warn" : ""}>
            {today.input_output_ratio.toFixed(1)}:1
          </span>
        </Stat>
      )}
      {hasRouter && (
        <Stat label="calls" title="LLM calls today, model A / model B">
          {r.api_calls_today} / {r.model_b_calls_today}
        </Stat>
      )}

      <Divider />
      <Stat label="topics">{status.topics_total}</Stat>
      <Stat label="core memory" title="Tokens used of the core-memory budget">
        {status.core_tokens}/{status.core_max}
      </Stat>
      <Stat label="finetune set">{status.finetune_count}</Stat>
      <Stat label="project">{status.current_project || "—"}</Stat>

      {autonomic && (
        <>
          <Divider />
          <Stat label="autonomic" title="Background levers">
            <span className="inline-flex items-center gap-1">
              <Dot
                ok={autonomic.enabled && autonomic.scheduler_running}
                title="Autonomic scheduler"
              />
              {autonomic.registered_levers.length} levers
            </span>
          </Stat>
        </>
      )}

      {embeddings && (
        <Stat
          label="memory index"
          title={
            embeddings.embedder.last_error
              ? `embedder: ${embeddings.embedder.last_error}`
              : `${embeddings.coverage.embedded}/${embeddings.coverage.total_notes} notes embedded`
          }
        >
          <span className="inline-flex items-center gap-1">
            <Dot ok={!embOff} title="Embeddings" />
            {embOff ? (
              <span className="text-warn">text-only</span>
            ) : (
              <span className="font-mono text-[11px]">
                {embeddings.embedder.model}
              </span>
            )}
          </span>
        </Stat>
      )}

      {/* Problems last and loud — they were the same weight as everything
          else, which is how a warning goes unread. */}
      <div className="ml-auto flex items-center gap-2">
        {pendingCount ? (
          <Badge tone="warn" title="Proposals waiting for your review">
            {pendingCount} pending
          </Badge>
        ) : null}
        {embeddings && embeddings.coverage.missing > 0 && !embOff && (
          <Badge tone="warn">{embeddings.coverage.missing} unembedded</Badge>
        )}
        {!hasRouter && (
          <Badge tone="danger">router: {r?.error || "unavailable"}</Badge>
        )}
      </div>
    </footer>
  );
}
