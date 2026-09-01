/** Facts the agent carries into every single turn.
 *
 * This lived at the top of the Knowledge screen, above four unrelated
 * forms, where it was the first thing anyone saw and almost never the
 * thing they came for. It is standing configuration — a small block of
 * text edited rarely and read on every request — so it belongs beside the
 * other things that define who the agent is.
 *
 * The old UI showed a raw token count and a textarea. The count matters
 * because the block is charged on EVERY turn, so it is shown as a budget
 * with a warning as it fills, and each fact can be removed individually
 * rather than by editing a blob and hoping.
 */
import { useEffect, useState } from "react";
import { addCoreFact, deleteCoreFact, fetchCore } from "../../api";
import { Badge, Button, Card, EmptyState, Spinner, cx } from "../../ui";

export default function CoreMemoryTab({
  flash,
}: {
  flash: (msg: string) => void;
}) {
  const [content, setContent] = useState("");
  const [tokens, setTokens] = useState(0);
  const [max, setMax] = useState(4000);
  const [loading, setLoading] = useState(true);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetchCore();
      setContent(r.content || "");
      setTokens(r.tokens || 0);
      setMax(r.max || 4000);
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const add = async () => {
    const fact = draft.trim();
    if (!fact) return;
    setBusy(true);
    try {
      await addCoreFact(fact);
      setDraft("");
      flash("Added to core memory");
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (line: string) => {
    if (!confirm(`Remove this from every future turn?\n\n${line}`)) return;
    try {
      await deleteCoreFact(line);
      flash("Removed");
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  // The file is markdown; the facts are its bullet lines. Headings and
  // prose are shown as context but are not individually removable.
  const facts = content
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.startsWith("- ") || l.startsWith("* "))
    .map((l) => l.slice(2).trim())
    .filter(Boolean);

  const pct = Math.min(100, Math.round((tokens / Math.max(1, max)) * 100));
  const tone = pct >= 90 ? "danger" : pct >= 70 ? "warn" : "ok";

  return (
    <div className="max-w-3xl space-y-4">
      <Card
        title="Budget"
        subtitle="This block is included in every request, so its size is a running cost, not a one-off."
        actions={
          <Badge tone={tone}>
            {tokens} / {max} tokens
          </Badge>
        }
      >
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-hover">
          <div
            className={cx(
              "h-full rounded-full transition-all",
              tone === "danger" ? "bg-danger" : tone === "warn" ? "bg-warn" : "bg-ok",
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
        {pct >= 70 && (
          <p className="mt-2 text-xs text-warn">
            Nearly full. Every fact here is re-sent on each turn — remove what
            the agent can look up instead.
          </p>
        )}
      </Card>

      <Card title="Facts" subtitle="Always in context, in the order they were added.">
        <div className="mb-3 flex gap-2">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
            placeholder="e.g. My working hours are 10:00–19:00 Yerevan time"
            className="flex-1 text-sm"
            aria-label="New core fact"
          />
          <Button kind="primary" onClick={add} disabled={busy || !draft.trim()}>
            {busy ? <Spinner /> : "Add"}
          </Button>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-sm text-ink-dim">
            <Spinner /> Loading…
          </div>
        )}

        {!loading && facts.length === 0 && (
          <EmptyState title="No standing facts">
            Add something the agent should never have to look up — who you are,
            how you work, what it must never do.
          </EmptyState>
        )}

        <ul className="divide-y divide-edge">
          {facts.map((f, i) => (
            <li key={i} className="group flex items-start gap-2 py-2">
              <span className="min-w-0 flex-1 text-sm">{f}</span>
              <Button
                kind="danger"
                size="sm"
                className="opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100"
                onClick={() => remove(f)}
              >
                Remove
              </Button>
            </li>
          ))}
        </ul>
      </Card>

      <Card
        title="Raw file"
        subtitle="What the agent actually reads, including any headings."
      >
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-edge bg-canvas p-3 font-mono text-xs text-ink-dim">
          {content || "(empty)"}
        </pre>
      </Card>
    </div>
  );
}
