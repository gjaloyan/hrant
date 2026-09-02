/** The whole system prompt, in the order the model reads it.
 *
 * The Prompt tab beside this one edits sections through a dropdown, one
 * at a time, and nothing anywhere showed the result — so "do we have all
 * the system prompts in here?" had no answer you could look up.
 *
 * It also makes the split visible, which is the actual answer: the
 * thirteen rule modules a pipeline profile can override are about a third
 * of what the model receives. Identity, permissions and capabilities come
 * from stores the profile was never meant to touch. That is the design —
 * a profile overlays RULES, identity is content — but a design nobody can
 * see reads as a missing feature.
 */
import { useEffect, useState } from "react";
import { fetchAssembledPrompt, type PromptPart } from "../../api";
import { Badge, Button, Card, Spinner, cx } from "../../ui";

const CHANNELS = ["telegram", "webui", "voice"];

export default function AssembledPrompt({
  flash,
}: {
  flash: (msg: string) => void;
}) {
  const [parts, setParts] = useState<PromptPart[]>([]);
  const [perTurn, setPerTurn] = useState<string[]>([]);
  const [total, setTotal] = useState(0);
  const [channel, setChannel] = useState("telegram");
  const [open, setOpen] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchAssembledPrompt(channel)
      .then((d) => {
        setParts(d.parts || []);
        setPerTurn(d.per_turn || []);
        setTotal(d.total_chars || 0);
      })
      .catch((e) => flash("Error: " + e.message))
      .finally(() => setLoading(false));
  }, [channel]);

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-ink-dim">
        <Spinner /> Assembling…
      </div>
    );

  return (
    <div className="max-w-3xl space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-ink-dim">Shown for</span>
        <select
          value={channel}
          onChange={(e) => setChannel(e.target.value)}
          className="text-sm"
          aria-label="Channel"
        >
          {CHANNELS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <span className="ml-auto text-xs text-ink-dim">
          {total.toLocaleString()} characters before the per-turn blocks
        </span>
      </div>

      {parts.map((p) => {
        const isOpen = open === p.name;
        const pct = total ? Math.round((p.chars / total) * 100) : 0;
        return (
          <Card
            key={p.name}
            title={p.name}
            subtitle={p.source}
            actions={
              <div className="flex items-center gap-2">
                {p.profile_can_override ? (
                  <Badge tone="accent">profile can change this</Badge>
                ) : (
                  <Badge tone="neutral" title={`Edited in ${p.edit_in}`}>
                    {p.edit_in.replace("Settings → ", "")}
                  </Badge>
                )}
                <Button
                  kind="ghost"
                  size="sm"
                  onClick={() => setOpen(isOpen ? null : p.name)}
                >
                  {isOpen ? "Hide" : "Show"}
                </Button>
              </div>
            }
          >
            <div className="flex items-center gap-3">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-hover">
                <div
                  className={cx(
                    "h-full rounded-full",
                    p.profile_can_override ? "bg-accent" : "bg-edge-strong",
                  )}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="w-24 text-right text-xs text-ink-dim">
                {p.chars.toLocaleString()} · {pct}%
              </span>
            </div>
            {isOpen && (
              <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-edge bg-canvas p-3 font-mono text-[11px] text-ink-dim">
                {p.text || "(empty)"}
              </pre>
            )}
          </Card>
        );
      })}

      <Card
        title="Added per turn"
        subtitle="These depend on the question being asked, so they cannot be shown before there is one."
      >
        <ul className="space-y-1 text-sm text-ink-dim">
          {perTurn.map((t) => (
            <li key={t} className="flex gap-2">
              <span className="text-ink-faint">·</span>
              {t}
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
