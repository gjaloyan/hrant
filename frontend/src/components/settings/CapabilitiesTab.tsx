/** What the agent can do.
 *
 * This rendered `/api/capabilities` — the block that goes into the SYSTEM
 * PROMPT, where each description is clipped to 100 characters because
 * every one is billed on every turn. So a person browsing the tool list
 * read sentences cut mid-word ("Headless-Chromium deep-research via
 * Vercel Labs `agent-browser` CLI. Drives a real browser to read J"), in
 * a wall of monospace with no search. The longest real description is
 * 2556 characters; the screen was showing four per cent of it.
 *
 * Truncation is a prompt concern. This reads the registry instead, and
 * answers the question the screen exists for: what can it do, and can it
 * reach that right now.
 */
import { useEffect, useMemo, useState } from "react";
import { fetchCapabilityTools, type CapabilityTool, type CapabilitySkill } from "../../api";
import { Badge, Button, Card, EmptyState, Spinner } from "../../ui";

export default function CapabilitiesTab({
  flash,
}: {
  flash: (msg: string) => void;
}) {
  const [tools, setTools] = useState<CapabilityTool[]>([]);
  const [skills, setSkills] = useState<CapabilitySkill[]>([]);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetchCapabilityTools();
      setTools(r.tools || []);
      setSkills(r.skills || []);
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return tools;
    return tools.filter((t) =>
      `${t.name} ${t.description}`.toLowerCase().includes(needle),
    );
  }, [tools, q]);

  const always = shown.filter((t) => t.always_on);
  const gated = shown.filter((t) => !t.always_on);

  const Row = ({ t }: { t: CapabilityTool }) => {
    const isOpen = open === t.name;
    // The first sentence is what the model reads first and what a reader
    // needs; the rest is available on demand rather than by default.
    const first = t.description.split(/(?<=\.)\s/)[0] || t.description;
    const hasMore = t.description.length > first.length + 4;
    return (
      <li className="border-t border-edge/60 py-2 first:border-t-0">
        <button
          onClick={() => setOpen(isOpen ? null : t.name)}
          className="w-full text-left"
        >
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-sm">{t.name}</span>
            {t.origin !== "builtin" && <Badge tone="accent">{t.origin}</Badge>}
            {t.bundle && (
              <Badge
                tone="warn"
                title={`The model must call load_tool_bundle("${t.bundle}") before it can use this`}
              >
                needs “{t.bundle}”
              </Badge>
            )}
          </span>
          <span className="mt-0.5 block text-xs text-ink-dim">
            {isOpen ? t.description : first}
            {hasMore && !isOpen && (
              <span className="text-accent"> …more</span>
            )}
          </span>
        </button>
      </li>
    );
  };

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-ink-dim">
        <Spinner /> Loading…
      </div>
    );

  return (
    <div className="max-w-3xl space-y-4">
      <div className="flex items-center gap-2">
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search tools…"
          className="flex-1 text-sm"
          aria-label="Search tools"
        />
        <Button kind="ghost" size="sm" onClick={load}>
          Refresh
        </Button>
      </div>

      {shown.length === 0 && (
        <EmptyState title="No match">
          Nothing here mentions “{q}”.
        </EmptyState>
      )}

      {always.length > 0 && (
        <Card
          title="Always available"
          subtitle="Reachable on any turn, without the model having to load anything first."
          actions={<Badge>{always.length}</Badge>}
          bodyClass="p-0 px-4 py-2"
        >
          <ul>
            {always.map((t) => (
              <Row key={t.name} t={t} />
            ))}
          </ul>
        </Card>
      )}

      {gated.length > 0 && (
        <Card
          title="Behind a bundle"
          subtitle="The model has to load the bundle before it can use these — a round trip it does not always think to make."
          actions={<Badge tone="warn">{gated.length}</Badge>}
          bodyClass="p-0 px-4 py-2"
        >
          <ul>
            {gated.map((t) => (
              <Row key={t.name} t={t} />
            ))}
          </ul>
        </Card>
      )}

      <Card
        title="Skills"
        subtitle="Learned procedures the agent can load for a task."
        actions={<Badge>{skills.length}</Badge>}
      >
        {skills.length === 0 ? (
          <EmptyState title="No skills yet">
            The agent writes one when it solves something worth repeating.
          </EmptyState>
        ) : (
          <ul className="divide-y divide-edge/60">
            {skills.map((s) => (
              <li key={s.name} className="py-2">
                <span className="flex items-center gap-2">
                  <span className="text-sm font-medium">{s.name}</span>
                  {!s.enabled && <Badge tone="neutral">off</Badge>}
                </span>
                <span className="block text-xs text-ink-dim">
                  {s.description}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
