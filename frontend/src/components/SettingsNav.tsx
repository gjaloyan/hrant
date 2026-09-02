/** The settings rail.
 *
 * Twenty-two destinations sat in one ungrouped column, in the order they
 * happened to be built — "Soul" next to "Providers" next to "Reminders" —
 * and the list ran off the bottom of the screen behind the status bar, so
 * the last few were reachable only by scrolling a list you had no reason
 * to believe continued. Nothing said what any of them did, and there was
 * no way to search.
 *
 * Same twenty-two screens, grouped by what you came here to change, each
 * with a one-line description, and a filter that matches the description
 * and synonyms as well as the title — you should find "temperature" by
 * typing "temperature", not by knowing it lives under Engine.
 */
import { useMemo, useState } from "react";
import { cx } from "../ui";

export type SettingsGroup = {
  group: string;
  items: { id: string; label: string; hint: string; keywords?: string }[];
};

export const SETTINGS_NAV: SettingsGroup[] = [
  {
    group: "Character",
    items: [
      { id: "soul", label: "Soul", hint: "Who the agent is, in its own words", keywords: "personality values identity soul.md character" },
      { id: "identity", label: "Identity", hint: "Name, role, how it introduces itself", keywords: "name role bio" },
      { id: "user", label: "User Profile", hint: "What it knows about you", keywords: "about me owner profile facts preferences" },
      { id: "core", label: "Core Memory", hint: "Facts carried into every turn", keywords: "core memory always context standing facts budget tokens" },
      { id: "conversation", label: "Recent Context", hint: "The last turns the agent still carries", keywords: "conversation history recent context short term memory clear forget buffer" },
      { id: "voice", label: "Voice", hint: "Speech in and out — models, language", keywords: "tts stt whisper speech audio armenian transcribe" },
    ],
  },
  {
    group: "Thinking",
    items: [
      { id: "providers", label: "Providers", hint: "Which models it can call, and failover", keywords: "llm model api key openai anthropic router fallback quota" },
      { id: "reasoning", label: "Reasoning", hint: "Depth, effort, when to think harder", keywords: "effort depth critical thinking levels" },
      { id: "modelrouting", label: "Model Routing", hint: "Send cheap work to a cheaper model", keywords: "cost cheap small model per task classification keyword routing price save money" },
      { id: "pipeline", label: "Pipeline", hint: "Stages of a turn, and their profiles", keywords: "stages profile turn flow" },
      { id: "engine", label: "Engine", hint: "Temperature, context, token budgets", keywords: "temperature context window tokens budget sampling" },
      { id: "capabilities", label: "Capabilities", hint: "Every tool and skill it can reach", keywords: "tools skills what can it do list browser terminal bundle" },
    ],
  },
  {
    group: "Memory",
    items: [
      { id: "memory", label: "Search Index", hint: "How notes are embedded for semantic search", keywords: "embedding embeddings ollama openai cohere vector semantic search backfill index" },
      { id: "digests", label: "Memory Digests", hint: "Nightly consolidation of what it learned", keywords: "consolidation nightly summary sleep replay" },
      { id: "kgraph", label: "Fact Index", hint: "Facts and topics pulled out by consolidation", keywords: "graph kgraph facts topics entities consolidation nodes rebuild" },
    ],
  },
  {
    group: "People & Access",
    items: [
      { id: "roles", label: "Roles & Contacts", hint: "Who may talk to it, and as what", keywords: "owner trusted guest permissions telegram access contacts family" },
      { id: "channels", label: "Channels", hint: "Telegram, web, and what it watches", keywords: "telegram bot webui watch feed subscription" },
    ],
  },
  {
    group: "Automation",
    items: [
      { id: "skills", label: "Skills", hint: "Learned procedures it can load", keywords: "skill procedure playbook recipe" },
      { id: "jobs", label: "Jobs", hint: "Background and scheduled work", keywords: "background job cron scheduled task queue" },
      { id: "subagents", label: "Subagents", hint: "Delegation and worker teams", keywords: "delegate worker team parallel" },
      { id: "reminders", label: "Reminders", hint: "Scheduled messages and follow-ups", keywords: "reminder schedule notification todo alarm" },
      { id: "selfmods", label: "Self-Modifications", hint: "Code changes it proposed to itself", keywords: "self modification patch propose diff review" },
    ],
  },
  {
    group: "System",
    items: [
      { id: "status", label: "System Status", hint: "Health of every moving part", keywords: "health status uptime diagnostics" },
      { id: "logs", label: "Logs", hint: "Raw output, for when something is wrong", keywords: "log debug error trace output" },
    ],
  },
];

export default function SettingsNav({
  tab,
  setTab,
  footer,
}: {
  tab: string;
  setTab: (id: any) => void;
  footer?: React.ReactNode;
}) {
  const [q, setQ] = useState("");

  const groups = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return SETTINGS_NAV;
    return SETTINGS_NAV.map((g) => ({
      group: g.group,
      items: g.items.filter((i) =>
        `${i.label} ${i.hint} ${i.keywords || ""}`.toLowerCase().includes(needle),
      ),
    })).filter((g) => g.items.length > 0);
  }, [q]);

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-edge bg-canvas/60 md:w-64">
      <div className="border-b border-edge p-2">
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search settings…"
          aria-label="Search settings"
          className="w-full text-sm"
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {groups.length === 0 && (
          <p className="px-2 py-4 text-xs text-ink-faint">
            Nothing matches “{q}”.
          </p>
        )}
        {groups.map((g) => (
          <div key={g.group} className="mb-3">
            <p className="px-2 pb-1 text-micro font-semibold uppercase text-ink-faint">
              {g.group}
            </p>
            <ul className="space-y-0.5">
              {g.items.map((i) => (
                <li key={i.id}>
                  <button
                    onClick={() => setTab(i.id)}
                    title={i.hint}
                    aria-current={tab === i.id ? "page" : undefined}
                    className={cx(
                      "w-full rounded-lg px-2 py-1.5 text-left",
                      tab === i.id
                        ? "bg-accent-soft text-accent"
                        : "text-ink-dim hover:bg-surface-hover hover:text-ink",
                    )}
                  >
                    <span className="block truncate text-sm font-medium">
                      {i.label}
                    </span>
                    {/* The description is why the search works and why a
                        stranger can find anything here at all. */}
                    <span className="block truncate text-[11px] text-ink-faint">
                      {i.hint}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {footer && <div className="border-t border-edge p-2">{footer}</div>}
    </aside>
  );
}
