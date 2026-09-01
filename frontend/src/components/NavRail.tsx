/** Grouped navigation.
 *
 * Eleven equal-weight pills in one scrolling row said everything on this
 * agent is equally important and equally unrelated — Chat sat beside
 * Fine-Tune with nothing to suggest you use one constantly and the other
 * twice a month. Grouping is the cheapest thing that fixes it: four short
 * lists you can scan, instead of one long line you have to read.
 *
 * Desktop gets a rail with visible group names. Mobile keeps a single
 * scrolling strip — a collapsed rail on a phone would trade one problem
 * for a worse one — but ordered by group so related things sit together.
 */
import { cx } from "../ui";

export type Tab =
  | "chat"
  | "projects"
  | "goals"
  | "knowledge"
  | "graph"
  | "sessions"
  | "intelligence"
  | "autonomic"
  | "finetune"
  | "usage"
  | "settings";

type Item = { id: Tab; label: string; icon: string; hint: string };

export const NAV: { group: string; items: Item[] }[] = [
  {
    group: "Work",
    items: [
      { id: "chat", label: "Chat", icon: "💬", hint: "Talk to the agent" },
      { id: "projects", label: "Tasks & Projects", icon: "✅", hint: "Todo list, trackers, calendar" },
      { id: "goals", label: "Goals", icon: "🎯", hint: "What the agent is driving at" },
    ],
  },
  {
    group: "Memory",
    items: [
      { id: "knowledge", label: "Knowledge", icon: "📚", hint: "What it has learned" },
      { id: "graph", label: "Graph", icon: "🔗", hint: "How the notes connect" },
      { id: "sessions", label: "Sessions", icon: "🗂", hint: "Past conversations" },
    ],
  },
  {
    group: "Agent",
    items: [
      { id: "intelligence", label: "Intelligence", icon: "🧠", hint: "Evaluator, meta-learner, self-modifier" },
      { id: "autonomic", label: "Autonomic", icon: "🦾", hint: "Background levers" },
      { id: "finetune", label: "Fine-Tune", icon: "🎓", hint: "Training runs" },
    ],
  },
  {
    group: "System",
    items: [
      { id: "usage", label: "Usage", icon: "📈", hint: "Tokens and cost" },
      { id: "settings", label: "Settings", icon: "⚙️", hint: "Everything configurable" },
    ],
  },
];

const ALL = NAV.flatMap((g) => g.items);
export const labelOf = (id: Tab) => ALL.find((i) => i.id === id)?.label ?? id;

export default function NavRail({
  tab,
  setTab,
  badges,
}: {
  tab: Tab;
  setTab: (t: Tab) => void;
  badges?: Partial<Record<Tab, number>>;
}) {
  const Dot = ({ id }: { id: Tab }) => {
    const n = badges?.[id];
    if (!n) return null;
    return (
      <span className="ml-auto shrink-0 rounded-full bg-accent px-1.5 py-px text-micro font-semibold text-white">
        {n > 99 ? "99+" : n}
      </span>
    );
  };

  return (
    <>
      {/* Desktop rail */}
      <nav className="hidden md:flex w-52 shrink-0 flex-col gap-2.5 overflow-y-auto border-r border-edge bg-surface/60 px-2 pb-4 pt-3">
        {NAV.map((g) => (
          <div key={g.group}>
            <p className="px-2 pb-1 text-micro font-semibold uppercase text-ink-faint">
              {g.group}
            </p>
            <ul className="space-y-0.5">
              {g.items.map((it) => (
                <li key={it.id}>
                  <button
                    onClick={() => setTab(it.id)}
                    title={it.hint}
                    aria-current={tab === it.id ? "page" : undefined}
                    className={cx(
                      "flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-sm",
                      tab === it.id
                        ? "bg-accent-soft text-accent font-medium"
                        : "text-ink-dim hover:bg-surface-hover hover:text-ink",
                    )}
                  >
                    <span className="w-5 text-center">{it.icon}</span>
                    <span className="truncate">{it.label}</span>
                    <Dot id={it.id} />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Mobile strip — grouped order, dividers where a group ends */}
      <nav className="md:hidden flex gap-1 overflow-x-auto no-scrollbar border-b border-edge px-2 py-2">
        {NAV.map((g, gi) => (
          <div key={g.group} className="flex gap-1">
            {gi > 0 && <span className="my-1 w-px shrink-0 bg-edge" />}
            {g.items.map((it) => (
              <button
                key={it.id}
                onClick={() => setTab(it.id)}
                aria-label={it.label}
                title={it.label}
                className={cx(
                  "shrink-0 rounded-lg px-2.5 py-1.5 text-sm",
                  tab === it.id
                    ? "bg-accent-soft text-accent"
                    : "text-ink-dim hover:bg-surface-hover",
                )}
              >
                {it.icon}
              </button>
            ))}
          </div>
        ))}
      </nav>
    </>
  );
}
