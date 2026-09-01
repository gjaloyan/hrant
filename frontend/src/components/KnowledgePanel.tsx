/** What the agent knows.
 *
 * The old screen stacked five unrelated workflows down a 320px column —
 * a core-memory editor, remember/forget inputs, a topic-research form, a
 * quick-note box, a gap counter and a graph counter — with a "Save note"
 * button rendered as the loudest thing on the page. Cards showed "x67"
 * with nothing to say what 67 counted, and the filter box sat in the far
 * corner from the list it filtered.
 *
 * This screen now answers one question: what does it know, and how do I
 * add to it. Core memory moved to Settings → Character (it is a standing
 * configuration, edited rarely). Graph statistics moved to the Graph tab,
 * which is where someone looking for them would go. Knowledge gaps became
 * something you can act on instead of a number in a corner.
 */
import { useEffect, useMemo, useState } from "react";
import {
  deleteNote,
  fetchGaps,
  fetchKnowledge,
  forceLearn,
  quickNote,
  type GapEntry,
  type IndexEntry,
} from "../api";
import { Badge, Button, EmptyState, Flash, Spinner, cx } from "../ui";

function TeachDialog({
  open,
  onClose,
  onDone,
}: {
  open: boolean;
  onClose: () => void;
  onDone: (msg: string) => void;
}) {
  // Two ways to add knowledge that used to be two separate forms competing
  // for space. They are the same intent — "learn this" — so they are one
  // dialog with a choice, shown only when asked for.
  const [mode, setMode] = useState<"research" | "note">("research");
  const [topic, setTopic] = useState("");
  const [depth, setDepth] = useState<"quick" | "deep">("quick");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  if (!open) return null;

  const submit = async () => {
    setBusy(true);
    try {
      if (mode === "research") {
        if (!topic.trim()) return;
        await forceLearn(topic.trim(), depth);
        onDone(`Researching “${topic.trim()}” (${depth})`);
      } else {
        if (!note.trim()) return;
        const r = await quickNote(note.trim());
        onDone(`Saved as “${r.topic}”`);
      }
      setTopic("");
      setNote("");
      onClose();
    } catch (e: any) {
      onDone("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 p-4 pt-24"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-xl2 border border-edge bg-surface shadow-pop"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="border-b border-edge px-4 py-3">
          <h3 className="font-semibold">Teach the agent</h3>
        </header>
        <div className="space-y-4 p-4">
          <div className="inline-flex rounded-lg border border-edge bg-canvas p-0.5 text-xs">
            {(["research", "note"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cx(
                  "rounded-md px-3 py-1",
                  mode === m
                    ? "bg-accent-soft font-medium text-accent"
                    : "text-ink-dim hover:text-ink",
                )}
              >
                {m === "research" ? "Research a topic" : "Save a note"}
              </button>
            ))}
          </div>

          {mode === "research" ? (
            <>
              <label className="block">
                <span className="mb-1 block text-sm font-medium">Topic</span>
                <input
                  autoFocus
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && submit()}
                  placeholder="e.g. Armenian bankruptcy procedure"
                  className="w-full"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-sm font-medium">Depth</span>
                <select
                  value={depth}
                  onChange={(e) => setDepth(e.target.value as any)}
                  className="w-full"
                >
                  <option value="quick">Quick — one pass, a few minutes</option>
                  <option value="deep">Deep — follows sources, slower and costs more</option>
                </select>
              </label>
            </>
          ) : (
            <label className="block">
              <span className="mb-1 block text-sm font-medium">Note</span>
              <textarea
                autoFocus
                rows={5}
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Something the agent should remember. It picks the topic name."
                className="w-full resize-none"
              />
            </label>
          )}
        </div>
        <footer className="flex justify-end gap-2 border-t border-edge px-4 py-3">
          <Button kind="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            kind="primary"
            onClick={submit}
            disabled={busy || (mode === "research" ? !topic.trim() : !note.trim())}
          >
            {busy ? <Spinner /> : mode === "research" ? "Research" : "Save"}
          </Button>
        </footer>
      </div>
    </div>
  );
}

export default function KnowledgePanel({
  onSelectTopic,
}: {
  onSelectTopic: (topic: string) => void;
}) {
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [gaps, setGaps] = useState<GapEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState("");
  const [category, setCategory] = useState("");
  const [showGaps, setShowGaps] = useState(false);
  const [teaching, setTeaching] = useState(false);
  const [msg, setMsg] = useState("");

  const flash = (t: string) => {
    setMsg(t);
    setTimeout(() => setMsg(""), 3500);
  };

  const load = async () => {
    setLoading(true);
    try {
      const [k, g] = await Promise.all([fetchKnowledge(), fetchGaps()]);
      setIndex((k as any).index || (k as any).topics || []);
      setGaps(((g as any).gaps || []).filter((x: GapEntry) => !x.has_note_now));
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const remove = async (topic: string) => {
    if (!confirm(`Delete everything the agent knows about “${topic}”?`)) return;
    try {
      await deleteNote(topic);
      flash(`Deleted “${topic}”`);
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const learnGap = async (topic: string) => {
    try {
      await forceLearn(topic, "quick");
      flash(`Researching “${topic}”`);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const categories = useMemo(
    () => [...new Set(index.map((e) => e.category))].sort(),
    [index],
  );

  const grouped = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const rows = index.filter((e) => {
      if (category && e.category !== category) return false;
      if (!needle) return true;
      return `${e.topic} ${e.keywords.join(" ")}`.toLowerCase().includes(needle);
    });
    const out = new Map<string, IndexEntry[]>();
    for (const e of rows) {
      const list = out.get(e.category) || [];
      list.push(e);
      out.set(e.category, list);
    }
    for (const list of out.values())
      list.sort((a, b) => b.access_count - a.access_count);
    return [...out.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [index, q, category]);

  const shown = grouped.reduce((n, [, l]) => n + l.length, 0);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Search sits WITH the list it filters, not in the opposite corner. */}
      <div className="flex flex-wrap items-center gap-2 border-b border-edge px-3 py-2">
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search topics and keywords…"
          className="min-w-[10rem] flex-1 text-sm"
          aria-label="Search knowledge"
        />
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="text-sm"
          aria-label="Filter by category"
        >
          <option value="">All categories</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        {gaps.length > 0 && (
          <Button
            kind={showGaps ? "default" : "ghost"}
            size="sm"
            onClick={() => setShowGaps((v) => !v)}
            title="Questions the agent could not answer from what it knows"
          >
            {gaps.length} gap{gaps.length === 1 ? "" : "s"}
          </Button>
        )}
        <Button kind="primary" size="sm" onClick={() => setTeaching(true)}>
          ＋ Teach
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* Gaps are actionable here — each one can be researched on the
            spot. As a bare counter in a corner it was information nobody
            could do anything with. */}
        {showGaps && gaps.length > 0 && (
          <div className="border-b border-edge bg-warn/5 px-4 py-3">
            <h3 className="mb-2 text-micro font-semibold uppercase text-warn">
              Asked about, not known
            </h3>
            <ul className="space-y-1">
              {gaps.slice(0, 12).map((g) => (
                <li
                  key={g.topic}
                  className="flex items-center gap-2 text-sm"
                >
                  <span className="truncate">{g.topic}</span>
                  <span className="text-xs text-ink-faint">
                    asked {g.count}×
                  </span>
                  <Button
                    kind="ghost"
                    size="sm"
                    className="ml-auto"
                    onClick={() => learnGap(g.topic)}
                  >
                    Research
                  </Button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6">
          {loading && (
            <div className="flex items-center gap-2 text-sm text-ink-dim">
              <Spinner /> Loading…
            </div>
          )}

          {!loading && shown === 0 && (
            <EmptyState
              icon="📚"
              title={index.length === 0 ? "Nothing learned yet" : "No match"}
              action={
                index.length === 0 ? (
                  <Button kind="primary" onClick={() => setTeaching(true)}>
                    Teach the agent something
                  </Button>
                ) : undefined
              }
            >
              {index.length === 0
                ? "Everything the agent researches or is told to remember lands here."
                : "Try a different word, or clear the category filter."}
            </EmptyState>
          )}

          {grouped.map(([cat, entries]) => (
            <section key={cat} className="mb-6">
              <h3 className="mb-2 text-micro font-semibold uppercase text-ink-dim">
                {cat} <span className="text-ink-faint">({entries.length})</span>
              </h3>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {entries.map((e) => (
                  <div
                    key={e.topic}
                    className="group flex flex-col rounded-xl2 border border-edge bg-surface p-3 text-left transition-colors hover:border-edge-strong"
                  >
                    <button
                      onClick={() => onSelectTopic(e.topic)}
                      className="min-w-0 text-left"
                      title={e.topic}
                    >
                      <span className="line-clamp-2 text-sm font-medium">
                        {e.topic}
                      </span>
                    </button>
                    {e.keywords.length > 0 && (
                      <p className="mt-1 line-clamp-2 text-xs text-ink-faint">
                        {e.keywords.slice(0, 5).join(" · ")}
                      </p>
                    )}
                    <div className="mt-2 flex items-center gap-2">
                      {/* "x67" said nothing. This says what 67 counts. */}
                      <Badge
                        tone="neutral"
                        title="How many times the agent has consulted this note"
                      >
                        used {e.access_count}×
                      </Badge>
                      <Button
                        kind="danger"
                        size="sm"
                        className="ml-auto opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100"
                        onClick={() => remove(e.topic)}
                        title={`Delete “${e.topic}”`}
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>

      <TeachDialog
        open={teaching}
        onClose={() => setTeaching(false)}
        onDone={(m) => {
          flash(m);
          load();
        }}
      />
      <Flash text={msg} />
    </div>
  );
}
