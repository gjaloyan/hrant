import { useEffect, useState } from "react";
import {
  addCoreFact,
  deleteCoreFact,
  deleteNote,
  fetchCore,
  fetchGaps,
  fetchGraphEntities,
  fetchGraphNeighbors,
  fetchGraphStats,
  fetchKnowledge,
  forceLearn,
  GapEntry,
  GraphEntity,
  GraphNeighbor,
  GraphStats,
  IndexEntry,
  quickNote,
  reindexGraph,
} from "../api";

export default function KnowledgePanel({
  onSelectTopic,
}: {
  onSelectTopic: (topic: string) => void;
}) {
  const [byCat, setByCat] = useState<Record<string, IndexEntry[]>>({});
  const [core, setCore] = useState<string>("");
  const [coreTokens, setCoreTokens] = useState(0);
  const [coreMax, setCoreMax] = useState(0);
  const [fact, setFact] = useState("");
  const [forgetText, setForgetText] = useState("");
  const [learnTopic, setLearnTopic] = useState("");
  const [learnDepth, setLearnDepth] = useState<"quick" | "deep">("quick");
  const [quickNoteText, setQuickNoteText] = useState("");
  const [filter, setFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [gaps, setGaps] = useState<{ open: GapEntry[]; closed: GapEntry[] }>({
    open: [],
    closed: [],
  });
  const [showGaps, setShowGaps] = useState(false);
  const [msg, setMsg] = useState("");

  // Knowledge Graph state
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  const [showGraph, setShowGraph] = useState(false);
  const [graphEntities, setGraphEntities] = useState<GraphEntity[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [neighbors, setNeighbors] = useState<GraphNeighbor[]>([]);
  const [entityFilter, setEntityFilter] = useState("");
  const [reindexing, setReindexing] = useState(false);

  const refresh = async () => {
    try {
      const [k, c, g, gs] = await Promise.all([
        fetchKnowledge(),
        fetchCore(),
        fetchGaps(),
        fetchGraphStats().catch(() => null),
      ]);
      setByCat(k.by_category || {});
      setCore(c.content || "");
      setCoreTokens(c.tokens || 0);
      setCoreMax(c.max || 0);
      setGaps({ open: g.open, closed: g.closed });
      if (gs) setGraphStats(gs);
    } catch (e: any) {
      setMsg("Error loading: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 3000);
  };

  const handleAddFact = async () => {
    if (!fact.trim()) return;
    const res = await addCoreFact(fact.trim());
    setFact("");
    flash(res.message);
    refresh();
  };

  const handleForget = async () => {
    if (!forgetText.trim()) return;
    const res = await deleteCoreFact(forgetText.trim());
    setForgetText("");
    flash(res.message);
    refresh();
  };

  const handleLearn = async () => {
    if (!learnTopic.trim()) return;
    setBusy(true);
    try {
      await forceLearn(learnTopic.trim(), learnDepth);
      setLearnTopic("");
      flash("Note created");
      await refresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleQuickNote = async () => {
    if (!quickNoteText.trim()) return;
    try {
      const res = await quickNote(quickNoteText.trim());
      setQuickNoteText("");
      flash(`Note saved: ${res.topic}`);
      await refresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleDelete = async (topic: string) => {
    if (!confirm(`Delete note "${topic}"?`)) return;
    await deleteNote(topic);
    flash("Deleted");
    refresh();
  };

  const filterFn = (t: IndexEntry) =>
    !filter.trim() ||
    t.topic.toLowerCase().includes(filter.toLowerCase()) ||
    t.keywords.some((k) => k.toLowerCase().includes(filter.toLowerCase()));

  const totalTopics = Object.values(byCat).reduce((s, arr) => s + arr.length, 0);

  return (
    <div className="flex flex-1 min-h-0">
      {/* Left: Core Memory + Actions */}
      <aside className="w-80 border-r border-slate-800 bg-slate-950/60 p-3 overflow-y-auto text-sm space-y-4">
        {/* Core Memory */}
        <section>
          <h2 className="font-bold mb-2">Core Memory</h2>
          <div className="text-xs opacity-60 mb-1">
            {coreTokens}/{coreMax} tokens
          </div>
          <pre className="bg-slate-900 rounded p-2 whitespace-pre-wrap max-h-48 overflow-y-auto text-xs">
            {core || "(empty)"}
          </pre>
          <div className="flex gap-1 mt-2">
            <input
              className="flex-1 bg-slate-900 rounded px-2 py-1 text-xs outline-none"
              placeholder="remember fact..."
              value={fact}
              onChange={(e) => setFact(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddFact()}
            />
            <button
              onClick={handleAddFact}
              className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 text-xs"
            >
              +
            </button>
          </div>
          <div className="flex gap-1 mt-1">
            <input
              className="flex-1 bg-slate-900 rounded px-2 py-1 text-xs outline-none"
              placeholder="forget about..."
              value={forgetText}
              onChange={(e) => setForgetText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleForget()}
            />
            <button
              onClick={handleForget}
              className="bg-rose-700 hover:bg-rose-600 rounded px-2 text-xs"
            >
              -
            </button>
          </div>
        </section>

        {/* Learn Topic */}
        <section>
          <h2 className="font-bold mb-2">Learn Topic</h2>
          <div className="flex gap-1">
            <input
              className="flex-1 bg-slate-900 rounded px-2 py-1 text-xs outline-none"
              placeholder="topic to learn..."
              value={learnTopic}
              onChange={(e) => setLearnTopic(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLearn()}
            />
            <select
              className="bg-slate-900 rounded px-1 text-xs"
              value={learnDepth}
              onChange={(e) => setLearnDepth(e.target.value as "quick" | "deep")}
            >
              <option value="quick">quick</option>
              <option value="deep">deep</option>
            </select>
            <button
              onClick={handleLearn}
              disabled={busy}
              className="bg-sky-700 hover:bg-sky-600 rounded px-2 text-xs disabled:opacity-50"
            >
              {busy ? "..." : "learn"}
            </button>
          </div>
        </section>

        {/* Quick Note */}
        <section>
          <h2 className="font-bold mb-2">Quick Note</h2>
          <textarea
            className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none resize-none"
            rows={2}
            placeholder="save a quick note without research..."
            value={quickNoteText}
            onChange={(e) => setQuickNoteText(e.target.value)}
          />
          <button
            onClick={handleQuickNote}
            className="mt-1 bg-emerald-700 hover:bg-emerald-600 rounded px-3 py-1 text-xs w-full"
          >
            Save note
          </button>
        </section>

        {/* Gaps */}
        <section>
          <button
            onClick={() => setShowGaps(!showGaps)}
            className="font-bold text-left w-full flex justify-between items-center"
          >
            <span>Knowledge Gaps</span>
            <span className="text-xs opacity-60">
              {gaps.open.length} open / {gaps.closed.length} closed
            </span>
          </button>
          {showGaps && (
            <div className="mt-2 space-y-1 text-xs max-h-48 overflow-y-auto">
              {gaps.open.length === 0 && gaps.closed.length === 0 && (
                <div className="opacity-50">(no gaps recorded)</div>
              )}
              {gaps.open.map((g) => (
                <div key={g.topic} className="flex justify-between bg-rose-900/30 rounded px-2 py-1">
                  <span>{g.topic}</span>
                  <span className="opacity-60">x{g.count}</span>
                </div>
              ))}
              {gaps.closed.slice(0, 10).map((g) => (
                <div key={g.topic} className="flex justify-between bg-slate-800 rounded px-2 py-1 opacity-60">
                  <span>{g.topic}</span>
                  <span>x{g.count} (closed)</span>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Knowledge Graph */}
        <section>
          <button
            onClick={async () => {
              if (!showGraph) {
                try {
                  const ents = await fetchGraphEntities();
                  setGraphEntities(ents.entities || []);
                } catch { /* ignore */ }
              }
              setShowGraph(!showGraph);
            }}
            className="font-bold text-left w-full flex justify-between items-center"
          >
            <span>Knowledge Graph</span>
            <span className="text-xs opacity-60">
              {graphStats
                ? `${graphStats.entities} entities / ${graphStats.edges} edges`
                : "..."}
            </span>
          </button>
          {showGraph && (
            <div className="mt-2 space-y-2">
              {/* Stats */}
              {graphStats && (
                <div className="grid grid-cols-3 gap-1 text-xs">
                  <div className="bg-slate-800 rounded p-1.5 text-center">
                    <div className="text-emerald-400 font-bold">{graphStats.entities}</div>
                    <div className="opacity-50">entities</div>
                  </div>
                  <div className="bg-slate-800 rounded p-1.5 text-center">
                    <div className="text-sky-400 font-bold">{graphStats.edges}</div>
                    <div className="opacity-50">edges</div>
                  </div>
                  <div className="bg-slate-800 rounded p-1.5 text-center">
                    <div className="text-amber-400 font-bold">{graphStats.notes_indexed}</div>
                    <div className="opacity-50">notes</div>
                  </div>
                </div>
              )}

              {/* Reindex */}
              <button
                onClick={async () => {
                  setReindexing(true);
                  try {
                    const res = await reindexGraph();
                    flash(`Reindexed: ${res.notes} notes, ${res.triples} triples`);
                    const [gs, ents] = await Promise.all([
                      fetchGraphStats(),
                      fetchGraphEntities(),
                    ]);
                    setGraphStats(gs);
                    setGraphEntities(ents.entities || []);
                  } catch (e: any) {
                    flash("Reindex error: " + e.message);
                  } finally {
                    setReindexing(false);
                  }
                }}
                disabled={reindexing}
                className="w-full bg-violet-800 hover:bg-violet-700 rounded py-1 text-xs disabled:opacity-50"
              >
                {reindexing ? "Reindexing..." : "Reindex all notes"}
              </button>

              {/* Entity filter */}
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none"
                placeholder="filter entities..."
                value={entityFilter}
                onChange={(e) => setEntityFilter(e.target.value)}
              />

              {/* Entity list */}
              <div className="max-h-60 overflow-y-auto space-y-0.5">
                {graphEntities
                  .filter((e) =>
                    !entityFilter.trim() ||
                    e.name.toLowerCase().includes(entityFilter.toLowerCase())
                  )
                  .slice(0, 50)
                  .map((e) => (
                    <button
                      key={e.name}
                      onClick={async () => {
                        setSelectedEntity(e.name);
                        try {
                          const res = await fetchGraphNeighbors(e.name);
                          setNeighbors(res.neighbors || []);
                        } catch {
                          setNeighbors([]);
                        }
                      }}
                      className={`w-full text-left flex justify-between rounded px-2 py-0.5 text-xs hover:bg-slate-700 transition-colors ${
                        selectedEntity === e.name ? "bg-slate-700" : "bg-slate-800/50"
                      }`}
                    >
                      <span className="truncate">{e.name}</span>
                      <span className="opacity-50 shrink-0 ml-1">{e.edges}</span>
                    </button>
                  ))}
                {graphEntities.length === 0 && (
                  <div className="opacity-50 text-xs text-center py-2">
                    No entities. Click "Reindex" to build the graph.
                  </div>
                )}
              </div>

              {/* Neighbors */}
              {selectedEntity && (
                <div className="border-t border-slate-700 pt-2">
                  <div className="text-xs font-bold mb-1 text-emerald-400">
                    {selectedEntity} ({neighbors.length} connections)
                  </div>
                  <div className="max-h-40 overflow-y-auto space-y-0.5">
                    {neighbors.map((n, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-1 text-xs bg-slate-800/50 rounded px-2 py-0.5"
                      >
                        <span className="text-sky-400 opacity-70 shrink-0">
                          {n.relation}
                        </span>
                        <span className="mx-1 opacity-30">→</span>
                        <button
                          onClick={async () => {
                            setSelectedEntity(n.target);
                            try {
                              const res = await fetchGraphNeighbors(n.target);
                              setNeighbors(res.neighbors || []);
                            } catch {
                              setNeighbors([]);
                            }
                          }}
                          className="text-emerald-300 hover:underline truncate"
                        >
                          {n.target}
                        </button>
                        <span className="opacity-30 shrink-0 ml-auto text-[10px]">
                          [{n.note}]
                        </span>
                      </div>
                    ))}
                    {neighbors.length === 0 && (
                      <div className="opacity-50 text-xs">(no connections)</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        <button
          onClick={refresh}
          className="w-full bg-slate-800 hover:bg-slate-700 rounded py-1 text-xs"
        >
          Refresh
        </button>

        {msg && (
          <div className="bg-sky-900/50 text-xs rounded p-2">{msg}</div>
        )}
      </aside>

      {/* Right: Knowledge Base */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold">Knowledge Base ({totalTopics} topics)</h2>
          <input
            className="bg-slate-900 rounded px-3 py-1.5 text-sm w-64 outline-none focus:ring-1 focus:ring-sky-600"
            placeholder="filter topics..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>

        {Object.entries(byCat).map(([cat, items]) => {
          const filtered = items.filter(filterFn);
          if (!filtered.length) return null;
          return (
            <div key={cat} className="mb-4">
              <div className="text-xs uppercase opacity-60 mb-2 font-semibold">
                {cat} ({filtered.length})
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {filtered
                  .sort((a, b) => b.access_count - a.access_count)
                  .map((t) => (
                    <div
                      key={t.path}
                      className="bg-slate-800 rounded p-2 text-sm flex items-center justify-between group hover:bg-slate-700 transition-colors"
                    >
                      <button
                        onClick={() => onSelectTopic(t.topic)}
                        className="text-left flex-1 min-w-0"
                      >
                        <div className="truncate">{t.topic}</div>
                        <div className="text-xs opacity-50">
                          x{t.access_count} · {t.keywords.slice(0, 3).join(", ")}
                        </div>
                      </button>
                      <button
                        onClick={() => handleDelete(t.topic)}
                        className="opacity-0 group-hover:opacity-60 hover:opacity-100 text-rose-400 px-2 ml-2 shrink-0"
                        title="Delete note"
                      >
                        x
                      </button>
                    </div>
                  ))}
              </div>
            </div>
          );
        })}

        {totalTopics === 0 && (
          <div className="opacity-50 text-sm text-center mt-8">
            Knowledge base is empty. Use "Learn Topic" or chat with the agent to build it up.
          </div>
        )}
      </div>
    </div>
  );
}
