import { Component, ReactNode, useEffect, useRef, useState } from "react";
import {
  KGraphFull,
  KGraphNode,
  KGraphNodeKind,
  KGraphSearchResult,
  KGraphStats,
  KGraphNeighborhood,
  fetchKGraph,
  fetchKGraphNode,
  fetchKGraphStats,
  rebuildKGraph,
  searchKGraph,
} from "../../api";

type Props = { flash: (msg: string) => void };

const KIND_COLOR: Record<KGraphNodeKind, string> = {
  fact:    "#4FCF6A",  // emerald
  topic:   "#FF8C00",  // warm orange (matches CLI accent)
  skill:   "#FFC107",  // amber
  project: "#5FB3FF",  // sky
  entity:  "#9CA3AF",  // slate
};

const KIND_LABEL: Record<KGraphNodeKind, string> = {
  fact: "Fact",
  topic: "Topic",
  skill: "Skill",
  project: "Project",
  entity: "Entity",
};

type View = "explorer" | "graph";

export default function KnowledgeGraphTab({ flash }: Props) {
  const [stats, setStats] = useState<KGraphStats | null>(null);
  const [view, setView] = useState<View>("explorer");
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<KGraphNodeKind | "">("");
  const [results, setResults] = useState<KGraphSearchResult[]>([]);
  const [selected, setSelected] = useState<KGraphNeighborhood | null>(null);
  const [graph, setGraph] = useState<KGraphFull | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshStats = async () => {
    try {
      const s = await fetchKGraphStats();
      setStats(s);
    } catch (e: any) {
      flash("Load failed: " + e.message);
    }
  };

  useEffect(() => {
    refreshStats();
  }, []);

  // Re-run search whenever the query / kind filter changes, debounced.
  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const r = await searchKGraph(query, kindFilter || undefined, 50);
        setResults(r.results);
      } catch (e: any) {
        flash("Search failed: " + e.message);
      }
    }, 200);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, kindFilter]);

  const handleOpen = async (node_id: string) => {
    try {
      const n = await fetchKGraphNode(node_id);
      setSelected(n);
    } catch (e: any) {
      flash("Open failed: " + e.message);
    }
  };

  const handleRebuild = async () => {
    if (!confirm("Re-derive the graph from memory_facts + skills + goals? Wipes any LLM-proposed edges.")) return;
    setLoading(true);
    try {
      const r = await rebuildKGraph();
      flash(
        `Rebuilt: ${r.stats.facts} facts, ${r.stats.topics} topics, ` +
        `${r.stats.skills} skills, ${r.stats.projects} projects, ` +
        `${r.stats.edges} edges`,
      );
      await refreshStats();
      setSelected(null);
      setGraph(null);
    } catch (e: any) {
      flash("Rebuild failed: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchToGraph = async () => {
    setView("graph");
    if (graph !== null) return;
    setLoading(true);
    try {
      const g = await fetchKGraph();
      setGraph(g);
    } catch (e: any) {
      flash("Graph load failed: " + e.message);
      setView("explorer");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0 gap-3">
      {/* Header: stats + view switch + rebuild */}
      <div className="bg-slate-800 rounded p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex gap-2">
            <button
              onClick={() => setView("explorer")}
              className={`text-xs px-2.5 py-1 rounded ${
                view === "explorer"
                  ? "bg-accent-soft text-accent font-medium"
                  : "bg-slate-700 hover:bg-slate-600 text-slate-300"
              }`}
            >
              Explorer
            </button>
            <button
              onClick={handleSwitchToGraph}
              className={`text-xs px-2.5 py-1 rounded ${
                view === "graph"
                  ? "bg-accent-soft text-accent font-medium"
                  : "bg-slate-700 hover:bg-slate-600 text-slate-300"
              }`}
            >
              Graph view
            </button>
            <button
              onClick={handleRebuild}
              disabled={loading}
              className="text-xs px-2.5 py-1 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-40"
            >
              Rebuild
            </button>
          </div>
        </div>
        {stats?.load_error && (
          <div className="mb-2 bg-red-950/40 border border-red-800/50 rounded p-2 text-xs text-red-200">
            <span className="font-semibold">Graph file warning:</span>{" "}
            {stats.load_error}
          </div>
        )}
        {stats && (
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
            <div>
              <span className="text-slate-500">Nodes:</span> {stats.total_nodes}{" "}
              <span className="text-slate-500">
                ({Object.entries(stats.by_kind)
                  .filter(([_, n]) => n > 0)
                  .map(([k, n]) => `${n} ${k}`)
                  .join(", ")})
              </span>
            </div>
            <div>
              <span className="text-slate-500">Edges:</span> {stats.total_edges}
            </div>
            {stats.top_topics.length > 0 && (
              <div className="col-span-2 mt-1">
                <span className="text-slate-500">Top topics:</span>{" "}
                {stats.top_topics.slice(0, 8).map((t, i) => (
                  <button
                    key={t.id}
                    onClick={() => handleOpen(t.id)}
                    className="hover:text-amber-300 mr-2"
                    title={`${t.degree} connections`}
                  >
                    <span style={{ color: KIND_COLOR.topic }}>{t.label}</span>
                    <span className="text-slate-500"> ({t.degree})</span>
                    {i < stats.top_topics.slice(0, 8).length - 1 ? "," : ""}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main content */}
      {view === "explorer" ? (
        <div className="flex gap-4 flex-1 min-h-0">
          {/* Left: search + results */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search facts / topics / skills..."
                className="flex-1 bg-slate-900 rounded px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
              />
              <select
                value={kindFilter}
                onChange={(e) => setKindFilter(e.target.value as KGraphNodeKind | "")}
                className="bg-slate-900 rounded px-2 py-1.5 text-sm"
              >
                <option value="">All kinds</option>
                <option value="fact">Facts</option>
                <option value="topic">Topics</option>
                <option value="skill">Skills</option>
                <option value="project">Projects</option>
                <option value="entity">Entities</option>
              </select>
            </div>
            <div className="flex-1 overflow-y-auto space-y-1">
              {!query.trim() ? (
                <div className="text-slate-500 text-sm italic p-4">
                  Start typing to search, or click a top topic above.
                </div>
              ) : results.length === 0 ? (
                <div className="text-slate-500 text-sm italic p-4">
                  No matches.
                </div>
              ) : (
                results.map((r) => (
                  <div
                    key={r.id}
                    onClick={() => handleOpen(r.id)}
                    className={`bg-slate-800 hover:bg-slate-700/80 rounded p-2 cursor-pointer border ${
                      selected?.node.id === r.id
                        ? "border-sky-500"
                        : "border-transparent"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs px-1.5 py-0.5 rounded font-mono"
                        style={{ color: KIND_COLOR[r.kind] }}
                      >
                        {KIND_LABEL[r.kind]}
                      </span>
                      <span className="text-sm text-slate-200 truncate flex-1">
                        {r.label}
                      </span>
                      <span className="text-xs text-slate-500">{r.degree}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Right: detail */}
          <div className="w-[460px] shrink-0 border-l border-slate-800 pl-4 flex flex-col min-h-0">
            {selected ? (
              <NodeDetail
                neighborhood={selected}
                onOpen={handleOpen}
                onClose={() => setSelected(null)}
              />
            ) : (
              <div className="text-slate-500 text-sm italic">
                Click a search result or a top topic to inspect it.
              </div>
            )}
          </div>
        </div>
      ) : (
        <GraphErrorBoundary>
          <GraphView
            graph={graph}
            loading={loading}
            onNodeClick={handleOpen}
            selectedId={selected?.node.id}
          />
        </GraphErrorBoundary>
      )}
    </div>
  );
}


/* ─── Detail pane ────────────────────────────────────────────────── */


function NodeDetail({
  neighborhood,
  onOpen,
  onClose,
}: {
  neighborhood: KGraphNeighborhood;
  onOpen: (id: string) => void;
  onClose: () => void;
}) {
  const { node, neighbors } = neighborhood;

  return (
    <>
      <div className="mb-3 flex items-center justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span
              className="text-xs px-1.5 py-0.5 rounded font-mono"
              style={{ color: KIND_COLOR[node.kind] }}
            >
              {KIND_LABEL[node.kind]}
            </span>
            <span className="text-xs text-slate-500 font-mono truncate">
              {node.id}
            </span>
          </div>
          <div className="text-sm text-slate-200 break-words">{node.label}</div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-500 hover:text-slate-300 text-sm ml-2"
        >
          ✕
        </button>
      </div>

      {Object.keys(node.metadata || {}).length > 0 && (
        <section className="mb-3">
          <div className="text-xs text-slate-400 mb-1">METADATA</div>
          <pre className="bg-slate-900/60 rounded p-2 text-xs text-slate-300 overflow-x-auto">
            {JSON.stringify(node.metadata, null, 2)}
          </pre>
        </section>
      )}

      <section className="flex-1 overflow-y-auto">
        <div className="text-xs text-slate-400 mb-1">
          NEIGHBOURS ({neighbors.length})
        </div>
        {neighbors.length === 0 ? (
          <div className="text-xs text-slate-500 italic">
            (no connections)
          </div>
        ) : (
          <div className="space-y-1">
            {neighbors.map((n, i) => (
              <div
                key={i}
                onClick={() => onOpen(n.node.id)}
                className="bg-slate-900/60 hover:bg-slate-800 rounded p-2 cursor-pointer text-xs"
              >
                <div className="flex items-center gap-1 text-slate-500">
                  <span>{n.direction === "out" ? "→" : "←"}</span>
                  <span className="font-mono">{n.edge.kind}</span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span
                    className="text-[10px] font-mono"
                    style={{ color: KIND_COLOR[n.node.kind] }}
                  >
                    {KIND_LABEL[n.node.kind]}
                  </span>
                  <span className="text-slate-200 truncate">
                    {n.node.label}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  );
}


/* ─── Graph view (simple SVG force-directed) ──────────────────────── */


type Positioned = KGraphNode & { x: number; y: number };

function GraphView({
  graph,
  loading,
  onNodeClick,
  selectedId,
}: {
  graph: KGraphFull | null;
  loading: boolean;
  onNodeClick: (id: string) => void;
  selectedId?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 600 });
  const [layout, setLayout] = useState<LayoutResult | null>(null);
  const [layoutPending, setLayoutPending] = useState(false);

  // Audit #21: use a ResizeObserver instead of window resize so
  // we re-measure when the parent panel resizes (e.g. user drags
  // the settings sidebar) and when the tab transitions out of a
  // display:none state where clientWidth would have been 0 on
  // initial mount.
  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const update = () => {
      const w = node.clientWidth;
      const h = node.clientHeight;
      if (w > 0 && h > 0) {
        setDims((prev) =>
          prev.w === w && prev.h === h ? prev : { w, h },
        );
      }
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  // Audit #9: run the force-directed sim ASYNCHRONOUSLY after
  // first paint. The previous useMemo version computed inside the
  // render cycle and blocked the browser for ~500ms on a 200-node
  // graph. Now we render an empty SVG immediately, then a
  // setTimeout(0) trampoline kicks off the layout — the browser
  // can paint the "computing layout..." state and stay responsive.
  useEffect(() => {
    if (!graph || graph.nodes.length === 0 || dims.w === 0) {
      setLayout(null);
      return;
    }
    setLayoutPending(true);
    let cancelled = false;
    const handle = window.setTimeout(() => {
      const result = runForceLayout(graph.nodes, graph.edges, dims.w, dims.h);
      if (cancelled) return;
      setLayout(result);
      setLayoutPending(false);
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [graph, dims.w, dims.h]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400">
        Loading graph…
      </div>
    );
  }
  if (!graph) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500 italic">
        No graph data — click Rebuild.
      </div>
    );
  }
  if (!layout) {
    return (
      <div
        ref={containerRef}
        className="flex-1 flex items-center justify-center text-slate-400 bg-slate-900 rounded min-h-0"
      >
        {layoutPending ? "Computing layout…" : "Sizing…"}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 bg-slate-900 rounded overflow-hidden min-h-0"
    >
      <svg width={dims.w} height={dims.h} viewBox={`0 0 ${dims.w} ${dims.h}`}>
        {/* Edges */}
        {graph.edges.map((e, i) => {
          const s = layout.byId[e.source];
          const t = layout.byId[e.target];
          if (!s || !t) return null;
          return (
            <line
              key={i}
              x1={s.x} y1={s.y} x2={t.x} y2={t.y}
              stroke="#475569" strokeWidth={0.8} opacity={0.5}
            />
          );
        })}
        {/* Nodes */}
        {layout.nodes.map((n) => {
          const r = 4 + Math.min(8, Math.sqrt(n.weight) * 2);
          const selected = n.id === selectedId;
          return (
            <g
              key={n.id}
              onClick={() => onNodeClick(n.id)}
              style={{ cursor: "pointer" }}
            >
              <circle
                cx={n.x} cy={n.y} r={r}
                fill={KIND_COLOR[n.kind]}
                stroke={selected ? "#fff" : "#0f172a"}
                strokeWidth={selected ? 2 : 1}
              />
              {/* Show labels only for topics + skills + projects — */}
              {/* fact labels would clutter the view. */}
              {(n.kind === "topic" || n.kind === "skill" || n.kind === "project") && (
                <text
                  x={n.x + r + 3} y={n.y + 3}
                  fill="#cbd5e1" fontSize={10}
                  pointerEvents="none"
                >
                  {n.label.length > 24 ? n.label.slice(0, 24) + "…" : n.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}


/* ─── Error boundary ─────────────────────────────────────────────── */

// Audit #10: the SVG renderer + force layout do arithmetic on
// floating-point node positions; a corrupt graph file (NaN
// weights, undefined edge endpoints from manual edits, etc.)
// could throw inside the render. Without a boundary, that
// cascades up and crashes the whole Settings panel with a white
// screen. ErrorBoundary keeps the rest of Settings alive and
// shows a recovery hint.

type ErrorBoundaryState = { error: Error | null };
type ErrorBoundaryProps = { children: ReactNode; fallback?: ReactNode };

class GraphErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("KnowledgeGraphTab caught:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div className="flex-1 bg-red-950/40 rounded p-4 text-red-200 text-sm">
            <div className="font-semibold mb-1">Graph view crashed.</div>
            <div className="text-xs text-red-300/80">
              {this.state.error.message}
            </div>
            <div className="text-xs text-red-300/60 mt-2">
              Try Rebuild. If the error persists, delete{" "}
              <span className="font-mono">~/.hrant/data/knowledge/graph.json</span>{" "}
              and rebuild from sources.
            </div>
          </div>
        )
      );
    }
    return this.props.children;
  }
}


/* ─── Force-directed layout (no dependency) ───────────────────────── */


type LayoutResult = {
  nodes: Positioned[];
  byId: Record<string, Positioned>;
};

function runForceLayout(
  nodes: KGraphNode[],
  edges: { source: string; target: string }[],
  width: number,
  height: number,
): LayoutResult {
  // Place nodes on a circle to start, then iterate a basic
  // spring/repulsion sim. Not as pretty as d3-force but no dep
  // and good enough for ≤300 nodes.
  const positioned: Positioned[] = nodes.map((n, i) => {
    const angle = (i / nodes.length) * Math.PI * 2;
    const r = Math.min(width, height) * 0.35;
    return {
      ...n,
      x: width / 2 + Math.cos(angle) * r,
      y: height / 2 + Math.sin(angle) * r,
    };
  });
  const byId: Record<string, Positioned> = {};
  // Build a `nodeId → array index` Map once. Audit #1: the old
  // code called `positioned.indexOf(s)` inside the per-edge loop
  // for every iteration of the sim — O(N × E × iters) overall.
  // At 200 nodes / 400 edges / 150 iters that was ~24M indexOf
  // calls each scanning ~100 entries → ~2 billion comparisons,
  // synchronously inside useMemo. Browser tab froze. A Map
  // lookup is O(1), so the same workload now drops to ~24M map
  // hits → <100ms.
  const indexOf = new Map<string, number>();
  positioned.forEach((n, i) => {
    byId[n.id] = n;
    indexOf.set(n.id, i);
  });

  const iterations = nodes.length > 200 ? 80 : 150;
  const k = Math.sqrt((width * height) / nodes.length) * 0.6;  // ideal distance
  const repel = k * k;
  const attractK = 1 / k;

  for (let iter = 0; iter < iterations; iter++) {
    const cooling = 1 - iter / iterations;
    const dx: number[] = new Array(positioned.length).fill(0);
    const dy: number[] = new Array(positioned.length).fill(0);

    // Repulsion (all pairs — O(N²), fine at ≤300 nodes).
    for (let i = 0; i < positioned.length; i++) {
      for (let j = i + 1; j < positioned.length; j++) {
        const a = positioned[i];
        const b = positioned[j];
        let dxi = a.x - b.x;
        let dyi = a.y - b.y;
        let dist2 = dxi * dxi + dyi * dyi;
        if (dist2 < 1) {
          dxi = Math.random() - 0.5;
          dyi = Math.random() - 0.5;
          dist2 = 1;
        }
        const force = repel / dist2;
        const dist = Math.sqrt(dist2);
        const fx = (dxi / dist) * force;
        const fy = (dyi / dist) * force;
        dx[i] += fx;
        dy[i] += fy;
        dx[j] -= fx;
        dy[j] -= fy;
      }
    }

    // Attraction (along edges). O(1) index lookup per endpoint.
    for (const e of edges) {
      const si = indexOf.get(e.source);
      const ti = indexOf.get(e.target);
      if (si === undefined || ti === undefined) continue;
      const s = positioned[si];
      const t = positioned[ti];
      const dxi = s.x - t.x;
      const dyi = s.y - t.y;
      const dist = Math.sqrt(dxi * dxi + dyi * dyi) || 1;
      const force = dist * dist * attractK;
      const fx = (dxi / dist) * force;
      const fy = (dyi / dist) * force;
      dx[si] -= fx;
      dy[si] -= fy;
      dx[ti] += fx;
      dy[ti] += fy;
    }

    // Apply with cooling + clamp to viewport.
    for (let i = 0; i < positioned.length; i++) {
      const n = positioned[i];
      const dxi = dx[i];
      const dyi = dy[i];
      const disp = Math.sqrt(dxi * dxi + dyi * dyi) || 1;
      const limited = Math.min(disp, k * 4 * cooling);
      n.x += (dxi / disp) * limited;
      n.y += (dyi / disp) * limited;
      n.x = Math.max(20, Math.min(width - 20, n.x));
      n.y = Math.max(20, Math.min(height - 20, n.y));
    }
  }

  return { nodes: positioned, byId };
}
