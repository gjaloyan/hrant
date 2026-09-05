import { useCallback, useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { forceCollide } from "d3-force";
import {
  fetchKGraphStats,
  fetchKGraphNode,
  searchKGraph,
  rebuildKGraph,
  KGraphStats,
  KGraphNeighborhood,
  KGraphSearchResult,
  FullGraph,
  GraphNeighbor,
} from "../api";

/* This screen was wired to `/api/graph/*` — the LEGACY knowledge_graph,
 * whose endpoints answer `{"nodes":[],"links":[]}` and
 * `{"entities":0,"edges":0}`. So it rendered an empty canvas while the
 * graph it was meant to show sat under `/api/kgraph/*` with 12,229 nodes
 * and 21,434 edges (measured 2026-09-05, after the edges finally
 * persisted). The API bindings for it already existed; nothing had ever
 * called them.
 *
 * It does not load the whole graph. `/api/kgraph` is 4.3 MB and a force
 * layout of 12k nodes is not a picture of anything. You arrive at a hub
 * — a top topic or a search hit — and expand outward, which is also how
 * anyone actually reads a graph.
 */

const EXPAND_CAP = 40;   // neighbours drawn per expansion

/** One neighbourhood in the shape the canvas already renders. */
function toCanvas(hood: KGraphNeighborhood): FullGraph {
  const nodes = [
    { id: hood.node.id, name: hood.node.label, connections: hood.neighbor_count },
  ];
  const links: FullGraph["links"] = [];
  for (const n of (hood.neighbors || []).slice(0, EXPAND_CAP)) {
    nodes.push({ id: n.node.id, name: n.node.label, connections: 1 });
    links.push({
      source: n.edge.source,
      target: n.edge.target,
      relation: n.edge.kind,
      note: n.node.kind,
      weight: n.edge.weight ?? 1,
    });
  }
  return { nodes, links };
}

/** Merge an expansion into what is already on screen, without duplicates. */
function mergeGraph(base: FullGraph, add: FullGraph): FullGraph {
  const byId = new Map(base.nodes.map((n) => [n.id, n]));
  for (const n of add.nodes) if (!byId.has(n.id)) byId.set(n.id, n);
  const key = (l: FullGraph["links"][number]) =>
    `${l.source}|${l.target}|${l.relation}`;
  const seen = new Set(base.links.map(key));
  const links = [...base.links];
  for (const l of add.links) if (!seen.has(key(l))) { seen.add(key(l)); links.push(l); }
  return { nodes: [...byId.values()], links };
}

type NodeObj = {
  id: string;
  name: string;
  connections: number;
  x?: number;
  y?: number;
};

// Color palette for different relation types
const RELATION_COLORS: Record<string, string> = {
  related_to: "#38bdf8",
  mentions: "#a78bfa",
  keyword: "#fbbf24",
  has: "#34d399",
  uses: "#f87171",
  causes: "#fb923c",
};

function relationColor(rel: string): string {
  for (const [key, color] of Object.entries(RELATION_COLORS)) {
    if (rel.includes(key)) return color;
  }
  return "#94a3b8";
}

/** Helper: extract string id from a link endpoint (could be string or object after d3 processes it) */
function linkId(endpoint: string | NodeObj): string {
  return typeof endpoint === "string" ? endpoint : endpoint.id;
}

export default function GraphViewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const [stats, setStats] = useState<KGraphStats | null>(null);
  const [hits, setHits] = useState<KGraphSearchResult[]>([]);
  const [graphData, setGraphData] = useState<FullGraph>({ nodes: [], links: [] });
  const [selected, setSelected] = useState<NodeObj | null>(null);
  const [neighbors, setNeighbors] = useState<GraphNeighbor[]>([]);
  const [hovered, setHovered] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [msg, setMsg] = useState("");
  const [dims, setDims] = useState({ width: 800, height: 600 });
  const [showLabels, setShowLabels] = useState(true);
  const [showRelations, setShowRelations] = useState(false);
  const [filter, setFilter] = useState("");
  const [initialFit, setInitialFit] = useState(false);

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 4000);
  };

  /** Draw one node's neighbourhood; `merge` keeps what is already there. */
  const showNode = useCallback(async (id: string, merge = false) => {
    try {
      const hood = await fetchKGraphNode(id);
      const next = toCanvas(hood);
      setGraphData((cur) => (merge ? mergeGraph(cur, next) : next));
      if (!merge) setInitialFit(false);
    } catch (e: any) {
      flash("Cannot open that node: " + e.message);
    }
  }, []);

  const loadGraph = useCallback(async () => {
    setLoading(true);
    setInitialFit(false);
    try {
      const s = await fetchKGraphStats();
      setStats(s);
      // Start at the busiest topic rather than at 12k nodes at once.
      const start = (s.top_topics || [])[0]?.id;
      if (start) await showNode(start);
      else setGraphData({ nodes: [], links: [] });
    } catch (e: any) {
      flash("Error loading graph: " + e.message);
    } finally {
      setLoading(false);
    }
  }, [showNode]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // Configure forces: balanced — compact but labels don't overlap
  useEffect(() => {
    if (!loading && graphData.nodes.length > 0 && graphRef.current) {
      const fg = graphRef.current;
      fg.d3Force("charge")?.strength(-50).distanceMax(250);
      fg.d3Force("link")?.distance(40);
      // Collision radius based on label width to prevent text overlap
      fg.d3Force("collide", forceCollide()
        .radius((node: any) => {
          const nameLen = (node.name || "").length;
          return Math.max(10, 4 + nameLen * 2.5);
        })
        .strength(0.9)
        .iterations(3)
      );
      fg.d3Force("center")?.strength(0.15);
      fg.d3ReheatSimulation();
    }
  }, [loading, graphData]);

  // Auto-fit graph after initial render
  useEffect(() => {
    if (!initialFit && !loading && graphData.nodes.length > 0 && graphRef.current) {
      const timer = setTimeout(() => {
        graphRef.current?.zoomToFit(400, 60);
        setInitialFit(true);
      }, 1200);
      return () => clearTimeout(timer);
    }
  }, [loading, graphData, initialFit]);

  // Track container size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      setDims({ width: el.clientWidth, height: el.clientHeight });
    };
    update();
    const observer = new ResizeObserver(() => update());
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const handleReindex = async () => {
    setReindexing(true);
    try {
      const res = await rebuildKGraph();
      const st = res.stats || {};
      flash(
        `Rebuilt: ${st.facts ?? 0} facts, ${st.topics ?? 0} topics, ` +
        `${st.edges ?? 0} links`,
      );
      await loadGraph();
    } catch (e: any) {
      flash("Reindex error: " + e.message);
    } finally {
      setReindexing(false);
    }
  };

  // Click node → load neighbors from API (reliable, not dependent on d3 internal state)
  const handleNodeClick = useCallback(
    async (node: NodeObj) => {
      setSelected(node);
      try {
        const hood = await fetchKGraphNode(node.id);
        // Expanding is how you read a graph: the clicked node's own
        // neighbours join what is on screen instead of replacing it.
        setGraphData((cur) => mergeGraph(cur, toCanvas(hood)));
        setNeighbors(
          (hood.neighbors || []).slice(0, EXPAND_CAP).map((n) => ({
            source: n.edge.source,
            target: n.edge.target,
            relation: n.edge.kind,
            note: n.node.label,
            weight: n.edge.weight ?? 1,
          })) as GraphNeighbor[],
        );
      } catch {
        setNeighbors([]);
      }
      if (graphRef.current) {
        graphRef.current.centerAt(node.x, node.y, 400);
        graphRef.current.zoom(2.5, 400);
      }
    },
    []
  );

  const navigateToEntity = useCallback(
    (entityName: string) => {
      // Find node in current graph data (d3 mutates nodes in place, so they have x/y)
      const found = (graphData.nodes as NodeObj[]).find(
        (n) => n.id === entityName || n.name === entityName
      );
      if (found) {
        handleNodeClick(found);
      }
    },
    [graphData.nodes, handleNodeClick]
  );

  // Compute sidebar width for layout
  const sidebarW = selected ? 320 : 0;
  const graphW = Math.max(100, dims.width - sidebarW);

  // Highlighted node ids (selected + its neighbors)
  const highlightSet = new Set<string>();
  if (selected) {
    highlightSet.add(selected.id);
    for (const n of neighbors) {
      highlightSet.add(n.target);
      highlightSet.add(n.source);
    }
  }

  // Node paint
  const nodeCanvasObject = useCallback(
    (node: NodeObj, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const label = node.name;
      const conn = node.connections || 0;
      // Most entities have exactly one link, so the old floor of 4px in
      // slate-500 rendered them as near-invisible specks on the dark
      // canvas — the graph looked empty even when it was not.
      const size = Math.max(5.5, Math.min(14, 4 + conn * 1.1));
      const isSelected = selected?.id === node.id;
      const isNeighbor = highlightSet.has(node.id) && !isSelected;
      const isHovered = hovered === node.id;

      // Glow for selected
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, size + 4, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(251, 191, 36, 0.15)";
        ctx.fill();
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, size, 0, 2 * Math.PI);
      ctx.fillStyle = isSelected
        ? "#f59e0b"
        : isHovered
        ? "#38bdf8"
        : isNeighbor
        ? "#6ee7b7"
        : conn > 8
        ? "#34d399"
        : conn > 3
        ? "#818cf8"
        : "#8b98b4";   // leaf nodes: readable against the canvas, still quiet
      ctx.fill();

      if (isSelected || isHovered) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2 / globalScale;
        ctx.stroke();
      }

      // Far zoom: only hub nodes (many connections). Close zoom: all labels.
      const showThisLabel =
        isSelected || isHovered || isNeighbor ||
        (showLabels && (
          globalScale > 1.5 ||                          // zoomed in — show all
          (globalScale > 0.8 && conn > 3) ||            // medium — show hubs
          conn > 6                                      // far — only big hubs
        ));
      if (showThisLabel) {
        // Font: fixed screen-space size so it's readable at any zoom
        const fontSize = 12 / globalScale;
        ctx.font = `${isSelected || isHovered ? "bold " : ""}${fontSize}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";

        const textW = ctx.measureText(label).width;
        const pad = 2 / globalScale;
        const bgX = node.x! - textW / 2 - pad;
        const bgY = node.y! + size + 2 / globalScale;
        const bgW = textW + pad * 2;
        const bgH = fontSize + pad * 2;

        // Background pill
        ctx.fillStyle = "rgba(2, 6, 23, 0.85)";
        ctx.fillRect(bgX, bgY, bgW, bgH);

        if (isSelected || isHovered) {
          ctx.strokeStyle = isSelected ? "rgba(251,191,36,0.5)" : "rgba(56,189,248,0.4)";
          ctx.lineWidth = 1 / globalScale;
          ctx.strokeRect(bgX, bgY, bgW, bgH);
        }

        ctx.fillStyle = isSelected ? "#fbbf24" : isHovered ? "#38bdf8" : isNeighbor ? "#6ee7b7" : "#cbd5e1";
        ctx.fillText(label, node.x!, bgY + pad);
      }
    },
    [selected, hovered, showLabels, highlightSet]
  );

  // Link paint
  const linkCanvasObject = useCallback(
    (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const src = link.source;
      const tgt = link.target;
      if (!src?.x || !tgt?.x) return;

      const srcId = linkId(src);
      const tgtId = linkId(tgt);
      const isHighlighted = selected && (
        highlightSet.has(srcId) && highlightSet.has(tgtId)
      );

      ctx.beginPath();
      ctx.moveTo(src.x, src.y);
      ctx.lineTo(tgt.x, tgt.y);
      ctx.strokeStyle = isHighlighted
        ? relationColor(link.relation)
        : "rgba(100,116,139,0.25)";
      ctx.lineWidth = isHighlighted
        ? Math.max(1, 2 / globalScale)
        : Math.max(0.5, 1 / globalScale);
      ctx.stroke();

      // Relation label on highlighted links when toggled
      if (showRelations && isHighlighted && globalScale > 1.2) {
        const midX = (src.x + tgt.x) / 2;
        const midY = (src.y + tgt.y) / 2;
        const fontSize = Math.max(8, 9 / globalScale);
        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = "center";
        ctx.fillStyle = relationColor(link.relation);
        ctx.fillText(link.relation, midX, midY - 4);
      }
    },
    [selected, showRelations, highlightSet]
  );

  return (
    <div className="flex flex-col h-full w-full overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-edge bg-surface/60 shrink-0 flex-wrap">
        <h2 className="font-bold text-sm whitespace-nowrap">Knowledge Graph</h2>

        {stats && (
          <div className="flex gap-2 text-xs text-ink-dim">
            <span>{stats.total_nodes.toLocaleString()} nodes</span>
            <span className="text-ink-faint">·</span>
            <span>{stats.total_edges.toLocaleString()} links</span>
            <span className="text-ink-faint">·</span>
            <span>showing {graphData.nodes.length} around {selected?.name
              ?? stats.top_topics?.[0]?.label ?? "a hub"}</span>
          </div>
        )}

        {/* Searches the WHOLE graph, not the handful currently drawn —
            with 12k nodes, filtering what is on screen finds nothing you
            did not already have. */}
        <div className="relative">
          <input
            type="search"
            className="w-56 text-xs"
            placeholder="Find a topic, entity or fact…"
            value={filter}
            onChange={async (e) => {
              const q = e.target.value;
              setFilter(q);
              if (q.trim().length < 2) { setHits([]); return; }
              try {
                const r = await searchKGraph(q.trim(), undefined, 8);
                setHits(r.results || []);
              } catch { setHits([]); }
            }}
          />
          {hits.length > 0 && (
            <ul className="absolute z-20 mt-1 max-h-72 w-80 overflow-auto rounded-lg
                           border border-edge bg-surface shadow-lg">
              {hits.map((h) => (
                <li key={h.id}>
                  <button
                    className="w-full px-2 py-1.5 text-left hover:bg-surface-hover"
                    onClick={() => {
                      setHits([]);
                      setFilter("");
                      showNode(h.id);
                    }}
                  >
                    <span className="block truncate text-xs">{h.label}</span>
                    <span className="block text-[10px] text-ink-faint">
                      {h.kind} · {h.degree} link{h.degree === 1 ? "" : "s"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <label className="flex items-center gap-1 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
          />
          Labels
        </label>

        <label className="flex items-center gap-1 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showRelations}
            onChange={(e) => setShowRelations(e.target.checked)}
          />
          Relations
        </label>

        <button
          onClick={() => graphRef.current?.zoomToFit(400, 40)}
          className="rounded-md border border-edge-strong px-2 py-1 text-xs text-ink-dim hover:bg-surface-hover hover:text-ink"
          title="Fit the whole graph on screen"
        >
          Fit
        </button>

        <button
          onClick={handleReindex}
          disabled={reindexing}
          className="rounded-md border border-edge-strong px-2 py-1 text-xs text-ink-dim hover:bg-surface-hover hover:text-ink disabled:opacity-50"
          title="Re-derive every link from the notes. Takes a while."
        >
          {reindexing ? "Rebuilding…" : "Rebuild"}
        </button>

        <button
          onClick={loadGraph}
          className="rounded-md border border-edge-strong px-2 py-1 text-xs text-ink-dim hover:bg-surface-hover hover:text-ink"
        >
          Refresh
        </button>

        {msg && <span className="text-xs text-sky-400 ml-auto">{msg}</span>}
      </div>

      {/* Main area */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Graph canvas */}
        <div ref={containerRef} className="relative min-w-0 flex-1 bg-canvas">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm opacity-50">
              Loading graph...
            </div>
          ) : graphData.nodes.length === 0 ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-sm opacity-50 gap-3">
              <div>Knowledge graph is empty.</div>
              <div className="text-xs">
                Click "Reindex" to build the graph from existing notes.
              </div>
            </div>
          ) : (
            <ForceGraph2D
              ref={graphRef}
              width={graphW}
              height={dims.height}
              /* No client-side filtering any more. It hid the canvas
                 while you typed — the match is almost never among the
                 forty nodes currently drawn — and the box now searches
                 the whole graph and takes you there instead. */
              graphData={graphData}
              nodeId="id"
              nodeCanvasObject={nodeCanvasObject as any}
              nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
                const size = Math.max(4, Math.min(12, 3 + (node.connections || 0) * 1.0));
                ctx.beginPath();
                ctx.arc(node.x!, node.y!, size + 4, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
              }}
              linkCanvasObject={linkCanvasObject as any}
              linkDirectionalParticles={0}
              onNodeClick={handleNodeClick as any}
              onNodeHover={(node: any) => setHovered(node?.id || null)}
              onBackgroundClick={() => {
                setSelected(null);
                setNeighbors([]);
              }}
              cooldownTicks={200}
              d3AlphaDecay={0.015}
              d3VelocityDecay={0.2}
              enableZoomInteraction={true}
              enablePanInteraction={true}
              backgroundColor="rgba(0,0,0,0)"
            />
          )}
        </div>

        {/* Side panel */}
        {selected && (
          <aside className="w-80 shrink-0 border-l border-slate-800 bg-slate-900/95 overflow-y-auto text-sm">
            <div className="p-3 space-y-3">
              {/* Header */}
              <div className="flex justify-between items-start gap-2">
                <h3 className="font-bold text-emerald-400 text-base break-all leading-tight">
                  {selected.name}
                </h3>
                <button
                  onClick={() => { setSelected(null); setNeighbors([]); }}
                  className="text-xs opacity-50 hover:opacity-100 shrink-0 bg-slate-800 rounded px-2 py-0.5"
                >
                  close
                </button>
              </div>

              <div className="text-xs opacity-60">
                {neighbors.length} connections
              </div>

              {/* Connection list */}
              <div>
                <div className="text-[10px] font-semibold opacity-50 mb-1.5 uppercase tracking-wider">
                  Connections
                </div>
                <div className="space-y-0.5 max-h-[60vh] overflow-y-auto">
                  {neighbors.map((n, i) => (
                    <button
                      key={i}
                      onClick={() => navigateToEntity(n.target)}
                      className="w-full text-left flex items-center gap-1 text-xs bg-slate-800/60 hover:bg-slate-700 rounded px-2 py-1 transition-colors group"
                    >
                      <span className="opacity-30 shrink-0">{"\u2192"}</span>
                      <span
                        className="shrink-0 px-1 rounded text-[10px]"
                        style={{
                          color: relationColor(n.relation),
                          backgroundColor: relationColor(n.relation) + "20",
                        }}
                      >
                        {n.relation}
                      </span>
                      <span className="text-emerald-300 truncate group-hover:underline">
                        {n.target}
                      </span>
                      {n.note && (
                        <span className="opacity-20 text-[9px] ml-auto shrink-0">
                          {n.note}
                        </span>
                      )}
                    </button>
                  ))}
                  {neighbors.length === 0 && (
                    <div className="text-xs opacity-40 py-2">(no connections)</div>
                  )}
                </div>
              </div>

              {/* Legend */}
              <div className="border-t border-slate-700 pt-2">
                <div className="text-[10px] font-semibold opacity-50 mb-1 uppercase tracking-wider">
                  Relation types
                </div>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(RELATION_COLORS).map(([rel, color]) => (
                    <span
                      key={rel}
                      className="text-[10px] px-1.5 py-0.5 rounded"
                      style={{ color, backgroundColor: color + "20" }}
                    >
                      {rel}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
