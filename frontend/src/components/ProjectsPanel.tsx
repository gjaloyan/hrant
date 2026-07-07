import { useEffect, useState } from "react";
import {
  addProjectContext,
  addProjectDecision,
  addProjectIssue,
  createProject,
  endProject,
  fetchProjectDetail,
  fetchProjects,
} from "../api";
import TrackerBoard from "./TrackerBoard";
import TrackerCalendar from "./TrackerCalendar";
import FramesView from "./FramesView";

export default function ProjectsPanel({ onRefresh }: { onRefresh: () => void }) {
  const [current, setCurrent] = useState<string | null>(null);
  const [all, setAll] = useState<string[]>([]);
  const [overview, setOverview] = useState("");
  const [newName, setNewName] = useState("");
  const [contextText, setContextText] = useState("");
  const [decWhat, setDecWhat] = useState("");
  const [decWhy, setDecWhy] = useState("");
  const [issueProblem, setIssueProblem] = useState("");
  const [issueFix, setIssueFix] = useState("");
  const [msg, setMsg] = useState("");
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [view, setView] = useState<"journal" | "trackers" | "calendar" | "frames">(
    "trackers",
  );

  const refresh = async () => {
    try {
      const p = await fetchProjects();
      setCurrent(p.current);
      setAll(p.all);
      if (p.current) {
        loadOverview(p.current);
        setSelectedProject(p.current);
      }
    } catch (e: any) {
      setMsg("Error: " + e.message);
    }
  };

  const loadOverview = async (name: string) => {
    try {
      const d = await fetchProjectDetail(name);
      setOverview(d.overview || "(empty)");
    } catch {
      setOverview("(could not load)");
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 3000);
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    try {
      const res = await createProject(newName.trim());
      setNewName("");
      flash(res.message);
      refresh();
      onRefresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleEnd = async () => {
    if (!current) return;
    if (!confirm(`End project "${current}"?`)) return;
    try {
      const res = await endProject(current);
      flash(res.message);
      setOverview("");
      refresh();
      onRefresh();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleContext = async () => {
    if (!current || !contextText.trim()) return;
    try {
      const res = await addProjectContext(current, contextText.trim());
      setContextText("");
      flash(res.message);
      loadOverview(current);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleDecision = async () => {
    if (!current || !decWhat.trim() || !decWhy.trim()) return;
    try {
      const res = await addProjectDecision(current, decWhat.trim(), decWhy.trim());
      setDecWhat("");
      setDecWhy("");
      flash(res.message);
      loadOverview(current);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleIssue = async () => {
    if (!current || !issueProblem.trim() || !issueFix.trim()) return;
    try {
      const res = await addProjectIssue(current, issueProblem.trim(), issueFix.trim());
      setIssueProblem("");
      setIssueFix("");
      flash(res.message);
      loadOverview(current);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleSelectProject = (name: string) => {
    setSelectedProject(name);
    loadOverview(name);
  };

  return (
    <div className="flex flex-1 min-h-0">
      {/* Left: project list + actions */}
      <aside className="w-80 border-r border-slate-800 bg-slate-950/60 p-3 overflow-y-auto text-sm space-y-4">
        {/* Create project */}
        <section>
          <h2 className="font-bold mb-2">Create Project</h2>
          <div className="flex gap-1">
            <input
              className="flex-1 bg-slate-900 rounded px-2 py-1 text-xs outline-none"
              placeholder="project name..."
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            />
            <button
              onClick={handleCreate}
              className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 text-xs"
            >
              create
            </button>
          </div>
        </section>

        {/* Active project */}
        <section>
          <h2 className="font-bold mb-2">
            Active: {current ? <span className="text-emerald-400">{current}</span> : <span className="opacity-50">none</span>}
          </h2>
          {current && (
            <button
              onClick={handleEnd}
              className="bg-rose-700 hover:bg-rose-600 rounded px-3 py-1 text-xs w-full"
            >
              End project
            </button>
          )}
        </section>

        {/* Add context */}
        {current && (
          <section>
            <h2 className="font-bold mb-2">Add Context</h2>
            <textarea
              className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none resize-none"
              rows={2}
              placeholder="project context..."
              value={contextText}
              onChange={(e) => setContextText(e.target.value)}
            />
            <button
              onClick={handleContext}
              className="mt-1 bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs w-full"
            >
              Add context
            </button>
          </section>
        )}

        {/* Add decision */}
        {current && (
          <section>
            <h2 className="font-bold mb-2">Record Decision</h2>
            <input
              className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none mb-1"
              placeholder="what we decided..."
              value={decWhat}
              onChange={(e) => setDecWhat(e.target.value)}
            />
            <input
              className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none"
              placeholder="because..."
              value={decWhy}
              onChange={(e) => setDecWhy(e.target.value)}
            />
            <button
              onClick={handleDecision}
              className="mt-1 bg-violet-700 hover:bg-violet-600 rounded px-3 py-1 text-xs w-full"
            >
              Record decision
            </button>
          </section>
        )}

        {/* Add issue */}
        {current && (
          <section>
            <h2 className="font-bold mb-2">Record Issue</h2>
            <input
              className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none mb-1"
              placeholder="problem..."
              value={issueProblem}
              onChange={(e) => setIssueProblem(e.target.value)}
            />
            <input
              className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none"
              placeholder="fix / solution..."
              value={issueFix}
              onChange={(e) => setIssueFix(e.target.value)}
            />
            <button
              onClick={handleIssue}
              className="mt-1 bg-amber-700 hover:bg-amber-600 rounded px-3 py-1 text-xs w-full"
            >
              Record issue
            </button>
          </section>
        )}

        {/* All projects list */}
        <section>
          <h2 className="font-bold mb-2">All Projects ({all.length})</h2>
          {all.length === 0 && <div className="opacity-50 text-xs">(none)</div>}
          {all.map((name) => (
            <button
              key={name}
              onClick={() => handleSelectProject(name)}
              className={`w-full text-left px-2 py-1 rounded text-xs ${
                selectedProject === name
                  ? "bg-sky-700"
                  : "bg-slate-800 hover:bg-slate-700"
              } ${name === current ? "border-l-2 border-emerald-400" : ""} mb-1`}
            >
              {name}
              {name === current && (
                <span className="ml-1 text-emerald-400">(active)</span>
              )}
            </button>
          ))}
        </section>

        {msg && <div className="bg-sky-900/50 text-xs rounded p-2">{msg}</div>}
      </aside>

      {/* Right: view switch (Trackers board / Calendar / Journal) */}
      <div className="flex flex-1 min-w-0 flex-col">
        <div className="flex gap-1 border-b border-slate-800 px-3 py-2 text-xs">
          {(["trackers", "calendar", "frames", "journal"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded px-3 py-1 ${
                view === v ? "bg-sky-700 text-white" : "bg-slate-800"
              }`}
            >
              {v === "trackers"
                ? "Trackers"
                : v === "calendar"
                ? "Calendar"
                : v === "frames"
                ? "Frames"
                : "Journal"}
            </button>
          ))}
        </div>
        {view === "trackers" && <TrackerBoard />}
        {view === "calendar" && <TrackerCalendar />}
        {view === "frames" && <FramesView />}
        {view === "journal" && (
          <div className="flex-1 overflow-y-auto p-4">
            {selectedProject ? (
              <>
                <h2 className="text-lg font-bold mb-4">
                  Project: {selectedProject}
                  {selectedProject === current && (
                    <span className="ml-2 text-sm text-emerald-400">
                      (active)
                    </span>
                  )}
                </h2>
                <pre className="whitespace-pre-wrap text-sm bg-slate-900 rounded p-4 max-w-3xl">
                  {overview}
                </pre>
              </>
            ) : (
              <div className="opacity-50 text-sm text-center mt-8">
                Select a project or create a new one.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
