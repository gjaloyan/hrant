import { useCallback, useEffect, useState } from "react";
import {
  fetchGoals,
  addGoal,
  completeGoal,
  pauseGoal,
  resumeGoal,
  failGoal,
  deleteGoal,
  updateGoalPriority,
  fetchBgStatus,
  bgLearn,
  bgCancel,
  bgProcessGoals,
  GoalData,
  GoalStats,
  BgStatus,
} from "../api";

const TYPE_COLORS: Record<string, string> = {
  user: "#38bdf8",
  learning: "#34d399",
  improvement: "#fbbf24",
  proactive: "#a78bfa",
};

const STATUS_COLORS: Record<string, string> = {
  active: "#34d399",
  paused: "#fbbf24",
  completed: "#64748b",
  failed: "#f87171",
};

function PriorityBar({ priority }: { priority: number }) {
  return (
    <div className="flex gap-0.5 items-center" title={`Priority: ${priority}/10`}>
      {Array.from({ length: 10 }, (_, i) => (
        <div
          key={i}
          className="w-1.5 h-3 rounded-sm"
          style={{
            backgroundColor:
              i < priority
                ? priority >= 8
                  ? "#f87171"
                  : priority >= 5
                  ? "#fbbf24"
                  : "#64748b"
                : "rgba(100,116,139,0.2)",
          }}
        />
      ))}
    </div>
  );
}

export default function GoalsPanel() {
  const [goals, setGoals] = useState<GoalData[]>([]);
  const [stats, setStats] = useState<GoalStats | null>(null);
  const [bgStatus, setBgStatus] = useState<BgStatus | null>(null);
  const [selected, setSelected] = useState<GoalData | null>(null);
  const [msg, setMsg] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [formDesc, setFormDesc] = useState("");
  const [formPriority, setFormPriority] = useState(5);
  const [formType, setFormType] = useState("user");
  const [formContext, setFormContext] = useState("");
  const [learnTopic, setLearnTopic] = useState("");
  const [filter, setFilter] = useState<"all" | "active" | "completed">("all");

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 4000);
  };

  const load = useCallback(async () => {
    try {
      const [goalsData, bg] = await Promise.all([fetchGoals(), fetchBgStatus()]);
      setGoals(goalsData.goals);
      setStats(goalsData.stats);
      setBgStatus(bg);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  const handleAdd = async () => {
    if (!formDesc.trim()) return;
    try {
      await addGoal({
        description: formDesc.trim(),
        priority: formPriority,
        goal_type: formType,
        context: formContext.trim(),
      });
      setFormDesc("");
      setFormContext("");
      setShowForm(false);
      load();
      flash("Goal added");
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleAction = async (action: string, id: string) => {
    try {
      if (action === "complete") await completeGoal(id);
      else if (action === "pause") await pauseGoal(id);
      else if (action === "resume") await resumeGoal(id);
      else if (action === "fail") await failGoal(id);
      else if (action === "delete") {
        if (!confirm("Delete this goal?")) return;
        await deleteGoal(id);
        if (selected?.id === id) setSelected(null);
      }
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handlePriority = async (id: string, p: number) => {
    try {
      await updateGoalPriority(id, p);
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleBgLearn = async () => {
    if (!learnTopic.trim()) return;
    try {
      await bgLearn(learnTopic.trim());
      setLearnTopic("");
      flash("Background learning started");
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleProcessGoals = async () => {
    try {
      const res = await bgProcessGoals();
      flash(res.started > 0 ? "Processing proactive goal..." : "No goals to process");
      load();
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const filtered = goals.filter((g) => {
    if (filter === "active") return g.status === "active" || g.status === "paused";
    if (filter === "completed") return g.status === "completed" || g.status === "failed";
    return true;
  });

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      {/* Left: goal list */}
      <div className="w-96 shrink-0 border-r border-slate-800 bg-slate-950/60 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-3 border-b border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="font-bold text-sm">Goals</h2>
            <button
              onClick={() => setShowForm(!showForm)}
              className="bg-sky-800 hover:bg-sky-700 rounded px-2 py-0.5 text-xs"
            >
              + Add
            </button>
          </div>

          {/* Stats row */}
          {stats && (
            <div className="flex gap-2 text-xs flex-wrap">
              <span className="text-emerald-400">{stats.active} active</span>
              <span className="text-amber-400">{stats.paused} paused</span>
              <span className="text-slate-400">{stats.completed} done</span>
              <span className="text-rose-400">{stats.failed} failed</span>
              <span className="opacity-40">|</span>
              <span className="opacity-50">{stats.interaction_count} interactions</span>
              <span className="opacity-50">next check in {stats.next_proactive_check_in}</span>
            </div>
          )}

          {/* Filter tabs */}
          <div className="flex gap-1">
            {(["all", "active", "completed"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-2 py-0.5 rounded text-xs ${
                  filter === f ? "bg-sky-700" : "bg-slate-800 hover:bg-slate-700"
                }`}
              >
                {f}
              </button>
            ))}
          </div>

          {/* Add form */}
          {showForm && (
            <div className="bg-slate-800 rounded p-2 space-y-2">
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none"
                placeholder="Goal description..."
                value={formDesc}
                onChange={(e) => setFormDesc(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
              />
              <div className="flex gap-2 items-center">
                <label className="text-[10px] opacity-50">Priority:</label>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={formPriority}
                  onChange={(e) => setFormPriority(Number(e.target.value))}
                  className="flex-1"
                />
                <span className="text-xs w-4 text-center">{formPriority}</span>
              </div>
              <div className="flex gap-2 items-center">
                <label className="text-[10px] opacity-50">Type:</label>
                <select
                  value={formType}
                  onChange={(e) => setFormType(e.target.value)}
                  className="bg-slate-900 rounded px-2 py-0.5 text-xs"
                >
                  <option value="user">User</option>
                  <option value="learning">Learning</option>
                  <option value="improvement">Improvement</option>
                </select>
              </div>
              <input
                className="w-full bg-slate-900 rounded px-2 py-1 text-xs outline-none"
                placeholder="Context (why this goal?)..."
                value={formContext}
                onChange={(e) => setFormContext(e.target.value)}
              />
              <div className="flex gap-1">
                <button
                  onClick={handleAdd}
                  className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-0.5 text-xs"
                >
                  Create
                </button>
                <button
                  onClick={() => setShowForm(false)}
                  className="bg-slate-700 rounded px-2 py-0.5 text-xs"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Goal list */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {filtered.length === 0 && (
            <div className="text-xs opacity-40 p-2">No goals yet</div>
          )}
          {filtered.map((g) => (
            <button
              key={g.id}
              onClick={() => setSelected(g)}
              className={`w-full text-left rounded p-2 transition-colors text-xs ${
                selected?.id === g.id
                  ? "bg-sky-800"
                  : g.status === "active"
                  ? "bg-slate-800/60 hover:bg-slate-700"
                  : "bg-slate-800/30 hover:bg-slate-800/50 opacity-70"
              }`}
            >
              <div className="flex justify-between items-start gap-1">
                <span className="font-medium">{g.description}</span>
                <div className="flex gap-1 shrink-0 items-center">
                  <span
                    className="text-[9px] rounded px-1"
                    style={{
                      color: TYPE_COLORS[g.goal_type] || "#94a3b8",
                      backgroundColor: (TYPE_COLORS[g.goal_type] || "#94a3b8") + "20",
                    }}
                  >
                    {g.goal_type}
                  </span>
                  <span
                    className="text-[9px] rounded px-1"
                    style={{
                      color: STATUS_COLORS[g.status] || "#94a3b8",
                      backgroundColor: (STATUS_COLORS[g.status] || "#94a3b8") + "20",
                    }}
                  >
                    {g.status}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <PriorityBar priority={g.priority} />
                <span className="opacity-40 text-[10px]">{g.created.slice(0, 16)}</span>
                {g.subtasks.length > 0 && (
                  <span className="opacity-40 text-[10px]">
                    {g.subtasks.filter((s) => s.status === "done").length}/{g.subtasks.length} subtasks
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>

        {/* Background task status */}
        <div className="border-t border-slate-800 p-3 space-y-2">
          <div className="text-[10px] font-semibold opacity-50 uppercase tracking-wider">
            Background Tasks
          </div>
          {bgStatus?.busy && bgStatus.current ? (
            <div className="bg-emerald-900/30 rounded p-2 text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-emerald-400">{bgStatus.current.description}</span>
                <button
                  onClick={async () => { await bgCancel(); load(); }}
                  className="text-rose-400 hover:text-rose-300 text-[10px]"
                >
                  cancel
                </button>
              </div>
              <div className="opacity-50 text-[10px]">
                started: {bgStatus.current.started}
              </div>
            </div>
          ) : (
            <div className="text-xs opacity-40">Idle</div>
          )}
          <div className="flex gap-1">
            <input
              className="flex-1 bg-slate-900 rounded px-2 py-0.5 text-xs outline-none"
              placeholder="Learn topic..."
              value={learnTopic}
              onChange={(e) => setLearnTopic(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleBgLearn()}
            />
            <button
              onClick={handleBgLearn}
              disabled={bgStatus?.busy}
              className="bg-violet-800 hover:bg-violet-700 rounded px-2 py-0.5 text-xs disabled:opacity-50"
            >
              Learn
            </button>
          </div>
          <button
            onClick={handleProcessGoals}
            disabled={bgStatus?.busy}
            className="w-full bg-slate-800 hover:bg-slate-700 rounded px-2 py-0.5 text-xs disabled:opacity-50"
          >
            Process proactive goals
          </button>
        </div>

        {msg && (
          <div className="p-2 text-xs text-sky-400 border-t border-slate-800">{msg}</div>
        )}
      </div>

      {/* Right: goal detail */}
      <div className="flex-1 overflow-y-auto p-4">
        {!selected ? (
          <div className="flex items-center justify-center h-full opacity-40 text-sm">
            Select a goal to view details
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-between items-start gap-4">
              <div>
                <h3 className="font-bold text-lg">{selected.description}</h3>
                <div className="flex gap-3 text-xs mt-1 flex-wrap">
                  <span
                    className="rounded px-2 py-0.5"
                    style={{
                      color: TYPE_COLORS[selected.goal_type] || "#94a3b8",
                      backgroundColor: (TYPE_COLORS[selected.goal_type] || "#94a3b8") + "20",
                    }}
                  >
                    {selected.goal_type}
                  </span>
                  <span
                    className="rounded px-2 py-0.5"
                    style={{
                      color: STATUS_COLORS[selected.status] || "#94a3b8",
                      backgroundColor: (STATUS_COLORS[selected.status] || "#94a3b8") + "20",
                    }}
                  >
                    {selected.status}
                  </span>
                  <span className="opacity-50">Created: {selected.created}</span>
                  {selected.completed && (
                    <span className="opacity-50">Completed: {selected.completed}</span>
                  )}
                  {selected.source && (
                    <span className="opacity-50">Source: {selected.source}</span>
                  )}
                </div>
              </div>
              {/* Action buttons */}
              <div className="flex gap-1 shrink-0">
                {selected.status === "active" && (
                  <>
                    <button
                      onClick={() => handleAction("complete", selected.id)}
                      className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-1 text-xs"
                    >
                      Complete
                    </button>
                    <button
                      onClick={() => handleAction("pause", selected.id)}
                      className="bg-amber-700 hover:bg-amber-600 rounded px-2 py-1 text-xs"
                    >
                      Pause
                    </button>
                    <button
                      onClick={() => handleAction("fail", selected.id)}
                      className="bg-rose-700 hover:bg-rose-600 rounded px-2 py-1 text-xs"
                    >
                      Fail
                    </button>
                  </>
                )}
                {selected.status === "paused" && (
                  <button
                    onClick={() => handleAction("resume", selected.id)}
                    className="bg-sky-700 hover:bg-sky-600 rounded px-2 py-1 text-xs"
                  >
                    Resume
                  </button>
                )}
                <button
                  onClick={() => handleAction("delete", selected.id)}
                  className="bg-rose-800 hover:bg-rose-700 rounded px-2 py-1 text-xs"
                >
                  Delete
                </button>
              </div>
            </div>

            {/* Priority slider */}
            <div className="bg-slate-800 rounded p-3">
              <div className="flex items-center gap-3">
                <span className="text-xs opacity-50">Priority:</span>
                <PriorityBar priority={selected.priority} />
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={selected.priority}
                  onChange={(e) => handlePriority(selected.id, Number(e.target.value))}
                  className="flex-1"
                  disabled={selected.status !== "active" && selected.status !== "paused"}
                />
                <span className="text-sm font-bold w-6 text-center">{selected.priority}</span>
              </div>
            </div>

            {/* Context */}
            {selected.context && (
              <div className="bg-slate-800 rounded p-3">
                <div className="text-[10px] font-semibold opacity-50 uppercase tracking-wider mb-1">
                  Context
                </div>
                <div className="text-sm whitespace-pre-wrap">{selected.context}</div>
              </div>
            )}

            {/* Subtasks */}
            {selected.subtasks.length > 0 && (
              <div className="bg-slate-800 rounded p-3">
                <div className="text-[10px] font-semibold opacity-50 uppercase tracking-wider mb-2">
                  Subtasks ({selected.subtasks.filter((s) => s.status === "done").length}/{selected.subtasks.length})
                </div>
                <div className="space-y-1">
                  {selected.subtasks.map((st, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <span className={st.status === "done" ? "text-emerald-400" : "opacity-40"}>
                        {st.status === "done" ? "[x]" : "[ ]"}
                      </span>
                      <div>
                        <div className={st.status === "done" ? "line-through opacity-60" : ""}>
                          {st.description}
                        </div>
                        {st.result && (
                          <div className="text-xs text-emerald-400 mt-0.5">{st.result}</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Progress notes */}
            {selected.progress_notes.length > 0 && (
              <div className="bg-slate-800 rounded p-3">
                <div className="text-[10px] font-semibold opacity-50 uppercase tracking-wider mb-2">
                  Progress Log
                </div>
                <div className="space-y-0.5 text-xs">
                  {selected.progress_notes.map((note, i) => (
                    <div key={i} className="opacity-70">{note}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
