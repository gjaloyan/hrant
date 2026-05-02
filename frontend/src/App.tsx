import { useCallback, useEffect, useRef, useState } from "react";
import Chat, { type ChatHandle } from "./components/Chat";
import FinetunePanel from "./components/FinetunePanel";
import GoalsPanel from "./components/GoalsPanel";
import GraphViewer from "./components/GraphViewer";
import KnowledgePanel from "./components/KnowledgePanel";
import NoteViewer from "./components/NoteViewer";
import ProjectsPanel from "./components/ProjectsPanel";
import SessionsPanel from "./components/SessionsPanel";
import IntelligencePanel from "./components/IntelligencePanel";
import AutonomicPanel from "./components/AutonomicPanel";
import UsagePage from "./components/UsagePage";
import SettingsPanel from "./components/SettingsPanel";
import StatusBar from "./components/StatusBar";
import { fetchStatus, newSession, fetchAutonomicStatus, fetchPending, fetchEmbeddingsStatus, StatusPayload, AutonomicStatus, EmbeddingsStatusResponse } from "./api";

type Tab = "chat" | "goals" | "knowledge" | "graph" | "sessions" | "intelligence" | "autonomic" | "usage" | "projects" | "finetune" | "settings";

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "💬" },
  { id: "goals", label: "Goals", icon: "🎯" },
  { id: "sessions", label: "Sessions", icon: "📊" },
  { id: "knowledge", label: "Knowledge", icon: "📚" },
  { id: "graph", label: "Graph", icon: "🔗" },
  { id: "intelligence", label: "Intelligence", icon: "🧠" },
  { id: "autonomic", label: "Autonomic", icon: "🦾" },
  { id: "usage", label: "Usage", icon: "📈" },
  { id: "projects", label: "Projects", icon: "📁" },
  { id: "finetune", label: "Fine-Tune", icon: "🎓" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];

export default function App() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [autonomic, setAutonomic] = useState<AutonomicStatus | null>(null);
  const [pendingCount, setPendingCount] = useState<number>(0);
  const [embeddings, setEmbeddings] = useState<EmbeddingsStatusResponse | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("chat");
  const chatRef = useRef<ChatHandle>(null);

  const refresh = async () => {
    const [statusResult, autonomicResult, pendingResult, embeddingsResult] = await Promise.allSettled([
      fetchStatus(),
      fetchAutonomicStatus(),
      fetchPending(),
      fetchEmbeddingsStatus(),
    ]);
    if (statusResult.status === "fulfilled") setStatus(statusResult.value);
    if (autonomicResult.status === "fulfilled") setAutonomic(autonomicResult.value);
    if (pendingResult.status === "fulfilled") setPendingCount(pendingResult.value.pending.length);
    if (embeddingsResult.status === "fulfilled") setEmbeddings(embeddingsResult.value);
  };

  const handleNewSession = useCallback(async () => {
    try {
      await newSession();
      chatRef.current?.clearMessages();
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 10000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <header className="border-b border-slate-800 bg-slate-900/80 sticky top-0 z-30">
        {/* Title row */}
        <div className="px-3 sm:px-4 py-2 flex items-center justify-between gap-2">
          <h1 className="font-bold text-base sm:text-lg truncate">
            <span className="hidden sm:inline">Self-Learning </span>Agent
          </h1>
          <span className="text-[10px] sm:hidden text-slate-400 truncate">
            {TABS.find((t) => t.id === tab)?.label}
          </span>
        </div>
        {/* Tabs row — horizontally scrollable on mobile, icon-only under sm */}
        <nav
          className="flex gap-1 text-sm overflow-x-auto px-2 sm:px-4 pb-2 scroll-smooth
                     [scrollbar-width:none] [-ms-overflow-style:none]
                     [&::-webkit-scrollbar]:hidden"
        >
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              aria-label={t.label}
              title={t.label}
              className={`shrink-0 rounded transition-colors px-2.5 py-1.5 sm:px-3 ${
                tab === t.id
                  ? "bg-sky-700 text-white"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              <span>{t.icon}</span>
              <span className="hidden sm:inline ml-1">{t.label}</span>
            </button>
          ))}
        </nav>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Chat is always mounted but hidden when not active — preserves state across tab switches */}
        <div className={`flex flex-1 min-w-0 ${tab !== "chat" ? "hidden" : ""}`}>
          <Chat ref={chatRef} onRefreshStatus={refresh} project={status?.current_project ?? null} onNewSession={handleNewSession} />
        </div>
        {tab === "goals" && <GoalsPanel />}
        {tab === "sessions" && <SessionsPanel />}
        {tab === "knowledge" && <KnowledgePanel onSelectTopic={setSelectedTopic} />}
        {tab === "graph" && <GraphViewer />}
        {tab === "intelligence" && <IntelligencePanel />}
        {tab === "autonomic" && <AutonomicPanel />}
        {tab === "usage" && <UsagePage />}
        {tab === "projects" && <ProjectsPanel onRefresh={refresh} />}
        {tab === "finetune" && <FinetunePanel />}
        {tab === "settings" && <SettingsPanel />}
      </div>

      <StatusBar status={status} autonomic={autonomic} pendingCount={pendingCount} embeddings={embeddings} />
      <NoteViewer topic={selectedTopic} onClose={() => setSelectedTopic(null)} />
    </div>
  );
}
