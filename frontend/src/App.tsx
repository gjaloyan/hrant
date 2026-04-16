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
import UsagePage from "./components/UsagePage";
import SettingsPanel from "./components/SettingsPanel";
import StatusBar from "./components/StatusBar";
import { fetchStatus, newSession, StatusPayload } from "./api";

type Tab = "chat" | "goals" | "knowledge" | "graph" | "sessions" | "intelligence" | "usage" | "projects" | "finetune" | "settings";

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "💬" },
  { id: "goals", label: "Goals", icon: "🎯" },
  { id: "sessions", label: "Sessions", icon: "📊" },
  { id: "knowledge", label: "Knowledge", icon: "📚" },
  { id: "graph", label: "Graph", icon: "🔗" },
  { id: "intelligence", label: "Intelligence", icon: "🧠" },
  { id: "usage", label: "Usage", icon: "📈" },
  { id: "projects", label: "Projects", icon: "📁" },
  { id: "finetune", label: "Fine-Tune", icon: "🎓" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];

export default function App() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("chat");
  const chatRef = useRef<ChatHandle>(null);

  const refresh = async () => {
    try {
      setStatus(await fetchStatus());
    } catch { /* ignore */ }
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
      <header className="px-4 py-2 border-b border-slate-800 bg-slate-900/80 flex items-center gap-4">
        <h1 className="font-bold text-lg">Self-Learning Agent</h1>
        <nav className="flex gap-1 text-sm">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-3 py-1.5 rounded transition-colors ${
                tab === t.id
                  ? "bg-sky-700 text-white"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              {t.icon} {t.label}
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
        {tab === "usage" && <UsagePage />}
        {tab === "projects" && <ProjectsPanel onRefresh={refresh} />}
        {tab === "finetune" && <FinetunePanel />}
        {tab === "settings" && <SettingsPanel />}
      </div>

      <StatusBar status={status} />
      <NoteViewer topic={selectedTopic} onClose={() => setSelectedTopic(null)} />
    </div>
  );
}
