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
import NavRail, { labelOf, type Tab } from "./components/NavRail";
import { Button } from "./ui";
import { SpeakerProvider } from "./ui/speakers";
import {
  fetchStatus,
  newSession,
  fetchAutonomicStatus,
  fetchPending,
  fetchEmbeddingsStatus,
  StatusPayload,
  AutonomicStatus,
  EmbeddingsStatusResponse,
} from "./api";

export default function App() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [autonomic, setAutonomic] = useState<AutonomicStatus | null>(null);
  const [pendingCount, setPendingCount] = useState<number>(0);
  const [embeddings, setEmbeddings] =
    useState<EmbeddingsStatusResponse | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("chat");
  const chatRef = useRef<ChatHandle>(null);

  const refresh = async () => {
    const [statusResult, autonomicResult, pendingResult, embeddingsResult] =
      await Promise.allSettled([
        fetchStatus(),
        fetchAutonomicStatus(),
        fetchPending(),
        fetchEmbeddingsStatus(),
      ]);
    if (statusResult.status === "fulfilled") setStatus(statusResult.value);
    if (autonomicResult.status === "fulfilled")
      setAutonomic(autonomicResult.value);
    if (pendingResult.status === "fulfilled")
      setPendingCount(pendingResult.value.pending.length);
    if (embeddingsResult.status === "fulfilled")
      setEmbeddings(embeddingsResult.value);
  };

  const handleNewSession = useCallback(async () => {
    try {
      await newSession();
      chatRef.current?.clearMessages();
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 10000);
    return () => clearInterval(t);
  }, []);

  return (
    <SpeakerProvider>
    <div className="flex h-screen flex-col">
      <header className="sticky top-0 z-30 border-b border-edge bg-surface/80 backdrop-blur">
        <div className="flex items-center justify-between gap-3 px-3 py-2 sm:px-4">
          <div className="flex min-w-0 items-baseline gap-2">
            <h1 className="truncate text-[15px] font-semibold">Hrant</h1>
            {/* The current screen used to be shown only on mobile, so on a
                wide window nothing named the page you were on. */}
            <span className="truncate text-xs text-ink-dim">
              {labelOf(tab)}
            </span>
          </div>
          {/* "New chat" was buried beside the composer. It is one of the two
              things anyone does most, so it belongs where it can be found. */}
          <Button
            kind="ghost"
            size="sm"
            onClick={() => {
              setTab("chat");
              handleNewSession();
            }}
            title="Start a fresh conversation"
          >
            ＋ New chat
          </Button>
        </div>
        <div className="md:hidden">
          <NavRail tab={tab} setTab={setTab} badges={{ settings: pendingCount }} />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="hidden md:flex">
          <NavRail tab={tab} setTab={setTab} badges={{ settings: pendingCount }} />
        </div>

        {/* Chat stays mounted but hidden — switching tabs must not throw
            away a conversation in progress. */}
        <div
          className={`flex min-w-0 flex-1 ${tab !== "chat" ? "hidden" : ""}`}
        >
          <Chat
            ref={chatRef}
            onRefreshStatus={refresh}
            project={status?.current_project ?? null}
          />
        </div>
        {tab === "goals" && <GoalsPanel />}
        {tab === "sessions" && <SessionsPanel />}
        {tab === "knowledge" && (
          <KnowledgePanel onSelectTopic={setSelectedTopic} />
        )}
        {tab === "graph" && <GraphViewer />}
        {tab === "intelligence" && <IntelligencePanel />}
        {tab === "autonomic" && <AutonomicPanel />}
        {tab === "usage" && <UsagePage />}
        {tab === "projects" && <ProjectsPanel onRefresh={refresh} />}
        {tab === "finetune" && <FinetunePanel />}
        {tab === "settings" && <SettingsPanel />}
      </div>

      <StatusBar
        status={status}
        autonomic={autonomic}
        pendingCount={pendingCount}
        embeddings={embeddings}
      />
      <NoteViewer
        topic={selectedTopic}
        onClose={() => setSelectedTopic(null)}
      />
    </div>
    </SpeakerProvider>
  );
}
