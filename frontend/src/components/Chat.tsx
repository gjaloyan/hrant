import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import {
  AgentAnswer, chatStream, StreamEvent, addFromChat, addCorrection, fetchCurrentSession,
  fetchActiveModel, setActiveModel, clearActiveModel,
  type ActiveModelSelection, type AvailableModel,
} from "../api";

type Msg =
  | { role: "user"; text: string }
  | { role: "agent"; text: string; meta?: AgentAnswer; progress?: string[] };

export type ChatHandle = {
  clearMessages: () => void;
};

function ConfidenceBadge({ c }: { c: number }) {
  const color =
    c >= 90 ? "bg-emerald-600" : c >= 70 ? "bg-amber-600" : "bg-rose-600";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs ${color}`}>
      {c}%
    </span>
  );
}

const Chat = forwardRef<ChatHandle, {
  onRefreshStatus: () => void;
  project: string | null;
  onNewSession?: () => void;
}>(function Chat({ onRefreshStatus, project, onNewSession }, ref) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [correcting, setCorrecting] = useState<number | null>(null);
  const [correctionText, setCorrectionText] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [activeModel, setActiveModelState] = useState<ActiveModelSelection | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useImperativeHandle(ref, () => ({
    clearMessages: () => setMsgs([]),
  }));

  const scroll = () =>
    setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 50);

  const loadModels = async () => {
    try {
      const data = await fetchActiveModel();
      setAvailableModels(data.models || []);
      if (data.active && data.active.provider_id) {
        setActiveModelState(data.active as ActiveModelSelection);
      } else {
        setActiveModelState(null);
      }
    } catch { /* ignore */ }
  };

  // Load current session's turns on mount (survives server restart)
  useEffect(() => {
    if (loaded) return;
    loadModels();
    (async () => {
      try {
        const data = await fetchCurrentSession();
        if (data.session && data.session.turns && data.session.turns.length > 0) {
          const restored: Msg[] = [];
          for (const t of data.session.turns) {
            restored.push({ role: "user", text: t.user });
            restored.push({
              role: "agent",
              text: t.answer,
              meta: {
                answer: t.answer,
                verification: {
                  confidence: t.confidence ?? 0,
                  verified_claims: [],
                  unverified_claims: [],
                  contradictions: [],
                  notes_used: [],
                },
                learned_topics: [],
                used_topics: t.topics ?? [],
                is_chat: t.is_chat,
              },
            });
          }
          setMsgs(restored);
          setTimeout(() => endRef.current?.scrollIntoView({ behavior: "auto" }), 100);
        }
      } catch { /* ignore — first load with empty session */ }
      setLoaded(true);
    })();
  }, [loaded]);

  const send = async () => {
    if (!input.trim() || busy) return;
    const text = input.trim();
    setInput("");
    setMsgs((m) => [...m, { role: "user", text }]);
    setBusy(true);

    const progress: string[] = [];
    setMsgs((m) => [...m, { role: "agent", text: "", progress }]);
    scroll();

    await chatStream(text, project, (ev: StreamEvent) => {
      if (ev.type === "progress") {
        progress.push(`${ev.event}: ${ev.message}`);
        setMsgs((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          if (last.role === "agent") last.progress = [...progress];
          return copy;
        });
        scroll();
      } else if (ev.type === "answer") {
        setMsgs((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          if (last.role === "agent") {
            last.text = ev.data.answer;
            last.meta = ev.data;
          }
          return copy;
        });
        scroll();
      } else if (ev.type === "error") {
        setMsgs((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          if (last.role === "agent") last.text = "Error: " + ev.message;
          return copy;
        });
      }
    });

    setBusy(false);
    onRefreshStatus();
  };

  const handleAddToFinetune = async (msgIdx: number) => {
    const agentMsg = msgs[msgIdx];
    if (agentMsg.role !== "agent" || !agentMsg.meta) return;
    let userText = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        userText = msgs[i].text;
        break;
      }
    }
    if (!userText) return;
    try {
      await addFromChat({
        question: userText,
        answer: agentMsg.meta.answer,
        used_topics: agentMsg.meta.used_topics,
        confidence: agentMsg.meta.verification.confidence,
        project,
      });
      alert("Added to finetune queue");
    } catch (e: any) {
      alert("Error: " + e.message);
    }
  };

  const handleCorrection = async (msgIdx: number) => {
    const agentMsg = msgs[msgIdx];
    if (agentMsg.role !== "agent" || !correctionText.trim()) return;
    let userText = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        userText = msgs[i].text;
        break;
      }
    }
    if (!userText) return;
    try {
      await addCorrection({
        question: userText,
        wrong_answer: agentMsg.text,
        corrected_answer: correctionText.trim(),
        project,
      });
      setCorrecting(null);
      setCorrectionText("");
      alert("Correction saved");
    } catch (e: any) {
      alert("Error: " + e.message);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-w-0">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {msgs.length === 0 && (
          <div className="opacity-50 text-sm text-center mt-8">
            Ask a question. The agent will analyze the task, load knowledge,
            answer, and verify itself.
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : ""}>
            <div
              className={`inline-block max-w-[85%] rounded-lg p-3 text-sm whitespace-pre-wrap ${
                m.role === "user" ? "bg-sky-800" : "bg-slate-800"
              }`}
            >
              {/* Progress indicators while loading — compact, no tool details */}
              {m.role === "agent" && m.progress && m.progress.length > 0 && !m.text && (
                <div className="text-xs opacity-60 space-y-0.5">
                  {m.progress
                    .filter((p) => !p.startsWith("tool:") && !p.startsWith("tool_error:"))
                    .slice(-6)
                    .map((p, j) => (
                      <div key={j}>· {p}</div>
                    ))}
                </div>
              )}
              {/* Compact thinking trace after answer — show key stages */}
              {m.role === "agent" && m.meta?.thinking_trace && m.meta.thinking_trace.length > 0 && m.text && (
                <details className="mb-2">
                  <summary className="text-[10px] opacity-40 cursor-pointer hover:opacity-60">
                    thinking: {m.meta.thinking_trace.filter(s => !s.event.startsWith("tool")).length} steps
                    {m.meta.thinking_trace.length > 0 && ` · ${m.meta.thinking_trace[m.meta.thinking_trace.length - 1].ts.toFixed(1)}s`}
                  </summary>
                  <div className="text-[10px] opacity-50 mt-1 space-y-0.5 max-h-24 overflow-y-auto">
                    {m.meta.thinking_trace
                      .filter(s => !s.event.startsWith("tool"))
                      .map((s, j) => (
                        <div key={j}>
                          <span className="text-slate-500">{s.ts.toFixed(1)}s</span>{" "}
                          <span className="text-slate-400">{s.event}:</span> {s.message}
                        </div>
                      ))}
                  </div>
                </details>
              )}
              {m.text}

              {/* Verification footer */}
              {m.role === "agent" && m.meta && !m.meta.is_chat && (
                <div className="mt-2 pt-2 border-t border-slate-700 text-xs opacity-80 space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <ConfidenceBadge c={m.meta.verification.confidence} />
                    {m.meta.used_topics.length > 0 && (
                      <span>topics: {m.meta.used_topics.join(", ")}</span>
                    )}
                    {m.meta.learned_topics.length > 0 && (
                      <span>new: {m.meta.learned_topics.join(", ")}</span>
                    )}
                  </div>
                  {m.meta.verification.unverified_claims.length > 0 && (
                    <details>
                      <summary className="cursor-pointer">
                        unverified ({m.meta.verification.unverified_claims.length})
                      </summary>
                      <ul className="mt-1 pl-4 list-disc">
                        {m.meta.verification.unverified_claims.map((c, j) => (
                          <li key={j}>{c}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                  {m.meta.verification.contradictions.length > 0 && (
                    <details>
                      <summary className="cursor-pointer text-rose-400">
                        contradictions ({m.meta.verification.contradictions.length})
                      </summary>
                      <ul className="mt-1 pl-4 list-disc">
                        {m.meta.verification.contradictions.map((c, j) => (
                          <li key={j}>{c}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              )}

              {/* Token usage */}
              {m.role === "agent" && m.meta?.token_usage && m.meta.token_usage.total_tokens > 0 && (
                <div className="mt-1 text-[10px] opacity-50 flex gap-3 flex-wrap">
                  <span>tokens: {m.meta.token_usage.input_tokens.toLocaleString()} in / {m.meta.token_usage.output_tokens.toLocaleString()} out</span>
                  {m.meta.token_usage.cache_read_tokens > 0 && (
                    <span>cache: {m.meta.token_usage.cache_read_tokens.toLocaleString()} read</span>
                  )}
                  <span>${m.meta.token_usage.cost_usd.toFixed(4)}</span>
                  <span>{m.meta.token_usage.llm_calls} calls</span>
                </div>
              )}

              {/* Action buttons for agent messages */}
              {m.role === "agent" && m.meta && m.text && (
                <div className="mt-2 pt-2 border-t border-slate-700 flex gap-2 text-xs">
                  <button
                    onClick={() => handleAddToFinetune(i)}
                    className="bg-amber-800 hover:bg-amber-700 rounded px-2 py-0.5"
                    title="Add this Q&A to finetune queue"
                  >
                    + finetune
                  </button>
                  <button
                    onClick={() => {
                      setCorrecting(correcting === i ? null : i);
                      setCorrectionText("");
                    }}
                    className="bg-rose-800 hover:bg-rose-700 rounded px-2 py-0.5"
                    title="Submit a correction for this answer"
                  >
                    correction
                  </button>
                </div>
              )}

              {/* Correction input */}
              {correcting === i && (
                <div className="mt-2 space-y-1">
                  <textarea
                    className="w-full bg-slate-950 rounded p-2 text-xs"
                    rows={3}
                    placeholder="Enter the correct answer..."
                    value={correctionText}
                    onChange={(e) => setCorrectionText(e.target.value)}
                  />
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleCorrection(i)}
                      className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-0.5 text-xs"
                    >
                      save correction
                    </button>
                    <button
                      onClick={() => setCorrecting(null)}
                      className="bg-slate-700 rounded px-2 py-0.5 text-xs"
                    >
                      cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="border-t border-slate-800 p-3 space-y-2">
        {/* Model selector */}
        <div className="flex items-center gap-2 relative">
          <button
            onClick={() => { setShowModelPicker(!showModelPicker); if (!showModelPicker) loadModels(); }}
            className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 rounded px-2.5 py-1 text-xs transition-colors"
            title="Select model"
          >
            <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            <span className="text-slate-300">
              {activeModel ? `${activeModel.provider_name} / ${activeModel.model}` : "Default model"}
            </span>
            <svg className="w-3 h-3 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showModelPicker && (
            <div className="absolute bottom-full left-0 mb-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 min-w-[300px] max-h-[400px] overflow-y-auto">
              <div className="p-2 border-b border-slate-700">
                <button
                  onClick={async () => {
                    await clearActiveModel();
                    setActiveModelState(null);
                    setShowModelPicker(false);
                  }}
                  className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors ${
                    !activeModel ? "bg-sky-700 text-white" : "hover:bg-slate-700 text-slate-300"
                  }`}
                >
                  Default (from config)
                </button>
              </div>
              <div className="p-2 space-y-1">
                {(() => {
                  const grouped: Record<string, AvailableModel[]> = {};
                  for (const m of availableModels) {
                    const key = m.provider_name;
                    if (!grouped[key]) grouped[key] = [];
                    grouped[key].push(m);
                  }
                  return Object.entries(grouped).map(([provName, models]) => (
                    <div key={provName}>
                      <div className="text-[10px] text-slate-500 uppercase tracking-wider px-3 py-1">{provName}</div>
                      {models.map((m) => {
                        const isActive = activeModel?.provider_id === m.provider_id && activeModel?.model === m.model;
                        return (
                          <button
                            key={`${m.provider_id}:${m.model}`}
                            onClick={async () => {
                              try {
                                const res = await setActiveModel(m.provider_id, m.model);
                                setActiveModelState(res.active);
                              } catch { /* ignore */ }
                              setShowModelPicker(false);
                            }}
                            className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors ${
                              isActive ? "bg-sky-700 text-white" : "hover:bg-slate-700 text-slate-300"
                            }`}
                          >
                            {m.model}
                          </button>
                        );
                      })}
                    </div>
                  ));
                })()}
                {availableModels.length === 0 && (
                  <div className="text-xs text-slate-500 px-3 py-2">No providers configured</div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="flex gap-2">
          <textarea
            className="flex-1 bg-slate-900 rounded p-2 text-sm resize-none outline-none focus:ring-1 focus:ring-sky-600"
            rows={2}
            placeholder="Ask something..."
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <div className="flex flex-col gap-1">
            <button
              onClick={send}
              disabled={busy}
              className="bg-sky-700 hover:bg-sky-600 rounded px-4 py-1 disabled:opacity-50 transition-colors text-sm"
            >
              {busy ? "..." : "Send"}
            </button>
            {onNewSession && (
              <button
                onClick={onNewSession}
                disabled={busy}
                className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs disabled:opacity-50"
                title="Start a new session"
              >
                New
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

export default Chat;
