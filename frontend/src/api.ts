// ---- Types ----

export type VerificationResult = {
  confidence: number;
  verified_claims: string[];
  unverified_claims: string[];
  contradictions: string[];
  notes_used: string[];
};

export type TokenUsage = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  cost_usd: number;
  llm_calls: number;
};

export type ToolCallDetail = {
  name: string;
  args: Record<string, unknown>;
  result: string;             // truncated preview (first ~4000 chars)
  result_truncated?: boolean; // true if backend truncated the body
  result_full_len?: number;   // original body length in chars
  is_error: boolean;
  duration_ms: number;
};

export type ThinkingStep = {
  ts: number;
  event: string;
  message: string;
  tokens_so_far: number;
  tool_call?: ToolCallDetail | null;
};

export type LLMCallDetail = {
  label: string;
  task_type: string;
  model: string;
  system_redacted: string;
  user_redacted: string;
  response_preview: string;
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
};

export type Claim = {
  id: string;
  text: string;
  status: "verified" | "unverified" | "contradicted";
  evidence_ids: string[];
  risk: "low" | "medium" | "high";
};

export type EvidenceItem = {
  id: string;
  source_type: "tool" | "memory" | "user" | "model";
  source_ref: string;
  quote?: string;
  confidence: number;
};

export type AgentAnswer = {
  answer: string;
  verification: VerificationResult;
  learned_topics: string[];
  used_topics: string[];
  project?: string | null;
  is_chat?: boolean;
  token_usage?: TokenUsage | null;
  thinking_trace?: ThinkingStep[];
  llm_calls?: LLMCallDetail[];
  claims?: Claim[];
  evidence?: EvidenceItem[];
  turn_id?: string;
  // Phase 15A: durable job record id. Set by the /api/chat handler
  // so the WebUI can deep-link a streamed answer to its Jobs-tab
  // entry without cross-referencing by timestamp.
  job_id?: string;
  // AskUserQuestion follow-up: when the agent called the `ask_user`
  // tool, the turn ends with this populated. The chat UI renders
  // an AskUserCard with clickable options instead of the plain
  // answer body.
  question?: PendingQuestion | null;
};

export type QuestionOption = {
  id: string;
  label: string;
  description?: string;
};

export type PendingQuestion = {
  question_id: string;
  asked_at: number;
  question: string;
  why: string;
  header: string;
  options: QuestionOption[];
  multi_select: boolean;
  default_option_id: string;
  asker_speaker_id: string;
  asker_chat_id: number | null;
  channel: string;
  answered: boolean;
  answer_at?: number | null;
  answer_choice?: string;
  answer_text?: string;
};

export async function answerPendingQuestion(
  question_id: string,
  choice: string,
  text: string = "",
): Promise<AgentAnswer & {
  answered_question_id?: string;
  answer_choice?: string;
  answer_text?: string;
  already_answered?: boolean;
}> {
  return json_post("/api/chat/answer-question", {
    question_id, choice, text,
  });
}

export async function fetchOpenQuestions(): Promise<{
  questions: PendingQuestion[];
}> {
  return json_get("/api/chat/open-questions");
}


// ─── Logs tab ─────────────────────────────────────────────────────

export type LogLevel = "debug" | "info" | "warning" | "error" | "critical";
export type LogSource = "python" | "tool" | "job" | "supervisor" | "agent";

export type LogEvent = {
  ts: number;
  level: LogLevel;
  source: LogSource;
  logger: string;
  message: string;
  meta: Record<string, unknown>;
  request_id: string;
};

export type LogSourcesPayload = {
  levels: LogLevel[];
  sources: LogSource[];
};

export type FetchLogsParams = {
  level?: LogLevel;
  source?: LogSource;
  search?: string;
  limit?: number;
  before_ts?: number;
};

export async function fetchLogs(
  params: FetchLogsParams = {},
): Promise<{ events: LogEvent[]; count: number }> {
  const q = new URLSearchParams();
  if (params.level) q.set("level", params.level);
  if (params.source) q.set("source", params.source);
  if (params.search) q.set("search", params.search);
  if (params.limit != null) q.set("limit", String(params.limit));
  if (params.before_ts != null) q.set("before_ts", String(params.before_ts));
  return json_get(`/api/logs?${q.toString()}`);
}

export async function fetchLogSources(): Promise<LogSourcesPayload> {
  return json_get("/api/logs/sources");
}

/** Triggers a browser download of the current ring buffer. */
export function downloadLogs(format: "jsonl" | "txt"): void {
  const a = document.createElement("a");
  a.href = `/api/logs/download?format=${format}`;
  a.click();
}


// ─── Reasoning routing (hybrid reasoning effort) ────────────────────

export type ReasoningRoutingConfig = {
  routing: Record<string, string>;
  fallback: string;
  override: string;
  updated_at: number;
  valid_levels: string[];
  defaults: Record<string, string>;
};

export async function fetchReasoningRouting(): Promise<ReasoningRoutingConfig> {
  return json_get("/api/reasoning-routing");
}

export async function saveReasoningRouting(
  routing: Record<string, string>,
  fallback: string,
): Promise<{ ok: boolean; routing: Record<string, string>; fallback: string }> {
  return json_put("/api/reasoning-routing", { routing, fallback });
}

export async function setReasoningOverride(
  level: string,
): Promise<{ ok: boolean; override: string }> {
  return json_put("/api/reasoning-routing/override", { level });
}

// Round A: full TurnWorkspace JSON returned by GET /api/turns/<id>.
// Same shape as what backend.workspace.save_turn writes per turn.
export type TurnArtifact = {
  turn_id: string;
  channel?: string;
  task: string;
  answer: string;
  project?: string | null;
  verification: VerificationResult;
  claims?: Claim[];
  evidence?: EvidenceItem[];
  thinking_trace?: ThinkingStep[];
  llm_calls?: LLMCallDetail[];
  tool_call_order?: Array<{
    name: string;
    args: Record<string, unknown>;
    result_preview: string;
    result_truncated?: boolean;
    result_full_len?: number;
    is_error?: boolean;
  }>;
  token_usage?: TokenUsage;
};

export type ConversationTurnRow = {
  ts: string;
  user: string;
  answer: string;
  intent?: string;
  is_chat?: boolean;
  confidence?: number;
  topics?: string[];
  channel?: string;
  turn_id?: string;
};

export type IndexEntry = {
  topic: string;
  category: string;
  path: string;
  keywords: string[];
  access_count: number;
  updated: string;
  project?: string | null;
};

export type RouterStats = {
  date: string;
  api_calls_today: number;
  api_cost_today: number;
  model_b_calls_today: number;
  total_a_calls: number;
  total_b_calls: number;
  last_reason: string;
  model_a_id?: string;
  model_b_id?: string;
  budget_usd?: number;
  model_a_available?: boolean;
  model_b_available?: boolean;
};

export type StatusPayload = {
  topics_total: number;
  by_category: Record<string, number>;
  core_tokens: number;
  core_max: number;
  finetune_count: number;
  current_project: string | null;
  mode: string;
  finetune_enabled: boolean;
  training_location: string;
  model_a: string | null;
  model_b: string | null;
  model_version: string | null;
  router: RouterStats | { error: string };
};

export type FinetuneExample = {
  id: string;
  score: number;
  user: string;
  assistant: string;
  metadata: {
    source_notes: string[];
    confidence: number;
    project: string | null;
    timestamp: string;
    verified: boolean;
    category: string;
    boosted: boolean;
    original_wrong_answer?: string | null;
  };
};

export type FinetuneStatusPayload = {
  total: number;
  curated: number;
  ready: boolean;
  min_required: number;
  by_category: Record<string, number>;
};

export type ModelVersion = {
  tag: string;
  model_id: string;
  created: string;
  examples_count: number;
  notes: string;
};

export type ModelVersionsState = {
  current: string;
  versions: ModelVersion[];
  rollback_enabled: boolean;
};

export type ConversationTurn = {
  ts: string;
  user: string;
  answer: string;
  intent: string;
  is_chat: boolean;
  confidence?: number;
  topics?: string[];
  // Round A: pointer to the on-disk TurnWorkspace artefact for lazy
  // loading the thinking_trace + claims/evidence on demand. Empty
  // for turns recorded before this round (chat fast-path / preference
  // branch never write a turn artefact).
  turn_id?: string;
  channel?: string;
  // Round F-pre: cheap summary fields persisted alongside the row
  // so WebUI badges (token bar + tool / LLM call counts) restore
  // immediately on page refresh, without waiting for the lazy
  // /api/turns/<id> fetch.
  token_usage?: TokenUsage | null;
  n_tool_calls?: number;
  n_llm_calls?: number;
};

export type GapEntry = {
  topic: string;
  count: number;
  last: string;
  has_note_now: boolean;
};

export type StreamEvent =
  // Round B: progress events optionally carry a structured tool_call
  // payload (every `tool`/`tool_error` step has one). When present,
  // the WebUI can append a live ToolCallCard before the final answer
  // arrives. Text-only events (think, solve, verify, …) omit it.
  | { type: "progress"; event: string; message: string; tool_call?: ToolCallDetail | null }
  | { type: "answer"; data: AgentAnswer }
  | { type: "error"; message: string };

// ---- Helpers ----

async function json_get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function json_post<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(text || `${r.status}`);
  }
  return r.json();
}

async function json_put<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

async function json_delete<T>(url: string, body?: unknown): Promise<T> {
  const opts: RequestInit = { method: "DELETE" };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

// ---- SSE stream reader ----

async function readSSE(url: string, body: unknown, onEvent: (e: any) => void): Promise<void> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.body) return;
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // Normalize \r\n to \n so SSE splitting works regardless of line endings
    buf = buf.replace(/\r\n/g, "\n");
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const p of parts) {
      const line = p.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(5).trim()));
      } catch { /* ignore */ }
    }
  }
}

// ---- Status ----

export const fetchStatus = () => json_get<StatusPayload>("/api/status");

// ---- Chat ----

export async function chatStream(
  message: string,
  project: string | null,
  onEvent: (e: StreamEvent) => void,
  attachments: string[] = [],
  channel: string | null = null,
  speaker_id: string | null = null,
): Promise<void> {
  // `channel` picks which conversation bucket the turn lands in
  // (null = backend default "webui"). `speaker_id` is the primary
  // per-user partition — null = backend default "webui:default".
  const body: Record<string, unknown> = { message, project, attachments };
  if (channel) body.channel = channel;
  if (speaker_id) body.speaker_id = speaker_id;
  return readSSE("/api/chat", body, onEvent);
}

// ---- Knowledge ----

export const fetchKnowledge = () =>
  json_get<{ topics: IndexEntry[]; by_category: Record<string, IndexEntry[]> }>("/api/knowledge");

export async function fetchNote(topic: string) {
  const r = await fetch(`/api/knowledge/${encodeURIComponent(topic)}`);
  if (!r.ok) return null;
  return r.json();
}

export const deleteNote = (topic: string) =>
  json_delete<{ ok: boolean }>(`/api/knowledge/${encodeURIComponent(topic)}`);

export const forceLearn = (topic: string, depth: "deep" | "quick" = "quick") =>
  json_post<any>("/api/knowledge/learn", { topic, depth, category: "profession" });

export const quickNote = (text: string) =>
  json_post<{ topic: string; path: string }>("/api/knowledge/quick-note", { text });

// ---- Core Memory ----

export const fetchCore = () =>
  json_get<{ content: string; tokens: number; max: number }>("/api/core-memory");

export const addCoreFact = (fact: string) =>
  json_post<{ message: string }>("/api/core-memory", { fact, source: "ui" });

export const deleteCoreFact = (search_text: string) =>
  json_delete<{ message: string }>("/api/core-memory", { search_text });

// ---- Gaps ----

export const fetchGaps = () =>
  json_get<{ gaps: GapEntry[]; open: GapEntry[]; closed: GapEntry[] }>("/api/gaps");

// ---- Capabilities ----

export const fetchCapabilities = () =>
  json_get<{ block: string }>("/api/capabilities");

// ---- Conversation ----
// Round A: per-channel filter. Channel=null returns all turns
// (back-compat for callers that pre-date the channel split).
export const fetchConversation = (
  channel: string | null = null,
  n = 50,
) => {
  const qs = new URLSearchParams();
  if (channel) qs.set("channel", channel);
  qs.set("n", String(n));
  return json_get<{
    channel: string | null;
    turns: ConversationTurn[];
    count: number;
  }>(`/api/conversation?${qs.toString()}`);
};

export const clearConversation = () =>
  json_delete<{ ok: boolean }>("/api/conversation");

// ---- Identity ----

export type IdentityProfileMeta = {
  speaker_id: string;
  path: string;
  size: number;
  modified: string;
};

export const fetchIdentity = (speaker_id?: string) => {
  const qs = speaker_id ? `?speaker_id=${encodeURIComponent(speaker_id)}` : "";
  return json_get<{
    soul: string;
    identity: string;
    user_profile: string;
    speaker_id: string | null;
  }>(`/api/identity${qs}`);
};

export const fetchIdentityProfiles = () =>
  json_get<{ profiles: IdentityProfileMeta[] }>("/api/identity/profiles");

export const updateIdentity = (
  file: string,
  content: string,
  speaker_id?: string,
) =>
  json_put<{ ok: boolean }>("/api/identity", { file, content, speaker_id });

export const fetchIdentityHistory = () =>
  json_get<{ versions: { timestamp: string; path: string; size: number }[] }>("/api/identity/history");

// ---- Projects ----

export const fetchProjects = () =>
  json_get<{ current: string | null; all: string[] }>("/api/projects");

export const createProject = (name: string) =>
  json_post<{ message: string }>("/api/projects", { name });

export const fetchProjectDetail = (name: string) =>
  json_get<{ overview: string }>(`/api/projects/${encodeURIComponent(name)}`);

export const endProject = (name: string) =>
  json_post<{ message: string }>(`/api/projects/${encodeURIComponent(name)}/end`);

export const addProjectContext = (name: string, text: string) =>
  json_post<{ message: string }>(`/api/projects/${encodeURIComponent(name)}/context`, { text });

export const addProjectDecision = (name: string, what: string, why: string) =>
  json_post<{ message: string }>(`/api/projects/${encodeURIComponent(name)}/decision`, { what, why });

export const addProjectIssue = (name: string, problem: string, fix: string) =>
  json_post<{ message: string }>(`/api/projects/${encodeURIComponent(name)}/issue`, { problem, fix });

// ---- Goals ----

export type GoalData = {
  id: string;
  description: string;
  priority: number;
  goal_type: string;
  status: string;
  subtasks: { description: string; status: string; result: string }[];
  created: string;
  completed: string | null;
  context: string;
  source: string;
  progress_notes: string[];
};

export type GoalStats = {
  total: number;
  active: number;
  paused: number;
  completed: number;
  failed: number;
  by_type: Record<string, number>;
  interaction_count: number;
  next_proactive_check_in: number;
};

export const fetchGoals = () =>
  json_get<{ goals: GoalData[]; stats: GoalStats }>("/api/goals");

export const addGoal = (data: {
  description: string;
  priority?: number;
  goal_type?: string;
  context?: string;
  subtasks?: string[];
}) => json_post<{ goal: GoalData }>("/api/goals", data);

export const completeGoal = (id: string) =>
  json_post<{ ok: boolean }>(`/api/goals/${id}/complete`);

export const pauseGoal = (id: string) =>
  json_post<{ ok: boolean }>(`/api/goals/${id}/pause`);

export const resumeGoal = (id: string) =>
  json_post<{ ok: boolean }>(`/api/goals/${id}/resume`);

export const failGoal = (id: string) =>
  json_post<{ ok: boolean }>(`/api/goals/${id}/fail`);

export const deleteGoal = (id: string) =>
  json_delete<{ ok: boolean }>(`/api/goals/${id}`);

export const updateGoalPriority = (id: string, priority: number) =>
  json_put<{ ok: boolean }>(`/api/goals/${id}/priority`, { priority });

// ---- Sessions ----

export type SessionSummary = {
  id: string;
  speaker_id: string;
  // session_key isolates conversation threads (Wife DM vs Wife group
  // share speaker_id, get distinct session_keys). thread_label is
  // the human-readable form rendered by the server: "DM (bot tg-main)",
  // "Group -1009 (bot tg-main)", "WebUI", etc.
  session_key: string;
  thread_label: string;
  started: string;
  ended: string | null;
  title: string;
  archived: boolean;
  turn_count: number;
  avg_confidence: number;
  intents: Record<string, number>;
  topics_used: string[];
  duration_seconds: number | null;
};

export type SessionDetail = SessionSummary & {
  turns: ConversationTurn[];
};

export type SpeakerSummary = {
  speaker_id: string;
  session_count: number;
  last_active: string;
};

export type ThreadSummary = {
  session_key: string;
  speaker_id: string;
  thread_label: string;
  session_count: number;
  last_active: string;
};

export type SessionStats = {
  total_sessions: number;
  total_turns: number;
  intents: Record<string, number>;
  daily_counts: Record<string, number>;
  confidence_over_time: { session_id: string; date: string; avg_confidence: number; turns: number }[];
  archived_count: number;
};

export const fetchSessions = (
  includeArchived = false,
  speaker_id?: string,
  session_key?: string,
) => {
  const params = new URLSearchParams();
  params.set("include_archived", String(includeArchived));
  if (speaker_id) params.set("speaker_id", speaker_id);
  if (session_key) params.set("session_key", session_key);
  return json_get<{
    sessions: SessionSummary[];
    current_by_speaker: Record<string, string>;
    current_by_session_key: Record<string, string>;
    default_speaker: string;
  }>(`/api/sessions?${params}`);
};

export const fetchSpeakers = () =>
  json_get<{ speakers: SpeakerSummary[]; default_speaker: string }>(
    "/api/sessions/speakers",
  );

export const fetchThreads = () =>
  json_get<{ threads: ThreadSummary[]; default_speaker: string }>(
    "/api/sessions/threads",
  );

export const fetchSessionStats = () =>
  json_get<SessionStats>("/api/sessions/stats");

export const fetchCurrentSession = (speaker_id?: string) => {
  const qs = speaker_id ? `?speaker_id=${encodeURIComponent(speaker_id)}` : "";
  return json_get<{ session: SessionDetail | null; speaker_id?: string }>(
    `/api/sessions/current${qs}`,
  );
};

// Round A: lazy-load full TurnWorkspace artefact for a single turn.
// Used when the user clicks to expand a tool card on a historical
// chat message — we don't ship the full thinking_trace in the
// session list to keep that payload small.
export const fetchTurn = (turn_id: string) =>
  json_get<TurnArtifact>(`/api/turns/${encodeURIComponent(turn_id)}`);

// (channel-aware fetchConversation lives in the // ---- Conversation ----
// section above — exported earlier so the existing memory page +
// the new WebUI history loader share one definition.)

export const fetchSession = (id: string) =>
  json_get<{ session: SessionDetail }>(`/api/sessions/${id}`);

export const newSession = (speaker_id?: string) => {
  const qs = speaker_id ? `?speaker_id=${encodeURIComponent(speaker_id)}` : "";
  return json_post<{ session: SessionDetail }>(`/api/sessions/new${qs}`);
};

export const deleteSession = (id: string) =>
  json_delete<{ ok: boolean }>(`/api/sessions/${id}`);

export const archiveSessions = (days: number) =>
  json_post<{ archived: number }>("/api/sessions/archive", { days });

// ---- Knowledge Graph ----

export type GraphStats = {
  entities: number;
  edges: number;
  notes_indexed: number;
};

export type GraphEntity = {
  name: string;
  edges: number;
};

export type GraphNeighbor = {
  source: string;
  target: string;
  relation: string;
  note: string;
  weight: number;
};

export const fetchGraphStats = () =>
  json_get<GraphStats>("/api/graph");

export const fetchGraphEntities = () =>
  json_get<{ entities: GraphEntity[] }>("/api/graph/entities");

export const fetchGraphNeighbors = (entity: string) =>
  json_get<{ entity: string; neighbors: GraphNeighbor[] }>(
    `/api/graph/neighbors/${encodeURIComponent(entity)}`
  );

export const reindexGraph = () =>
  json_post<{ notes: number; triples: number; errors: number }>("/api/graph/reindex");

export type GraphNode = {
  id: string;
  name: string;
  connections: number;
};

export type GraphLink = {
  source: string;
  target: string;
  relation: string;
  note: string;
  weight: number;
};

export type FullGraph = {
  nodes: GraphNode[];
  links: GraphLink[];
};

export const fetchFullGraph = () =>
  json_get<FullGraph>("/api/graph/full");

// ---- Finetune ----

export const fetchFinetuneStatus = () =>
  json_get<FinetuneStatusPayload>("/api/finetune/status");

export const fetchFinetuneExamples = () =>
  json_get<{ items: FinetuneExample[] }>("/api/finetune/examples");

export const deleteFinetuneExample = (id: string) =>
  json_delete<{ ok: boolean }>(`/api/finetune/examples/${id}`);

export const boostFinetuneExample = (id: string) =>
  json_post<{ ok: boolean }>(`/api/finetune/examples/${id}/boost`);

export const editFinetuneExample = (id: string, patch: { assistant?: string; boosted?: boolean }) =>
  json_put<{ ok: boolean }>(`/api/finetune/examples/${id}`, patch);

export const addFromChat = (data: {
  question: string;
  answer: string;
  used_topics: string[];
  confidence: number;
  project?: string | null;
}) => json_post<{ ok: boolean }>("/api/finetune/add-from-chat", data);

export const addCorrection = (data: {
  question: string;
  wrong_answer: string;
  corrected_answer: string;
  project?: string | null;
}) => json_post<{ ok: boolean; id: string }>("/api/finetune/correction", data);

export const fetchModelVersions = () =>
  json_get<ModelVersionsState>("/api/model/versions");

export const switchModelVersion = (tag: string) =>
  json_post<{ message: string }>("/api/finetune/switch", { tag });

export const rollbackModel = () =>
  json_post<{ message: string }>("/api/finetune/rollback");

export async function startFinetune(onEvent: (e: any) => void): Promise<void> {
  return readSSE("/api/finetune/start", undefined, onEvent);
}

export const exportForCloud = () =>
  json_post<any>("/api/finetune/export-cloud");

export const importGguf = (path: string, tag?: string) =>
  json_post<any>("/api/finetune/import-gguf", { path, tag: tag || null });

export const compareModels = () =>
  json_post<any>("/api/finetune/compare");

// ---- Memory (conversation facts) ----

export type MemoryStats = {
  total_facts_logged: number;
  memory_edges_in_graph: number;
  memory_entities: number;
  graph_total_entities: number;
  graph_total_edges: number;
};

export type MemoryFact = {
  summary: string;
  triples: [string, string, string][];
  tags: string[];
  category: string;
  confidence: number;
  ts: string;
  source_turn: string;
};

export type RecalledFact = {
  entity: string;
  relation: string;
  target: string;
  weight: number;
};

export const fetchMemoryStats = () =>
  json_get<MemoryStats>("/api/memory");

export const fetchMemoryFacts = (limit = 50) =>
  json_get<{ facts: MemoryFact[] }>(`/api/memory/facts?limit=${limit}`);

export const recallMemory = (query: string, limit = 10) =>
  json_post<{ facts: RecalledFact[]; block: string }>("/api/memory/recall", { query, limit });

// ---- Meta-Learner ----

export type MetaLearnerStats = {
  total_failures: number;
  by_root_cause: Record<string, number>;
  by_domain: Record<string, number>;
  avg_severity: number;
  patterns_count: number;
  patterns: { pattern: string; frequency: number; domains: string[]; suggested_fix: string; priority: number }[];
};

export type FailureEntry = {
  ts: string;
  question: string;
  answer_preview: string;
  confidence: number;
  contradictions: string[];
  unverified: string[];
  intent: string;
  analysis: {
    root_cause: string;
    missing_topic: string | null;
    error_pattern: string | null;
    domain: string;
    fix_action: string;
    fix_detail: string;
    severity: number;
  } | null;
};

export const fetchMetaLearner = () =>
  json_get<MetaLearnerStats>("/api/meta-learner");

export const fetchFailures = () =>
  json_get<{ failures: FailureEntry[] }>("/api/meta-learner/failures");

export const extractPatterns = () =>
  json_post<{ patterns: any[] }>("/api/meta-learner/extract-patterns");

// ---- Evaluator ----

export type EvalDayReport = {
  date: string;
  total_interactions: number;
  tasks: number;
  chats: number;
  avg_confidence: number;
  total_contradictions: number;
  total_unverified: number;
  topics_used: string[];
  by_intent: Record<string, number>;
  low_confidence_count: number;
  high_confidence_count: number;
};

export type EvalStats = {
  total_logged: number;
  total_tasks: number;
  total_chats: number;
  overall_avg_confidence: number;
  today: EvalDayReport;
  weekly_trend: EvalDayReport[];
  regressions: { domain: string; this_week_avg: number; last_week_avg: number; drop: number; sample_size: number }[];
  suggestions: { type: string; suggestion: string; priority: number; topic?: string; intent?: string; avg_confidence?: number }[];
};

export const fetchEvalStats = () =>
  json_get<EvalStats>("/api/eval");

export const fetchEvalToday = () =>
  json_get<EvalDayReport>("/api/eval/today");

export const fetchEvalTrend = () =>
  json_get<{ trend: EvalDayReport[] }>("/api/eval/trend");

export const fetchEvalRegressions = () =>
  json_get<{ regressions: any[] }>("/api/eval/regressions");

export const fetchEvalSuggestions = () =>
  json_get<{ suggestions: any[] }>("/api/eval/suggestions");

// ---- Analogy Engine ----

export type AnalogyPattern = {
  pattern_id: string;
  pattern: string;
  abstract_form: string;
  domain: string;
  mechanism: string;
  applicable_when: string;
  examples: { question: string; domain: string }[];
};

export const fetchAnalogies = () =>
  json_get<{ patterns: AnalogyPattern[]; stats: { total_patterns: number; by_domain: Record<string, number>; total_examples: number } }>("/api/analogies");

// ---- Self-Modifier ----

export type ModProposal = {
  id: string;
  module: string;
  title: string;
  description: string;
  old_code: string;
  new_code: string;
  impact: string;
  risk: string;
  reasoning: string;
  status: string;
  created: string;
  reviewed: string | null;
  review_note: string;
};

export type SelfModifierStats = {
  total: number;
  by_status: Record<string, number>;
  by_impact: Record<string, number>;
  modules: string[];
};

export const fetchSelfModifier = () =>
  json_get<SelfModifierStats>("/api/self-modifier");

export const fetchProposals = (status?: string) =>
  json_get<{ proposals: ModProposal[] }>(`/api/self-modifier/proposals${status ? `?status=${status}` : ""}`);

export const analyzeModule = (module: string) =>
  json_post<{ proposals: ModProposal[] }>("/api/self-modifier/analyze", { module });

export const approveProposal = (id: string, note?: string) =>
  json_post<{ ok: boolean }>(`/api/self-modifier/proposals/${id}/approve`, { note: note || "" });

export const rejectProposal = (id: string, note?: string) =>
  json_post<{ ok: boolean }>(`/api/self-modifier/proposals/${id}/reject`, { note: note || "" });

export const applyProposal = (id: string) =>
  json_post<{ ok: boolean; message: string }>(`/api/self-modifier/proposals/${id}/apply`);

export const deleteProposal = (id: string) =>
  json_delete<{ ok: boolean }>(`/api/self-modifier/proposals/${id}`);

// ---- Token Usage ----

export type UsageCallRecord = {
  ts: string;
  task_type: string;
  model: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  cost_usd: number;
  duration_ms: number;
  prompt_preview: string;
};

export type UsageStats = {
  total_calls: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  by_task_type: Record<string, { calls: number; input: number; output: number; cost: number }>;
  by_model: Record<string, { calls: number; input: number; output: number; cost: number }>;
};

export const fetchUsageStats = () =>
  json_get<UsageStats>("/api/usage");

export const fetchUsageCalls = (limit = 50) =>
  json_get<{ calls: UsageCallRecord[] }>(`/api/usage/calls?limit=${limit}`);

export type RequestTrace = {
  ts: string;
  question: string;
  trace: ThinkingStep[];
  usage: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    cost_usd?: number;
    llm_calls?: number;
  };
};

export const fetchUsageTraces = (limit = 20) =>
  json_get<{ traces: RequestTrace[] }>(`/api/usage/traces?limit=${limit}`);


// ---- Channels ----
export type ChannelConfig = {
  id: string;
  name: string;
  type: string; // "telegram"
  enabled: boolean;
  auto_start: boolean;
  config: Record<string, any>;
  status?: string;
  runtime_status?: string;
  created?: string;
  updated?: string;
  last_started?: string;
};

export const fetchChannels = () =>
  json_get<{ channels: ChannelConfig[] }>("/api/channels");

export const createChannel = (ch: {
  id: string;
  name: string;
  type: string;
  enabled?: boolean;
  auto_start?: boolean;
  config?: Record<string, any>;
}) => json_post<ChannelConfig>("/api/channels", ch);

export const updateChannel = (
  channelId: string,
  data: { name?: string; enabled?: boolean; auto_start?: boolean; config?: Record<string, any> },
) => json_put<ChannelConfig>(`/api/channels/${encodeURIComponent(channelId)}`, data);

export const deleteChannel = (channelId: string) =>
  json_delete<{ ok: boolean }>(`/api/channels/${encodeURIComponent(channelId)}`);

export const startChannel = (channelId: string) =>
  json_post<{ ok: boolean; status: string }>(`/api/channels/${encodeURIComponent(channelId)}/start`);

export const stopChannel = (channelId: string) =>
  json_post<{ ok: boolean; status: string }>(`/api/channels/${encodeURIComponent(channelId)}/stop`);

export const testChannel = (channelId: string) =>
  json_post<{ ok: boolean; bot_name?: string; bot_username?: string; error?: string }>(
    `/api/channels/${encodeURIComponent(channelId)}/test`,
  );


// ---- Providers (multi-LLM) ----
export type OAuthStatus = {
  authenticated: boolean;
  expires_at?: string;
  expired?: boolean;
  has_refresh?: boolean;
  scope?: string;
};

export type ProviderConfig = {
  id: string;
  name: string;
  type: string;
  enabled: boolean;
  is_default?: boolean;
  auth_type?: string; // "api_key" | "oauth" | "none"
  api_key_masked?: string;
  api_key_env?: string;
  base_url?: string;
  models: string[];
  default_model?: string;
  max_tokens?: number;
  temperature?: number;
  oauth?: Record<string, string>;
  oauth_status?: OAuthStatus;
  created?: string;
  updated?: string;
};

export type ProviderTypeInfo = {
  label: string;
  base_url: string;
  key_env_default: string;
  models: string[];
  supports_tools: boolean;
  auth_types?: string[];
};

export const fetchProviders = () =>
  json_get<{ providers: ProviderConfig[]; types: Record<string, ProviderTypeInfo> }>("/api/providers");

export const fetchProviderTypes = () =>
  json_get<{ types: Record<string, ProviderTypeInfo>; pricing: Record<string, Record<string, number>> }>(
    "/api/providers/types",
  );

export const createProvider = (p: {
  id: string;
  name: string;
  type: string;
  enabled?: boolean;
  api_key?: string;
  api_key_env?: string;
  base_url?: string;
  models?: string[];
  default_model?: string;
  max_tokens?: number;
  temperature?: number;
  auth_type?: string;
  oauth?: Record<string, string>;
  aws?: { access_key_id: string; secret_access_key: string; region: string };
}) => json_post<{ ok: boolean; provider: ProviderConfig }>("/api/providers", p);

export const updateProvider = (
  providerId: string,
  data: Record<string, any>,
) => json_put<{ ok: boolean }>(`/api/providers/${encodeURIComponent(providerId)}`, data);

export const deleteProvider = (providerId: string) =>
  json_delete<{ ok: boolean }>(`/api/providers/${encodeURIComponent(providerId)}`);

export const testProvider = (providerId: string) =>
  json_post<{ ok: boolean; models?: string[]; model?: string; message?: string; error?: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/test`,
  );

// Live catalog from an OpenAI-compatible gateway (OpenRouter etc.) —
// lets the Settings UI change a provider's model without hand-editing
// the providers.json models list.
export const fetchProviderRemoteModels = (providerId: string) =>
  json_get<{ ok: boolean; models: string[]; count?: number; error?: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/models/remote`,
  );

export type OllamaModel = {
  name: string;
  size: number;
  modified: string;
  family: string;
  parameters: string;
  quantization: string;
};

export const fetchOllamaModels = () =>
  json_get<{ ok: boolean; models: OllamaModel[]; error?: string }>("/api/providers/ollama/models");

export const pullOllamaModel = (model: string) =>
  json_post<{ ok: boolean; message?: string; error?: string }>("/api/providers/ollama/pull", { model });

export const deleteOllamaModel = (model: string) =>
  json_delete<{ ok: boolean; error?: string }>(`/api/providers/ollama/models/${encodeURIComponent(model)}`);

// ---- Provider Auth / OAuth ----
export const updateProviderAuth = (
  providerId: string,
  authType: string,
  oauth?: Record<string, string>,
) =>
  json_put<{ ok: boolean }>(`/api/providers/${encodeURIComponent(providerId)}/auth`, {
    auth_type: authType,
    oauth: oauth || {},
  });

export const fetchOAuthStatus = (providerId: string) =>
  json_get<OAuthStatus>(`/api/providers/${encodeURIComponent(providerId)}/oauth/status`);

export const getOAuthAuthorizeUrl = (providerId: string) =>
  json_post<{ url: string; redirect_uri: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/oauth/authorize-url`,
  );

export const oauthClientCredentials = (providerId: string) =>
  json_post<{ ok: boolean; message?: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/oauth/client-credentials`,
  );

export const oauthRevoke = (providerId: string) =>
  json_post<{ ok: boolean }>(`/api/providers/${encodeURIComponent(providerId)}/oauth/revoke`);

export const oauthExchangeUrl = (providerId: string, url: string) =>
  json_post<{ ok: boolean; message?: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/oauth/exchange-url`,
    { url },
  );

export const oauthManualToken = (providerId: string, access_token: string, refresh_token?: string) =>
  json_post<{ ok: boolean; message?: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/oauth/manual-token`,
    { access_token, refresh_token: refresh_token || "", expires_in: 86400 },
  );

export const fetchAuthTypes = () =>
  json_get<{
    auth_types: Record<string, { label: string; fields: string[] }>;
    oauth_presets: Record<string, Record<string, string>>;
  }>("/api/providers/auth-types");

export interface ProviderConnectInfo {
  key_url: string;
  key_instructions: string;
  docs_url: string;
  extra_fields?: string[];
}

export const fetchProviderConnectInfo = () =>
  json_get<{ connect_info: Record<string, ProviderConnectInfo> }>("/api/providers/connect-info");

export type CodexStatus = {
  logged_in: boolean;
  reason?: string;
  email?: string;
  plan_type?: string;
  account_id?: string;
  expires_at?: string;
  expires_in_seconds?: number;
  expired?: boolean;
  auth_provider?: string;
  last_refresh?: string;
};

export const fetchCodexStatus = () =>
  json_get<CodexStatus>("/api/providers/codex/status");

export type CodexModel = {
  slug: string;
  display_name: string;
  description?: string;
  default_reasoning_level?: string;
  supported_in_api?: boolean;
  visibility?: string;
};

export type CodexModelsResponse = {
  ok: boolean;
  reason?: string;
  models: CodexModel[];
  fetched_at?: string;
  client_version?: string;
  source?: "cache_file" | "fallback";
};

export const fetchCodexModels = () =>
  json_get<CodexModelsResponse>("/api/providers/codex/models");

export type CopilotStatus = {
  logged_in: boolean;
  reason?: string;
  user?: string;
  source?: string;
  bearer_cached?: boolean;
  bearer_expires_at?: string;
  checked_paths?: string[];
};

export const fetchCopilotStatus = () =>
  json_get<CopilotStatus>("/api/providers/copilot/status");


// ---------- Embeddings (memory layer) ----------

export type EmbedderConfig = {
  backend?: "auto" | "llama_cpp" | "ollama" | "openai" | "cohere" | "disabled";
  llama_cpp?: { url?: string; model?: string };
  ollama?: { url?: string; model?: string };
  openai?: { provider_id?: string; model?: string };
  cohere?: { model?: string };
};

export type EmbedderStatus = {
  backend: "llama_cpp" | "ollama" | "openai" | "cohere" | "disabled" | null;
  model: string | null;
  dim: number | null;
  last_error: string | null;
  config: EmbedderConfig;
};

export type VectorStoreStats = {
  count: number;
  dim: number | null;
  backend: string | null;
  model: string | null;
};

export type EmbeddingsCoverage = {
  total_notes: number;
  embedded: number;
  missing: number;
  stale_store: boolean;
  reason: "ok" | "embedder_disabled" | "store_dim_or_model_mismatch" | "missing_notes";
};

export type EmbeddingsStatusResponse = {
  embedder: EmbedderStatus;
  vector_store: VectorStoreStats;
  coverage: EmbeddingsCoverage;
};

export type EmbeddingsBackfillResponse = {
  ok: boolean;
  reason?: string;
  backend?: string;
  model?: string;
  dim?: number;
  embedded?: number;
  skipped?: number;
  errors?: number;
  total?: number;
};

export const fetchEmbeddingsStatus = () =>
  json_get<EmbeddingsStatusResponse>("/api/memory/embeddings/status");

export const updateEmbeddingsConfig = (cfg: EmbedderConfig) =>
  json_put<EmbeddingsStatusResponse & { config: EmbedderConfig }>(
    "/api/memory/embeddings/config",
    cfg,
  );

export const resetEmbedder = () =>
  json_post<EmbeddingsStatusResponse>("/api/memory/embeddings/reset");

export const backfillEmbeddings = (force: boolean = false) =>
  json_post<EmbeddingsBackfillResponse>(
    `/api/memory/embeddings/backfill?force=${force ? "true" : "false"}`,
  );


// ---------- Attachments + voice transcription ----------

export type AttachmentMeta = {
  sha256: string;
  kind: "image" | "audio" | "file";
  mime_type: string;
  size: number;
  filename: string;
  transcript: string;
  created: string;
};

export async function uploadAttachment(file: File | Blob, filename?: string): Promise<AttachmentMeta> {
  const fd = new FormData();
  fd.append("file", file, filename || (file instanceof File ? file.name : "blob"));
  const r = await fetch("/api/attachments", { method: "POST", body: fd });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(text || `${r.status}`);
  }
  return r.json();
}

export async function transcribeAudio(
  blob: Blob,
  filename = "voice.ogg",
  language?: string,
): Promise<{ text: string; sha256: string; backend: string }> {
  const fd = new FormData();
  fd.append("file", blob, filename);
  if (language) fd.append("language", language);
  const r = await fetch("/api/transcribe", { method: "POST", body: fd });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(text || `${r.status}`);
  }
  return r.json();
}

export const attachmentUrl = (sha: string) => `/api/attachments/${sha}`;


// ---------- Voice config (STT / TTS) + Tailscale discovery ----------

export type TranscriberConfig = {
  backend?: "auto" | "local_whisper" | "whisper_cpp" | "openai_whisper" | "disabled";
  local_whisper?: { url?: string; model?: string };
  whisper_cpp?: { url?: string; model?: string };
  openai_whisper?: { model?: string };
};

export type TranscriberStatus = {
  backend: string | null;
  model?: string | null;
  last_error?: string | null;
  config?: TranscriberConfig;
};

export type TtsConfig = {
  backend?: "auto" | "edge_tts" | "local_piper" | "openai_tts" | "disabled";
  edge_tts?: { voice?: string; voice_ru?: string };
  local_piper?: { url?: string; voice?: string; voice_ru?: string };
  openai_tts?: { model?: string; voice?: string };
};

export type TtsStatus = {
  backend: string | null;
  voice?: string | null;
  last_error?: string | null;
  config?: TtsConfig;
};

export const fetchTranscribeConfig = () =>
  json_get<TranscriberConfig>("/api/transcribe/config");

export const fetchTranscribeStatus = () =>
  json_get<TranscriberStatus>("/api/transcribe/status");

export const putTranscribeConfig = (cfg: Partial<TranscriberConfig>) =>
  json_put<{ ok: boolean; config: TranscriberConfig; transcriber: TranscriberStatus }>(
    "/api/transcribe/config",
    cfg,
  );

export const resetTranscriber = () =>
  json_post<{ ok: boolean; transcriber: TranscriberStatus }>("/api/transcribe/reset");

export const fetchTtsConfig = () => json_get<TtsConfig>("/api/tts/config");
export const fetchTtsStatus = () => json_get<TtsStatus>("/api/tts/status");

export const putTtsConfig = (cfg: Partial<TtsConfig>) =>
  json_put<{ ok: boolean; config: TtsConfig; tts: TtsStatus }>("/api/tts/config", cfg);

export const resetTts = () =>
  json_post<{ ok: boolean; tts: TtsStatus }>("/api/tts/reset");

// Tailscale / LAN service discovery — probes the host for
// whisper / piper / ollama and (optionally) writes URLs back into
// the per-service configs.
export type DiscoverResult = {
  host: string;
  error?: string;
  found?: Record<string, { ok: boolean; url?: string; reason?: string }>;
  applied?: Record<string, string>;
};

export const runDiscover = (host?: string, apply = false) => {
  const params = new URLSearchParams();
  if (host) params.set("host", host);
  if (apply) params.set("apply", "true");
  const qs = params.toString();
  return json_get<DiscoverResult>(`/api/discover${qs ? `?${qs}` : ""}`);
};


// ---------- Engine config (router + verification overrides) ----------

export type EngineRouterCfg = {
  daily_api_budget_usd?: number;
  estimated_cost_per_call_usd?: number;
  api_ping_cache_seconds?: number;
  fallback_to_local?: boolean;
  tool_synth_max_tokens?: number;
  tool_loop_input_budget?: number;
};

export type EngineVerificationCfg = {
  enabled?: boolean;
  min_confidence?: number;
  require_sources?: boolean;
  critic_threshold?: number;
  critic_max_retries?: number;
  critic_retry_token_budget?: number;
};

export type EngineWorkspaceCfg = {
  inbox_retention_days?: number;
  outbox_retention_days?: number;
  notes_retention_days?: number;
  turns_retention_days?: number;
};

export type EngineKnowledgeCfg = {
  core_memory_max_tokens?: number;
  auto_promote_threshold?: number;
  finetune_min_examples?: number;
  note_max_tokens?: number;
};

export type EngineConfigEnvelope = {
  effective: {
    router: EngineRouterCfg;
    verification: EngineVerificationCfg;
    workspace: EngineWorkspaceCfg;
    knowledge: EngineKnowledgeCfg;
  };
  overrides: {
    router?: EngineRouterCfg;
    verification?: EngineVerificationCfg;
    workspace?: EngineWorkspaceCfg;
    knowledge?: EngineKnowledgeCfg;
  };
  schema: {
    router: string[];
    verification: string[];
    workspace: string[];
    knowledge: string[];
  };
};

export type EngineConfigUpdateResp = EngineConfigEnvelope & {
  applied: {
    router?: EngineRouterCfg;
    verification?: EngineVerificationCfg;
    workspace?: EngineWorkspaceCfg;
    knowledge?: EngineKnowledgeCfg;
  };
  rejected: string[];
  ok: boolean;
};

export const fetchEngineConfig = () =>
  json_get<EngineConfigEnvelope>("/api/engine/config");

export const putEngineConfig = (cfg: {
  router?: EngineRouterCfg;
  verification?: EngineVerificationCfg;
  workspace?: EngineWorkspaceCfg;
  knowledge?: EngineKnowledgeCfg;
}) => json_put<EngineConfigUpdateResp>("/api/engine/config", cfg);

export const resetEngineConfig = () =>
  json_post<EngineConfigEnvelope & { ok: boolean }>("/api/engine/config/reset");


// ---------- Roles, relationships, scheduled messages (Phase 11) ----

export type Role = "owner" | "trusted" | "guest";

export type RoleEntry = { role: Role; label?: string };

export type RolesState = {
  owner_speaker_ids: string[];
  speakers: Record<string, RoleEntry>;
  seen_speakers: { speaker_id: string; session_count: number; last_active: string }[];
  default_speaker: string;
};

export const fetchRoles = () => json_get<RolesState>("/api/roles");

export const setRole = (speaker_id: string, role: Role, label?: string) =>
  json_put<{ ok: boolean; speaker_id: string; entry: RoleEntry }>(
    `/api/roles/${encodeURIComponent(speaker_id)}`,
    { role, label },
  );

export const fetchRelationships = () =>
  json_get<{ relationships: Record<string, string> }>("/api/relationships");

export const setRelationships = (relationships: Record<string, string>) =>
  json_put<{ ok: boolean; relationships: Record<string, string> }>(
    "/api/relationships",
    { relationships },
  );

export const fetchTelegramContacts = () =>
  json_get<{
    contacts: Record<
      string,
      { chat_id: number; username?: string; label?: string; last_seen?: string }
    >;
  }>("/api/contacts/telegram");

export type ScheduledMessage = {
  id: string;
  target_speaker: string;
  text: string;
  due_at: string;
  requested_by: string;
  requested_at: string;
  status: "pending" | "delivering" | "sent" | "failed" | "cancelled";
  delivered_at: string | null;
  last_error: string;
};

export const fetchScheduledMessages = (status?: string) => {
  const qs = status ? `?status=${encodeURIComponent(status)}` : "";
  return json_get<{ messages: ScheduledMessage[] }>(
    `/api/scheduled-messages${qs}`,
  );
};

export const cancelScheduledMessage = (id: string) =>
  json_delete<{ ok: boolean }>(
    `/api/scheduled-messages/${encodeURIComponent(id)}`,
  );

// Reminders tab (2026-06-12): explicit-time creation from Settings.
// Exactly one of due_at (ISO 8601 UTC Z) / delay_minutes.
export const createScheduledMessage = (p: {
  text: string;
  due_at?: string;
  delay_minutes?: number;
  target_speaker?: string;
}) => json_post<{ ok: boolean; message: ScheduledMessage }>(
  "/api/scheduled-messages", p,
);


// ---------- Skills (Phase 12B/C) ----------

export type SkillMeta = {
  name: string;
  description: string;
  triggers: string[];
  when_to_use: string;
  body: string;
  source: "builtin" | "user";
  enabled: boolean;
  path: string;
};

export type SkillFull = SkillMeta & { raw_md: string };

export const fetchSkills = () =>
  json_get<{ skills: SkillMeta[]; user_dir: string }>("/api/skills");

export const fetchSkill = (name: string) =>
  json_get<SkillFull>(`/api/skills/${encodeURIComponent(name)}`);

export const upsertSkill = (name: string, raw_md: string) =>
  json_put<{ ok: boolean } & SkillMeta>(
    `/api/skills/${encodeURIComponent(name)}`,
    { raw_md },
  );

export const deleteSkill = (name: string) =>
  json_delete<{ ok: boolean; deleted: string }>(
    `/api/skills/${encodeURIComponent(name)}`,
  );

export const setSkillEnabled = (name: string, enabled: boolean) =>
  json_post<{ ok: boolean } & SkillMeta>(
    `/api/skills/${encodeURIComponent(name)}/enabled`,
    { enabled },
  );

export const reloadSkills = () =>
  json_post<{ ok: boolean; count: number }>("/api/skills/reload");

export type SkillInstallRequest = {
  source_type: "git" | "zip" | "local";
  source: string;
  name?: string;
  subdir?: string;
};

export const installSkill = (body: SkillInstallRequest) =>
  json_post<{ ok: boolean; name: string; skill: SkillMeta }>(
    "/api/skills/install",
    body,
  );


// ---------- Self-modifications (patch overlay) ----------

export type SelfModPatch = {
  id: string;
  slug: string;
  file: string;
  title: string;
  created: string;
  status: "applied" | "needs_review" | "reverted";
  patch_filename: string;
  last_error: string;
};

export const fetchSelfMods = () =>
  json_get<{ patches: SelfModPatch[] }>("/api/self-mods");

export const revertSelfMod = (patch_id: string) =>
  json_post<{ ok: boolean; reverted: string }>(`/api/self-mods/${patch_id}/revert`);

export const revertAllSelfMods = () =>
  json_post<{ ok: boolean }>("/api/self-mods/revert-all");

export type SelfModArchive = {
  archive_id: string;
  archived_at: string;
  patch_count: number;
  entries: SelfModPatch[];
};

export const fetchSelfModHistory = () =>
  json_get<{ archives: SelfModArchive[] }>("/api/self-mods/history");

export const restoreSelfModFromHistory = (
  archive_id: string,
  patch_filename: string,
) =>
  json_post<{ ok: boolean; entry: SelfModPatch }>(
    `/api/self-mods/history/${encodeURIComponent(archive_id)}/${encodeURIComponent(patch_filename)}/restore`,
  );

export const restoreSelfModArchive = (archive_id: string) =>
  json_post<{
    ok: boolean;
    restored: string[];
    failed_at: string | null;
    error: string | null;
  }>(`/api/self-mods/history/${encodeURIComponent(archive_id)}/restore`);


// ---------- Active model selection ----------
export interface ActiveModelSelection {
  provider_id: string;
  provider_type: string;
  model: string;
  provider_name: string;
}

export interface AvailableModel {
  provider_id: string;
  provider_name: string;
  provider_type: string;
  model: string;
  is_default: boolean;
}

export const fetchActiveModel = () =>
  json_get<{ active: ActiveModelSelection | Record<string, never>; models: AvailableModel[] }>("/api/active-model");

export const setActiveModel = (provider_id: string, model: string) =>
  json_put<{ ok: boolean; active: ActiveModelSelection }>("/api/active-model", { provider_id, model });

export const clearActiveModel = () =>
  json_delete<{ ok: boolean }>("/api/active-model");


// ---------- Autonomic (Model X) ----------

export type AutonomicStatus = {
  enabled: boolean;
  enabled_path: string;
  scheduler_running: boolean;
  registered_levers: string[];
};

export type TickEntry = {
  ts: string;
  source: string;
  lever: string | null;
  params: Record<string, unknown>;
  reason: string;
  rule_name: string | null;
  executed: boolean;
  note: string;
};

export type LeverReport = {
  lever: string;
  params: Record<string, unknown>;
  started_at: string;
  finished_at: string;
  status: string;
  outcome: Record<string, unknown>;
  cost: {
    tokens_in: number;
    tokens_out: number;
    seconds: number;
    usd: number;
  };
  reason: string;
  follow_ups: string[];
};

export type PendingEntry = {
  id: string;
  lever: string;
  params: Record<string, unknown>;
  requested_at: string;
  status: string;
};

export type ImmuneSignature = {
  id: string;
  pattern: Record<string, unknown>;
  severity: string;
  fix_lever: string;
  fix_params: Record<string, unknown>;
  observed_count: number;
  success_rate: number | null;
};

export const fetchAutonomicStatus = () =>
  json_get<AutonomicStatus>("/api/autonomic/status");

export const fetchTicks = (limit = 50) =>
  json_get<{ ticks: TickEntry[] }>(`/api/autonomic/ticks?limit=${limit}`);

export const fetchLeverHistory = (name: string, limit = 10) =>
  json_get<{ lever: string; reports: LeverReport[] }>(
    `/api/autonomic/levers/${encodeURIComponent(name)}?limit=${limit}`,
  );

export const fetchPending = () =>
  json_get<{ pending: PendingEntry[] }>("/api/autonomic/pending");

export const enqueuePending = (lever: string, params: Record<string, unknown>) =>
  json_post<{ id: string; status: string }>("/api/autonomic/pending", { lever, params });

export const approvePending = (id: string) =>
  json_post<LeverReport>(`/api/autonomic/pending/${id}/approve`);

export const rejectPending = (id: string) =>
  json_post<{ ok: boolean; rejected_id: string }>(`/api/autonomic/pending/${id}/reject`);

export const fetchImmune = () =>
  json_get<{ signatures: ImmuneSignature[] }>("/api/autonomic/immune");

export const toggleKillSwitch = (enabled: boolean) =>
  json_post<{ enabled: boolean }>("/api/autonomic/kill-switch", { enabled });

// Autonomic scheduler settings (currently just tick_interval_seconds).
// Distinct from /api/engine/config because changing the interval is a
// live side-effect on the scheduler, not a CONFIG mutation.
export type AutonomicSettings = {
  effective: { tick_interval_seconds: number | null };
  saved: { tick_interval_seconds?: number };
  range_seconds: { min: number; max: number };
};

export const fetchAutonomicSettings = () =>
  json_get<AutonomicSettings>("/api/autonomic/settings");

export const putAutonomicSettings = (tick_interval_seconds: number) =>
  json_put<{
    ok: boolean;
    applied_live: boolean;
    effective: { tick_interval_seconds: number };
  }>("/api/autonomic/settings", { tick_interval_seconds });


// ---------- Jobs (Phase 15A: durable per-turn records) ----------

export type JobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "interrupted"
  | "cancelled";

export type Job = {
  id: string;
  status: JobStatus;
  channel: string;
  speaker_id: string;
  session_id: string | null;
  prompt: string;
  response: string | null;
  error: string | null;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  completed_at: number | null;
  reply_to: Record<string, any>;
  tool_calls: Array<{
    name: string;
    args_summary?: string;
    ok: boolean;
    error?: string | null;
    elapsed_ms?: number;
  }>;
  attempts: Array<{
    provider_id: string;
    model: string;
    ok: boolean;
    error?: string | null;
    elapsed_ms?: number;
    started_at?: number;
  }>;
  retry_count: number;
  interrupted_count: number;
};

export type JobsListResponse = {
  total: number;       // count matching the same filter as `jobs[]`
  total_all?: number;  // count across all jobs (no filter applied)
  limit: number;
  offset: number;
  jobs: Job[];
};

export const fetchJobs = (params: {
  status?: JobStatus;
  channel?: string;
  speaker_id?: string;
  limit?: number;
  offset?: number;
} = {}) => {
  const q = new URLSearchParams();
  if (params.status) q.set("status", params.status);
  if (params.channel) q.set("channel", params.channel);
  if (params.speaker_id) q.set("speaker_id", params.speaker_id);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.offset !== undefined) q.set("offset", String(params.offset));
  const qs = q.toString();
  return json_get<JobsListResponse>("/api/jobs" + (qs ? `?${qs}` : ""));
};

export const fetchJob = (id: string) =>
  json_get<Job>(`/api/jobs/${encodeURIComponent(id)}`);

export const fetchJobStats = () =>
  json_get<Record<JobStatus, number>>("/api/jobs/_/stats");

export const retryJob = (id: string) =>
  json_post<{ new_job_id: string; prompt: string; channel: string }>(
    `/api/jobs/${encodeURIComponent(id)}/retry`,
    {},
  );

export const cancelJob = (id: string) =>
  json_post<{ ok: boolean; status: string; note?: string }>(
    `/api/jobs/${encodeURIComponent(id)}/cancel`,
    {},
  );

export const deleteJob = (id: string) =>
  json_delete<{ ok: boolean }>(`/api/jobs/${encodeURIComponent(id)}`);

export const cleanupJobs = (max_age_days: number, keep_failed: boolean = true) =>
  json_post<{ ok: boolean; deleted_count: number; deleted: string[] }>(
    `/api/jobs/_/cleanup?max_age_days=${encodeURIComponent(
      String(max_age_days),
    )}&keep_failed=${keep_failed}`,
    {},
  );


// ---------- Failover (Phase 15B: multi-provider chain) ----------

export type FailoverChainEntry = {
  provider_id: string;
  model: string;
};

export type FailoverConfig = {
  enabled: boolean;
  chain: FailoverChainEntry[];
  retry_on: string[];
  max_attempts: number;
};

export type FailoverInfo = {
  config: FailoverConfig;
  categories: string[];
  default_retry_on: string[];
};

export const fetchFailover = () =>
  json_get<FailoverInfo>("/api/failover");

export const putFailover = (cfg: FailoverConfig) =>
  json_put<{ ok: boolean; config: FailoverConfig }>(
    "/api/failover",
    cfg,
  );

export const reorderFailover = (from_index: number, to_index: number) =>
  json_post<{ ok: boolean; config: FailoverConfig }>(
    "/api/failover/reorder",
    { from_index, to_index },
  );

export const toggleFailover = (enabled: boolean) =>
  json_post<{ ok: boolean; config: FailoverConfig }>(
    "/api/failover/toggle",
    { enabled },
  );


// ---------- Consolidation (Phase 16A: daily memory digests) ----------

export type DigestFact = {
  text: string;
  related_topics: string[];
  confidence: number;
  category: string;
  promoted: boolean;
  reason_if_skipped?: string | null;
};

export type ProfileUpdate = {
  speaker_id: string;
  profile_path: string;
  appended_text: string;
  pre_size_bytes: number;
};

export type Digest = {
  date: string;
  started_at: number;
  completed_at: number;
  window_start_ts: number;
  window_end_ts: number;
  speakers_active: string[];
  channels_active: string[];
  turns_analyzed: number;
  jobs_failed: number;
  jobs_interrupted: number;
  narrative: string;
  new_facts: DigestFact[];
  profile_updates: ProfileUpdate[];
  open_threads: string[];
  links_added: Array<Record<string, unknown>>;
  tokens_used: number;
  estimated_cost_usd: number;
  status: "in_progress" | "success" | "partial" | "failed" | "skipped";
  skip_reason?: string | null;
  error?: string | null;
};

export type DigestSummary = {
  date: string;
  status: string;
  narrative_preview: string;
  new_facts_count: number;
  open_threads_count: number;
  turns_analyzed: number;
  speakers_active: string[];
  completed_at: number;
  tokens_used: number;
};

export type ConsolidationStatus = {
  state: {
    last_run_at: number;
    last_run_status: string;
    last_run_digest: string | null;
    last_run_error: string | null;
    last_run_duration_seconds: number;
    last_run_tokens_used: number;
    last_run_cost_usd_estimate: number;
    total_runs: number;
    last_run_speakers_seen: string[];
    last_run_jobs_analyzed: number;
    last_run_facts_added: number;
  };
  would_fire_now: boolean;
  gate_reason: string;
  now: number;
  cooldown_remaining_seconds: number;
  idle_for_seconds: number | null;
  config: {
    idle_threshold_seconds: number;
    cooldown_seconds: number;
    min_jobs_for_run: number;
    tick_seconds: number;
  };
};

export const fetchConsolidationStatus = () =>
  json_get<ConsolidationStatus>("/api/consolidation/status");

export const runConsolidation = (dry_run: boolean = false) =>
  json_post<Digest>(
    `/api/consolidation/run?force=true&dry_run=${dry_run}`,
    {},
  );

export const fetchDigests = (limit: number = 30) =>
  json_get<{ total: number; digests: DigestSummary[] }>(
    `/api/consolidation/digests?limit=${encodeURIComponent(String(limit))}`,
  );

export const fetchDigest = (date_str: string) =>
  json_get<Digest>(`/api/consolidation/digests/${encodeURIComponent(date_str)}`);


// ---------- Memory knowledge graph (Phase 16C) ----------
//
// Distinct from the older `GraphNode`/`GraphStats`/`fetchGraphStats` etc.
// further up in this file — those describe the note-triples graph
// served at `/api/graph/*`. The KGraph types below describe the
// memory consolidation knowledge graph at `/api/kgraph/*`
// (facts ↔ topics ↔ skills ↔ projects ↔ entities). The two graphs
// coexist; eventually we may unify them.

export type KGraphNodeKind = "fact" | "topic" | "skill" | "project" | "entity";
export type KGraphEdgeKind =
  | "is_about"
  | "uses"
  | "mentions"
  | "relates_to"
  | "continues";

export type KGraphNode = {
  id: string;
  kind: KGraphNodeKind;
  label: string;
  weight: number;
  metadata: Record<string, any>;
};

export type KGraphEdge = {
  source: string;
  target: string;
  kind: KGraphEdgeKind;
  weight: number;
  metadata: Record<string, any>;
};

export type KGraphFull = {
  version: number;
  updated_at: number;
  node_count: number;
  edge_count: number;
  nodes: KGraphNode[];
  edges: KGraphEdge[];
};

export type KGraphStats = {
  total_nodes: number;
  total_edges: number;
  by_kind: Record<KGraphNodeKind, number>;
  edge_kinds: Record<string, number>;
  top_topics: Array<{ id: string; label: string; degree: number }>;
  // Phase 16C audit #19: non-null when graph.json couldn't be
  // loaded (corrupt JSON, forward-version schema). The WebUI
  // shows a "rebuild required" banner when set.
  load_error?: string | null;
};

export type KGraphSearchResult = KGraphNode & { degree: number };

export type KGraphNeighborhood = {
  node: KGraphNode;
  neighbors: Array<{
    edge: KGraphEdge;
    node: KGraphNode;
    direction: "in" | "out";
  }>;
  neighbor_count: number;
};

export const fetchKGraphStats = () =>
  json_get<KGraphStats>("/api/kgraph/stats");

export const fetchKGraph = () =>
  json_get<KGraphFull>("/api/kgraph");

export const searchKGraph = (q: string, kind?: KGraphNodeKind, limit: number = 30) => {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (kind) params.set("kind", kind);
  return json_get<{ q: string; kind: string | null; results: KGraphSearchResult[] }>(
    `/api/kgraph/search?${params.toString()}`,
  );
};

export const fetchKGraphNode = (node_id: string) =>
  json_get<KGraphNeighborhood>(
    `/api/kgraph/node/${encodeURIComponent(node_id)}`,
  );

export const rebuildKGraph = () =>
  json_post<{ ok: boolean; stats: Record<string, number> }>(
    "/api/kgraph/rebuild",
    {},
  );


// ---------- Subagents (multi-role pool) ----------

export type SubagentStatus = "running" | "completed" | "failed";

export type SubagentToolCall = {
  name: string;
  args: Record<string, any>;
  result_preview: string;
  is_error: boolean;
};

export type SubagentSession = {
  id: string;
  status: SubagentStatus;
  role: string;
  task: string;
  parent_job_id: string;
  parent_speaker: string;
  created_at: number;
  started_at: number;
  completed_at: number;
  answer: string;
  error: string;
  iterations: number;
  tool_summary: Record<string, number>;
  tool_calls: SubagentToolCall[];
  elapsed_ms: number;
};

export type SubagentListResponse = {
  total: number;
  limit: number;
  offset: number;
  subagents: SubagentSession[];
};

export type SubagentStats = {
  running: number;
  completed: number;
  failed: number;
  total_persisted: number;
};

export const fetchSubagents = (params: {
  status?: SubagentStatus;
  role?: string;
  limit?: number;
  offset?: number;
} = {}) => {
  const q = new URLSearchParams();
  if (params.status) q.set("status", params.status);
  if (params.role) q.set("role", params.role);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.offset !== undefined) q.set("offset", String(params.offset));
  const qs = q.toString();
  return json_get<SubagentListResponse>(
    "/api/subagents" + (qs ? `?${qs}` : ""),
  );
};

export const fetchActiveSubagents = () =>
  json_get<{ count: number; subagents: SubagentSession[] }>(
    "/api/subagents/active",
  );

export const fetchSubagent = (id: string) =>
  json_get<SubagentSession>(`/api/subagents/${encodeURIComponent(id)}`);

export const fetchSubagentStats = () =>
  json_get<SubagentStats>("/api/subagents/_/stats");

export const fetchSubagentRoles = () =>
  json_get<{ roles: Record<string, string> }>("/api/subagents/roles");


// ─── Pipeline Profiles ────────────────────────────────────────────

export type PipelineProfile = {
  id: string;
  name: string;
  description: string;
  created_at: number;
  updated_at: number;
  engine_overrides: Record<string, Record<string, unknown>>;
  reasoning_overrides: {
    routing?: Record<string, string>;
    fallback?: string;
  };
  prompt_overrides: {
    sections?: Record<string, string | null>;
  };
  logging_overrides: {
    root?: string;
    modules?: Record<string, string>;
  };
};

export type PipelineProfileSummary = {
  id: string;
  name: string;
  description: string;
  created_at: number;
  updated_at: number;
};

export type PipelineHistoryEntry = PipelineProfileSummary & {
  timestamp: number;
};

export type SystemPromptSectionsPayload = {
  order: string[];
  sections: Record<string, string>;
};

export async function fetchPipelineProfiles(): Promise<{
  profiles: PipelineProfileSummary[];
}> {
  return json_get("/api/pipeline-profiles");
}

export async function fetchPipelineProfile(id: string): Promise<PipelineProfile> {
  return json_get(`/api/pipeline-profiles/${encodeURIComponent(id)}`);
}

export async function fetchActivePipelineProfile(): Promise<{ active_id: string }> {
  return json_get("/api/pipeline-profiles/active");
}

export async function setActivePipelineProfile(id: string): Promise<{ active_id: string }> {
  return json_put("/api/pipeline-profiles/active", { id });
}

export async function createPipelineProfile(p: PipelineProfile): Promise<PipelineProfile> {
  return json_post("/api/pipeline-profiles", p);
}

export async function updatePipelineProfile(p: PipelineProfile): Promise<PipelineProfile> {
  return json_put(`/api/pipeline-profiles/${encodeURIComponent(p.id)}`, p);
}

export async function deletePipelineProfile(id: string): Promise<{ deleted: string }> {
  return json_delete(`/api/pipeline-profiles/${encodeURIComponent(id)}`);
}

export async function fetchPipelineProfileHistory(id: string): Promise<{
  history: PipelineHistoryEntry[];
}> {
  return json_get(`/api/pipeline-profiles/${encodeURIComponent(id)}/history`);
}

export async function restorePipelineProfile(
  id: string,
  timestamp: number,
): Promise<PipelineProfile> {
  return json_post(`/api/pipeline-profiles/${encodeURIComponent(id)}/restore/${timestamp}`, {});
}

export async function fetchSystemPromptSections(): Promise<SystemPromptSectionsPayload> {
  return json_get("/api/pipeline-profiles/system-prompt-sections");
}
