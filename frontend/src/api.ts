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

export type ThinkingStep = {
  ts: number;
  event: string;
  message: string;
  tokens_so_far: number;
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
};

export type GapEntry = {
  topic: string;
  count: number;
  last: string;
  has_note_now: boolean;
};

export type StreamEvent =
  | { type: "progress"; event: string; message: string }
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
  onEvent: (e: StreamEvent) => void
): Promise<void> {
  return readSSE("/api/chat", { message, project }, onEvent);
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

export const fetchConversation = () =>
  json_get<{ turns: ConversationTurn[]; count: number }>("/api/conversation");

export const clearConversation = () =>
  json_delete<{ ok: boolean }>("/api/conversation");

// ---- Identity ----

export const fetchIdentity = () =>
  json_get<{ soul: string; identity: string; user_profile: string }>("/api/identity");

export const updateIdentity = (file: string, content: string) =>
  json_put<{ ok: boolean }>("/api/identity", { file, content });

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

export type SessionStats = {
  total_sessions: number;
  total_turns: number;
  intents: Record<string, number>;
  daily_counts: Record<string, number>;
  confidence_over_time: { session_id: string; date: string; avg_confidence: number; turns: number }[];
  archived_count: number;
};

export const fetchSessions = (includeArchived = false) =>
  json_get<{ sessions: SessionSummary[]; current_id: string | null }>(
    `/api/sessions?include_archived=${includeArchived}`
  );

export const fetchSessionStats = () =>
  json_get<SessionStats>("/api/sessions/stats");

export const fetchCurrentSession = () =>
  json_get<{ session: SessionDetail | null }>("/api/sessions/current");

export const fetchSession = (id: string) =>
  json_get<{ session: SessionDetail }>(`/api/sessions/${id}`);

export const newSession = () =>
  json_post<{ session: SessionDetail }>("/api/sessions/new");

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
