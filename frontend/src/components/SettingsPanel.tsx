import { Suspense, lazy, useEffect, useState } from "react";
// Audit #26: lazy-load Settings tabs so the initial bundle doesn't
// pay for the kitchen-sink Settings panel on chat-only sessions.
// IdentityEditor + UserProfileTab + StatusTab are tiny and used on
// almost every open of Settings — keep them eager. Everything else
// (memory graph viz, jobs, digests, force-directed layout, etc.)
// loads on demand. The dynamic import() boundary lets Vite split
// each into its own chunk; the gzipped initial JS bundle drops
// from ~170 KB to ~110 KB.
import IdentityEditor from "./settings/IdentityEditor";
import UserProfileTab from "./settings/UserProfileTab";
import StatusTab from "./settings/StatusTab";
const ConversationTab = lazy(() => import("./settings/ConversationTab"));
const CapabilitiesTab = lazy(() => import("./settings/CapabilitiesTab"));
const MemoryTab = lazy(() => import("./settings/MemoryTab"));
const VoiceTab = lazy(() => import("./settings/VoiceTab"));
const EngineTab = lazy(() => import("./settings/EngineTab"));
const SelfModsTab = lazy(() => import("./settings/SelfModsTab"));
const RolesTab = lazy(() => import("./settings/RolesTab"));
const SkillsTab = lazy(() => import("./settings/SkillsTab"));
const JobsTab = lazy(() => import("./settings/JobsTab"));
const FailoverPanel = lazy(() => import("./settings/FailoverPanel"));
const MemoryDigestsTab = lazy(() => import("./settings/MemoryDigestsTab"));
const KnowledgeGraphTab = lazy(() => import("./settings/KnowledgeGraphTab"));
import {
  clearConversation,
  compareModels,
  ConversationTurn,
  fetchCapabilities,
  fetchConversation,
  fetchIdentity,
  fetchIdentityHistory,
  fetchStatus,
  StatusPayload,
  updateIdentity,
  fetchChannels,
  createChannel,
  updateChannel,
  deleteChannel,
  startChannel,
  stopChannel,
  testChannel,
  type ChannelConfig,
  fetchProviders,
  createProvider,
  updateProvider,
  deleteProvider,
  testProvider,
  fetchOllamaModels,
  pullOllamaModel,
  deleteOllamaModel,
  type ProviderConfig,
  type ProviderTypeInfo,
  type OllamaModel,
  updateProviderAuth,
  getOAuthAuthorizeUrl,
  oauthClientCredentials,
  oauthRevoke,
  oauthExchangeUrl,
  oauthManualToken,
  fetchAuthTypes,
  fetchProviderConnectInfo,
  type ProviderConnectInfo,
  fetchCodexStatus,
  type CodexStatus,
  fetchCodexModels,
  type CodexModel,
  fetchCopilotStatus,
  type CopilotStatus,
  fetchActiveModel,
  setActiveModel,
  clearActiveModel,
  type ActiveModelSelection,
  type AvailableModel,
} from "../api";

type IdentityTab = "soul" | "identity" | "user" | "providers" | "channels" | "memory" | "voice" | "engine" | "selfmods" | "roles" | "skills" | "jobs" | "digests" | "kgraph" | "conversation" | "capabilities" | "status";

export default function SettingsPanel() {
  const [tab, setTab] = useState<IdentityTab>("soul");
  const [soul, setSoul] = useState("");
  const [identity, setIdentity] = useState("");
  const [userProfile, setUserProfile] = useState("");
  const [dirty, setDirty] = useState(false);
  const [msg, setMsg] = useState("");
  const [conversation, setConversation] = useState<ConversationTurn[]>([]);
  const [convCount, setConvCount] = useState(0);
  const [capabilities, setCapabilities] = useState("");
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [history, setHistory] = useState<{ timestamp: string; path: string; size: number }[]>([]);
  const [channels, setChannels] = useState<ChannelConfig[]>([]);
  const [showAddChannel, setShowAddChannel] = useState(false);
  const [newChName, setNewChName] = useState("");
  const [newChType, setNewChType] = useState("telegram");
  const [newChToken, setNewChToken] = useState("");
  const [newChAllowed, setNewChAllowed] = useState("");
  const [testResult, setTestResult] = useState<Record<string, string>>({});
  // Providers
  const [providers, setProviders] = useState<ProviderConfig[]>([]);
  const [providerTypes, setProviderTypes] = useState<Record<string, ProviderTypeInfo>>({});
  const [showAddProvider, setShowAddProvider] = useState(false);
  const [newProv, setNewProv] = useState({ name: "", type: "openai", apiKey: "", baseUrl: "", model: "", authMethod: "api_key" as string });
  const [provTestResult, setProvTestResult] = useState<Record<string, string>>({});
  const [activeModelSel, setActiveModelSel] = useState<ActiveModelSelection | null>(null);
  const [availModels, setAvailModels] = useState<AvailableModel[]>([]);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [ollamaPullName, setOllamaPullName] = useState("");
  const [ollamaPulling, setOllamaPulling] = useState(false);
  const [oauthPresets, setOauthPresets] = useState<Record<string, Record<string, string>>>({});
  const [connectInfo, setConnectInfo] = useState<Record<string, ProviderConnectInfo>>({});
  const [wizardStep, setWizardStep] = useState(0);
  const [connectWizard, setConnectWizard] = useState<string | null>(null);
  const [editingAuth, setEditingAuth] = useState<string | null>(null);
  const [authForm, setAuthForm] = useState({
    authType: "api_key",
    clientId: "",
    clientSecret: "",
    tokenUrl: "",
    authorizeUrl: "",
    scope: "",
    grantType: "authorization_code",
  });
  const [oauthPasteUrl, setOauthPasteUrl] = useState("");
  const [oauthManualTok, setOauthManualTok] = useState("");
  const [oauthBusy, setOauthBusy] = useState(false);
  const [codexStatus, setCodexStatus] = useState<CodexStatus | null>(null);
  const [codexModels, setCodexModels] = useState<CodexModel[]>([]);
  const [codexModelsSource, setCodexModelsSource] = useState<string>("");
  const [copilotStatus, setCopilotStatus] = useState<CopilotStatus | null>(null);
  const [awsForm, setAwsForm] = useState({ access_key_id: "", secret_access_key: "", region: "us-east-1" });

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 3000);
  };

  const loadIdentity = async () => {
    try {
      const data = await fetchIdentity();
      setSoul(data.soul);
      setIdentity(data.identity);
      setUserProfile(data.user_profile);
      setDirty(false);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const loadConversation = async () => {
    try {
      const data = await fetchConversation();
      setConversation(data.turns);
      setConvCount(data.count);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const loadCapabilities = async () => {
    try {
      const data = await fetchCapabilities();
      setCapabilities(data.block);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const loadStatus = async () => {
    try {
      setStatus(await fetchStatus());
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const loadHistory = async () => {
    try {
      const data = await fetchIdentityHistory();
      setHistory(data.versions);
    } catch { /* ignore */ }
  };

  const loadChannels = async () => {
    try {
      const data = await fetchChannels();
      setChannels(data.channels || []);
    } catch { /* ignore */ }
  };

  const loadProviders = async () => {
    try {
      const data = await fetchProviders();
      setProviders(data.providers || []);
      if (data.types) setProviderTypes(data.types);
    } catch { /* ignore */ }
    try {
      const at = await fetchAuthTypes();
      if (at.oauth_presets) setOauthPresets(at.oauth_presets);
    } catch { /* ignore */ }
    try {
      const ci = await fetchProviderConnectInfo();
      if (ci.connect_info) setConnectInfo(ci.connect_info);
    } catch { /* ignore */ }
    try {
      const cs = await fetchCodexStatus();
      setCodexStatus(cs);
    } catch { /* ignore */ }
    try {
      const cm = await fetchCodexModels();
      setCodexModels(cm.models || []);
      setCodexModelsSource(cm.source || "");
    } catch { /* ignore */ }
    try {
      const cs = await fetchCopilotStatus();
      setCopilotStatus(cs);
    } catch { /* ignore */ }
    try {
      const am = await fetchActiveModel();
      setAvailModels(am.models || []);
      if (am.active && am.active.provider_id) {
        setActiveModelSel(am.active as ActiveModelSelection);
      } else {
        setActiveModelSel(null);
      }
    } catch { /* ignore */ }
  };

  const loadOllamaModels = async () => {
    try {
      const data = await fetchOllamaModels();
      if (data.ok) setOllamaModels(data.models || []);
    } catch { /* ignore */ }
  };

  useEffect(() => {
    loadIdentity();
    loadConversation();
    loadCapabilities();
    loadStatus();
    loadHistory();
    loadChannels();
    loadProviders();
    loadOllamaModels();
  }, []);

  const handleSave = async (file: string, content: string) => {
    try {
      await updateIdentity(file, content);
      setDirty(false);
      flash(`Saved ${file}.md`);
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleClearConversation = async () => {
    if (!confirm("Clear all conversation history?")) return;
    try {
      await clearConversation();
      setConversation([]);
      setConvCount(0);
      flash("Conversation cleared");
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const handleCompare = async () => {
    try {
      const res = await compareModels();
      flash(
        `Compare: old=${res.old_score?.toFixed(3)}, new=${res.new_score?.toFixed(3)}, improvement=${res.improvement?.toFixed(3)}`
      );
    } catch (e: any) {
      flash("Error: " + e.message);
    }
  };

  const TABS: { id: IdentityTab; label: string }[] = [
    { id: "soul", label: "Soul" },
    { id: "identity", label: "Identity" },
    { id: "user", label: "User Profile" },
    { id: "providers", label: "Providers" },
    { id: "channels", label: "Channels" },
    { id: "memory", label: "Memory" },
    { id: "voice", label: "Voice" },
    { id: "engine", label: "Engine" },
    { id: "selfmods", label: "Self-Modifications" },
    { id: "roles", label: "Roles & Contacts" },
    { id: "skills", label: "Skills" },
    { id: "jobs", label: "Jobs" },
    { id: "digests", label: "Memory Digests" },
    { id: "kgraph", label: "Knowledge Graph" },
    { id: "conversation", label: "Conversation" },
    { id: "capabilities", label: "Capabilities" },
    { id: "status", label: "System Status" },
  ];

  const editorProps = (file: string, value: string, setter: (v: string) => void) => ({
    file, value, setter, dirty,
    onSave: handleSave, onReload: loadIdentity, setDirty,
  });

  return (
    <div className="flex flex-1 min-h-0">
      {/* Sidebar */}
      <aside className="w-48 border-r border-slate-800 bg-slate-950/60 p-3 text-sm space-y-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`w-full text-left px-2 py-1.5 rounded transition-colors ${
              tab === t.id
                ? "bg-sky-700 text-white"
                : "bg-slate-800 hover:bg-slate-700 text-slate-300"
            }`}
          >
            {t.label}
          </button>
        ))}
        {msg && <div className="mt-4 bg-sky-900/50 text-xs rounded p-2">{msg}</div>}
      </aside>

      {/* Content. Lazy-loaded tabs render under a Suspense
          boundary; the fallback shows a tiny placeholder so the
          panel doesn't blank-flash while the chunk fetches. */}
      <div className="flex-1 p-4 overflow-y-auto flex flex-col">
        <Suspense fallback={
          <div className="text-slate-500 text-sm italic">Loading…</div>
        }>
        {tab === "soul" && <IdentityEditor {...editorProps("soul", soul, setSoul)} />}
        {tab === "identity" && <IdentityEditor {...editorProps("identity", identity, setIdentity)} />}
        {tab === "user" && (
          <UserProfileTab
            userProfile={userProfile}
            setUserProfile={setUserProfile}
            dirty={dirty}
            onSave={handleSave}
            onReload={loadIdentity}
            setDirty={setDirty}
            history={history}
          />
        )}

        {tab === "conversation" && (
          <ConversationTab
            conversation={conversation}
            convCount={convCount}
            onRefresh={loadConversation}
            onClear={handleClearConversation}
          />
        )}

        {tab === "capabilities" && (
          <CapabilitiesTab capabilities={capabilities} onRefresh={loadCapabilities} />
        )}

        {tab === "memory" && <MemoryTab flash={flash} />}

        {tab === "voice" && <VoiceTab flash={flash} />}

        {tab === "engine" && <EngineTab flash={flash} />}

        {tab === "selfmods" && <SelfModsTab flash={flash} />}

        {tab === "roles" && <RolesTab flash={flash} />}

        {tab === "skills" && <SkillsTab flash={flash} />}

        {tab === "jobs" && <JobsTab flash={flash} />}

        {tab === "digests" && <MemoryDigestsTab flash={flash} />}

        {tab === "kgraph" && <KnowledgeGraphTab flash={flash} />}

        {tab === "providers" && (
          <div className="space-y-4">
            {/* Active Model Selector */}
            <div className="bg-slate-800 rounded p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold">Active Model</h4>
                {activeModelSel && (
                  <button
                    onClick={async () => {
                      await clearActiveModel();
                      setActiveModelSel(null);
                      flash("Reset to default model");
                    }}
                    className="text-[10px] bg-slate-700 hover:bg-slate-600 rounded px-2 py-0.5"
                  >
                    Reset to default
                  </button>
                )}
              </div>
              <select
                className="w-full bg-slate-900 rounded px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-sky-600"
                value={activeModelSel ? `${activeModelSel.provider_id}:${activeModelSel.model}` : "__default__"}
                onChange={async (e) => {
                  const val = e.target.value;
                  if (val === "__default__") {
                    await clearActiveModel();
                    setActiveModelSel(null);
                    flash("Using default model");
                  } else {
                    const [pid, ...rest] = val.split(":");
                    const model = rest.join(":");
                    try {
                      const res = await setActiveModel(pid, model);
                      setActiveModelSel(res.active);
                      flash(`Switched to ${res.active.provider_name} / ${res.active.model}`);
                    } catch (err: any) {
                      flash("Error: " + err.message);
                    }
                  }
                }}
              >
                <option value="__default__">Default (from config)</option>
                {(() => {
                  const grouped: Record<string, AvailableModel[]> = {};
                  for (const m of availModels) {
                    if (!grouped[m.provider_name]) grouped[m.provider_name] = [];
                    grouped[m.provider_name].push(m);
                  }
                  return Object.entries(grouped).map(([provName, models]) => (
                    <optgroup key={provName} label={provName}>
                      {models.map((m) => (
                        <option key={`${m.provider_id}:${m.model}`} value={`${m.provider_id}:${m.model}`}>
                          {m.model}
                        </option>
                      ))}
                    </optgroup>
                  ));
                })()}
              </select>
              <div className="text-[10px] text-slate-500 mt-1.5">
                Model changes take effect immediately — no restart needed.
              </div>
            </div>

            {/* Failover chain — what to try when the active model returns
                a retryable error (rate limit / 5xx / timeout / auth). */}
            <FailoverPanel
              flash={flash}
              providers={providers}
              availableModels={availModels}
            />

            <div className="flex justify-between items-center">
              <h3 className="font-bold">LLM Providers</h3>
              <div className="flex gap-2">
                <button onClick={loadProviders} className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs">
                  Refresh
                </button>
                <button
                  onClick={() => setShowAddProvider(!showAddProvider)}
                  className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs"
                >
                  + Add Provider
                </button>
              </div>
            </div>

            {showAddProvider && (
              <div className="bg-slate-800 rounded p-4 space-y-4">
                <h4 className="text-sm font-semibold">Connect Provider</h4>

                {/* Step 1: Choose provider type */}
                <div>
                  <label className="text-xs text-slate-400 block mb-2">Choose provider</label>
                  <div className="grid grid-cols-3 gap-2">
                    {Object.entries(providerTypes).map(([k, v]) => (
                      <button
                        key={k}
                        onClick={() => {
                          setNewProv({
                            ...newProv,
                            type: k,
                            name: v.label,
                            baseUrl: v.base_url || "",
                            model: v.models?.[0] || "",
                            authMethod: v.auth_types?.[0] || "api_key",
                          });
                          setWizardStep(1);
                        }}
                        className={`text-left p-2.5 rounded border text-xs transition-colors ${
                          newProv.type === k && wizardStep >= 1
                            ? "border-sky-500 bg-sky-900/30"
                            : "border-slate-700 bg-slate-900 hover:border-slate-500"
                        }`}
                      >
                        <div className="font-semibold text-slate-200">{v.label}</div>
                        <div className="text-slate-500 text-[10px] mt-0.5">
                          {v.models?.slice(0, 2).join(", ") || (k === "ollama" ? "Local models" : "Custom")}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Step 2: Connection wizard */}
                {wizardStep >= 1 && (() => {
                  const ci = connectInfo[newProv.type];
                  const ptype = providerTypes[newProv.type];
                  const isOllama = newProv.type === "ollama";
                  const needsBaseUrl = connectInfo[newProv.type]?.extra_fields?.includes("base_url") || false;
                  return (
                    <div className="bg-slate-900 rounded p-4 space-y-3">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="w-6 h-6 rounded-full bg-sky-700 flex items-center justify-center text-xs font-bold">1</div>
                        <span className="text-sm font-medium">
                          {isOllama ? "Configure Ollama" : "Get API Key"}
                        </span>
                      </div>

                      {ci && !isOllama && (
                        <div className="space-y-2">
                          {ci.key_url && (
                            <a
                              href={ci.key_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-2 bg-sky-700 hover:bg-sky-600 text-white rounded px-3 py-1.5 text-xs font-medium transition-colors"
                            >
                              Open {ptype?.label || newProv.type} Dashboard
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                              </svg>
                            </a>
                          )}
                          <div className="text-[11px] text-slate-400 whitespace-pre-wrap leading-relaxed pl-1">
                            {ci.key_instructions}
                          </div>
                          {ci.docs_url && (
                            <a href={ci.docs_url} target="_blank" rel="noopener noreferrer"
                              className="text-[10px] text-sky-400 hover:text-sky-300 underline">
                              Documentation
                            </a>
                          )}
                        </div>
                      )}

                      {isOllama && (
                        <div className="text-[11px] text-slate-400 leading-relaxed pl-1">
                          No API key needed. Make sure Ollama is running locally.
                        </div>
                      )}

                      <div className="flex items-center gap-2 mt-3">
                        <div className="w-6 h-6 rounded-full bg-sky-700 flex items-center justify-center text-xs font-bold">2</div>
                        <span className="text-sm font-medium">Connect</span>
                      </div>

                      <div className="space-y-2">
                        <div>
                          <label className="text-xs text-slate-400 block mb-1">Name</label>
                          <input
                            className="w-full bg-slate-800 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
                            placeholder={ptype?.label || "My Provider"}
                            value={newProv.name}
                            onChange={(e) => setNewProv({ ...newProv, name: e.target.value })}
                          />
                        </div>

                        {/* Auth method tabs */}
                        {!isOllama && (ptype?.auth_types?.length ?? 0) > 0 && (
                          <div>
                            <label className="text-xs text-slate-400 block mb-1">Auth Method</label>
                            <div className="flex gap-1">
                              {(ptype?.auth_types || ["api_key"]).map((at: string) => (
                                <button
                                  key={at}
                                  onClick={() => setNewProv({ ...newProv, authMethod: at })}
                                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                                    newProv.authMethod === at
                                      ? "bg-sky-700 text-white"
                                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                                  }`}
                                >
                                  {at === "api_key"
                                    ? "API Key"
                                    : at === "oauth"
                                    ? "OAuth"
                                    : at === "codex_subscription"
                                    ? "Codex Login"
                                    : at === "copilot_subscription"
                                    ? "Copilot Login"
                                    : at === "aws_credentials"
                                    ? "AWS Credentials"
                                    : "No Auth"}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* API Key fields */}
                        {!isOllama && newProv.authMethod === "api_key" && (
                          <div>
                            <label className="text-xs text-slate-400 block mb-1">API Key</label>
                            <input
                              className="w-full bg-slate-800 rounded px-2 py-1.5 text-sm font-mono outline-none focus:ring-1 focus:ring-sky-600"
                              placeholder="Paste your API key here..."
                              type="password"
                              value={newProv.apiKey}
                              onChange={(e) => setNewProv({ ...newProv, apiKey: e.target.value })}
                            />
                          </div>
                        )}

                        {/* OAuth fields */}
                        {!isOllama && newProv.authMethod === "oauth" && (
                          <div className="space-y-3 bg-slate-800/50 rounded p-3">
                            {/* Preset indicator */}
                            {oauthPresets[newProv.type] && (
                              <div className="flex items-center gap-2 bg-emerald-900/30 border border-emerald-800/50 rounded px-2.5 py-1.5">
                                <svg className="w-4 h-4 text-emerald-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                                <span className="text-xs text-emerald-300">{ptype?.label} OAuth preset loaded (PKCE)</span>
                              </div>
                            )}

                            {/* Step 1: Open auth page */}
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <div className="w-5 h-5 rounded-full bg-violet-700 flex items-center justify-center text-[10px] font-bold flex-shrink-0">1</div>
                                <span className="text-xs font-medium text-slate-200">Open authorization page</span>
                              </div>
                              <p className="text-[10px] text-slate-500 ml-7">
                                Click the button to open the sign-in page. After you authorize, <strong className="text-slate-300">copy the URL</strong> from the browser address bar before the page closes.
                              </p>
                              <button
                                disabled={oauthBusy || (!oauthPresets[newProv.type] && !authForm.authorizeUrl)}
                                onClick={async () => {
                                  setOauthBusy(true);
                                  try {
                                    const provId = newProv.type + "_oauth";
                                    const preset = oauthPresets[newProv.type];
                                    const oauthCfg: Record<string, string> = {};
                                    if (preset) {
                                      Object.assign(oauthCfg, preset);
                                    } else {
                                      if (authForm.authorizeUrl) oauthCfg.authorize_url = authForm.authorizeUrl;
                                      if (authForm.tokenUrl) oauthCfg.token_url = authForm.tokenUrl;
                                      if (authForm.clientId) oauthCfg.client_id = authForm.clientId;
                                      if (authForm.clientSecret) oauthCfg.client_secret = authForm.clientSecret;
                                      if (authForm.scope) oauthCfg.scope = authForm.scope;
                                    }
                                    try {
                                      await createProvider({
                                        id: provId,
                                        name: newProv.name || ptype?.label || provId,
                                        type: newProv.type,
                                        enabled: true,
                                        base_url: newProv.baseUrl || undefined,
                                        default_model: newProv.model || undefined,
                                        auth_type: "oauth",
                                        oauth: oauthCfg,
                                      });
                                    } catch { /* provider may already exist */ }
                                    const res = await getOAuthAuthorizeUrl(provId);
                                    if (res.url) {
                                      window.open(res.url, "_blank", "width=700,height=700");
                                      flash("Authorization page opened — sign in, then copy the redirect URL");
                                    } else {
                                      flash("Failed to generate authorize URL");
                                    }
                                    const plist = await fetchProviders();
                                    setProviders(plist.providers || []);
                                  } catch (e: any) {
                                    flash("Error: " + (e.message || String(e)));
                                  } finally {
                                    setOauthBusy(false);
                                  }
                                }}
                                className="ml-7 bg-violet-700 hover:bg-violet-600 disabled:opacity-40 rounded px-4 py-2 text-xs font-medium flex items-center gap-2"
                              >
                                {oauthBusy ? (
                                  <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                  </svg>
                                ) : (
                                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                  </svg>
                                )}
                                Authorize
                              </button>
                            </div>

                            {/* Step 2: Paste redirect URL */}
                            <div className="space-y-2 border-t border-slate-700 pt-3">
                              <div className="flex items-center gap-2">
                                <div className="w-5 h-5 rounded-full bg-violet-700 flex items-center justify-center text-[10px] font-bold flex-shrink-0">2</div>
                                <span className="text-xs font-medium text-slate-200">Paste redirect URL</span>
                              </div>
                              <p className="text-[10px] text-slate-500 ml-7">
                                After signing in, copy the full URL from the browser address bar and paste it here:
                              </p>
                              <div className="ml-7">
                                <div className="flex gap-1">
                                  <input
                                    className="flex-1 bg-slate-800 rounded px-2 py-1.5 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                                    placeholder="http://localhost:1455/auth/callback?code=abc123&state=..."
                                    value={oauthPasteUrl}
                                    onChange={(e) => setOauthPasteUrl(e.target.value)}
                                  />
                                  <button
                                    disabled={!oauthPasteUrl || oauthBusy}
                                    onClick={async () => {
                                      setOauthBusy(true);
                                      try {
                                        const provId = newProv.type + "_oauth";
                                        const res = await oauthExchangeUrl(provId, oauthPasteUrl);
                                        if (res.ok) {
                                          flash("Connected! Token received.");
                                          setOauthPasteUrl("");
                                          const plist = await fetchProviders();
                                          setProviders(plist.providers || []);
                                          setWizardStep(0);
                                        } else {
                                          flash("Exchange failed: " + JSON.stringify(res));
                                        }
                                      } catch (e: any) {
                                        flash("Error: " + (e.message || String(e)));
                                      } finally {
                                        setOauthBusy(false);
                                      }
                                    }}
                                    className="bg-violet-700 hover:bg-violet-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs font-medium"
                                  >Connect</button>
                                </div>
                              </div>
                            </div>

                            {/* Advanced: custom OAuth endpoints for non-preset providers */}
                            {!oauthPresets[newProv.type] && (
                              <details className="border-t border-slate-700 pt-2">
                                <summary className="text-[10px] text-slate-500 cursor-pointer hover:text-slate-400">
                                  OAuth endpoint settings
                                </summary>
                                <div className="mt-2 space-y-2">
                                  <div>
                                    <label className="text-[10px] text-slate-500 block mb-0.5">Authorize URL</label>
                                    <input className="w-full bg-slate-800 rounded px-2 py-1 text-xs outline-none focus:ring-1 focus:ring-sky-600"
                                      placeholder="https://provider.com/oauth/authorize"
                                      value={authForm.authorizeUrl} onChange={(e) => setAuthForm({ ...authForm, authorizeUrl: e.target.value })} />
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-slate-500 block mb-0.5">Token URL</label>
                                    <input className="w-full bg-slate-800 rounded px-2 py-1 text-xs outline-none"
                                      placeholder="https://provider.com/oauth/token"
                                      value={authForm.tokenUrl} onChange={(e) => setAuthForm({ ...authForm, tokenUrl: e.target.value })} />
                                  </div>
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <label className="text-[10px] text-slate-500 block mb-0.5">Client ID</label>
                                      <input className="w-full bg-slate-800 rounded px-2 py-1 text-xs outline-none"
                                        value={authForm.clientId} onChange={(e) => setAuthForm({ ...authForm, clientId: e.target.value })} />
                                    </div>
                                    <div>
                                      <label className="text-[10px] text-slate-500 block mb-0.5">Client Secret</label>
                                      <input className="w-full bg-slate-800 rounded px-2 py-1 text-xs outline-none" type="password"
                                        value={authForm.clientSecret} onChange={(e) => setAuthForm({ ...authForm, clientSecret: e.target.value })} />
                                    </div>
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-slate-500 block mb-0.5">Scope</label>
                                    <input className="w-full bg-slate-800 rounded px-2 py-1 text-xs outline-none"
                                      value={authForm.scope} onChange={(e) => setAuthForm({ ...authForm, scope: e.target.value })} />
                                  </div>
                                </div>
                              </details>
                            )}
                          </div>
                        )}

                        {/* Codex Subscription block — uses ~/.codex/auth.json */}
                        {newProv.authMethod === "codex_subscription" && (
                          <div className="space-y-2 bg-slate-800/50 rounded p-3">
                            {codexStatus?.logged_in ? (
                              <div className="space-y-1 text-xs">
                                <div className="flex items-center gap-2">
                                  <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
                                  <span className="text-emerald-300 font-semibold">
                                    Codex login detected
                                  </span>
                                </div>
                                <div className="text-slate-300">
                                  <span className="text-slate-500">email:</span> {codexStatus.email || "—"}
                                </div>
                                <div className="text-slate-300">
                                  <span className="text-slate-500">plan:</span>{" "}
                                  <span className="font-mono">{codexStatus.plan_type || "—"}</span>
                                </div>
                                <div className="text-slate-300">
                                  <span className="text-slate-500">expires:</span>{" "}
                                  {codexStatus.expires_at || "—"}
                                  {codexStatus.expired ? (
                                    <span className="ml-2 text-rose-400">(expired — will auto-refresh)</span>
                                  ) : null}
                                </div>
                                <div className="text-[10px] text-slate-500 pt-1">
                                  Auth file: <span className="font-mono">~/.codex/auth.json</span>. Token is read on each request and refreshed automatically.
                                </div>
                              </div>
                            ) : (
                              <div className="space-y-2 text-xs">
                                <div className="flex items-center gap-2">
                                  <span className="inline-block w-2 h-2 rounded-full bg-amber-400" />
                                  <span className="text-amber-300 font-semibold">
                                    No Codex login found
                                  </span>
                                </div>
                                <ol className="list-decimal ml-5 text-slate-400 space-y-0.5">
                                  <li>Install Codex CLI: <a href="https://github.com/openai/codex" target="_blank" rel="noreferrer" className="text-sky-400 underline">github.com/openai/codex</a></li>
                                  <li>Run <span className="font-mono bg-slate-900 px-1 rounded">codex login</span> and complete the browser sign-in</li>
                                  <li>Click Refresh below</li>
                                </ol>
                                {codexStatus?.reason ? (
                                  <div className="text-[10px] text-slate-500">
                                    Reason: <span className="font-mono">{codexStatus.reason}</span>
                                  </div>
                                ) : null}
                              </div>
                            )}
                            <button
                              onClick={async () => {
                                try {
                                  const [cs, cm] = await Promise.all([
                                    fetchCodexStatus(),
                                    fetchCodexModels(),
                                  ]);
                                  setCodexStatus(cs);
                                  setCodexModels(cm.models || []);
                                  setCodexModelsSource(cm.source || "");
                                  flash(
                                    cs.logged_in
                                      ? `Codex login OK — ${cm.models?.length ?? 0} model(s) loaded`
                                      : "Still no Codex login",
                                  );
                                } catch (e: any) {
                                  flash("Error: " + (e.message || String(e)));
                                }
                              }}
                              className="bg-slate-700 hover:bg-slate-600 rounded px-2 py-1 text-[10px]"
                            >
                              Refresh
                            </button>
                          </div>
                        )}

                        {/* GitHub Copilot Subscription block */}
                        {newProv.authMethod === "copilot_subscription" && (
                          <div className="space-y-2 bg-slate-800/50 rounded p-3">
                            {copilotStatus?.logged_in ? (
                              <div className="space-y-1 text-xs">
                                <div className="flex items-center gap-2">
                                  <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
                                  <span className="text-emerald-300 font-semibold">
                                    GitHub Copilot login detected
                                  </span>
                                </div>
                                <div className="text-slate-300">
                                  <span className="text-slate-500">user:</span>{" "}
                                  {copilotStatus.user || "—"}
                                </div>
                                <div className="text-slate-300">
                                  <span className="text-slate-500">source:</span>{" "}
                                  <span className="font-mono text-[10px] break-all">{copilotStatus.source || "—"}</span>
                                </div>
                                {copilotStatus.bearer_cached && (
                                  <div className="text-slate-300">
                                    <span className="text-slate-500">bearer expires:</span>{" "}
                                    {copilotStatus.bearer_expires_at || "—"}
                                  </div>
                                )}
                                <div className="text-[10px] text-slate-500 pt-1">
                                  Persistent oauth_token stays in your editor's auth file. We exchange it for a short-lived bearer per request.
                                </div>
                              </div>
                            ) : (
                              <div className="space-y-2 text-xs">
                                <div className="flex items-center gap-2">
                                  <span className="inline-block w-2 h-2 rounded-full bg-amber-400" />
                                  <span className="text-amber-300 font-semibold">
                                    No GitHub Copilot login found
                                  </span>
                                </div>
                                <ol className="list-decimal ml-5 text-slate-400 space-y-0.5">
                                  <li>Sign in to Copilot in VS Code (Copilot extension), JetBrains, or run <span className="font-mono bg-slate-900 px-1 rounded">gh auth login</span> with copilot scope.</li>
                                  <li>Click Refresh below.</li>
                                </ol>
                                {copilotStatus?.checked_paths && copilotStatus.checked_paths.length > 0 && (
                                  <details className="text-[10px] text-slate-500">
                                    <summary className="cursor-pointer">checked paths</summary>
                                    <ul className="mt-1 ml-3 font-mono">
                                      {copilotStatus.checked_paths.map((p) => (
                                        <li key={p} className="break-all">{p}</li>
                                      ))}
                                    </ul>
                                  </details>
                                )}
                              </div>
                            )}
                            <button
                              onClick={async () => {
                                try {
                                  const cs = await fetchCopilotStatus();
                                  setCopilotStatus(cs);
                                  flash(cs.logged_in ? `Copilot login OK (${cs.user || ""})` : "Still no Copilot login");
                                } catch (e: any) {
                                  flash("Error: " + (e.message || String(e)));
                                }
                              }}
                              className="bg-slate-700 hover:bg-slate-600 rounded px-2 py-1 text-[10px]"
                            >
                              Refresh
                            </button>
                          </div>
                        )}

                        {/* AWS Bedrock credentials block */}
                        {newProv.authMethod === "aws_credentials" && (
                          <div className="space-y-2 bg-slate-800/50 rounded p-3">
                            <div className="text-[10px] text-slate-400">
                              Stored in <span className="font-mono">knowledge/providers.json</span> (gitignored). Requires <span className="font-mono">pip install boto3</span> in the venv.
                            </div>
                            <div>
                              <label className="text-[10px] text-slate-500 block mb-0.5">Access Key ID</label>
                              <input
                                className="w-full bg-slate-800 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                                placeholder="AKIA..."
                                value={awsForm.access_key_id}
                                onChange={(e) => setAwsForm({ ...awsForm, access_key_id: e.target.value })}
                              />
                            </div>
                            <div>
                              <label className="text-[10px] text-slate-500 block mb-0.5">Secret Access Key</label>
                              <input
                                type="password"
                                className="w-full bg-slate-800 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                                placeholder="..."
                                value={awsForm.secret_access_key}
                                onChange={(e) => setAwsForm({ ...awsForm, secret_access_key: e.target.value })}
                              />
                            </div>
                            <div>
                              <label className="text-[10px] text-slate-500 block mb-0.5">Region</label>
                              <input
                                className="w-full bg-slate-800 rounded px-2 py-1 text-xs font-mono outline-none focus:ring-1 focus:ring-sky-600"
                                placeholder="us-east-1"
                                value={awsForm.region}
                                onChange={(e) => setAwsForm({ ...awsForm, region: e.target.value })}
                              />
                            </div>
                          </div>
                        )}

                        {needsBaseUrl && (
                          <div>
                            <label className="text-xs text-slate-400 block mb-1">Base URL</label>
                            <input
                              className="w-full bg-slate-800 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
                              placeholder={isOllama ? "http://localhost:11434" : "https://api.example.com/v1"}
                              value={newProv.baseUrl}
                              onChange={(e) => setNewProv({ ...newProv, baseUrl: e.target.value })}
                            />
                          </div>
                        )}
                        <div>
                          <label className="text-xs text-slate-400 block mb-1">Default Model</label>
                          <input
                            className="w-full bg-slate-800 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
                            placeholder={
                              newProv.type === "openai_codex" && codexModels[0]
                                ? codexModels[0].slug
                                : ptype?.models?.[0] || "model-name"
                            }
                            value={newProv.model}
                            onChange={(e) => setNewProv({ ...newProv, model: e.target.value })}
                          />
                          {newProv.type === "openai_codex" ? (
                            codexModels.length > 0 ? (
                              <>
                                <div className="flex gap-1 mt-1 flex-wrap">
                                  {codexModels.map((m) => (
                                    <button
                                      key={m.slug}
                                      onClick={() => setNewProv({ ...newProv, model: m.slug })}
                                      title={m.description || m.display_name}
                                      className={`text-[10px] rounded px-1.5 py-0.5 ${
                                        newProv.model === m.slug
                                          ? "bg-sky-700 text-white"
                                          : "bg-slate-700 hover:bg-slate-600"
                                      }`}
                                    >
                                      {m.display_name}
                                    </button>
                                  ))}
                                </div>
                                <div className="text-[10px] text-slate-500 mt-1">
                                  {codexModelsSource === "cache_file"
                                    ? `Loaded from ~/.codex/models_cache.json (${codexModels.length} models)`
                                    : `Using fallback list (${codexModels.length} models) — run any \`codex\` command to populate ~/.codex/models_cache.json`}
                                </div>
                              </>
                            ) : (
                              <div className="text-[10px] text-amber-400 mt-1">
                                No Codex models found — run <span className="font-mono">codex login</span>, then any <span className="font-mono">codex</span> command, then click Refresh below.
                              </div>
                            )
                          ) : (
                            ptype?.models?.length > 0 && (
                              <div className="flex gap-1 mt-1 flex-wrap">
                                {ptype.models.map((m: string) => (
                                  <button
                                    key={m}
                                    onClick={() => setNewProv({ ...newProv, model: m })}
                                    className={`text-[10px] rounded px-1.5 py-0.5 ${
                                      newProv.model === m
                                        ? "bg-sky-700 text-white"
                                        : "bg-slate-700 hover:bg-slate-600"
                                    }`}
                                  >
                                    {m}
                                  </button>
                                ))}
                              </div>
                            )
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2 mt-3">
                        <div className="w-6 h-6 rounded-full bg-sky-700 flex items-center justify-center text-xs font-bold">3</div>
                        <span className="text-sm font-medium">Connect & Test</span>
                      </div>

                      {provTestResult["__new__"] && (
                        <div className={`text-xs rounded px-2 py-1.5 ${
                          provTestResult["__new__"]?.startsWith("Connected")
                            ? "bg-emerald-900/40 text-emerald-300"
                            : "bg-rose-900/40 text-rose-300"
                        }`}>
                          {provTestResult["__new__"]}
                        </div>
                      )}

                      <div className="flex gap-2 pt-1">
                        <button
                          onClick={async () => {
                            const name = newProv.name.trim() || ptype?.label || newProv.type;
                            const id = name.toLowerCase().replace(/[^a-z0-9]+/g, "-");
                            const useOAuth = newProv.authMethod === "oauth";
                            try {
                              const isAws = newProv.authMethod === "aws_credentials";
                              await createProvider({
                                id,
                                name,
                                type: newProv.type,
                                enabled: true,
                                api_key: useOAuth || isAws ? "" : newProv.apiKey.trim(),
                                api_key_env: ptype?.key_env_default || "",
                                base_url: newProv.baseUrl.trim() || ptype?.base_url || "",
                                models: newProv.model ? [newProv.model] : ptype?.models || [],
                                default_model: newProv.model || ptype?.models?.[0] || "",
                                auth_type: newProv.authMethod,
                                ...(useOAuth ? {
                                  oauth: {
                                    client_id: authForm.clientId,
                                    client_secret: authForm.clientSecret,
                                    authorize_url: authForm.authorizeUrl,
                                    token_url: authForm.tokenUrl,
                                    scope: authForm.scope,
                                    grant_type: authForm.grantType,
                                  },
                                } : {}),
                                ...(isAws ? {
                                  aws: {
                                    access_key_id: awsForm.access_key_id.trim(),
                                    secret_access_key: awsForm.secret_access_key.trim(),
                                    region: awsForm.region.trim() || "us-east-1",
                                  },
                                } : {}),
                              });

                              // If OAuth with pasted redirect URL — exchange code
                              if (useOAuth && oauthPasteUrl.trim()) {
                                try {
                                  await oauthExchangeUrl(id, oauthPasteUrl.trim());
                                } catch { /* ignore */ }
                              }
                              // If OAuth with manual token — save it
                              else if (useOAuth && oauthManualTok.trim()) {
                                try {
                                  await oauthManualToken(id, oauthManualTok.trim());
                                } catch { /* ignore */ }
                              }

                              const hasToken = oauthPasteUrl.trim() || oauthManualTok.trim();

                              // Auto-test (skip if OAuth without token yet)
                              if (!useOAuth || hasToken) {
                                try {
                                  const res = await testProvider(id);
                                  if (res.ok) {
                                    const extra = res.models?.length ? ` (${res.models.slice(0, 3).join(", ")})` : "";
                                    setProvTestResult((r) => ({ ...r, "__new__": `Connected${extra}` }));
                                  } else {
                                    setProvTestResult((r) => ({ ...r, "__new__": `Created, but test failed: ${res.error}` }));
                                  }
                                } catch { /* test failed, provider still created */ }
                              }

                              setShowAddProvider(false);
                              setWizardStep(0);
                              setNewProv({ name: "", type: "openai", apiKey: "", baseUrl: "", model: "", authMethod: "api_key" });
                              setOauthManualTok("");
                              setOauthPasteUrl("");
                              setProvTestResult((r) => { const n = { ...r }; delete n["__new__"]; return n; });
                              loadProviders();
                              flash("Provider connected!");
                            } catch (e: any) {
                              flash("Error: " + e.message);
                            }
                          }}
                          className="bg-emerald-700 hover:bg-emerald-600 rounded px-4 py-1.5 text-xs font-medium"
                        >
                          Connect
                        </button>
                        <button
                          onClick={() => {
                            setShowAddProvider(false);
                            setWizardStep(0);
                            setNewProv({ name: "", type: "openai", apiKey: "", baseUrl: "", model: "", authMethod: "api_key" });
                            setOauthManualTok("");
                            setOauthPasteUrl("");
                            setProvTestResult((r) => { const n = { ...r }; delete n["__new__"]; return n; });
                          }}
                          className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1.5 text-xs"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* Provider cards */}
            <div className="space-y-3">
              {providers.map((p) => (
                <div key={p.id} className="bg-slate-800 rounded p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div>
                        <div className="font-semibold text-sm flex items-center gap-2">
                          {p.name}
                          {p.is_default && (
                            <span className="text-[10px] bg-sky-900/60 text-sky-300 rounded px-1.5 py-0.5">DEFAULT</span>
                          )}
                        </div>
                        <div className="text-xs text-slate-500">
                          {providerTypes[p.type]?.label || p.type} · {p.default_model || p.models?.[0] || "no model"}
                        </div>
                      </div>
                    </div>
                    <label className="flex items-center gap-1 text-xs">
                      <input
                        type="checkbox"
                        checked={p.enabled}
                        onChange={async () => {
                          try {
                            await updateProvider(p.id, { enabled: !p.enabled });
                            loadProviders();
                          } catch (e: any) {
                            flash("Error: " + e.message);
                          }
                        }}
                        className="rounded"
                      />
                      Enabled
                    </label>
                  </div>

                  <div className="text-xs text-slate-500 mb-2 space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span>Auth: <b className="text-slate-300">{p.auth_type || "api_key"}</b></span>
                      {(p.auth_type || "api_key") === "api_key" && (
                        <span>· Key: {p.api_key_masked || "(not set)"}</span>
                      )}
                      {p.auth_type === "oauth" && p.oauth_status && (
                        <span className={p.oauth_status.authenticated && !p.oauth_status.expired
                          ? "text-emerald-400" : "text-rose-400"}>
                          · {p.oauth_status.authenticated
                            ? (p.oauth_status.expired ? "Token expired" : "Connected")
                            : "Not authenticated"}
                        </span>
                      )}
                    </div>
                    {p.base_url && <div>URL: {p.base_url}</div>}
                    {p.models?.length > 0 && <div>Models: {p.models.join(", ")}</div>}
                  </div>

                  {/* Auth config panel */}
                  {editingAuth === p.id && (() => {
                    const ci = connectInfo[p.type];
                    const supportsOAuth = (providerTypes[p.type]?.auth_types || []).includes("oauth");
                    return (
                      <div className="bg-slate-900 rounded p-3 mb-2 space-y-3 text-xs">
                        {/* Quick connect: get API key link */}
                        {ci && ci.key_url && (
                          <div className="space-y-2">
                            <div className="text-slate-300 font-medium">Get API Key</div>
                            <a
                              href={ci.key_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-2 bg-sky-700 hover:bg-sky-600 text-white rounded px-3 py-1.5 text-xs font-medium transition-colors"
                            >
                              Open {providerTypes[p.type]?.label || p.type} Dashboard
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                              </svg>
                            </a>
                            <div className="text-[11px] text-slate-500 whitespace-pre-wrap leading-relaxed">{ci.key_instructions}</div>
                          </div>
                        )}

                        {/* Auth type selector */}
                        <div className="flex items-center gap-2 border-t border-slate-800 pt-2">
                          <label className="text-slate-400">Auth Type:</label>
                          <select
                            className="bg-slate-800 rounded px-2 py-1 text-xs outline-none"
                            value={authForm.authType}
                            onChange={(e) => setAuthForm({ ...authForm, authType: e.target.value })}
                          >
                            {(providerTypes[p.type]?.auth_types || ["api_key"]).map((at: string) => (
                              <option key={at} value={at}>{at}</option>
                            ))}
                          </select>
                        </div>

                        {/* OAuth config (collapsible, for advanced users) */}
                        {authForm.authType === "oauth" && supportsOAuth && (
                          <div className="space-y-2 border-t border-slate-800 pt-2">
                            <div className="flex items-center justify-between">
                              <span className="text-slate-300 font-medium">OAuth Configuration</span>
                              {oauthPresets[p.type] && (
                                <button
                                  onClick={() => {
                                    const preset = oauthPresets[p.type];
                                    setAuthForm({
                                      ...authForm,
                                      tokenUrl: preset.token_url || "",
                                      authorizeUrl: preset.authorize_url || "",
                                      scope: preset.scope || "",
                                      grantType: preset.grant_type || "authorization_code",
                                    });
                                  }}
                                  className="bg-slate-700 hover:bg-slate-600 rounded px-2 py-0.5 text-[10px]"
                                >
                                  Load preset
                                </button>
                              )}
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <label className="text-slate-500 block mb-0.5">Client ID</label>
                                <input className="w-full bg-slate-800 rounded px-2 py-1 outline-none" value={authForm.clientId}
                                  onChange={(e) => setAuthForm({ ...authForm, clientId: e.target.value })} />
                              </div>
                              <div>
                                <label className="text-slate-500 block mb-0.5">Client Secret</label>
                                <input className="w-full bg-slate-800 rounded px-2 py-1 outline-none" type="password" value={authForm.clientSecret}
                                  onChange={(e) => setAuthForm({ ...authForm, clientSecret: e.target.value })} />
                              </div>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <label className="text-slate-500 block mb-0.5">Token URL</label>
                                <input className="w-full bg-slate-800 rounded px-2 py-1 outline-none" placeholder="https://..." value={authForm.tokenUrl}
                                  onChange={(e) => setAuthForm({ ...authForm, tokenUrl: e.target.value })} />
                              </div>
                              <div>
                                <label className="text-slate-500 block mb-0.5">Authorize URL</label>
                                <input className="w-full bg-slate-800 rounded px-2 py-1 outline-none" placeholder="https://..." value={authForm.authorizeUrl}
                                  onChange={(e) => setAuthForm({ ...authForm, authorizeUrl: e.target.value })} />
                              </div>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <label className="text-slate-500 block mb-0.5">Scope</label>
                                <input className="w-full bg-slate-800 rounded px-2 py-1 outline-none" value={authForm.scope}
                                  onChange={(e) => setAuthForm({ ...authForm, scope: e.target.value })} />
                              </div>
                              <div>
                                <label className="text-slate-500 block mb-0.5">Grant Type</label>
                                <select className="w-full bg-slate-800 rounded px-2 py-1 outline-none" value={authForm.grantType}
                                  onChange={(e) => setAuthForm({ ...authForm, grantType: e.target.value })}>
                                  <option value="authorization_code">Authorization Code</option>
                                  <option value="client_credentials">Client Credentials</option>
                                </select>
                              </div>
                            </div>
                          </div>
                        )}

                        <div className="flex gap-2 pt-1">
                          <button
                            onClick={async () => {
                              try {
                                await updateProviderAuth(p.id, authForm.authType,
                                  authForm.authType === "oauth" ? {
                                    client_id: authForm.clientId,
                                    client_secret: authForm.clientSecret,
                                    token_url: authForm.tokenUrl,
                                    authorize_url: authForm.authorizeUrl,
                                    scope: authForm.scope,
                                    grant_type: authForm.grantType,
                                  } : undefined,
                                );
                                setEditingAuth(null);
                                loadProviders();
                                flash("Auth config saved");
                              } catch (e: any) { flash("Error: " + e.message); }
                            }}
                            className="bg-emerald-700 hover:bg-emerald-600 rounded px-2 py-0.5"
                          >
                            Save
                          </button>
                          <button onClick={() => setEditingAuth(null)} className="bg-slate-700 rounded px-2 py-0.5">
                            Cancel
                          </button>
                        </div>
                      </div>
                    );
                  })()}

                  {provTestResult[p.id] && (
                    <div className="text-xs mb-2 bg-slate-900 rounded px-2 py-1">{provTestResult[p.id]}</div>
                  )}

                  <div className="flex gap-2 flex-wrap">
                    <button
                      onClick={async () => {
                        try {
                          const res = await testProvider(p.id);
                          if (res.ok) {
                            const extra = res.models?.length ? ` (${res.models.slice(0, 3).join(", ")})` : "";
                            setProvTestResult((r) => ({ ...r, [p.id]: `Connected${extra}` }));
                          } else {
                            setProvTestResult((r) => ({ ...r, [p.id]: `Failed: ${res.error}` }));
                          }
                        } catch (e: any) {
                          setProvTestResult((r) => ({ ...r, [p.id]: `Error: ${e.message}` }));
                        }
                      }}
                      className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs"
                    >
                      Test
                    </button>
                    <button
                      onClick={() => {
                        if (editingAuth === p.id) { setEditingAuth(null); return; }
                        const oauth = p.oauth || {};
                        setAuthForm({
                          authType: p.auth_type || "api_key",
                          clientId: oauth.client_id || "",
                          clientSecret: "",
                          tokenUrl: oauth.token_url || "",
                          authorizeUrl: oauth.authorize_url || "",
                          scope: oauth.scope || "",
                          grantType: oauth.grant_type || "authorization_code",
                        });
                        setEditingAuth(p.id);
                      }}
                      className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
                    >
                      {editingAuth === p.id ? "Close Auth" : "Auth Config"}
                    </button>
                    {/* OAuth flow panel */}
                    {p.auth_type === "oauth" && (
                      <>
                        {p.oauth_status?.authenticated && !p.oauth_status?.expired ? (
                          <button
                            onClick={async () => {
                              try {
                                await oauthRevoke(p.id);
                                loadProviders();
                                flash("Token revoked");
                              } catch (e: any) { flash("Error: " + e.message); }
                            }}
                            className="bg-amber-800 hover:bg-amber-700 rounded px-3 py-1 text-xs"
                          >
                            Revoke Token
                          </button>
                        ) : (
                          <button
                            onClick={() => setConnectWizard(connectWizard === p.id ? null : p.id)}
                            className="bg-violet-700 hover:bg-violet-600 rounded px-3 py-1 text-xs"
                          >
                            {connectWizard === p.id ? "Close" : "Connect via OAuth"}
                          </button>
                        )}
                      </>
                    )}
                    {!p.is_default && (
                      <button
                        onClick={async () => {
                          if (!confirm(`Delete provider "${p.name}"?`)) return;
                          try {
                            await deleteProvider(p.id);
                            loadProviders();
                            flash("Provider deleted");
                          } catch (e: any) {
                            flash("Error: " + e.message);
                          }
                        }}
                        className="bg-rose-800 hover:bg-rose-700 rounded px-3 py-1 text-xs ml-auto"
                      >
                        Delete
                      </button>
                    )}
                  </div>

                  {/* OAuth Connection Wizard */}
                  {connectWizard === p.id && p.auth_type === "oauth" && (
                    <div className="bg-slate-900 rounded p-4 mt-2 space-y-3 text-xs">
                      <div className="text-sm font-semibold text-slate-200">OAuth Connection</div>

                      {/* Step 1: Get authorize link */}
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <div className="w-5 h-5 rounded-full bg-violet-700 flex items-center justify-center text-[10px] font-bold flex-shrink-0">1</div>
                          <span className="font-medium text-slate-300">Open authorization page</span>
                        </div>
                        <button
                          disabled={oauthBusy}
                          onClick={async () => {
                            try {
                              const res = await getOAuthAuthorizeUrl(p.id);
                              window.open(res.url, "_blank", "width=600,height=700");
                              flash("Authorization page opened — sign in and grant access");
                            } catch (e: any) { flash("Error: " + e.message); }
                          }}
                          className="bg-violet-700 hover:bg-violet-600 disabled:opacity-50 rounded px-3 py-1.5 text-xs font-medium flex items-center gap-1.5"
                        >
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                          Open Authorization Page
                        </button>
                        <div className="text-[10px] text-slate-500">
                          Sign in and grant access. If the page closes automatically, the token was received.
                        </div>
                      </div>

                      {/* Step 2: Manual paste redirect URL */}
                      <div className="space-y-2 border-t border-slate-800 pt-3">
                        <div className="flex items-center gap-2">
                          <div className="w-5 h-5 rounded-full bg-violet-700 flex items-center justify-center text-[10px] font-bold flex-shrink-0">2</div>
                          <span className="font-medium text-slate-300">Or paste the redirect URL</span>
                        </div>
                        <div className="text-[10px] text-slate-500">
                          If auto-callback didn't work, copy the full URL from your browser after authorization and paste it here:
                        </div>
                        <div className="flex gap-2">
                          <input
                            className="flex-1 bg-slate-800 rounded px-2 py-1.5 outline-none text-xs font-mono"
                            placeholder="http://localhost:8000/api/providers/oauth/callback?code=..."
                            value={oauthPasteUrl}
                            onChange={(e) => setOauthPasteUrl(e.target.value)}
                          />
                          <button
                            disabled={oauthBusy || !oauthPasteUrl.trim()}
                            onClick={async () => {
                              setOauthBusy(true);
                              try {
                                const res = await oauthExchangeUrl(p.id, oauthPasteUrl.trim());
                                flash(res.ok ? "Token received!" : "Failed");
                                setOauthPasteUrl("");
                                loadProviders();
                              } catch (e: any) { flash("Error: " + e.message); }
                              setOauthBusy(false);
                            }}
                            className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 rounded px-3 py-1.5 whitespace-nowrap"
                          >
                            Exchange
                          </button>
                        </div>
                      </div>

                      {/* Step 3: Manual token paste */}
                      <div className="space-y-2 border-t border-slate-800 pt-3">
                        <div className="flex items-center gap-2">
                          <div className="w-5 h-5 rounded-full bg-violet-700 flex items-center justify-center text-[10px] font-bold flex-shrink-0">3</div>
                          <span className="font-medium text-slate-300">Or paste token directly</span>
                        </div>
                        <div className="text-[10px] text-slate-500">
                          If you already have an access token (from dev tools, another app, etc.):
                        </div>
                        <div className="flex gap-2">
                          <input
                            className="flex-1 bg-slate-800 rounded px-2 py-1.5 outline-none text-xs font-mono"
                            placeholder="eyJ... or sk-..."
                            type="password"
                            value={oauthManualTok}
                            onChange={(e) => setOauthManualTok(e.target.value)}
                          />
                          <button
                            disabled={oauthBusy || !oauthManualTok.trim()}
                            onClick={async () => {
                              setOauthBusy(true);
                              try {
                                const res = await oauthManualToken(p.id, oauthManualTok.trim());
                                flash(res.ok ? "Token saved!" : "Failed");
                                setOauthManualTok("");
                                loadProviders();
                              } catch (e: any) { flash("Error: " + e.message); }
                              setOauthBusy(false);
                            }}
                            className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 rounded px-3 py-1.5 whitespace-nowrap"
                          >
                            Save Token
                          </button>
                        </div>
                      </div>

                      {/* Client credentials (server-to-server) */}
                      {p.oauth?.grant_type === "client_credentials" && (
                        <div className="space-y-2 border-t border-slate-800 pt-3">
                          <div className="text-slate-300 font-medium">Server-to-server (no browser)</div>
                          <button
                            disabled={oauthBusy}
                            onClick={async () => {
                              setOauthBusy(true);
                              try {
                                const res = await oauthClientCredentials(p.id);
                                flash(res.ok ? "Authenticated!" : "Failed");
                                loadProviders();
                              } catch (e: any) { flash("Error: " + e.message); }
                              setOauthBusy(false);
                            }}
                            className="bg-violet-700 hover:bg-violet-600 disabled:opacity-50 rounded px-3 py-1.5"
                          >
                            Authenticate (client_credentials)
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {providers.length === 0 && !showAddProvider && (
                <div className="text-sm opacity-50 text-center py-4">No providers configured</div>
              )}
            </div>

            {/* Ollama Local Models */}
            <div className="bg-slate-800 rounded p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold">Local Models (Ollama)</h4>
                <button onClick={loadOllamaModels} className="bg-slate-700 hover:bg-slate-600 rounded px-2 py-1 text-xs">
                  Refresh
                </button>
              </div>

              {ollamaModels.length > 0 ? (
                <div className="space-y-1">
                  {ollamaModels.map((m) => (
                    <div key={m.name} className="flex items-center gap-2 text-xs bg-slate-700/50 rounded p-2">
                      <span className="text-slate-200 font-mono flex-1">{m.name}</span>
                      <span className="text-slate-500">{m.parameters || m.family}</span>
                      <span className="text-slate-500">{(m.size / 1e9).toFixed(1)}GB</span>
                      {m.quantization && <span className="text-slate-600">{m.quantization}</span>}
                      <button
                        onClick={async () => {
                          if (!confirm(`Delete model "${m.name}"?`)) return;
                          try {
                            await deleteOllamaModel(m.name);
                            loadOllamaModels();
                            flash("Model deleted");
                          } catch (e: any) {
                            flash("Error: " + e.message);
                          }
                        }}
                        className="text-rose-400 hover:text-rose-300 text-[10px]"
                      >
                        delete
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-slate-500">No local models found (Ollama not running?)</div>
              )}

              <div className="flex gap-2">
                <input
                  className="flex-1 bg-slate-900 rounded px-2 py-1.5 text-xs outline-none focus:ring-1 focus:ring-sky-600"
                  placeholder="Model to pull (e.g. qwen2.5:7b-instruct)"
                  value={ollamaPullName}
                  onChange={(e) => setOllamaPullName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !ollamaPulling && ollamaPullName.trim() && (async () => {
                    setOllamaPulling(true);
                    try {
                      const res = await pullOllamaModel(ollamaPullName.trim());
                      flash(res.ok ? "Model pulled" : `Error: ${res.error}`);
                      if (res.ok) { setOllamaPullName(""); loadOllamaModels(); }
                    } catch (e: any) { flash("Error: " + e.message); }
                    setOllamaPulling(false);
                  })()}
                />
                <button
                  onClick={async () => {
                    if (!ollamaPullName.trim() || ollamaPulling) return;
                    setOllamaPulling(true);
                    try {
                      const res = await pullOllamaModel(ollamaPullName.trim());
                      flash(res.ok ? "Model pulled" : `Error: ${res.error}`);
                      if (res.ok) { setOllamaPullName(""); loadOllamaModels(); }
                    } catch (e: any) { flash("Error: " + e.message); }
                    setOllamaPulling(false);
                  }}
                  disabled={ollamaPulling || !ollamaPullName.trim()}
                  className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 rounded px-3 py-1 text-xs"
                >
                  {ollamaPulling ? "Pulling..." : "Pull"}
                </button>
              </div>
            </div>
          </div>
        )}

        {tab === "channels" && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-bold">Channels</h3>
              <div className="flex gap-2">
                <button
                  onClick={loadChannels}
                  className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
                >
                  Refresh
                </button>
                <button
                  onClick={() => setShowAddChannel(!showAddChannel)}
                  className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs"
                >
                  + Add Channel
                </button>
              </div>
            </div>

            {showAddChannel && (
              <div className="bg-slate-800 rounded p-4 space-y-3">
                <h4 className="text-sm font-semibold">New Channel</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-slate-400 block mb-1">Name</label>
                    <input
                      className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
                      placeholder="My Telegram Bot"
                      value={newChName}
                      onChange={(e) => setNewChName(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-400 block mb-1">Type</label>
                    <select
                      className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
                      value={newChType}
                      onChange={(e) => setNewChType(e.target.value)}
                    >
                      <option value="telegram">Telegram</option>
                    </select>
                  </div>
                </div>
                {newChType === "telegram" && (
                  <>
                    <div>
                      <label className="text-xs text-slate-400 block mb-1">Bot Token (from @BotFather)</label>
                      <input
                        className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm font-mono outline-none focus:ring-1 focus:ring-sky-600"
                        placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
                        value={newChToken}
                        onChange={(e) => setNewChToken(e.target.value)}
                        type="password"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-400 block mb-1">
                        Allowed users (comma-separated usernames or IDs, empty = all)
                      </label>
                      <input
                        className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
                        placeholder="username1, username2"
                        value={newChAllowed}
                        onChange={(e) => setNewChAllowed(e.target.value)}
                      />
                    </div>
                  </>
                )}
                <div className="flex gap-2">
                  <button
                    onClick={async () => {
                      if (!newChName.trim()) return;
                      const id = newChName.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-");
                      try {
                        await createChannel({
                          id,
                          name: newChName.trim(),
                          type: newChType,
                          enabled: false,
                          config:
                            newChType === "telegram"
                              ? {
                                  bot_token: newChToken.trim(),
                                  allowed_users: newChAllowed
                                    .split(",")
                                    .map((s) => s.trim())
                                    .filter(Boolean),
                                }
                              : {},
                        });
                        setShowAddChannel(false);
                        setNewChName("");
                        setNewChToken("");
                        setNewChAllowed("");
                        loadChannels();
                        flash("Channel created");
                      } catch (e: any) {
                        flash("Error: " + e.message);
                      }
                    }}
                    className="bg-emerald-700 hover:bg-emerald-600 rounded px-3 py-1 text-xs"
                  >
                    Create
                  </button>
                  <button
                    onClick={() => setShowAddChannel(false)}
                    className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {channels.length === 0 && !showAddChannel && (
              <div className="text-sm opacity-50 text-center py-8">
                No channels configured. Click "+ Add Channel" to connect Telegram or other services.
              </div>
            )}

            <div className="space-y-3">
              {channels.map((ch) => {
                const isRunning = ch.runtime_status === "running";
                return (
                  <div key={ch.id} className="bg-slate-800 rounded p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <span className="text-lg">
                          {ch.type === "telegram" ? "✈️" : "📡"}
                        </span>
                        <div>
                          <div className="font-semibold text-sm">{ch.name}</div>
                          <div className="text-xs text-slate-500">
                            {ch.type} · {ch.id}
                          </div>
                        </div>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isRunning
                              ? "bg-emerald-900/60 text-emerald-300"
                              : "bg-slate-700 text-slate-400"
                          }`}
                        >
                          {isRunning ? "RUNNING" : "STOPPED"}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <label className="flex items-center gap-1 text-xs">
                          <input
                            type="checkbox"
                            checked={ch.enabled}
                            onChange={async () => {
                              try {
                                await updateChannel(ch.id, { enabled: !ch.enabled });
                                loadChannels();
                              } catch (e: any) {
                                flash("Error: " + e.message);
                              }
                            }}
                            className="rounded"
                          />
                          Enabled
                        </label>
                        <label className="flex items-center gap-1 text-xs">
                          <input
                            type="checkbox"
                            checked={ch.auto_start}
                            onChange={async () => {
                              try {
                                await updateChannel(ch.id, { auto_start: !ch.auto_start });
                                loadChannels();
                              } catch (e: any) {
                                flash("Error: " + e.message);
                              }
                            }}
                            className="rounded"
                          />
                          Auto-start
                        </label>
                      </div>
                    </div>

                    {ch.type === "telegram" && (
                      <div className="text-xs text-slate-500 mb-3 space-y-0.5">
                        <div>
                          Token: {ch.config?.bot_token ? "••••" + ch.config.bot_token.slice(-8) : "(not set)"}
                        </div>
                        {ch.config?.allowed_users?.length > 0 && (
                          <div>Allowed: {ch.config.allowed_users.join(", ")}</div>
                        )}
                        {ch.last_started && <div>Last started: {ch.last_started}</div>}
                      </div>
                    )}

                    {testResult[ch.id] && (
                      <div className="text-xs mb-2 bg-slate-900 rounded px-2 py-1">
                        {testResult[ch.id]}
                      </div>
                    )}

                    <div className="flex gap-2">
                      {!isRunning ? (
                        <button
                          onClick={async () => {
                            try {
                              await startChannel(ch.id);
                              loadChannels();
                              flash("Channel started");
                            } catch (e: any) {
                              flash("Error: " + e.message);
                            }
                          }}
                          disabled={!ch.enabled}
                          className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 rounded px-3 py-1 text-xs"
                        >
                          Start
                        </button>
                      ) : (
                        <button
                          onClick={async () => {
                            try {
                              await stopChannel(ch.id);
                              loadChannels();
                              flash("Channel stopped");
                            } catch (e: any) {
                              flash("Error: " + e.message);
                            }
                          }}
                          className="bg-amber-700 hover:bg-amber-600 rounded px-3 py-1 text-xs"
                        >
                          Stop
                        </button>
                      )}
                      <button
                        onClick={async () => {
                          try {
                            const res = await testChannel(ch.id);
                            if (res.ok) {
                              setTestResult((p) => ({
                                ...p,
                                [ch.id]: `Connected: @${res.bot_username} (${res.bot_name})`,
                              }));
                            } else {
                              setTestResult((p) => ({ ...p, [ch.id]: `Failed: ${res.error}` }));
                            }
                          } catch (e: any) {
                            setTestResult((p) => ({ ...p, [ch.id]: `Error: ${e.message}` }));
                          }
                        }}
                        className="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs"
                      >
                        Test
                      </button>
                      <button
                        onClick={async () => {
                          if (!confirm(`Delete channel "${ch.name}"?`)) return;
                          try {
                            await deleteChannel(ch.id);
                            loadChannels();
                            flash("Channel deleted");
                          } catch (e: any) {
                            flash("Error: " + e.message);
                          }
                        }}
                        className="bg-rose-800 hover:bg-rose-700 rounded px-3 py-1 text-xs ml-auto"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === "status" && status && (
          <StatusTab status={status} onCompare={handleCompare} onRefresh={loadStatus} />
        )}
        </Suspense>
      </div>
    </div>
  );
}
