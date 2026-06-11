"""Pydantic data models."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Category = Literal["fundamentals", "profession", "projects", "personal"]
Confidence = Literal["unverified", "partial", "verified"]

# Categories for fine-tune examples (see FINETUNE_PIPELINE.md)
FinetuneCategory = Literal[
    "factual_qa",
    "troubleshooting",
    "procedure",
    "decision",
    "correction",
    "other",
]


class NoteFrontmatter(BaseModel):
    topic: str
    category: Category
    created: str
    updated: str
    keywords: list[str] = []
    source: str = ""
    confidence: Confidence = "unverified"
    access_count: int = 0
    project: Optional[str] = None


class Note(BaseModel):
    frontmatter: NoteFrontmatter
    body: str
    path: str = ""

    def to_markdown(self) -> str:
        fm = self.frontmatter
        kw = ", ".join(fm.keywords)
        lines = [
            "---",
            f"topic: {fm.topic}",
            f"category: {fm.category}",
            f"created: {fm.created}",
            f"updated: {fm.updated}",
            f"keywords: {kw}",
            f"source: {fm.source}",
            f"confidence: {fm.confidence}",
            f"access_count: {fm.access_count}",
        ]
        if fm.project:
            lines.append(f"project: {fm.project}")
        lines += ["---", "", self.body.strip(), ""]
        return "\n".join(lines)


class IndexEntry(BaseModel):
    topic: str
    category: Category
    path: str
    keywords: list[str]
    access_count: int = 0
    updated: str
    project: Optional[str] = None


class TaskAnalysis(BaseModel):
    """Legacy model kept for backward compatibility with tests."""
    required_topics: list[str]
    plan: list[str]
    confidence: int = Field(ge=0, le=100)
    reasoning: str = ""


class ThinkingResult(BaseModel):
    """Output of the universal thinking protocol.

    The thinking protocol forces the agent to reason before acting:
    1. UNDERSTAND — what is being asked, what type of question
    2. ASSESS — what I already know, what's in my NOTES/CORE MEMORY
    3. STRATEGIZE — what approach, which tools, in what order, why
    4. IDENTIFY — what knowledge topics I need to load
    """
    # Step 1: Understanding
    question_type: str = ""             # "factual", "self_analysis", "calculation",
                                        # "file_operation", "web_lookup", "creative",
                                        # "troubleshooting", "meta" (about the agent itself)
    core_question: str = ""             # Restatement: what exactly is being asked

    # Step 2: Assessment
    already_know: list[str] = []        # What's already in CORE MEMORY/NOTES that's relevant
    knowledge_gaps: list[str] = []      # What's missing

    # Step 3: Strategy
    approach: str = ""                  # Brief description of how to solve
    tools_needed: list[str] = []        # Which tools to use: ["web_search", "read_file", ...]
    tools_reasoning: str = ""           # WHY these tools (not just which)

    # Step 4: Topics (backward-compat with old flow)
    required_topics: list[str] = []     # Knowledge topics to load from KB
    plan: list[str] = []                # Ordered steps
    confidence: int = Field(ge=0, le=100, default=50)
    reasoning: str = ""                 # Overall reasoning summary

    # Step 5: Hierarchical decomposition (for complex tasks)
    subtasks: list[str] = []            # Independent subtasks for complex problems


class VerificationResult(BaseModel):
    confidence: int = Field(ge=0, le=100)
    verified_claims: list[str] = []
    unverified_claims: list[str] = []
    contradictions: list[str] = []
    notes_used: list[str] = []
    # Grader calibration (audit 2026-06-11). `confidence` stays the
    # conservative display scalar — clipped to 30 on an endpoint miss
    # so UI badges and gates remain strict. These two fields preserve
    # the SPLIT signal so the learning loop doesn't conflate "didn't
    # deliver the action" (process failure, root cause already known)
    # with "bad/hallucinated content" (needs LLM failure analysis):
    #   content_confidence — claim-verifiability score BEFORE any
    #     endpoint/psm clipping. None = no clip happened (confidence
    #     already IS the content score).
    #   endpoint_met — the delivery judgment. None = not evaluated.
    content_confidence: Optional[int] = None
    endpoint_met: Optional[bool] = None


class EvidenceItem(BaseModel):
    """A single piece of grounding for a claim.

    Phase A of the claim/evidence layer: the structure is populated from
    things that already exist (tool calls in the thinking trace, notes
    used by the verifier, the user's own message). Phase B will have the
    solver emit evidence directly, with quote+source bound at production
    time. This shape is the contract between the two.

    `source_type` values are deliberately small and stable so consumers
    (WebUI, eventual claim-by-claim verifier) can branch on them
    confidently:
      - "tool":   came from a tool call (read_file, locate_symbol, calc, …)
      - "memory": came from KG / notes / core memory
      - "user":   the user said it on the current turn
      - "model":  the model asserted it without external grounding
    """
    id: str
    source_type: str = "model"
    source_ref: str = ""        # e.g. "read_file:backend/llm.py:1026-1180"
    quote: str = ""             # supporting snippet, optional
    confidence: float = 1.0     # 0.0–1.0


class Claim(BaseModel):
    """One assertion the agent's answer makes.

    Phase A populates `status` from the verifier's existing
    {verified, unverified, contradicted} buckets and leaves
    `evidence_ids` empty by default (no automatic mapping yet).
    Phase B will carry evidence_ids end-to-end so the verifier can
    rule on each claim against the snippet that supports it.
    """
    id: str
    text: str
    status: str = "unverified"  # "verified" | "unverified" | "contradicted"
    evidence_ids: list[str] = []
    risk: str = "low"           # "low" | "medium" | "high"


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    # Per-stage attribution for the current request (populated from
    # TokenTracker.request_breakdown). Top-level key is the stage name
    # (`solve`, `verify`, `think`, `classify`, …); inner dict carries
    # calls / input_tokens / output_tokens / cost_usd / total_tokens.
    # Sorted by input_tokens descending so the heaviest stage is up
    # top; consumer just renders in iteration order.
    by_stage: dict[str, dict[str, float]] = {}


class ToolCallDetail(BaseModel):
    """Structured details for a tool call event in the thinking trace.

    The trace's free-form `message` field already carries a one-liner
    preview, but for the WebUI's "show more" expand we need a chunk
    of the actual result body. Kept off the message so the short
    summary stays short. Always present on tool-event steps; never
    on other event types.

    `result` carries a TRUNCATED preview, not the full body. A 60k
    `read_file` output going into every `AgentAnswer.thinking_trace`
    item used to bloat:
      - the API response shipped to WebUI on every chat reply
      - the dev-mode capture file persisted to disk
      - the SSE payload for streaming
    The verifier already gets the cap'd version through `tool_outputs`;
    the trace just needs enough for an operator to know what the
    tool returned, not the whole file.
    """
    name: str = ""
    args: dict = {}
    result: str = ""           # truncated preview, see RESULT_PREVIEW_CAP
    result_truncated: bool = False  # True if the original body was longer
    result_full_len: int = 0   # original byte length, for dev panel display
    is_error: bool = False
    duration_ms: int = 0


class ThinkingStep(BaseModel):
    """One step in the agent's thinking trace."""
    ts: float = 0.0          # monotonic timestamp (seconds from request start)
    event: str = ""           # stage name (core, think, solve, tool, verify...)
    message: str = ""         # human-readable description
    tokens_so_far: int = 0    # cumulative tokens at this point
    tool_call: Optional[ToolCallDetail] = None  # set only on `tool` / `tool_error` events


class LLMCallDetail(BaseModel):
    """Per-LLM-call record for the dev-mode prompt viewer.

    Captures what was actually sent to the LLM with file blobs
    REDACTED — `# SOUL\\n…soul.md body…` becomes
    `# SOUL\\n[file: knowledge/identity/soul.md, 1234 chars]`. The
    point is to see prompt STRUCTURE without dumping multi-kilobyte
    file contents back into the response (the file is a click away
    on disk anyway). For the same reason `system_redacted` and
    `user_redacted` are length-capped — see `verifier.detect_false_*`
    docstring for the same trade-off.
    """
    label: str = ""                   # free-text call-site label, e.g. "_verify"
    task_type: str = ""               # TaskType.value, e.g. "complex_solving"
    model: str = ""                   # provider's model id at call time
    system_redacted: str = ""         # system prompt, file blobs replaced with markers
    user_redacted: str = ""           # user prompt, same redaction rules
    response_preview: str = ""        # first 600 chars of LLM response
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class AgentAnswer(BaseModel):
    answer: str
    verification: VerificationResult
    learned_topics: list[str] = []
    used_topics: list[str] = []
    project: Optional[str] = None
    # True — if the agent took the short chat branch (small-talk, greeting).
    is_chat: bool = False
    token_usage: Optional[TokenUsage] = None
    thinking_trace: list[ThinkingStep] = []
    # Dev-mode payload: per-LLM-call captures with file blobs redacted.
    # Empty list when nothing was recorded (chat fast-path or skipped).
    llm_calls: list[LLMCallDetail] = []
    # Claim/evidence layer (P0). Populated from VerificationResult and
    # the thinking_trace by `claims.build_claims_and_evidence` so
    # consumers can render answers as "claim → its evidence" without
    # parsing the answer text themselves. Empty on chat fast-path.
    claims: list[Claim] = []
    evidence: list[EvidenceItem] = []
    # P1 TurnWorkspace persistence: stable id of the on-disk turn
    # record (`workspace/turns/<turn_id>.json`). Empty on chat
    # fast-path. Lets the WebUI / future evaluator pull the full
    # turn artifact (including tool_call_order with raw results)
    # without bloating this response payload.
    turn_id: str = ""
    # Pipeline mode picked for this turn. Surfaced so the UI can
    # show the user which tier ran (fast_chat / task_mode /
    # deep_agent) and the verifier behaviour reads correctly:
    # fast_chat skips analysis + verifier + retry + learning;
    # task_mode runs analysis + tools + memory but skips verifier
    # and retry loop; deep_agent is the full cycle. Empty string
    # on legacy paths that don't yet stamp it.
    mode: str = ""
    # AskUserQuestion follow-up. When the agent called the `ask_user`
    # tool, the turn ends with this field populated and `answer`
    # carries a short placeholder. The frontend renders a structured
    # AskUserCard from the payload; Telegram renders inline-keyboard
    # buttons. None means "no pending question — answer is final".
    question: Optional[dict] = None


class ChatRequest(BaseModel):
    message: str
    project: Optional[str] = None
    # Attachment sha256s — uploaded separately via /api/attachments and
    # referenced here by content-hash. Keeps the chat payload compact and
    # lets the same image be re-used across turns without re-uploading.
    attachments: list[str] = []
    # Coarse channel label kept for back-compat and UI grouping.
    # Default "webui". "telegram" means the WebUI is composing a turn
    # for the Telegram conversation context.
    channel: Optional[str] = None
    # Phase 10: speaker identity ('<channel>:<user_id>'). Primary
    # routing key for sessions, conversation memory, and per-speaker
    # user profile. Defaults to "webui:default" when omitted — the
    # local WebUI user. Telegram channel sets it to "telegram:<tg_user_id>"
    # so each Telegram user gets their own session+profile.
    speaker_id: Optional[str] = None


class LearnRequest(BaseModel):
    topic: str
    depth: Literal["deep", "quick"] = "quick"
    category: Category = "profession"


class CoreFactRequest(BaseModel):
    fact: str
    source: str = "user"


class CoreFactDelete(BaseModel):
    search_text: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class FinetuneMetadata(BaseModel):
    source_notes: list[str] = []
    confidence: int = 0
    project: Optional[str] = None
    timestamp: str = ""
    verified: bool = False
    category: FinetuneCategory = "factual_qa"
    boosted: bool = False
    original_wrong_answer: Optional[str] = None


class FinetunePair(BaseModel):
    """OpenAI-style fine-tune example."""
    id: str = ""          # stable short hash (computed on save)
    messages: list[ChatMessage]
    metadata: FinetuneMetadata = FinetuneMetadata()

    def user_text(self) -> str:
        for m in self.messages:
            if m.role == "user":
                return m.content
        return ""

    def assistant_text(self) -> str:
        for m in self.messages:
            if m.role == "assistant":
                return m.content
        return ""


# ---- fine-tune API models ----
class FinetuneStatus(BaseModel):
    total: int
    curated: int
    ready: bool
    min_required: int
    by_category: dict[str, int] = {}


class FinetuneEdit(BaseModel):
    assistant: Optional[str] = None
    boosted: Optional[bool] = None


class CorrectionRequest(BaseModel):
    question: str
    wrong_answer: str
    corrected_answer: str
    project: Optional[str] = None


# ---- model versions ----
class ModelVersion(BaseModel):
    tag: str              # "v0", "v1", ...
    model_id: str         # "mistral:7b" / "my-agent-v1"
    created: str
    examples_count: int = 0
    notes: str = ""


class ModelVersionsState(BaseModel):
    current: str = "v0"
    versions: list[ModelVersion] = []
    rollback_enabled: bool = True


class EvaluationResult(BaseModel):
    old_model: str
    new_model: str
    old_score: float
    new_score: float
    improvement: float
    should_upgrade: bool
    details: list[dict] = []


# ---- mode / cloud flow ----
class ModeInfo(BaseModel):
    mode: str
    finetune_enabled: bool
    training_location: str
    model_a: Optional[str] = None
    model_b: Optional[str] = None


class ImportGgufRequest(BaseModel):
    path: str
    tag: Optional[str] = None


class CloudExportResult(BaseModel):
    package_dir: str
    files: list[str] = []
    tag: str
    instructions: str = ""
