"""Pydantic-модели данных."""
from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

Category = Literal["fundamentals", "profession", "projects", "personal"]
Confidence = Literal["unverified", "partial", "verified"]

# Категории для fine-tune примеров (см. FINETUNE_PIPELINE.md)
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


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0


class ToolCallDetail(BaseModel):
    """Structured details for a tool call event in the thinking trace.

    The trace's free-form `message` field already carries a one-liner
    preview, but for the WebUI's "show more" expand we need the actual
    args dict and the full result body. Kept off the message so the
    short summary stays short. Always present on tool-event steps;
    never on other event types.
    """
    name: str = ""
    args: dict = {}
    result: str = ""
    is_error: bool = False
    duration_ms: int = 0


class ThinkingStep(BaseModel):
    """One step in the agent's thinking trace."""
    ts: float = 0.0          # monotonic timestamp (seconds from request start)
    event: str = ""           # stage name (core, think, solve, tool, verify...)
    message: str = ""         # human-readable description
    tokens_so_far: int = 0    # cumulative tokens at this point
    tool_call: Optional[ToolCallDetail] = None  # set only on `tool` / `tool_error` events


class AgentAnswer(BaseModel):
    answer: str
    verification: VerificationResult
    learned_topics: list[str] = []
    used_topics: list[str] = []
    project: Optional[str] = None
    # True — если агент пошёл по короткой chat-ветке (small-talk, приветствие).
    is_chat: bool = False
    token_usage: Optional[TokenUsage] = None
    thinking_trace: list[ThinkingStep] = []


class ChatRequest(BaseModel):
    message: str
    project: Optional[str] = None
    # Attachment sha256s — uploaded separately via /api/attachments and
    # referenced here by content-hash. Keeps the chat payload compact and
    # lets the same image be re-used across turns without re-uploading.
    attachments: list[str] = []


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
    id: str = ""          # стабильный короткий hash (вычисляется при сохранении)
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
