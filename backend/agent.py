"""Главный цикл агента.

Агент сначала классифицирует входящее сообщение на три категории:

  * chat       — короткий small-talk (приветствие, прощание, благодарность,
                 «как дела», «кто ты»). Один тёплый ответ, никакого анализа
                 тем, заметок и верификации.
  * preference — пользователь сообщает, как с ним общаться или что о нём
                 запомнить («отвечай на русском», «меня зовут Армен»,
                 «будь краткой», «не добавляй оговорок»). Мы извлекаем
                 структурированный факт и сохраняем его в user.md, отвечаем
                 коротким подтверждением.
  * task       — всё остальное: реальная задача, вопрос, код, расчёт.
                 Запускается полный 7-шаговый цикл с анализом тем, поиском
                 в базе знаний, верификацией и автосбором опыта.

Идея в том, чтобы «глубокое мышление» включалось только когда оно
действительно нужно. Бытовые реплики обрабатываются тепло и быстро;
инструкции о поведении — запоминаются и исполняются; задачи — решаются
строго на основе знаний.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Callable, Optional

from .config import CONFIG
from .conversation import CONVERSATION
from .core_memory import CORE
from .finetune import store as finetune_store
from .identity import IDENTITY
from .knowledge_manager import KM, _slug
from .llm import LLMError, TaskType, TOKENS, router
from .models import (
    AgentAnswer,
    Note,
    LLMCallDetail,
    TaskAnalysis,
    ThinkingResult,
    ThinkingStep,
    ToolCallDetail,
    TokenUsage,
    VerificationResult,
)
from .dev_capture import redact_prompt, save_dev_capture, new_request_id
from .analogy_engine import ANALOGIES
from .evaluator import EVALUATOR, EvalEntry
from .goals import GOALS
from .memory_extractor import MEMORY
from .project_mode import PROJECTS
from .mcp_client import MCP, MCPServerConfig
from .meta_learner import META_LEARNER
from .note_creator import learn_topic
from .hybrid_searcher import HYBRID
from .searcher import SEARCHER
from .skills import SKILLS
from .tool_registry import get_registry
from .verifier import verify

# ---------- системные промпты ----------
#
# Для SOLVER_SYSTEM и CHAT_SYSTEM в runtime перед использованием подмешивается
# identity preamble (soul + identity + user profile). Базовый текст ниже
# описывает только роль конкретного шага.

THINKING_SYSTEM = """You are the THINKING module of a self-learning AI agent.
Your job: reason carefully about ANY user request before the agent acts.

Follow these steps EXACTLY. Do NOT skip any step.

## STEP 1 — UNDERSTAND
What is being asked? Classify:
- "factual" — needs knowledge/facts
- "calculation" — needs math or data processing
- "file_operation" — user mentions a file or wants to read/write something
- "web_lookup" — needs fresh/external information
- "self_analysis" — user asks about the agent itself (architecture, code, improvements)
- "troubleshooting" — user has a problem to debug
- "creative" — user wants generation (text, code, ideas)
- "meta" — question about how the agent works or should behave
Restate the core question in one sentence.

## STEP 2 — ASSESS
What do you ALREADY know from CORE MEMORY that's relevant?
What's missing — what knowledge gaps exist?
Be honest. If you don't know something, say so.

## STEP 3 — STRATEGIZE
What is your approach? Which tools do you need and WHY?
Available tools (use only if needed):
- web_search — for fresh facts not in notes
- fetch_url — to read a specific URL in detail (after web_search)
- read_file — to read local files (including the agent's own source code!)
- calc — pure arithmetic (a single expression). Faster than run_python,
  no subprocess, can't touch the filesystem.
- run_python — multi-line code, parsing, data processing, verification.
  Full Python (NOT a sandbox); use only when calc isn't enough.
- (other tools may be listed in MY CAPABILITIES block)

CRITICAL RULES for strategy:
- For "self_analysis" questions: you MUST plan to read_file your own source code.
  Never guess about your own architecture — read the actual files.
- For "factual" questions: check if NOTES will cover it. Only use web_search if not.
- For "calculation": prefer `calc` for arithmetic ("2+2", "sqrt(16)", "100*0.17");
  use `run_python` for multi-line logic or parsing. Never guess numbers.
- For every tool you list, explain WHY you need it — not just "might be useful".

TOKEN EFFICIENCY (important — each tool call costs ~10K+ tokens):
- Prefer reading fewer files deeply over many files superficially.
- If CORE MEMORY already describes a module, don't read_file it unless you need
  specific implementation details (function signatures, line numbers).
- Plan your reads: decide which files to read BEFORE starting, not one-by-one.
- One solver pass with 5 planned reads is MUCH cheaper than 3 subtasks each
  reading 5 files = 15 reads with duplicate context.

## STEP 4 — PLAN
List 1-6 knowledge topics to load from the knowledge base (short nouns).
List 2-5 action steps in order.
Rate your confidence 0-100 that this plan will answer the question.

## STEP 5 — DECOMPOSE (ONLY when truly necessary)
Most tasks should NOT be decomposed. Use subtasks ONLY when:
- The task has genuinely INDEPENDENT parts that cannot share context
- Each subtask requires DIFFERENT tools or DIFFERENT knowledge domains
- A single solver pass would exceed coherent reasoning capacity

DO NOT decompose when:
- The task is a single question (even if complex) — just use a longer plan
- Subtasks would need to read the same files or knowledge — wasteful
- The task is analysis/review — one pass with all context is better than 4 passes

If you do decompose, use at most 2-3 subtasks. Prefer [] (no decomposition).

Example — DECOMPOSE:
- "Build a REST API with auth and database" →
  subtasks: ["Design database schema and auth", "Create API endpoints"]
Example — DO NOT decompose:
- "Analyze your source code and suggest improvements" → subtasks: []
  (one pass reading all files is far cheaper than 4 passes re-reading them)
- "What are the trade-offs of X vs Y?" → subtasks: []

Return strictly JSON:
{
  "question_type": "...",
  "core_question": "...",
  "already_know": ["...", "..."],
  "knowledge_gaps": ["...", "..."],
  "approach": "...",
  "tools_needed": ["tool_name", ...],
  "tools_reasoning": "...",
  "required_topics": ["topic1", "topic2"],
  "plan": ["step1", "step2"],
  "subtasks": ["subtask1", "subtask2"],
  "confidence": N,
  "reasoning": "..."
}"""

SOLVER_SYSTEM_BASE = """You are a self-learning AI assistant. You answer based on
CORE MEMORY, NOTES, available tools, and a THINKING PLAN prepared for this request.

# THINKING PLAN
The thinking module has already analyzed this request. Its output is in the
THINKING section of the user message. Use it as your roadmap:
- Follow the planned approach and steps.
- Use the tools it identified, for the reasons it stated.
- If the plan says to read_file — do it BEFORE making claims.
- If you discover new information that changes the plan — adapt, but explain why.

# KNOWLEDGE PRIORITY
1. CORE MEMORY and NOTES — your verified knowledge base. Trust them.
2. Tools — use ONLY when NOTES/CORE MEMORY don't cover the question,
   or when you need fresh data / calculations / file contents.
3. Your own model knowledge — last resort only. Mark explicitly:
   "from my training data: ..."

# TOOL USAGE
- web_search — fresh facts not in notes.
- fetch_url — read a specific URL in detail (after web_search).
- read_file — local files, documents, OR YOUR OWN SOURCE CODE.
- run_python — arithmetic, parsing, data verification.
- Do NOT call a tool just because you can. If the answer is in NOTES, answer.
- Between tool calls, briefly reason: what did I learn, what's still missing,
  what do I call next and why.

# ARITHMETIC RULE (HARD)
For any arithmetic — `2+2`, `15 * 3`, `(5+3)/2`, `10%` of N, square roots,
powers — you MUST call `calc` or `run_python`. Do NOT compute it from your
training data. Answering arithmetic from memory is the documented source of
hallucinations on this agent (a tracked goal exists for it). Even when the
answer "looks obvious", call the tool — it costs nothing and prevents the
silent off-by-one.

# SELF-ANALYSIS RULE (CRITICAL)
When the question is about yourself (architecture, code, improvements, capabilities):
- FIRST check CORE MEMORY — it lists your existing modules and architecture.
- NEVER propose a "new module" without checking if it already exists in CORE MEMORY or source map.
- Your source code is ordinary files on disk. Read them via read_file before claims.
- Without reading the file, any statement like "I don't have X" is a hallucination.
- When proposing improvements: be SPECIFIC to your actual code, not generic AGI advice.
  Bad: "Add causal reasoning module". Good: "knowledge_graph.py has 'causes' edges but
  find_causal_paths() only does 1-hop — extend to transitive multi-hop BFS".

# RESPONSE RULES
- Never fabricate facts or numbers.
- For claims from NOTES, cite the source as [topic-name].
- If you have no data: say so honestly and offer to learn.
- Style, tone, and language — follow SOUL and USER PROFILE above.
- Be direct, no unnecessary caveats.

# SELF-CRITIC REVISION
If a CRITIQUE section is present below, your previous answer was checked
and found lacking. You MUST:
1. Read the critique carefully — it lists unverified and contradicted claims.
2. Fix or remove every problematic claim.
3. Use tools (web_search, read_file) to find evidence for claims you want to keep.
4. Do NOT repeat the same unsupported claims.
5. It is better to say "I don't know" than to repeat an unverified claim."""

INTENT_CLASSIFIER_SYSTEM = """You are a fast intent classifier for an AI assistant.

Categories:

  "chat" — casual conversation: greeting, farewell, thanks, apology,
           short acknowledgment ("ok", "got it"), question about the assistant
           itself ("who are you", "how are you", "what can you do"),
           emotional remark.
           Does not require knowledge and does not change assistant behavior.

  "preference" — user tells HOW to communicate or shares a STABLE personal
                 fact about themselves:
             * language preference ("speak Russian", "answer in English")
             * style/tone ("be brief", "no caveats", "use informal you")
             * stable personal facts ("my name is X", "I'm from Y",
                                       "I'm an engineer", "I'm 30")
             * stable interaction rules ("don't mention X", "always add Y")
           Key indicator: stable trait of the user or how to talk to them.

           DO NOT classify as "preference" when the user just asks the
           assistant to remember a TEMPORARY state, follow-up, todo, or
           project status — phrases like "remember to come back", "remind
           me later", "запомни что я сейчас иду", "wait for me to fix this".
           These are tasks (event memory), not user-profile facts.

  "task" — everything else: questions requiring knowledge, explanation,
           search, code, calculation, analysis, instructions, problem-solving,
           AND any "remember X" where X is a temporary / follow-up / project
           state rather than a user trait.
           When in doubt between task and preference — choose task.

Return strictly JSON:
  {"intent": "chat" | "preference" | "task", "reason": "short justification"}"""

PREFERENCE_EXTRACTOR_SYSTEM = """You are a user preference extractor.
The user said something about how to communicate with them or shared a stable
personal fact. The USER PROFILE block (above) shows what we already know,
including their preferred language.

Extract the key point and return strictly JSON:
{
  "category": "language" | "style" | "about_user" | "rule" | "reject",
  "fact": "short third-person phrase (one sentence)",
  "acknowledgment": "warm brief confirmation in the USER PROFILE language (1 sentence)"
}

Rules:
- category:
    "language"   — about communication language;
    "style"      — about tone, brevity, formality, formatting;
    "about_user" — a STABLE fact about the user (name, age, profession, city,
                   relationships, long-term interests);
    "rule"       — an instruction "do/don't do X" in conversation;
    "reject"     — none of the above. Use this when the user is asking to
                   remember a TEMPORARY task state, follow-up, todo, future
                   review, or anything that isn't a stable user-profile fact.
- fact must be in the form "User ..." or "Respond ..." — short and to the point.
- acknowledgment — a natural phrase. If USER PROFILE specifies a preferred
  language, use THAT language regardless of which language the current message
  was written in. Otherwise fall back to the message language.
  No templates like "sure!", "great question!", no lists."""

CHAT_SYSTEM_BASE = """This is casual small-talk: greeting, farewell,
thanks, chitchat, "who are you" question, emotional remark.

Rules for this mode:
- Respond SHORT, warm, human-like. One or two sentences max.
- Do NOT list your capabilities or offer a menu of topics.
- Do NOT mention notes, sources, confidence, or "knowledge base".
- Take style, tone, character, and language from SOUL and USER PROFILE above.
- If USER PROFILE specifies a communication language — respond strictly in it.
- No markdown headers or bullet lists."""


def _format_llm_error_short(exc: BaseException, *, max_len: int = 200) -> str:
    """Compress an LLMError into a single-line user-facing message.

    Surfacing the actual error in chat is much more useful than a generic
    "API unavailable" fallback. We prefix with ⚠ so the UI clearly marks it
    as an error message, not a normal answer.
    """
    msg = str(exc)
    # Codex subscription quota — extract reset countdown
    if "usage_limit_reached" in msg or "quota exhausted" in msg:
        m = re.search(r'"resets_in_seconds"\s*:\s*(\d+)', msg)
        if m:
            sec = int(m.group(1))
            h = sec // 3600
            mi = (sec % 3600) // 60
            return f"⚠ Codex (ChatGPT) quota exhausted. Resets in {h}h {mi}m."
        return "⚠ Codex (ChatGPT) quota exhausted."
    # Server error with explicit "detail" field (Codex style)
    m = re.search(r'"detail"\s*:\s*"([^"]+)"', msg)
    if m:
        return f"⚠ {m.group(1)[:max_len]}"
    # OpenAI / Anthropic style {"error":{"message":"..."}}
    m = re.search(r'"message"\s*:\s*"([^"]+)"', msg)
    if m:
        return f"⚠ {m.group(1)[:max_len]}"
    # Default: first line, trimmed
    short = msg.split("\n")[0].strip()
    if len(short) > max_len:
        short = short[: max_len - 3] + "..."
    return f"⚠ {short}"


def _with_identity(base_system: str) -> str:
    """Склеивает identity preamble с конкретным system-промптом шага."""
    return f"{IDENTITY.preamble()}\n\n---\n\n{base_system}"


def _capabilities_block(compact: bool = False) -> str:
    """Динамический блок «MY CAPABILITIES» для system prompt.

    Перечисляет конкретные вещи, которые агент имеет и умеет прямо сейчас:
    зарегистрированные tools, загруженные skills, подключённые MCP серверы,
    и карту своего исходного кода на диске — чтобы при вопросах о себе агент
    знал, куда смотреть (read_file), а не фантазировал.

    `compact=True` skips the source map (~2-3 KB) and trims tool
    descriptions to 60 chars. Used for paths that don't need to know
    "where am I implemented?" — chat and intent classification.
    The full block stays for solver and self-analysis where the model
    may want to read its own code.
    """
    registry = get_registry()
    SKILLS.ensure_loaded()

    lines = ["# MY CAPABILITIES"]

    # --- Tools ---
    tools = registry.tools
    if tools:
        lines.append("\n## Инструменты (tools)")
        desc_cap = 60 if compact else 100
        for name, tool in sorted(tools.items()):
            origin_label = f"  [{tool.origin}]" if tool.origin != "builtin" else ""
            lines.append(f"- `{name}` — {tool.description[:desc_cap]}{origin_label}")

    # --- Skills ---
    if SKILLS.skills:
        lines.append("\n## Навыки (skills)")
        for sk in SKILLS.skills:
            triggers = ", ".join(sk.triggers[:5]) if sk.triggers else "—"
            desc_cap = 60 if compact else 80
            lines.append(f"- **{sk.name}**: {sk.description[:desc_cap]}  (triggers: {triggers})")

    # --- MCP ---
    if MCP.servers and not compact:
        lines.append("\n## Подключённые MCP-серверы")
        for srv_name, srv in MCP.servers.items():
            tool_count = len([t for t in tools if t.startswith(f"mcp_{srv_name}__")])
            lines.append(f"- `{srv_name}` ({tool_count} tools)")

    if compact:
        # Compact stops here. The big source map is only useful when
        # the model is going to actually read its own code; chat /
        # think don't need it and pay for those ~3k chars on every turn.
        return "\n".join(lines)

    # --- Source map (для вопросов о себе) ---
    root = Path(__file__).resolve().parent.parent
    lines.append("\n## Мой исходный код (source map)")
    lines.append(f"Корневая директория: `{root}`")
    lines.append("Ключевые файлы:")
    source_map = {
        # --- Core pipeline ---
        "backend/agent.py": "main agent loop: classify → think → solve → self-critic retry → verify",
        "backend/llm.py": "dual-model router (Claude+Ollama), tool-use loop, retry with backoff, token tracking",
        "backend/verifier.py": "answer verification (confidence, contradictions, tool output as evidence)",
        "backend/models.py": "Pydantic models (Note, TaskAnalysis, AgentAnswer, ThinkingResult, TokenUsage...)",
        # --- Knowledge & memory ---
        "backend/knowledge_manager.py": "CRUD notes + index + history + gap tracking",
        "backend/knowledge_graph.py": "LightRAG-inspired entity-relation graph, BFS traversal, JSON-backed",
        "backend/hybrid_searcher.py": "hybrid search: fuzzy keyword (60%) + graph traversal (40%)",
        "backend/memory_extractor.py": "IMPLEMENTED: extracts facts from conversations → stores as triples in knowledge graph (source=_memory)",
        "backend/core_memory.py": "persistent facts always in context",
        "backend/conversation.py": "sliding-window conversation memory (last N turns, persisted JSON)",
        "backend/searcher.py": "keyword + fuzzy search over knowledge base",
        # --- AGI modules (ALREADY IMPLEMENTED) ---
        "backend/meta_learner.py": "IMPLEMENTED: failure analysis, pattern extraction, corrective goal creation",
        "backend/evaluator.py": "IMPLEMENTED: per-day evaluation stats, confidence tracking, daily reports",
        "backend/analogy_engine.py": "IMPLEMENTED: pattern extraction from solutions, cross-domain analogy search, context_block for solver",
        "backend/self_modifier.py": "IMPLEMENTED: code analysis, patch proposals (approve/reject/apply), safe self-modification",
        "backend/goals.py": "IMPLEMENTED: goal manager with auto-suggestions from knowledge gaps, proactive learning",
        # --- Tools & skills ---
        "backend/tool_registry.py": "tool registry (register/execute)",
        "backend/builtin_tools.py": "builtin tools: web_search, fetch_url, read_file (with start_line/end_line), run_python",
        "backend/tools/calc.py": "safe AST-based arithmetic evaluator (backing for the `calc` skill)",
        "backend/skills/calc/handler.py": "registers the `calc` tool — use it for arithmetic instead of run_python",
        "backend/skills.py": "skill loader from SKILL.md + handler.py",
        "backend/mcp_client.py": "MCP client (sync-async bridge)",
        # --- Identity & config ---
        "backend/identity.py": "soul.md / identity.md / user.md",
        "backend/config.py": "config.yaml + MODE_PRESETS",
        "backend/note_creator.py": "note generation from web + auto entity extraction into knowledge graph",
        "backend/commands.py": "CLI command parser (remember, learn, status, gaps, graph...)",
        # --- Infrastructure ---
        "backend/main.py": "FastAPI server, SSE streaming, all API endpoints",
        "backend/sessions.py": "session management, turn persistence",
        "backend/finetune.py": "finetune data collection and queue",
        "backend/finetune_pipeline.py": "finetune training pipeline",
        "backend/project_mode.py": "project-scoped knowledge isolation",
        "backend/background.py": "background task runner",
        "cli.py": "CLI entry point (REPL)",
        "config.yaml": "mode and parameter configuration",
        "frontend/": "React + TypeScript UI: Chat, Graph, Knowledge, Sessions, Intelligence, Usage panels",
    }
    for path, desc in source_map.items():
        lines.append(f"- `{path}` — {desc}")
    lines.append("\nЧтобы узнать, как я устроен — зови `read_file` на любой из этих файлов.")
    lines.append("НЕ делай утверждений о своём коде, не прочитав файл.")

    return "\n".join(lines)


# Быстрый фильтр для самых очевидных случаев — без единого LLM-вызова.
# Сознательно узкий: ловит только бесспорную болтовню, всё неоднозначное
# уходит в LLM-классификатор.
# Arithmetic detector — any expression with two numbers (incl. decimals,
# percent, parentheses) and a binary operator. We don't try to be a full
# calculator parser; just enough that "2+2", "15 * 3", "(5+3)/2", "10%
# of 200", "2^10" trigger the marker. False positives are cheap (one
# wasted run_python call); false negatives are expensive (hallucinated
# arithmetic answers, which already showed up as a tracked goal).
_ARITHMETIC_RE = re.compile(
    r"(?:"
    r"\d+(?:\.\d+)?\s*[+\-*/^×÷%]\s*\d+(?:\.\d+)?"
    r"|\d+\s*\^\s*\d+"
    r"|sqrt\s*\(\s*\d"
    r"|\d+\s*%\s*(?:of|от)\s+\d+"
    r")",
    re.IGNORECASE,
)

# Word-form arithmetic ("сколько будет два плюс два", "calculate 2+2",
# "10 процентов от 250"). Triggers ONLY when the message ALSO contains
# at least one digit — keeps conversational uses of the words ("просто
# минус один") from misfiring.
_ARITHMETIC_WORDS_RE = re.compile(
    "(?:" + "|".join([
        # Russian operator words
        r"\bплюс\b", r"\bминус\b", r"\bумнож", r"\bраздел",
        r"\bделит", r"\bв\s+степен", r"\bкорень\s+из",
        r"\bпроцент(?:а|ов|ы)?\b",
        # Russian "compute it" verbs
        r"\bсколько\s+(?:будет|это|равно)\b",
        r"\bпосчита(?:й|ть|ем)",
        r"\bвычисл",
        r"\bподсчита(?:й|ть)",
        # English operator + verb words
        r"\bplus\b", r"\bminus\b", r"\bmultiply", r"\bdivid",
        r"\btimes\b",
        r"\bsquare\s+root\b", r"\bsqrt\b",
        r"\bto\s+the\s+power\b", r"\bpower\s+of\b",
        r"\bcalculat", r"\bcompute",
        r"\bsum\s+of\b", r"\bproduct\s+of\b", r"\bpercent\s+of\b",
    ]) + ")",
    re.IGNORECASE | re.UNICODE,
)
_ARITHMETIC_DIGIT_RE = re.compile(r"\d")


def _looks_like_arithmetic(s: str) -> bool:
    """True if `s` contains an arithmetic-looking expression. Used to
    force the solver into a calc/run_python path instead of letting it
    answer from training data — answering arithmetic from memory is the
    classic source of "2+2 = 5" hallucinations.

    Two complementary detectors:
      - symbolic form (`2+2`, `15 * 3`, `sqrt(16)`)
      - word form ("сколько будет 25 умножить на 3", "calculate 12 times 4")
    Word form requires at least one digit to fire so generic prose with
    "minus" / "plus" doesn't trigger.
    """
    if not s:
        return False
    if _ARITHMETIC_RE.search(s):
        return True
    if _ARITHMETIC_DIGIT_RE.search(s) and _ARITHMETIC_WORDS_RE.search(s):
        return True
    return False


# Tools that count as "reading the source" for self-analysis guard.
# Only these prove the solver actually grounded its claims in the
# current file contents — calc/web_search/run_python don't qualify
# even if they produced tool_context, because the agent could have
# pulled in unrelated content (a wiki page, an arithmetic result)
# and still hallucinated about its own architecture.
SOURCE_READ_TOOLS = frozenset({"read_file", "view_file"})


# Micro-acknowledgements: messages that are clearly just "I heard you"
# and don't deserve an LLM call. Tighter than _CHITCHAT_RE — that one
# also matches "who are you" / "how are you" which DO need a real
# answer from _chat_reply. _MICRO_ACK_RE matches only one-liners that
# the agent can answer with a static "✓".
_MICRO_ACK_RE = re.compile(
    # "continue" / "продолжай" / "go on" intentionally NOT here:
    # users frequently mean "do more work, pick up where you left
    # off" rather than an ack — that needs the full pipeline.
    r"^\s*(?:"
    r"ok|okay|окей|ок|"
    r"thanks?|thank\s*you|спасибо|благодарю|thx|спс|мерси|ty|"
    r"got\s*it|gotcha|понял|поняла|ясно|ага|угу|"
    r"cool|nice|круто|здорово|awesome|👍|"
    r"yes|yep|да|нет|no|nope"
    r")"
    r"[\s!?.,…)✀-➿\U0001F300-\U0001FAFF:]*$",
    re.IGNORECASE | re.UNICODE,
)


def _micro_ack_reply(task: str) -> str | None:
    """Static reply for trivial acknowledgements. Returns None when
    the message needs a real LLM response. Saves one LLM call (~300
    tokens) plus identity preamble (~5-10k tokens) per ack — the
    accumulated cost was real on long sessions where the user types
    "ok" / "thanks" / "continue" between substantive turns.

    Length cap: only fires for very short messages. A long message
    that happens to start with "ok" might actually be content
    ("ok so the issue is...") and needs a real reply.
    """
    if not task or len(task.strip()) > 30:
        return None
    if _MICRO_ACK_RE.match(task.strip()):
        return "✓"
    return None


_CHITCHAT_RE = re.compile(
    r"^\s*(?:"
    r"hi|hello|hey|yo|hola|"
    r"привет|здравствуй(?:те)?|хай|йо|салют|"
    r"bye|goodbye|пока|до\s*свидания|прощай|чао|"
    r"thanks?|thank\s*you|спасибо|благодарю|thx|спс|мерси|"
    r"ok|okay|окей|ок|понял|поняла|ясно|got\s*it|ага|угу|"
    r"как\s*(?:у\s*тебя\s*)?дела|как\s*ты|how\s*are\s*you|как\s*жизнь|"
    r"кто\s*ты|who\s*are\s*you|what\s*are\s*you|"
    r"доброе\s*утро|добрый\s*(?:день|вечер)|"
    r"good\s*(?:morning|day|evening|night)|"
    r"спокойной\s*ночи"
    r")"
    r"[\s!?.,…)\U0001F300-\U0001FAFF:)-]*$",
    re.IGNORECASE | re.UNICODE,
)


# Детектор вопросов агента о самом себе — используется в _chat_reply,
# чтобы подмешать capabilities block даже в лёгком режиме.
_SELF_QUESTION_RE = re.compile(
    r"(?:"
    r"кто\s*ты|что\s*ты\s*(?:умеешь|можешь|такое)|что\s*ты\s*за\s*(?:агент|бот|ии)"
    r"|что\s*ты\s*знаешь\s*о\s*себе"
    r"|расскажи\s*о\s*себе|опиши\s*себя"
    r"|какие\s*у\s*тебя\s*(?:инструменты|возможности|навыки|скиллы|tools)"
    r"|what\s*(?:can\s*you\s*do|are\s*(?:you|your\s*(?:tools|capabilities|skills)))"
    r"|who\s*are\s*you|tell\s*me\s*about\s*yourself|describe\s*yourself"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _is_self_question(text: str) -> bool:
    return bool(_SELF_QUESTION_RE.search(text))


# Wider net: catches code-review / self-improvement / token-usage
# requests that aren't phrased as "who are you" but DO require the
# agent to read its own source (and therefore want the full source
# map + git log + skipped KB notes). Keeps planning aligned with
# the eventual solver path.
_SELF_ANALYSIS_HINT_RE = re.compile(
    r"(?:"
    r"\byour\s+(?:code|source|architecture|implementation|prompts?)\b"
    r"|\bsource\s+code\b|\bself[\s-]?analysis\b|\bself[\s-]?review\b"
    r"|\btoken\s+usage\b|\boptim(?:i[zs]e|i[sz]ation|i[zs]ed)\b"
    r"|backend/[a-z_/]+\.py"
    r"|\bagent\.py\b|\bllm\.py\b|\bverifier\.py\b"
    # Russian
    r"|твой\s+код|своего?\s+код|исходны?й\s+код|архитектур"
    r"|оптимизац|самоанализ|проверь\s+(?:свой\s+)?код"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _looks_like_self_analysis_request(text: str) -> bool:
    """Cheap pre-classifier for self-analysis intent. The LLM
    classifier still runs and wins, but `_think`'s prompt-shaping
    needs the answer earlier — before we know `question_type`.
    Conservative: only matches phrases that strongly imply the
    agent will need to read its own code.
    """
    if not text:
        return False
    return bool(_SELF_ANALYSIS_HINT_RE.search(text))


# ---------- тип колбэка для прогресса ----------
ProgressCB = Callable[..., None]  # (event, message, tool_call: ToolCallDetail|None=None)


def _noop(_evt: str, _msg: str, _tc: "ToolCallDetail | None" = None) -> None:
    pass


# Лениво подключаем MCP-серверы один раз на процесс. Если в config.yaml ничего
# не задано — это no-op. Делаем модульно (а не в Agent.__init__), чтобы тесты
# с замоканным агентом не пытались поднимать subprocess.
_mcp_bootstrap_done = False


def _bootstrap_mcp() -> None:
    global _mcp_bootstrap_done
    if _mcp_bootstrap_done:
        return
    _mcp_bootstrap_done = True
    raw = CONFIG.mcp_servers
    if not raw:
        return
    configs = [
        MCPServerConfig(
            name=str(s.get("name") or f"server_{i}"),
            command=str(s.get("command") or ""),
            args=list(s.get("args") or []),
            env=dict(s.get("env") or {}),
            enabled=bool(s.get("enabled", True)),
        )
        for i, s in enumerate(raw)
    ]
    MCP.connect_all(configs)


# ---------- агент ----------
class Agent:
    def __init__(self, progress: Optional[ProgressCB] = None):
        self._user_progress = progress or _noop
        self._trace: list[ThinkingStep] = []
        self._llm_calls: list[LLMCallDetail] = []
        self._request_id: str = new_request_id()
        self._t0: float = 0.0
        _bootstrap_mcp()

    def _record_llm_call(
        self,
        *,
        label: str,
        task_type: "TaskType | str",
        system: str,
        user: str,
        response: str,
        duration_ms: int = 0,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        usage_before: dict | None = None,
    ) -> None:
        """Capture one LLM invocation with file blobs redacted, for the
        WebUI dev-mode panel. Best-effort: errors here must NEVER crash
        the agent — if redaction throws (regex catastrophic backtrack,
        encoding glitch), we still want the answer to ship.

        `usage_before` is a snapshot of `TOKENS.request_usage()` taken
        BEFORE the LLM call. We diff it against the current usage to
        attribute tokens / cost / model to THIS specific call. Without
        the diff trick we'd have to plumb counts through every LLM
        class — this keeps the call sites clean.
        """
        try:
            tt_value = task_type.value if hasattr(task_type, "value") else str(task_type)
            sys_red = redact_prompt(system or "")
            usr_red = redact_prompt(user or "")
            # Diff token counts against the snapshot. Solve's tool loop
            # makes many LLM iterations under one logical call; the diff
            # totals them, which is what dev-panel readers want anyway.
            if usage_before is not None and (input_tokens == 0 and output_tokens == 0):
                try:
                    after = TOKENS.request_usage()
                    input_tokens = max(0, int(after.get("input_tokens", 0)) - int(usage_before.get("input_tokens", 0)))
                    output_tokens = max(0, int(after.get("output_tokens", 0)) - int(usage_before.get("output_tokens", 0)))
                except Exception:
                    pass
            if not model:
                # Pull the most recently logged model from TokenTracker.
                try:
                    last = TOKENS.last_record()
                    if last is not None:
                        model = last.get("model") or model
                except Exception:
                    pass
            self._llm_calls.append(LLMCallDetail(
                label=label,
                task_type=tt_value,
                model=model,
                system_redacted=sys_red,
                user_redacted=usr_red,
                response_preview=(response or "")[:600],
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ))
        except Exception:
            pass

    def progress(
        self,
        event: str,
        message: str,
        tool_call: "ToolCallDetail | None" = None,
    ) -> None:
        """Record thinking step and forward to user callback.

        `tool_call` carries the structured tool-call payload (name, args,
        TRUNCATED result preview, plus result_truncated/result_full_len
        metadata so the WebUI panel can show "preview, 4000 of 50000
        chars"). The full body is NOT in the trace — it's available
        only to the verifier-side `tool_outputs` buffer and the
        immediate next iteration of the LLM tool-loop, both of which
        have their own (separate) caps. See `_on_tool_call` and
        `_compact_tool_result_for_llm`.
        """
        import time as _time
        elapsed = _time.monotonic() - self._t0 if self._t0 else 0.0
        usage = TOKENS.request_usage()
        self._trace.append(ThinkingStep(
            ts=round(elapsed, 2),
            event=event,
            message=message,
            tokens_so_far=usage["total_tokens"],
            tool_call=tool_call,
        ))
        # Round B: pass tool_call to the consumer callback when
        # available so the WebUI can render OpenClaw-style pills as
        # the agent works (vs. only after agent.run() finishes). The
        # try/except handles legacy callbacks that only accept (event,
        # message) — Telegram's progress streamer uses the 2-arg form
        # for the placeholder text, never the 3-arg form.
        try:
            self._user_progress(event, message, tool_call)
        except TypeError:
            self._user_progress(event, message)

    # Шаг 1
    def _load_core(self) -> str:
        self.progress("core", "загружаю core memory")
        return CORE.read()

    @staticmethod
    def _git_log_block(n: int = 50) -> str:
        """Recent commit log so the agent knows what its own code
        actually contains right now — not what it remembered from a
        snapshot in the knowledge base.

        For self-analysis turns this beats `MEMORY.recall_block` (which
        replays past observations about possibly-old code) and points
        the agent at fresh state. Best-effort: returns "" outside a
        git working tree or if the binary isn't present.
        """
        import subprocess
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        try:
            out = subprocess.run(
                ["git", "log", f"-{int(n)}", "--oneline", "--no-decorate"],
                cwd=repo, capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return ""
        if out.returncode != 0 or not out.stdout.strip():
            return ""
        return "# RECENT COMMITS (latest first — what the code looks like now)\n" + out.stdout.strip()

    def _shared_context(
        self,
        task: str,
        core: str | None = None,
        *,
        n_conv: int = 6,
        n_memory: int = 8,
        max_goals: int = 5,
        for_self_analysis: bool = False,
    ) -> str:
        """Common per-turn context block used by chat / think / solve.

        Soul / identity / user profile / name+language overrides are
        already in the system prompt via `_with_identity`. THIS block
        carries the per-turn state that tells the model "what we are
        doing right now":

          - CORE memory (long-term persistent facts)
          - CURRENT PROJECT (active workspace, if any)
          - ACTIVE GOALS (what the user is working on)
          - SHORT-TERM MEMORY (semantic recall of facts relevant to
            this query — pulled from the memory graph)
          - CONVERSATION (last `n_conv` turns)

        When `for_self_analysis=True` (the agent is being asked to
        review its own code), short-term memory is REPLACED with a
        recent git log. Memory holds past observations about possibly-
        old code state; for code review the only authoritative source
        is the file on disk + what changed lately. This is the fix for
        the recurring "agent finds bugs that were already fixed last
        commit" pattern — recall was reproducing stale findings.

        Empty sections are omitted so we don't waste tokens. Returned
        as a markdown-headed block ready to drop into the user message
        BEFORE the actual USER REQUEST.
        """
        parts: list[str] = []

        if core is None:
            try:
                core = CORE.read()
            except Exception:
                core = ""
        if core and core.strip():
            parts.append(f"# CORE MEMORY\n{core.strip()}")

        try:
            project = PROJECTS.current
        except Exception:
            project = None
        if project:
            parts.append(f"# CURRENT PROJECT\n{project}")

        try:
            goals_block = GOALS.context_block(max_goals=max_goals)
        except Exception:
            goals_block = ""
        if goals_block.strip():
            parts.append(goals_block.strip())

        if for_self_analysis:
            # Trade stale recall for fresh git log on self-analysis turns.
            git_block = self._git_log_block(n=50)
            if git_block:
                parts.append(git_block)
        else:
            try:
                memory_block = MEMORY.recall_block(task, max_facts=n_memory)
            except Exception:
                memory_block = ""
            if memory_block and memory_block.strip():
                parts.append(memory_block.strip())

        try:
            conv = CONVERSATION.context_block(
                n=n_conv,
                channel=getattr(self, "_channel", None),
            )
        except Exception:
            conv = ""
        if conv and conv.strip():
            parts.append(conv.strip())

        return "\n\n".join(parts)

    def _arithmetic_marker(self, task: str) -> str:
        """Notice prepended to the user prompt when arithmetic is
        detected. Forces the solver to run a calculator tool instead of
        answering from training data. Empty when the task isn't
        arithmetic, so unrelated turns aren't polluted.
        """
        if not _looks_like_arithmetic(task):
            return ""
        return (
            "[ARITHMETIC DETECTED] The user's message contains an "
            "arithmetic expression. You MUST call `calc` or `run_python` "
            "to compute the answer. Do NOT answer arithmetic from "
            "memory — answering from training data is the documented "
            "source of '2+2 = 5'-style hallucinations on this agent.\n\n"
        )

    def _attachment_marker(self) -> str:
        """Textual hint about attachments on the current turn.

        The image/audio bytes themselves go into the multimodal payload
        via `attachments=` on the LLM call, but the text prompt also
        needs to mention them — otherwise classifier / thinker / solver
        get only the user's question text and bias toward continuing
        whatever was discussed in conversation history.

        Each attachment is also mirrored into `workspace/inbox/<name>`
        (see workspace.py) so the model can call `read_file` on a path
        it actually sees in the prompt — that fixes the "file not found"
        loop where the LLM tried `read_file("contract.pdf")` against the
        cwd. The marker now lists those paths explicitly.
        """
        atts = getattr(self, "_attachments", None) or []
        if not atts:
            return ""
        try:
            from .attachments import ATTACHMENTS as _A
            metas = []
            for sha in atts:
                m = _A.get_meta(sha)
                if m is not None:
                    metas.append(m)
        except Exception:
            metas = []
        if not metas:
            return (
                f"[ATTACHMENT NOTICE] The user attached {len(atts)} item(s) on "
                f"THIS turn. Their question is grounded in the attached content.\n\n"
            )
        n_img = sum(1 for m in metas if getattr(m, "kind", "") == "image")
        n_audio = sum(1 for m in metas if getattr(m, "kind", "") == "audio")
        n_file = sum(
            1 for m in metas
            if getattr(m, "kind", "") not in ("image", "audio")
        )
        parts: list[str] = []
        if n_img:
            parts.append(f"{n_img} image" + ("s" if n_img > 1 else ""))
        if n_audio:
            parts.append(f"{n_audio} voice/audio message" + ("s" if n_audio > 1 else ""))
        if n_file:
            parts.append(f"{n_file} file" + ("s" if n_file > 1 else ""))
        what = ", ".join(parts) if parts else f"{len(atts)} attachment"
        # Per-item path lines so the model can `read_file(path)` directly.
        # Use getattr fallbacks because some test stubs and older serialised
        # records expose a smaller surface than the full Attachment dataclass.
        lines: list[str] = []
        for m in metas:
            kind = getattr(m, "kind", "attachment")
            label = getattr(m, "filename", "") or kind
            path = (
                getattr(m, "workspace_path", "")
                or "(not mirrored — bytes only available via multimodal)"
            )
            extra = ""
            transcript = getattr(m, "transcript", "") or ""
            if kind == "audio" and transcript:
                preview = transcript[:100].replace("\n", " ")
                extra = f" (transcript: {preview}{'…' if len(transcript) > 100 else ''})"
            lines.append(f"  - {kind} `{label}` → {path}{extra}")
        path_block = "\n".join(lines)
        return (
            f"[ATTACHMENT NOTICE] The user attached {what} on THIS turn. "
            f"Their question is grounded in the attached content — look at "
            f"the image / read the file / consider the audio transcript "
            f"FIRST, before relying on prior conversation history.\n"
            f"Workspace paths (use `read_file` for text/pdf/docx; image and "
            f"audio bytes are already attached to the LLM call):\n"
            f"{path_block}\n\n"
        )

    # Шаг 1.5 — классификация намерения (chat | preference | task)
    def _classify_intent(self, task: str) -> str:
        """Возвращает одну из строк: 'chat', 'preference', 'task'.

        Приоритеты:
          1. Длинное сообщение (>300 символов) — почти всегда task;
             preference обычно пишется коротко.
          2. Очевидный chitchat по regex — chat без LLM-вызова.
          3. Иначе — быстрый LLM-классификатор (3 категории).
             При ошибке — безопасный fallback 'task'.
        """
        trimmed = task.strip()
        if len(trimmed) > 300:
            return "task"
        # Arithmetic must take the task path so the solver can call
        # calc / run_python. Skip chitchat regex AND the LLM classifier
        # for this — both have been observed routing "2+2" to chat,
        # where the model answers from training data.
        if _looks_like_arithmetic(trimmed):
            return "task"
        if _CHITCHAT_RE.match(trimmed):
            return "chat"
        try:
            marker = self._attachment_marker()
            user_prompt = f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{marker}{trimmed}"
            import time as _t
            t0 = _t.monotonic()
            usage_before = TOKENS.request_usage()
            data = router().call_json(
                TaskType.CLASSIFICATION,
                INTENT_CLASSIFIER_SYSTEM,
                user_prompt,
                max_tokens=150,
                temperature=0.0,
            )
            self._record_llm_call(
                label="_classify_intent",
                task_type=TaskType.CLASSIFICATION,
                system=INTENT_CLASSIFIER_SYSTEM,
                user=user_prompt,
                response=str(data),
                duration_ms=int((_t.monotonic() - t0) * 1000),
                usage_before=usage_before,
            )
            intent = str(data.get("intent", "task")).strip().lower()
            if intent in ("chat", "preference", "task"):
                return intent
            return "task"
        except LLMError:
            # LLM retried and still failed — propagate to stop the pipeline
            raise

    # Быстрый chat-ответ: один LLM-вызов с identity preamble.
    def _chat_reply(self, task: str, core: str) -> str:
        self.progress("chat", "chatting...")
        system = _with_identity(CHAT_SYSTEM_BASE)
        if _is_self_question(task):
            system = f"{system}\n\n---\n\n{_capabilities_block()}"
        # Full per-turn context (core + project + goals + memory recall
        # + recent conversation) so even a quick chat reply knows who
        # the user is, what we're working on, and what was just said.
        ctx = self._shared_context(task, core, n_conv=4, n_memory=6)
        marker = self._attachment_marker() + self._arithmetic_marker(task)
        user = (
            f"{ctx}\n\n"
            f"# MESSAGE\n{marker}{task.strip()}"
        )
        attachments = getattr(self, "_attachments", None)
        try:
            import time as _t
            t0 = _t.monotonic()
            usage_before = TOKENS.request_usage()
            out = router().call(
                TaskType.QUICK_ANSWER,
                system,
                user,
                max_tokens=300, temperature=0.6,
                attachments=attachments,
            )
            self._record_llm_call(
                label="_chat_reply",
                task_type=TaskType.QUICK_ANSWER,
                system=system,
                user=user,
                response=out,
                duration_ms=int((_t.monotonic() - t0) * 1000),
                usage_before=usage_before,
            )
            return out
        except LLMError as e:
            # Surface the actual error short. Better than a generic fallback —
            # the user can see whether it's quota / bad model / network / etc.
            return _format_llm_error_short(e)

    @staticmethod
    def _chat_fallback(task: str) -> str:
        """Offline fallback for simple chat when LLM is unavailable."""
        t = task.strip().lower()
        if any(w in t for w in ("привет", "hello", "hi ", "hey", "хай", "салют")):
            return "Привет! API временно недоступен, но я на связи. Спроси что-нибудь позже или попробуй ещё раз."
        if any(w in t for w in ("пока", "bye", "до свидания", "goodbye")):
            return "Пока! До встречи."
        if any(w in t for w in ("спасибо", "thanks", "thank")):
            return "Пожалуйста!"
        if any(w in t for w in ("как дела", "как ты", "how are you")):
            return "API сейчас перегружен, но в целом я в порядке. Попробуй ещё раз через минуту."
        if _is_self_question(task):
            # Answer from identity files — no LLM needed
            try:
                soul = IDENTITY.soul()
                identity = IDENTITY.identity()
                # Extract a short self-description from identity.md
                lines = [l.strip() for l in identity.splitlines()
                         if l.strip() and not l.startswith("#")]
                short = " ".join(lines[:5]) if lines else ""
                if short:
                    return f"{short}\n\n_(API недоступен — ответ из identity.md)_"
            except Exception:
                pass
            return "Я — самообучающийся AI-агент. API сейчас недоступен, но спроси позже — расскажу подробнее."
        return "Сейчас API недоступен. Попробуй повторить через минуту — я буду готов помочь."

    # Preference — извлекаем структурированное предпочтение и сохраняем в нужное место.
    def _save_preference(self, task: str) -> tuple[str, str, str]:
        """Возвращает (category, fact, acknowledgment).

        Triage:
          language / style / about_user / rule → user.md (IDENTITY)
          reject                                → conversation memory only;
                                                  the extractor decided this
                                                  is not a stable profile fact.

        The LLM sees USER PROFILE in the system prompt so it can answer
        in the user's preferred language regardless of the message
        language (fix for "user wrote in English but expects Russian
        replies per user.md").
        """
        self.progress("preference", "запоминаю предпочтение")
        # Surface USER PROFILE so the extractor can pick the right language
        # for the acknowledgment AND make a confident reject/keep decision.
        profile_block = ""
        try:
            profile_block = (IDENTITY.user_profile() or "").strip()
        except Exception:
            profile_block = ""

        system_prompt = PREFERENCE_EXTRACTOR_SYSTEM
        if profile_block:
            system_prompt = (
                f"USER PROFILE:\n{profile_block}\n\n---\n\n{PREFERENCE_EXTRACTOR_SYSTEM}"
            )

        user_prompt = f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{task.strip()}"
        try:
            import time as _t
            t0 = _t.monotonic()
            usage_before = TOKENS.request_usage()
            data = router().call_json(
                TaskType.CLASSIFICATION,
                system_prompt,
                user_prompt,
                max_tokens=300, temperature=0.1,
            )
            self._record_llm_call(
                label="_save_preference",
                task_type=TaskType.CLASSIFICATION,
                system=system_prompt,
                user=user_prompt,
                response=str(data),
                duration_ms=int((_t.monotonic() - t0) * 1000),
                usage_before=usage_before,
            )
        except LLMError:
            # On extractor failure — DON'T blindly stuff raw input into user.md.
            # That's the bug that produced "User will fix everything..." rows.
            # Acknowledge in conversation memory only.
            return "reject", task.strip(), "Запомнил в контексте разговора."

        category = str(data.get("category", "about_user")).strip().lower()
        valid_profile = ("language", "style", "about_user", "rule")
        fact = str(data.get("fact", "")).strip() or task.strip()
        ack = str(data.get("acknowledgment", "")).strip() or "Запомнил."

        if category == "reject" or category not in valid_profile:
            # Conversation-scoped memory only. CONVERSATION.add_turn at the
            # caller already records the exchange — nothing to write here.
            self.progress("preference_skipped", "не сохраняю в user.md (не профильный факт)")
            return "reject", fact, ack

        IDENTITY.add_user_fact(fact, category=category)  # type: ignore[arg-type]
        self.progress("preference_saved", f"записано в user.md → {category}")
        return category, fact, ack

    # Step 2 — universal thinking
    def _think(self, task: str, core: str) -> ThinkingResult:
        """Universal thinking protocol: reason about ANY request before acting.

        This is the agent's "brain" — it decides what the question is,
        what it already knows, what tools to use, and why. The result
        drives all subsequent steps (knowledge loading, tool selection,
        solver prompting).
        """
        self.progress("think", "thinking...")
        # _think is the planning step — it picks question_type / tools /
        # required_topics. The full ~3k-char source map is only useful
        # for self-analysis (where it suggests where to look). For
        # everything else the compact view (tools + skills, no map) is
        # sufficient. Plus narrower convo/memory windows: 3/4 vs 6/10.
        # The previous defaults were tuned for solving, not classifying.
        #
        # Self-analysis also includes implicit code-review requests
        # ("check your token usage", "review the verifier", "look at
        # backend/agent.py"). These don't trip _is_self_question (which
        # is "who are you"-shaped) but DO need the full source map to
        # plan reads — and crucially, must NOT pull stale memory recall
        # because that's where "agent finds bugs already fixed last
        # commit" hallucinations come from.
        is_selfish = (
            _is_self_question(task)
            or _looks_like_self_analysis_request(task)
        )
        caps = _capabilities_block(compact=not is_selfish)
        ctx = self._shared_context(
            task, core,
            n_conv=6 if is_selfish else 3,
            n_memory=10 if is_selfish else 4,
            # Skip stale memory recall on planning step too — same
            # reason _solve does it later. Replaces the recall block
            # with a fresh git log via _git_log_block.
            for_self_analysis=is_selfish,
        )
        marker = self._attachment_marker() + self._arithmetic_marker(task)
        user = (
            f"{ctx}\n\n"
            f"# MY CAPABILITIES\n{caps}\n\n"
            f"# USER REQUEST\n{marker}{task}"
        )
        # Identity is part of who's thinking — without it the analyzer
        # can't tell that "Hrant?" is the user addressing the agent
        # by name, or that "ответь по-русски" matches the user's
        # pinned language preference. _think used to skip identity;
        # now it goes through _with_identity like chat / solve do.
        import time as _t
        _t0 = _t.monotonic()
        think_system = _with_identity(THINKING_SYSTEM)
        usage_before = TOKENS.request_usage()
        data = router().call_json(
            TaskType.TASK_ANALYSIS,
            think_system, user,
            max_tokens=1000, temperature=0.2,
        )
        self._record_llm_call(
            label="_think",
            task_type=TaskType.TASK_ANALYSIS,
            system=think_system,
            user=user,
            response=str(data),
            duration_ms=int((_t.monotonic() - _t0) * 1000),
            usage_before=usage_before,
        )
        result = ThinkingResult(
            question_type=str(data.get("question_type", "factual")).strip(),
            core_question=str(data.get("core_question", task)).strip(),
            already_know=list(data.get("already_know") or []),
            knowledge_gaps=list(data.get("knowledge_gaps") or []),
            approach=str(data.get("approach", "")).strip(),
            tools_needed=list(data.get("tools_needed") or []),
            tools_reasoning=str(data.get("tools_reasoning", "")).strip(),
            required_topics=list(data.get("required_topics") or []),
            plan=list(data.get("plan") or []),
            confidence=int(data.get("confidence", 50)),
            reasoning=str(data.get("reasoning", "")).strip(),
            subtasks=list(data.get("subtasks") or []),
        )
        if result.subtasks:
            self.progress(
                "decompose",
                f"complex task → {len(result.subtasks)} subtasks",
            )
        return result

    # Backward-compat shim: tests that mock _analyze still work.
    def _analyze(self, task: str, core: str) -> TaskAnalysis:
        t = self._think(task, core)
        return TaskAnalysis(
            required_topics=t.required_topics,
            plan=t.plan,
            confidence=t.confidence,
            reasoning=t.reasoning,
        )

    # Шаг 3
    def _ensure_knowledge(
        self,
        topics: list[str],
        project: str | None,
        *,
        allow_learning: bool = True,
    ) -> tuple[list[Note], list[str]]:
        """Resolve topics → loaded notes, optionally learning new ones.

        `allow_learning=False` is used by self_analysis turns: the agent
        is reading its own code via tools, not asking about external
        knowledge. Auto-`learn_topic()`-ing "agent architecture" or
        "tool loop" via web search would pollute the KB with content
        already encoded in the source files we just read.
        """
        loaded: list[Note] = []
        learned: list[str] = []
        loaded_slugs: set[str] = set()
        # Дедуп входного списка тем
        unique_topics = list(dict.fromkeys(t.strip() for t in topics if t and t.strip()))
        for topic in unique_topics:
            # min_raw_score=0.4 rejects "iodine"-vs-"blood sugar" style
            # weak matches where the KB has nothing actually relevant
            # but min-max normalization would scale the top noise hit
            # to 1.0 and load it as a "best" match.
            hit = HYBRID.find_best(topic, min_raw_score=0.4)
            if hit:
                hit_slug = _slug(hit.topic)
                if hit_slug in loaded_slugs:
                    continue
                self.progress("found", f"found: {hit.topic}")
                note = KM.get_note(hit.topic)
                if note:
                    loaded.append(note)
                    loaded_slugs.add(hit_slug)
                    continue
            # Miss. Always log to gap-tracker; only learn if allowed.
            KM.log_miss(topic)
            if not allow_learning:
                self.progress("skip_learn", f"skipped (self-analysis): {topic}")
                continue
            self.progress("learning", f"изучаю: {topic}")
            try:
                note = learn_topic(topic, depth="quick", project=project)
                note_slug = _slug(note.frontmatter.topic)
                if note_slug not in loaded_slugs:
                    loaded.append(note)
                    loaded_slugs.add(note_slug)
                learned.append(topic)
                self.progress("learned", f"создана заметка: {topic}")
            except Exception as e:
                self.progress("error", f"не удалось изучить {topic}: {e}")
        return loaded, learned

    # вспомогательно: собрать текст заметок
    def _notes_block(self, notes: list[Note], max_total_chars: int = 12000) -> str:
        """Собирает блок заметок для контекста с мягким ограничением.

        Если суммарный размер заметок превышает cap — режем самые длинные
        (по строкам с конца), чтобы уложиться. Это избавляет от blow-up
        контекста, когда анализатор попросил 6 больших тем сразу.
        """
        if not notes:
            return "(нет загруженных заметок)"

        # Дедуп по slug на всякий случай.
        seen: set[str] = set()
        unique: list[Note] = []
        for n in notes:
            s = _slug(n.frontmatter.topic)
            if s in seen:
                continue
            seen.add(s)
            unique.append(n)

        # Простая стратегия: равная квота на заметку.
        quota = max(400, max_total_chars // max(1, len(unique)))
        parts: list[str] = []
        for n in unique:
            body = n.body
            if len(body) > quota:
                body = body[:quota].rstrip() + "\n… [обрезано для контекста]"
            parts.append(
                f"### {n.frontmatter.topic} "
                f"(источник: {n.frontmatter.source or 'внутренний'})\n{body}"
            )
        return "\n\n".join(parts)

    # Step 4 — solve with thinking context
    def _solve(self, task: str, core: str, notes: list[Note],
               thinking: ThinkingResult | None = None,
               critique: str = "") -> tuple[str, str]:
        self.progress("solve", "composing answer...")
        # P0 Phase B: every _solve call resets the structured claims
        # the solver may emit. Last writer wins — the final visible
        # answer's claims are what the user (and the verifier) see.
        # If the LLM ignores the directive, this stays empty and
        # Phase A's verifier-bucket-based path produces claims.
        self._last_solver_claims: Optional[list[dict]] = None

        # Build the THINKING section that the solver will follow as roadmap
        think_block = ""
        if thinking:
            think_lines = [
                "# THINKING (prepared by the reasoning module)",
                f"**Question type:** {thinking.question_type}",
                f"**Core question:** {thinking.core_question}",
                f"**Approach:** {thinking.approach}",
            ]
            if thinking.tools_needed:
                think_lines.append(
                    f"**Tools to use:** {', '.join(thinking.tools_needed)}"
                )
                think_lines.append(f"**Why:** {thinking.tools_reasoning}")
            if thinking.already_know:
                think_lines.append(
                    f"**Already known:** {'; '.join(thinking.already_know)}"
                )
            if thinking.knowledge_gaps:
                think_lines.append(
                    f"**Knowledge gaps:** {'; '.join(thinking.knowledge_gaps)}"
                )
            if thinking.plan:
                think_lines.append("**Plan:**")
                for i, step in enumerate(thinking.plan, 1):
                    think_lines.append(f"  {i}. {step}")
            think_block = "\n".join(think_lines)

        critique_block = f"\n{critique}\n" if critique else ""

        # Unified per-turn context: core + project + goals + memory
        # recall + recent conversation. Goals + project landed here
        # alongside the existing memory/conv sections so the solver
        # knows what we're working on, not just the immediate query.
        is_self_analysis = bool(thinking and thinking.question_type == "self_analysis")
        ctx = self._shared_context(
            task, core, n_conv=6, n_memory=8,
            for_self_analysis=is_self_analysis,
        )
        marker = self._attachment_marker() + self._arithmetic_marker(task)
        # On self-analysis turns the NOTES block is dropped — KB notes
        # are snapshots and reproduce stale code state. The solver
        # MUST read live source via read_file. A short directive
        # replaces the section so the model knows why it's empty.
        if is_self_analysis:
            notes_section = (
                "# SOURCE OF TRUTH\n"
                "This is a self-analysis turn. KB notes about the code "
                "are intentionally omitted because they're snapshots and "
                "may report state that has since been fixed. Read the "
                "actual source files via `read_file` and the recent "
                "commits in `# RECENT COMMITS` above. Do NOT propose "
                "fixes for code you haven't read this turn.\n\n"
                "## TOKEN-EFFICIENT READING (mandatory)\n"
                "Backend source files are large — `agent.py` is "
                "~2000 lines, `llm.py` ~2500 lines. Reading the WHOLE "
                "file is wasteful and the result gets truncated when "
                "re-fed to me on the next iteration anyway (12k char "
                "cap per `read_file` result).\n\n"
                "RULES:\n"
                "1. Files <1000 lines: full read is fine.\n"
                "2. Files ≥1000 lines: ALWAYS pass `start_line` and "
                "`end_line` for the region you actually need. The "
                "file_reader prefixes each line with its number so you "
                "can quote them back unambiguously.\n"
                "3. Use `# RECENT COMMITS` above as your map: a commit "
                "message like `fix(verifier): ...` tells you to look "
                "in `backend/verifier.py`, narrow first via `grep` or "
                "the commit's described change before reading.\n"
                "4. Don't re-read the same file with bigger `max_chars` "
                "if a chunk got truncated — request a different "
                "`start_line`/`end_line` range instead."
            )
        else:
            notes_section = f"# NOTES\n{self._notes_block(notes)}"
        from .claims import SOLVER_CLAIMS_DIRECTIVE
        user = f"""{ctx}

{notes_section}

{think_block}
{critique_block}
# USER REQUEST
{marker}{task}{SOLVER_CLAIMS_DIRECTIVE}"""
        registry = get_registry()
        # Load skills lazily — registers their tools in the registry,
        # so SKILLS.ensure_loaded() MUST run before registry.to_anthropic_list().
        SKILLS.ensure_loaded()
        tools = registry.to_anthropic_list()

        # Match active skills by triggers
        matched_skills = SKILLS.match(task)
        if matched_skills:
            self.progress(
                "skill",
                f"active skills: {', '.join(s.name for s in matched_skills)}",
            )

        # System prompt: identity + solver base + capabilities + catalog + active skills
        system = _with_identity(SOLVER_SYSTEM_BASE)
        system = f"{system}\n\n---\n\n{_capabilities_block()}"
        catalog = SKILLS.catalog_block()
        if catalog:
            system = f"{system}\n\n---\n\n{catalog}"
        for sk in matched_skills:
            system = f"{system}\n\n---\n\n{sk.system_block()}"

        tool_outputs: list[str] = []

        # Adaptive caps per tool: a `read_file` of agent.py truncated to
        # 1500 chars left the verifier looking at the first 25 lines and
        # marking everything else "unverified". Bigger ceilings for tools
        # that legitimately produce file-sized output; the original 1500
        # stays for short tools (web snippets, calc results, etc).
        # Adaptive caps per tool. On self_analysis turns we widen
        # read_file/view_file because actual source files (agent.py
        # ~78k, llm.py ~98k) need more than the default 20k; the
        # verifier needs to see what the solver actually claimed
        # against, otherwise reviews flag arbitrary chunks as
        # "unverified". For non-self-analysis turns we keep the
        # tighter cap to limit tokens in the verifier prompt.
        if is_self_analysis:
            read_cap = 60000
        else:
            read_cap = 20000
        _tool_cap = {
            "read_file": read_cap,
            "view_file": read_cap,
            "read_note": 8000,
            "list_files": 4000,
            "glob": 4000,
            "grep": 4000,
            "search": 4000,
        }
        _DEFAULT_CAP = 1500

        def _on_tool_call(name: str, args: dict, result: str, is_error: bool) -> None:
            preview = (result or "").strip().splitlines()[0][:80] if result else ""
            tag = "tool_error" if is_error else "tool"
            # Structured detail rides alongside the one-liner so the
            # WebUI can render a compact summary by default and reveal
            # the result body on demand. We DON'T put the full body in
            # the trace — a 60k `read_file` of agent.py going into
            # every AgentAnswer would bloat the SSE / WebUI / dev
            # capture payloads even though the verifier-side
            # `tool_outputs` is already cap'd. Trace gets a 4000-char
            # preview; the full body is still alive in `tool_outputs`
            # for verification and on disk for read_file calls.
            _trace_result_cap = 4000
            full_result = result or ""
            full_len = len(full_result)
            preview_body = full_result[:_trace_result_cap]
            detail = ToolCallDetail(
                name=name,
                args=args or {},
                result=preview_body,
                result_truncated=full_len > _trace_result_cap,
                result_full_len=full_len,
                is_error=bool(is_error),
            )
            self.progress(
                tag,
                f"{name}({', '.join(args.keys())}) -> {preview}",
                tool_call=detail,
            )
            if result:
                if is_error:
                    # Errors are evidence too: "I couldn't read the
                    # file" / "web_search returned 503" / "run_python
                    # SyntaxError" — verifier needs these to confirm
                    # an answer like "I couldn't access X" or to flag
                    # an answer that claims success despite the error.
                    # Cap tighter than success cap; error messages are
                    # short and we don't want long stack traces eating
                    # the verifier prompt.
                    err_cap = 2000
                    snippet = result[:err_cap]
                    if len(result) > err_cap:
                        snippet += f"\n…[+{len(result) - err_cap} more chars truncated]"
                    tool_outputs.append(f"[{name} ERROR] {snippet}")
                else:
                    cap = _tool_cap.get(name, _DEFAULT_CAP)
                    snippet = result[:cap]
                    if len(result) > cap:
                        snippet += f"\n…[+{len(result) - cap} more chars truncated]"
                    tool_outputs.append(f"[{name}] {snippet}")

        import time as _t
        _t0 = _t.monotonic()
        usage_before = TOKENS.request_usage()
        answer = router().call_with_tools(
            TaskType.COMPLEX_SOLVING,
            system,
            user,
            tools=tools,
            execute_tool=registry.execute,
            max_tokens=4000,
            temperature=0.3,
            on_tool_call=_on_tool_call,
            attachments=getattr(self, "_attachments", None),
        )
        # Capture only the FIRST turn's system + user. Tool-loop iterations
        # repeat the same system and grow user with tool_results — those
        # are visible per-step via `tool_call` entries in thinking_trace.
        # The usage_before snapshot makes `input_tokens` / `output_tokens`
        # in the dev panel cover the WHOLE tool loop (all iterations) —
        # the right rollup for "cost of one solve call".
        self._record_llm_call(
            label="_solve",
            task_type=TaskType.COMPLEX_SOLVING,
            system=system,
            user=user,
            response=answer,
            duration_ms=int((_t.monotonic() - _t0) * 1000),
            usage_before=usage_before,
        )
        # P0 Phase B: pull the structured claims tail off the answer.
        # `cleaned_answer` is what the user (and verifier, and finetune
        # dataset, and dev capture) will see — never includes the
        # marker or JSON. `parsed_claims` rides on `self` for
        # `Agent.run` to feed into the claim/evidence builder.
        from .claims import extract_solver_claims_block
        cleaned_answer, parsed_claims = extract_solver_claims_block(answer)
        if parsed_claims is not None:
            self._last_solver_claims = parsed_claims
        tool_context = "\n\n".join(tool_outputs) if tool_outputs else ""
        return cleaned_answer, tool_context

    # Step 4b — build critique for self-critic loop
    def _build_critique(self, vr: VerificationResult, prev_answer: str) -> str:
        """Build a CRITIQUE block from verifier feedback for the retry solver."""
        parts = ["# CRITIQUE OF YOUR PREVIOUS ANSWER",
                 f"Your previous answer scored {vr.confidence}% confidence.",
                 "The verifier found the following problems:\n"]
        if vr.unverified_claims:
            parts.append("## Unverified claims (no evidence found):")
            for c in vr.unverified_claims:
                parts.append(f"- {c}")
        if vr.contradictions:
            parts.append("\n## Contradictions (conflicts with sources):")
            for c in vr.contradictions:
                parts.append(f"- {c}")
        parts.append(f"\n## Your previous answer (to revise):\n{prev_answer[:2000]}")
        parts.append("\nFix these issues. Use tools to find evidence. "
                     "Remove claims you cannot support.")
        return "\n".join(parts)

    # Step 5
    def _verify(self, task: str, answer: str, notes: list[Note],
                tool_context: str = "") -> VerificationResult:
        if not CONFIG.verification["enabled"]:
            return VerificationResult(confidence=100, notes_used=[n.frontmatter.topic for n in notes])
        self.progress("verify", "verifying answer...")

        usage_before = TOKENS.request_usage()

        def _capture(system, user, response, duration_ms):
            self._record_llm_call(
                label="_verify",
                task_type=TaskType.VERIFICATION,
                system=system,
                user=user,
                response=response,
                duration_ms=duration_ms,
                usage_before=usage_before,
            )

        # P0 Phase C: hand the verifier the solver's structured claims
        # plus the tool-call order so it can rule per-claim against the
        # exact evidence the solver cited. Both args are best-effort —
        # missing solver tail or empty trace falls back to Phase A
        # (legacy regex-based extraction). `tool_call_order` is built
        # from the live thinking_trace this turn so tool_N indexing
        # matches what the solver wrote in its tail.
        solver_claims = getattr(self, "_last_solver_claims", None)
        tool_call_order: list[dict] = []
        for step in self._trace:
            tc = step.tool_call
            if tc is None:
                continue
            tool_call_order.append({
                "name": tc.name,
                "args": tc.args or {},
                "result": tc.result or "",
                "is_error": bool(tc.is_error),
            })

        return verify(
            question=task,
            answer=answer,
            notes_text=self._notes_block(notes),
            used_topics=[n.frontmatter.topic for n in notes],
            tool_context=tool_context,
            on_llm_call=_capture,
            solver_claims=solver_claims,
            tool_call_order=tool_call_order,
        )

    # Шаг 6
    def _learn_from_experience(
        self,
        task: str,
        answer: str,
        notes: list[Note],
        vr: VerificationResult,
        project: str | None,
    ) -> None:
        self.progress("experience", "сохраняю опыт")
        # access_count уже ведётся в KM.get_note
        # Автосбор в finetune queue: confidence ≥ 85%, заметки есть, verified
        added = finetune_store().maybe_add_from_agent(
            question=task,
            answer=answer,
            source_notes=[n.frontmatter.topic for n in notes],
            confidence=vr.confidence,
            is_verified=not vr.contradictions,
            project=project,
        )
        if added:
            self.progress(
                "finetune",
                f"Q&A добавлено в finetune queue [{added.metadata.category}]",
            )

        # Extract reusable pattern from high-confidence answers
        if vr.confidence >= 90 and not vr.contradictions:
            try:
                domain = notes[0].frontmatter.category if notes else ""
                pattern = ANALOGIES.extract_pattern(task, answer, domain)
                if pattern:
                    self.progress("pattern", f"extracted: {pattern.pattern[:60]}")
            except Exception:
                pass

    # Шаг 7 — cleanup (нечего выгружать, заметки уже на диске)
    def _cleanup(self) -> None:
        self.progress("cleanup", "готово")

    def _extract_memories(
        self,
        user_msg: str,
        answer: str,
        intent: str,
        *,
        confidence: int = 100,
        contradictions: int = 0,
    ) -> None:
        """Extract memorable facts from conversation and store in graph.

        `confidence` + `contradictions` come from the verifier; the
        extractor uses them to drop agent-answer mining on low-confidence
        turns so we don't pollute the graph with wrong claims.
        """
        try:
            facts = MEMORY.extract_and_store(
                user_msg, answer, intent=intent,
                confidence=confidence, contradictions=contradictions,
            )
            if facts:
                summaries = "; ".join(f.summary for f in facts[:3])
                self.progress("memory_save", f"remembered {len(facts)} facts: {summaries[:100]}")
        except Exception:
            pass  # memory extraction is best-effort

    def _tick_goals(self) -> None:
        """Tick the goal manager and check for proactive learning opportunities."""
        try:
            GOALS.tick_interaction()
            if GOALS.should_check_proactive():
                gaps = KM.open_gaps(threshold=2, limit=5)
                if gaps:
                    created = GOALS.suggest_from_gaps(gaps, max_goals=3)
                    if created:
                        names = ", ".join(g.description for g in created)
                        self.progress("goals", f"new proactive goals: {names}")
        except Exception:
            pass  # goal tracking is best-effort

    # ---------- публичный вход ----------
    def _get_token_usage(self) -> TokenUsage:
        """Capture token usage for the current request."""
        u = TOKENS.request_usage()
        try:
            stages = TOKENS.request_breakdown().get("stages", {})
        except Exception:
            stages = {}
        return TokenUsage(
            input_tokens=u["input_tokens"],
            output_tokens=u["output_tokens"],
            total_tokens=u["total_tokens"],
            cache_read_tokens=u["cache_read_tokens"],
            cache_creation_tokens=u["cache_creation_tokens"],
            cost_usd=u["cost_usd"],
            llm_calls=u["llm_calls"],
            by_stage=stages,
        )

    def _persist_dev_capture(self, task: str, answer: str, confidence: int) -> None:
        """Save redacted LLM-call captures to `dev/` for offline review.
        Best-effort: never raises, never blocks the response."""
        if not self._llm_calls:
            return
        try:
            save_dev_capture(
                request_id=self._request_id,
                question=task,
                llm_calls=[c.model_dump() for c in self._llm_calls],
                answer_preview=answer or "",
                confidence=int(confidence or 0),
            )
        except Exception:
            pass

    def run(
        self,
        task: str,
        project: str | None = None,
        attachments: list[str] | None = None,
        *,
        channel: str = "webui",
    ) -> AgentAnswer:
        import time as _time
        TOKENS.reset_request()
        self._trace = []
        self._llm_calls = []
        self._request_id = new_request_id()
        # Cleared at start of each request so a flagged self-analysis
        # answer from a previous turn doesn't leak into this one.
        self._self_analysis_unverified = False
        self._t0 = _time.monotonic()
        # Stash attachments so _chat_reply / _solve can pick them up
        # without us threading the kwarg through every helper.
        self._attachments = attachments or None
        # Channel tag for conversation memory + turn record. Stored
        # on `self` so chat fast-path / preference branch / task
        # branch all tag the same way without each call site
        # passing it explicitly.
        self._channel = channel or "webui"
        try:
            core = self._load_core()

            # Branch 0: micro-ack. "ok" / "thanks" / "continue" /
            # "понял" — short messages that don't need any LLM call.
            # Hits before _classify_intent because the classifier
            # itself is an LLM call, and asking Sonnet to classify
            # "ok" is wasteful. ~300+ tokens saved per ack message.
            micro = _micro_ack_reply(task)
            if micro is not None:
                self.progress("micro_ack", "static reply (no LLM)")
                CONVERSATION.add_turn(
                    task, micro, intent="chat", is_chat=True,
                    channel=self._channel,
                )
                # Memory extraction gate: short acks have nothing to
                # extract — skipping saves another LLM call. The
                # lookups in EVALUATOR / GOALS still get the turn
                # via add_turn → recall_block.
                self._cleanup()
                self._tick_goals()
                self._persist_dev_capture(task, micro, 100)
                return AgentAnswer(
                    answer=micro,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    token_usage=self._get_token_usage(),
                    thinking_trace=self._trace,
                    llm_calls=self._llm_calls,
                )

            intent = self._classify_intent(task)

            # Branch 1: chitchat / small-talk. One warm reply, no pipeline.
            if intent == "chat":
                answer = self._chat_reply(task, core)
                CONVERSATION.add_turn(
                    task, answer, intent="chat", is_chat=True,
                    channel=self._channel,
                )
                # Memory extraction gate: skip on very short chat
                # turns ("привет", "thanks for the help") — the
                # extractor LLM call can't pull useful facts from
                # 5-15 chars, but it still costs ~300 input tokens.
                if len(task.strip()) >= 30:
                    self._extract_memories(task, answer, "chat")
                try:
                    EVALUATOR.log(EvalEntry(
                        question=task, intent="chat", confidence=100,
                        topics_used=[], contradictions=0, unverified=0,
                        verified=0, is_chat=True,
                    ))
                except Exception:
                    pass
                self._cleanup()
                self._tick_goals()
                self._persist_dev_capture(task, answer, 100)
                return AgentAnswer(
                    answer=answer,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    token_usage=self._get_token_usage(),
                    thinking_trace=self._trace,
                    llm_calls=self._llm_calls,
                )

            # Branch 2: preference — user configures the agent or shares
            # personal info. Save to user.md and acknowledge briefly.
            if intent == "preference":
                category, fact, ack = self._save_preference(task)
                # The "_(saved to user.md ...)" debug suffix only shows up
                # when AGI_DEBUG_MEMORY_ACK is enabled (or category=reject is
                # never decorated since nothing was actually saved). Plain
                # users see only the natural-language acknowledgment.
                debug_suffix = (
                    os.getenv("AGI_DEBUG_MEMORY_ACK", "").strip().lower()
                    in ("1", "true", "yes")
                )
                if debug_suffix and category != "reject":
                    reply = f"{ack}\n\n_(saved to user.md -> {category}: {fact})_"
                else:
                    reply = ack
                CONVERSATION.add_turn(
                    task, reply, intent="preference", is_chat=True,
                    channel=self._channel,
                )
                self._extract_memories(task, reply, "preference")
                self._cleanup()
                self._tick_goals()
                self._persist_dev_capture(task, reply, 100)
                return AgentAnswer(
                    answer=reply,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    token_usage=self._get_token_usage(),
                    thinking_trace=self._trace,
                    llm_calls=self._llm_calls,
                )

            # Branch 3: real task — full thinking → knowledge → solve → verify cycle.
            thinking = self._think(task, core)
            self.progress(
                "strategy",
                f"type={thinking.question_type}, "
                f"tools=[{', '.join(thinking.tools_needed)}], "
                f"confidence={thinking.confidence}%",
            )
            # self_analysis answers come from reading the agent's own code,
            # not from external research — disable web-driven auto-learning
            # for those turns to avoid polluting the KB with redundant notes.
            # AND skip topic loading entirely: KB notes about the code are
            # snapshots that reproduce stale findings ("agent finds bugs
            # already fixed last commit"). The git log + read_file are the
            # only authoritative sources for self-analysis.
            is_self_analysis = thinking.question_type == "self_analysis"
            if is_self_analysis:
                notes, learned = [], []
            else:
                notes, learned = self._ensure_knowledge(
                    thinking.required_topics, project, allow_learning=True,
                )

            # Load project-specific notes if in project mode. Skipped
            # on self-analysis because the solver gets a SOURCE OF
            # TRUTH directive there and must read live source — pulling
            # in stale project snapshots that the solver won't even
            # consult would just inflate the verifier prompt later.
            if project and not is_self_analysis:
                for entry in KM.list_topics():
                    if entry.project == project and all(
                        entry.topic != n.frontmatter.topic for n in notes
                    ):
                        note = KM.get_note(entry.topic)
                        if note:
                            notes.append(note)

            # Hierarchical decomposition: solve subtasks first, then synthesize.
            # Each subtask's answer is injected into the next subtask as
            # "already known" context, so the LLM doesn't repeat work.
            # Cap at 3 subtasks max to prevent token explosion.
            if thinking.subtasks:
                thinking.subtasks = thinking.subtasks[:3]
            if thinking.subtasks and len(thinking.subtasks) >= 2:
                subtask_results: list[str] = []
                # Tool evidence collected across subtasks so the final
                # verifier can fact-check synthesis claims against ALL
                # files the agent read, not just whatever the synthesis
                # solve happened to re-read. Especially important for
                # code review / self-analysis decomposed into chunks.
                subtask_tool_contexts: list[str] = []
                for i, subtask in enumerate(thinking.subtasks):
                    self.progress("subtask", f"[{i+1}/{len(thinking.subtasks)}] {subtask[:60]}")
                    enriched_thinking = thinking.model_copy()
                    if subtask_results:
                        # Inject prior subtask answers so LLM has context
                        # and won't re-read the same files
                        prior = "\n\n".join(subtask_results)
                        enriched_thinking.approach = (
                            f"{thinking.approach}\n\n"
                            f"PRIOR SUBTASK RESULTS (use this info, do NOT re-read "
                            f"files already analyzed here):\n{prior}"
                        )
                    sub_answer, sub_tool_ctx = self._solve(
                        subtask, core, notes, thinking=enriched_thinking,
                    )
                    subtask_results.append(f"## {subtask}\n{sub_answer}")
                    if sub_tool_ctx and sub_tool_ctx.strip():
                        subtask_tool_contexts.append(
                            f"--- subtask {i+1}: {subtask[:60]} ---\n{sub_tool_ctx}"
                        )
                # Synthesize: solve the full task with subtask results as context
                synthesis_context = "\n\n".join(subtask_results)
                thinking_with_synthesis = thinking.model_copy()
                thinking_with_synthesis.approach = (
                    f"Subtasks already solved. Synthesize into final answer.\n"
                    f"SUBTASK RESULTS:\n{synthesis_context}"
                )
                answer, tool_context = self._solve(task, core, notes, thinking=thinking_with_synthesis)
                # Prepend subtask evidence to the synthesis tool_context
                # so verify() sees both layers. Cap to keep verifier
                # prompt sane.
                if subtask_tool_contexts:
                    combined = "\n\n".join(subtask_tool_contexts)
                    if tool_context and tool_context.strip():
                        tool_context = f"{combined}\n\n--- synthesis ---\n{tool_context}"
                    else:
                        tool_context = combined
                    if len(tool_context) > 80000:
                        tool_context = tool_context[-80000:]
            else:
                answer, tool_context = self._solve(task, core, notes, thinking=thinking)

            # Hard enforcement for self_analysis: the only authoritative
            # source for "is X in the code" is read_file output. If the
            # solver answered without reading anything (no tool_context),
            # force ONE retry with a critique that names the rule. Without
            # this, skip_verify below would short-circuit verification and
            # we'd ship pure hallucinations on review questions.
            # Source-read check: tool_context being non-empty isn't
            # enough — solver might have called calc / web_search /
            # run_python and still not read a single source file. The
            # guard fires unless the trace shows at least one
            # read_file or view_file call THIS turn.
            source_files_read = any(
                step.tool_call is not None
                and step.tool_call.name in SOURCE_READ_TOOLS
                for step in self._trace
            )
            if is_self_analysis and not source_files_read:
                self.progress(
                    "self_analysis_guard",
                    "no read_file/view_file in self-analysis turn — forcing retry",
                )
                forced_critique = (
                    "STRICT GUARD: This is a self-analysis turn but you "
                    "answered without calling `read_file` or `view_file` "
                    "on any source. Claims about your own code MUST be "
                    "grounded in the actual file contents — read agent.py, "
                    "llm.py, verifier.py, or whichever modules your answer "
                    "references, THEN re-answer. Do not propose fixes "
                    "for code you have not read this turn. Other tools "
                    "(calc, web_search, run_python) do NOT count as "
                    "reading source."
                )
                try:
                    answer, tool_context = self._solve(
                        task, core, notes, thinking=thinking,
                        critique=forced_critique,
                    )
                except LLMError:
                    pass  # keep the original answer if retry fails

                # Re-check after the forced retry. If the model STILL
                # didn't call read_file/view_file, downgrade gracefully
                # instead of letting `skip_verify` short-circuit and
                # ship the unverified answer. Without this re-check, a
                # solver that ignored both attempts would still produce
                # a "confidence: thinking.confidence" verdict (often
                # 70-90) on pure narrative.
                source_files_read_after = any(
                    step.tool_call is not None
                    and step.tool_call.name in SOURCE_READ_TOOLS
                    for step in self._trace
                )
                if not source_files_read_after:
                    self.progress(
                        "self_analysis_guard",
                        "retry also skipped source reads — answer flagged",
                    )
                    answer = (
                        "⚠️ Self-analysis answer below was generated WITHOUT "
                        "reading the actual source files (the solver did not "
                        "call `read_file` or `view_file` even after a forced "
                        "retry). Treat any specific code claims as unverified.\n\n"
                        + (answer or "")
                    )
                    # Force a low-confidence VerificationResult right
                    # here so the downstream skip_verify path can't
                    # paper over the gap. Empty tool_context downstream
                    # means skip_verify would otherwise fire.
                    self._self_analysis_unverified = True

            # Skip verification for creative / meta / pure self_analysis where
            # there's no factual ground truth — saves 1 LLM call. BUT when
            # self_analysis came with tool_context (the agent actually
            # read_file'd its own code) the tool output IS evidence and
            # must be checked against — otherwise we ship hallucinated
            # claims about file contents. Same logic for any creative/meta
            # answer that ended up using tools.
            qtype = thinking.question_type if thinking else ""
            no_evidence_types = ("creative", "meta", "self_analysis")
            skip_verify = (
                qtype in no_evidence_types
                and not (tool_context or "").strip()
                # Don't skip verify when the self-analysis guard flagged
                # the answer — a low-confidence VerificationResult is
                # the whole point of flagging it.
                and not getattr(self, "_self_analysis_unverified", False)
            )
            if skip_verify:
                vr = VerificationResult(
                    confidence=thinking.confidence if thinking else 75,
                    notes_used=[n.frontmatter.topic for n in notes],
                )
            elif getattr(self, "_self_analysis_unverified", False):
                # Manual low-confidence verdict for the no-source-read
                # path (verifier has nothing to check against).
                vr = VerificationResult(
                    confidence=15,
                    unverified_claims=[
                        "self-analysis answer not grounded in source — "
                        "no read_file/view_file in trace"
                    ],
                    notes_used=[],
                )
                self._self_analysis_unverified = False  # reset for next request
            else:
                vr = self._verify(task, answer, notes, tool_context=tool_context)

            # Self-critic loop: if confidence is low, re-solve with verifier feedback.
            # Max retries configurable via config (default 2). Each retry injects the
            # critique (unverified claims + contradictions) so the solver can fix them.
            #
            # Guards against wasting tokens:
            # - Skip retry if confidence=0 with no notes (structural issue, retry won't help)
            # - Skip retry if verifier reported only "no loaded notes" (no evidence to verify against)
            # - Break on LLMError during retry (API down / rate limited)
            # - Break if token budget exceeded during retries
            critic_threshold = CONFIG.verification.get("critic_threshold", 50)
            max_retries = CONFIG.verification.get("critic_max_retries", 2)
            # Hard token budget for retries. Without this, a turn that
            # already burned ~50k tokens on think/solve/verify could
            # spend another ~50k per retry × 2 retries = potentially
            # $0.50+ on a single low-confidence answer. The comment
            # below the guard list claimed this was already implemented;
            # it wasn't. 60000 is a reasonable default — covers a
            # full self-review with one retry but stops a runaway.
            retry_token_budget = CONFIG.verification.get(
                "critic_retry_token_budget", 60000
            )
            retry = 0
            no_notes = not notes and not tool_context
            should_retry = (
                vr.confidence < critic_threshold
                and not no_notes  # don't retry if there's no evidence at all
            )
            # Accumulate tool_context across retry attempts. The solver
            # may call read_file on different files each pass; verifying
            # the FINAL answer needs all evidence the agent has gathered
            # so far, not just the last attempt's. Without this, evidence
            # collected on attempt 1 (e.g. agent.py contents) is invisible
            # to the verifier when attempt 2 only re-read llm.py.
            accumulated_tool_context = tool_context or ""
            # Cap so a runaway-retry case doesn't push the verifier
            # prompt past sensible limits.
            _MAX_ACCUMULATED_CTX = 80000
            while should_retry and retry < max_retries:
                # Pre-flight token check. Cheaper to bail with the
                # current answer than to spend another solve+verify
                # cycle with no headroom — the next call would fail
                # downstream on context-window or budget limits anyway.
                used_so_far = TOKENS.request_usage().get("total_tokens", 0)
                if used_so_far >= retry_token_budget:
                    self.progress(
                        "self_critic",
                        f"token budget exhausted ({used_so_far} ≥ "
                        f"{retry_token_budget}), stopping retries",
                    )
                    break
                retry += 1
                self.progress(
                    "self_critic",
                    f"confidence {vr.confidence}% < {critic_threshold}%, "
                    f"retrying ({retry}/{max_retries})...",
                )
                try:
                    critique = self._build_critique(vr, answer)
                    answer, tool_context = self._solve(
                        task, core, notes, thinking=thinking, critique=critique,
                    )
                    if tool_context and tool_context.strip():
                        # Append this attempt's evidence to the running
                        # collection (skip if we already have the same
                        # content — solver often re-reads identical
                        # files between retries).
                        if tool_context not in accumulated_tool_context:
                            accumulated_tool_context = (
                                f"{accumulated_tool_context}\n\n--- retry {retry} ---\n\n{tool_context}"
                                if accumulated_tool_context
                                else tool_context
                            )
                            if len(accumulated_tool_context) > _MAX_ACCUMULATED_CTX:
                                accumulated_tool_context = (
                                    accumulated_tool_context[-_MAX_ACCUMULATED_CTX:]
                                )
                    vr = self._verify(
                        task, answer, notes,
                        tool_context=accumulated_tool_context,
                    )
                    self.progress(
                        "self_critic",
                        f"retry {retry} confidence: {vr.confidence}%",
                    )
                except LLMError as e:
                    self.progress("self_critic", f"retry aborted: API error — {e}")
                    break
                # If confidence didn't improve at all, stop wasting tokens
                if vr.confidence < critic_threshold and vr.confidence == 0:
                    self.progress("self_critic", "confidence stuck at 0%, stopping retries")
                    break
                should_retry = vr.confidence < critic_threshold

            min_conf = CONFIG.verification["min_confidence"]
            if vr.confidence < min_conf:
                answer = (
                    f"⚠️ Низкая уверенность ({vr.confidence}%). "
                    f"Ответ может содержать неподтверждённые факты.\n\n{answer}"
                )

            self._learn_from_experience(task, answer, notes, vr, project)

            # Meta-learner: analyze failures and create corrective goals
            if vr.confidence < 60:
                try:
                    analysis = META_LEARNER.analyze_failure(
                        task, answer, vr,
                        intent=thinking.question_type if thinking else "task",
                    )
                    if analysis:
                        self.progress("meta_learn", f"failure analyzed: {analysis.get('root_cause', '?')}")
                except Exception:
                    pass

            used = [n.frontmatter.topic for n in notes]

            # Evaluator: log every interaction for performance tracking
            try:
                EVALUATOR.log(EvalEntry(
                    question=task,
                    intent=thinking.question_type if thinking else "task",
                    confidence=vr.confidence,
                    topics_used=used,
                    contradictions=len(vr.contradictions),
                    unverified=len(vr.unverified_claims),
                    verified=len(vr.verified_claims),
                ))
            except Exception:
                pass

            from .claims import build_claims_and_evidence
            claims, evidence = build_claims_and_evidence(
                vr, self._trace,
                user_message=task,
                solver_claims=getattr(self, "_last_solver_claims", None),
            )
            # P1 TurnWorkspace: persist a structured record per turn so
            # the WebUI / future evaluator / debugger can replay the
            # full turn (tool calls, claims, evidence, verification,
            # token usage) without bloating the live response payload.
            # Generated BEFORE CONVERSATION.add_turn so we can stamp
            # the conversation entry with its turn_id (Round A: WebUI
            # uses it for lazy-loading the full thinking_trace via
            # GET /api/turns/<id>).
            turn_id = ""
            try:
                from datetime import datetime as _dt
                from uuid import uuid4 as _uuid4
                from .workspace import get_workspace
                turn_id = (
                    f"{_dt.utcnow().strftime('%Y%m%d_%H%M%S')}"
                    f"_{_uuid4().hex[:8]}"
                )
                tool_call_order_dump = []
                for step in self._trace:
                    tc = step.tool_call
                    if tc is None:
                        continue
                    tool_call_order_dump.append({
                        "name": tc.name,
                        "args": tc.args or {},
                        "result_preview": tc.result or "",
                        "result_truncated": bool(tc.result_truncated),
                        "result_full_len": int(tc.result_full_len),
                        "is_error": bool(tc.is_error),
                    })
                get_workspace().save_turn(turn_id, {
                    "turn_id": turn_id,
                    "channel": getattr(self, "_channel", "webui"),
                    "task": task,
                    "answer": answer,
                    "project": project,
                    "verification": vr.model_dump(),
                    "claims": [c.model_dump() for c in claims],
                    "evidence": [e.model_dump() for e in evidence],
                    "thinking_trace": [s.model_dump() for s in self._trace],
                    "llm_calls": [c.model_dump() for c in self._llm_calls],
                    "tool_call_order": tool_call_order_dump,
                    "token_usage": self._get_token_usage().model_dump(),
                    "solver_claims_raw": getattr(self, "_last_solver_claims", None),
                })
            except Exception:
                turn_id = ""

            CONVERSATION.add_turn(
                task, answer,
                intent=thinking.question_type if thinking else "task",
                confidence=vr.confidence,
                topics_used=used,
                channel=getattr(self, "_channel", "webui"),
                turn_id=turn_id,
            )
            self._extract_memories(
                task, answer,
                thinking.question_type if thinking else "task",
                confidence=vr.confidence,
                contradictions=len(vr.contradictions),
            )
            self._cleanup()
            self._tick_goals()
            self._persist_dev_capture(task, answer, vr.confidence)

            return AgentAnswer(
                answer=answer,
                verification=vr,
                learned_topics=learned,
                used_topics=used,
                project=project,
                token_usage=self._get_token_usage(),
                thinking_trace=self._trace,
                llm_calls=self._llm_calls,
                claims=claims,
                evidence=evidence,
                turn_id=turn_id,
            )
        except LLMError as e:
            err_text = _format_llm_error_short(e)
            self._persist_dev_capture(task, err_text, 0)
            return AgentAnswer(
                answer=err_text,
                verification=VerificationResult(confidence=0),
                token_usage=self._get_token_usage(),
                thinking_trace=self._trace,
                llm_calls=self._llm_calls,
            )
