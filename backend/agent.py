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
    TaskAnalysis,
    ThinkingResult,
    ThinkingStep,
    TokenUsage,
    VerificationResult,
)
from .analogy_engine import ANALOGIES
from .evaluator import EVALUATOR, EvalEntry
from .goals import GOALS
from .memory_extractor import MEMORY
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
- run_python — for calculation, data processing, verification
- (other tools may be listed in MY CAPABILITIES block)

CRITICAL RULES for strategy:
- For "self_analysis" questions: you MUST plan to read_file your own source code.
  Never guess about your own architecture — read the actual files.
- For "factual" questions: check if NOTES will cover it. Only use web_search if not.
- For "calculation": prefer run_python over guessing numbers.
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


def _capabilities_block() -> str:
    """Динамический блок «MY CAPABILITIES» для system prompt.

    Перечисляет конкретные вещи, которые агент имеет и умеет прямо сейчас:
    зарегистрированные tools, загруженные skills, подключённые MCP серверы,
    и карту своего исходного кода на диске — чтобы при вопросах о себе агент
    знал, куда смотреть (read_file), а не фантазировал.
    """
    registry = get_registry()
    SKILLS.ensure_loaded()

    lines = ["# MY CAPABILITIES"]

    # --- Tools ---
    tools = registry.tools
    if tools:
        lines.append("\n## Инструменты (tools)")
        for name, tool in sorted(tools.items()):
            origin_label = f"  [{tool.origin}]" if tool.origin != "builtin" else ""
            lines.append(f"- `{name}` — {tool.description[:100]}{origin_label}")

    # --- Skills ---
    if SKILLS.skills:
        lines.append("\n## Навыки (skills)")
        for sk in SKILLS.skills:
            triggers = ", ".join(sk.triggers[:5]) if sk.triggers else "—"
            lines.append(f"- **{sk.name}**: {sk.description[:80]}  (triggers: {triggers})")

    # --- MCP ---
    if MCP.servers:
        lines.append("\n## Подключённые MCP-серверы")
        for srv_name, srv in MCP.servers.items():
            tool_count = len([t for t in tools if t.startswith(f"mcp_{srv_name}__")])
            lines.append(f"- `{srv_name}` ({tool_count} tools)")

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
        "backend/builtin_tools.py": "builtin tools: web_search, fetch_url, read_file, run_python, calc",
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


# ---------- тип колбэка для прогресса ----------
ProgressCB = Callable[[str, str], None]  # (event, message)


def _noop(_evt: str, _msg: str) -> None:
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
        self._t0: float = 0.0
        _bootstrap_mcp()

    def progress(self, event: str, message: str) -> None:
        """Record thinking step and forward to user callback."""
        import time as _time
        elapsed = _time.monotonic() - self._t0 if self._t0 else 0.0
        usage = TOKENS.request_usage()
        self._trace.append(ThinkingStep(
            ts=round(elapsed, 2),
            event=event,
            message=message,
            tokens_so_far=usage["total_tokens"],
        ))
        self._user_progress(event, message)

    # Шаг 1
    def _load_core(self) -> str:
        self.progress("core", "загружаю core memory")
        return CORE.read()

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
        if _CHITCHAT_RE.match(trimmed):
            return "chat"
        try:
            data = router().call_json(
                TaskType.CLASSIFICATION,
                INTENT_CLASSIFIER_SYSTEM,
                f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{trimmed}",
                max_tokens=150,
                temperature=0.0,
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
        conv = CONVERSATION.context_block(n=4)
        conv_section = f"\n\n{conv}" if conv else ""
        user = f"# CORE MEMORY\n{core.strip()}{conv_section}\n\n# MESSAGE\n{task.strip()}"
        attachments = getattr(self, "_attachments", None)
        try:
            return router().call(
                TaskType.QUICK_ANSWER,
                system,
                user,
                max_tokens=300, temperature=0.6,
                attachments=attachments,
            )
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

        try:
            data = router().call_json(
                TaskType.CLASSIFICATION,
                system_prompt,
                f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{task.strip()}",
                max_tokens=300, temperature=0.1,
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
        caps = _capabilities_block()
        conv = CONVERSATION.context_block(n=6)
        conv_section = f"\n\n{conv}" if conv else ""
        goals_section = ""
        goals_block = GOALS.context_block(max_goals=5)
        if goals_block:
            goals_section = f"\n\n{goals_block}"
        # Recall relevant facts from memory graph (previous conversations)
        memory_section = ""
        try:
            memory_block = MEMORY.recall_block(task, max_facts=10)
            if memory_block:
                memory_section = f"\n\n{memory_block}"
                self.progress("memory", f"recalled {memory_block.count(chr(10)) - 2} facts from memory")
        except Exception:
            pass
        user = (
            f"CORE MEMORY:\n{core}\n\n"
            f"MY CAPABILITIES:\n{caps}"
            f"{conv_section}{goals_section}{memory_section}\n\n"
            f"USER REQUEST:\n{task}"
        )
        data = router().call_json(
            TaskType.TASK_ANALYSIS,
            THINKING_SYSTEM, user, max_tokens=1000, temperature=0.2,
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
            hit = HYBRID.find_best(topic)
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

        conv = CONVERSATION.context_block(n=6)
        conv_section = f"\n{conv}\n" if conv else ""

        critique_block = f"\n{critique}\n" if critique else ""

        # Recall relevant facts from memory graph
        memory_block = ""
        try:
            mb = MEMORY.recall_block(task, max_facts=8)
            if mb:
                memory_block = f"\n{mb}\n"
        except Exception:
            pass

        user = f"""# CORE MEMORY
{core.strip()}

# NOTES
{self._notes_block(notes)}
{memory_block}
{think_block}
{conv_section}{critique_block}
# USER REQUEST
{task}"""
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
        _tool_cap = {
            "read_file": 12000,
            "view_file": 12000,
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
            self.progress(tag, f"{name}({', '.join(args.keys())}) -> {preview}")
            if result and not is_error:
                cap = _tool_cap.get(name, _DEFAULT_CAP)
                snippet = result[:cap]
                if len(result) > cap:
                    snippet += f"\n…[+{len(result) - cap} more chars truncated]"
                tool_outputs.append(f"[{name}] {snippet}")

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
        tool_context = "\n\n".join(tool_outputs) if tool_outputs else ""
        return answer, tool_context

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
        return verify(
            question=task,
            answer=answer,
            notes_text=self._notes_block(notes),
            used_topics=[n.frontmatter.topic for n in notes],
            tool_context=tool_context,
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

    def _extract_memories(self, user_msg: str, answer: str, intent: str) -> None:
        """Extract memorable facts from conversation and store in graph."""
        try:
            facts = MEMORY.extract_and_store(user_msg, answer, intent=intent)
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
        return TokenUsage(
            input_tokens=u["input_tokens"],
            output_tokens=u["output_tokens"],
            total_tokens=u["total_tokens"],
            cache_read_tokens=u["cache_read_tokens"],
            cache_creation_tokens=u["cache_creation_tokens"],
            cost_usd=u["cost_usd"],
            llm_calls=u["llm_calls"],
        )

    def run(
        self,
        task: str,
        project: str | None = None,
        attachments: list[str] | None = None,
    ) -> AgentAnswer:
        import time as _time
        TOKENS.reset_request()
        self._trace = []
        self._t0 = _time.monotonic()
        # Stash attachments so _chat_reply / _solve can pick them up
        # without us threading the kwarg through every helper.
        self._attachments = attachments or None
        try:
            core = self._load_core()

            intent = self._classify_intent(task)

            # Branch 1: chitchat / small-talk. One warm reply, no pipeline.
            if intent == "chat":
                answer = self._chat_reply(task, core)
                CONVERSATION.add_turn(task, answer, intent="chat", is_chat=True)
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
                return AgentAnswer(
                    answer=answer,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    token_usage=self._get_token_usage(),
                    thinking_trace=self._trace,
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
                CONVERSATION.add_turn(task, reply, intent="preference", is_chat=True)
                self._extract_memories(task, reply, "preference")
                self._cleanup()
                self._tick_goals()
                return AgentAnswer(
                    answer=reply,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    token_usage=self._get_token_usage(),
                    thinking_trace=self._trace,
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
            allow_learning = thinking.question_type != "self_analysis"
            notes, learned = self._ensure_knowledge(
                thinking.required_topics, project, allow_learning=allow_learning,
            )

            # Load project-specific notes if in project mode
            if project:
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
                    sub_answer, _ = self._solve(
                        subtask, core, notes, thinking=enriched_thinking,
                    )
                    subtask_results.append(f"## {subtask}\n{sub_answer}")
                # Synthesize: solve the full task with subtask results as context
                synthesis_context = "\n\n".join(subtask_results)
                thinking_with_synthesis = thinking.model_copy()
                thinking_with_synthesis.approach = (
                    f"Subtasks already solved. Synthesize into final answer.\n"
                    f"SUBTASK RESULTS:\n{synthesis_context}"
                )
                answer, tool_context = self._solve(task, core, notes, thinking=thinking_with_synthesis)
            else:
                answer, tool_context = self._solve(task, core, notes, thinking=thinking)

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
            )
            if skip_verify:
                vr = VerificationResult(
                    confidence=thinking.confidence if thinking else 75,
                    notes_used=[n.frontmatter.topic for n in notes],
                )
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
            retry = 0
            no_notes = not notes and not tool_context
            should_retry = (
                vr.confidence < critic_threshold
                and not no_notes  # don't retry if there's no evidence at all
            )
            while should_retry and retry < max_retries:
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
                    vr = self._verify(task, answer, notes, tool_context=tool_context)
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

            CONVERSATION.add_turn(
                task, answer,
                intent=thinking.question_type if thinking else "task",
                confidence=vr.confidence,
                topics_used=used,
            )
            self._extract_memories(task, answer, thinking.question_type if thinking else "task")
            self._cleanup()
            self._tick_goals()

            return AgentAnswer(
                answer=answer,
                verification=vr,
                learned_topics=learned,
                used_topics=used,
                project=project,
                token_usage=self._get_token_usage(),
                thinking_trace=self._trace,
            )
        except LLMError as e:
            return AgentAnswer(
                answer=_format_llm_error_short(e),
                verification=VerificationResult(confidence=0),
                token_usage=self._get_token_usage(),
                thinking_trace=self._trace,
            )
