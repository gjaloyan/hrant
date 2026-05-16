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
# Промпты живут в backend/prompts.py (легче редактировать без скролла через
# 2000-строчный agent.py). Re-import — тестовые/каналовые `from backend.agent
# import SOLVER_SYSTEM_BASE` продолжают работать. Identity preamble +
# capabilities блок добавляются в runtime через `_with_identity()` +
# `_capabilities_block()` ниже.
from .prompts import (  # noqa: E402
    CHAT_SYSTEM_BASE,
    INTENT_CLASSIFIER_SYSTEM,
    PREFERENCE_EXTRACTOR_SYSTEM,
    SOLVER_SYSTEM_BASE,
    THINKING_SYSTEM,
)



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


import contextvars as _contextvars

# Per-turn sticky-request hint resolved at `Agent.run()` entry.
# All `_with_identity` call sites (chat / think / solve / verify)
# read from this ContextVar, so they don't each need an extra
# kwarg threaded through. Default is empty string = no hint added.
_current_sticky_block: _contextvars.ContextVar[str] = _contextvars.ContextVar(
    "hrant_current_sticky_block", default=""
)


def _with_identity(
    base_system: str,
    *,
    speaker_id: str | None = None,
    sticky_block: str = "",
) -> str:
    """Склеивает identity preamble + agent state snapshot +
    (optional) sticky-request hint + per-speaker permissions с
    конкретным system-промптом шага.

    `speaker_id` selects which user_profile and role/permission
    block to inject. Default None falls back to the WebUI speaker
    (which is always owner) for back-compat.

    `sticky_block` is the rendered output of
    `backend.sticky_requests.render_sticky_block` for the current
    turn — empty string when no sticky pattern detected.

    The STATE SNAPSHOT block (`backend.self_state`) is what gives
    the agent line-of-sight to its own current settings — active
    TTS voice, active model, config file paths, tools registered.
    Without it, "change your voice to male" became "Понял" + a
    stored preference fact + no actual change; with it, the LLM
    sees the current voice + config path + the `set_setting`
    tool it can use to mutate the file."""
    from . import roles as _roles
    from . import self_state as _ss
    perms = _roles.permissions_block(speaker_id)
    try:
        state = _ss.current_self_state(speaker_id=speaker_id)
        snapshot = _ss.render_snapshot(state)
    except Exception:
        snapshot = ""  # never block a turn over the bookkeeping block
    # Fallback: if caller didn't pass sticky_block explicitly,
    # pull from the per-turn ContextVar that `Agent.run` sets.
    effective_sticky = sticky_block or _current_sticky_block.get()
    return (
        f"{IDENTITY.preamble(speaker_id=speaker_id)}\n\n"
        + (f"---\n\n{snapshot}\n\n" if snapshot else "")
        + (f"---\n\n{effective_sticky}\n" if effective_sticky else "")
        + f"---\n\n{base_system}\n\n"
        + f"---\n\n{perms}"
    )


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
        "backend/cli.py": "unified `hrant` CLI entry point (init/run/update/rollback/chat/...)",
        "backend/repl.py": "interactive REPL (`hrant chat`)",
        "backend/paths.py": "engine vs user-data path resolution",
        "backend/bootstrap.py": "first-run wizard helpers (copies knowledge_templates → data_dir)",
        "backend/updater.py": "git wrapper + history ledger for `hrant update`/`rollback`",
        "backend/self_mods.py": "local patch overlay (`~/.hrant/data/self_mods/`)",
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


# Profile-recall questions — "what is my X" / "do you remember Y" /
# "что моё X" / "помнишь Y" / etc. The agent already has user
# profile + recent conversation in its fast_chat context, so these
# don't need the full task_mode pipeline (think → notes → solve →
# verify). Pre-fix, the LLM intent classifier consistently labelled
# them "task" and the agent burned 4 LLM calls / $0.04 for what
# should have been a single chat_reply.
#
# Pattern intent: match short questions whose subject is the user's
# own profile, NOT general knowledge questions. "What is the capital
# of Armenia?" is NOT a recall question (no possessive pronoun);
# "What is my favorite color?" IS.
_PROFILE_RECALL_RE = re.compile(
    r"^\s*(?:"
    # English: "what/where/when/who is my X", "do you remember/know my Y",
    # "do you remember that I told you ...", "what did I tell you ...",
    r"(?:what|where|when|who|how)(?:'s|\s+is|\s+are)\s+(?:my|mine)\b"
    r"|do\s+you\s+(?:remember|know|recall)\s+(?:my|that\s+i|what\s+i)\b"
    r"|what\s+did\s+i\s+(?:tell|say|mention)\b"
    r"|tell\s+me\s+(?:about\s+myself|my\s+\w+)\b"
    # Russian: "что мой/моя/моё/мои X", "помнишь(/-ите) мой X",
    # "что ты обо мне знаешь", "что я тебе говорил"
    r"|что\s+(?:мо[яийё]|мои)\s+\w+"
    r"|какой\s+(?:мо[яийё]|мои)\s+\w+"
    r"|помн(?:ишь|ите)\s+(?:мо[яийё]|мои|что\s+я)\b"
    r"|что\s+ты\s+(?:обо\s+мне|про\s+меня)\s+знаешь"
    r"|что\s+я\s+тебе\s+(?:говорил|сказал)"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _looks_like_profile_recall(text: str) -> bool:
    """True for questions whose subject is the user's own stored
    profile / memory. Short questions only — long messages are
    almost certainly tasks even if they happen to contain a
    'remember' substring. See `_PROFILE_RECALL_RE` for the patterns."""
    if not text or len(text) > 200:
        return False
    return bool(_PROFILE_RECALL_RE.search(text))


# Directive verbs that mean "go DO something with a setting / config",
# NOT "save this as my preference". Pre-fix, "измени голос на мужской"
# routed to preference → saved to user_profile.md → no actual change.
# `_handle_preference` and the LLM classifier both saw it as
# "preference" because the wording ("change X") superficially matches
# how users phrase preferences. This regex catches the imperative-
# directive shape and forces it onto the task path where tools fire.
#
# Pattern intent:
#   - Imperative verb at the start of the message
#   - Optionally addressed to the agent ("ты", "you")
#   - The thing being changed is a setting / property / state attribute
#     ("голос", "voice", "language", "model", "config", "voice id",
#     "tone", "speed", "voice_id", ...)
# We deliberately don't match "I prefer male voice" / "always use
# male voice" — those ARE preferences. We match the *directive* shape
# only.
_DIRECTIVE_VERBS_RE = re.compile(
    r"^\s*(?:"
    # English directives
    r"(?:please\s+)?(?:change|switch|set|make|update|configure|use|"
    r"turn\s+(?:on|off)|enable|disable|apply|put|select|pick|"
    r"give\s+(?:me\s+)?(?:a\s+|the\s+)?[a-z]+\s+(?:voice|model|setting))\b"
    # Russian directives — most-common imperative forms the user
    # actually types in chat (the production audit caught
    # "измени голос на мужской" repeated 4 times).
    r"|(?:пожалуйста\s+)?(?:измен[иьяет]+|"
    r"смени|поменя[йли]+|"
    r"переключ[иьи]+|перенастро[йли]+|"
    r"сдела[йли]+|"
    r"поста[вьи]+|постав[ьи]+|"
    r"включ[иьи]+|выключ[иьи]+|"
    r"настро[йли]+|"
    r"вырубай|вруби|выруби)\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Mention of a system-side attribute that's in the agent's
# AGENT STATE SNAPSHOT. Used together with `_DIRECTIVE_VERBS_RE`
# — directive + system attribute = "the user wants me to apply
# this", not "the user wants me to remember this preference".
_SYSTEM_ATTRIBUTE_RE = re.compile(
    r"(?:"
    r"voice|tts|голос|озвучк[ауи]+|"
    r"language|язык|"
    r"model|модель|"
    r"provider|провайдер|"
    r"channel|канал|"
    r"tone|тон|стиль|"
    r"speed|скорост|"
    r"backend|бэкенд|"
    r"config|конфиг|настройк[уаи]+|"
    r"setting"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _looks_like_system_directive(text: str) -> bool:
    """True for messages that are imperatives targeting a system-side
    setting the agent owns. Routes to `task` (not preference) so the
    agent escalates to tool use instead of just saving a fact.

    Examples that match:
      "Измени голос на мужской"           → change voice
      "switch model to gpt-4o"            → change model
      "поставь язык русский"              → change language
      "Set the TTS voice to a male voice" → change voice

    Examples that DON'T match (still preferences):
      "I prefer male voice"     → no directive verb
      "Always use Russian"      → "always" qualifier, no directive
      "My favorite color is X"  → no system attribute"""
    if not text:
        return False
    if len(text) > 300:
        return False
    if not _DIRECTIVE_VERBS_RE.search(text):
        return False
    return bool(_SYSTEM_ATTRIBUTE_RE.search(text))


def _looks_like_system_setting_preference(text: str) -> bool:
    """True for preference-shaped messages that touch a system
    attribute the agent owns (`voice`, `model`, `language`, …).

    Used as a SECONDARY signal: the LLM intent classifier may
    label a message as `preference` because the wording ("I prefer
    X", "always use Y") looks like a stable trait — but if the
    subject of that preference is a config the agent can mutate,
    we should escalate to the task path with tools, not just save
    a fact. Same destination as `_looks_like_system_directive`,
    different entry signal.

    Decoupled from `_looks_like_system_directive` because that one
    requires an imperative verb at message start. A user-profile
    preference like "Respond using a male voice" has NO imperative
    verb — but it still touches the TTS config and should be
    applied, not just stored."""
    if not text:
        return False
    if len(text) > 300:
        return False
    # Must mention a system attribute we recognise; that's the
    # cheap gate.
    if not _SYSTEM_ATTRIBUTE_RE.search(text):
        return False
    # Heuristic words that hint "this is a request/preference about
    # behaviour" rather than e.g. "what voice are you using?" (recall)
    # or "how do I change the voice?" (question). Without this gate
    # the function fires on every TTS-related question too.
    preference_hint = re.search(
        r"(?:respond|reply|answer|use|prefer|always|"
        r"отвеча[йитл]+|использ[оуьте]+|"
        r"должен|должна|должно|"
        r"prefer|wanted?|want|need|должен)",
        text, re.IGNORECASE,
    )
    if not preference_hint:
        return False
    return True


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


# Pipeline-mode constants. Three explicit tiers:
#
#   fast_chat   — small-talk, status, preference, micro-ack. No
#                 thinking, no tools, no verifier, no learning.
#                 The agent's existing chat/preference branches
#                 already implement this — the constant is here so
#                 every code path can name the tier consistently.
#
#   task_mode   — normal Q&A and lightweight tasks. Full plan +
#                 tools + memory, but NO verifier, NO self-critic
#                 retry, NO inline learning. Default tier for
#                 anything classified as `task` that doesn't trip
#                 the deep_agent signals below.
#
#   deep_agent  — code review, self-analysis, complex investigation.
#                 Adds verifier, retry, full learning extraction on
#                 top of task_mode. Triggered when:
#                   - `_looks_like_self_analysis_request(task)` matches
#                   - thinking picks question_type=="self_analysis"
#                   - thinking returns subtasks (decomposition needed)
#                   - thinking.confidence < 60 (uncertain plan → verify)
#                   - task contains explicit "review/audit/исследуй"
#                     keywords (recognised pre-think for early routing)
PIPELINE_FAST_CHAT = "fast_chat"
PIPELINE_TASK_MODE = "task_mode"
PIPELINE_DEEP_AGENT = "deep_agent"

# Pre-think keywords that escalate task → deep_agent regardless of
# what thinking later decides. Conservative — only matches phrases
# that almost always mean "you should verify yourself thoroughly".
_DEEP_AGENT_HINT_RE = re.compile(
    r"\b(?:review|audit|investigate|deep[\s-]?dive|"
    r"исследуй|разбер[иё]|"
    r"провер[ьи]|"
    r"code\s+review|self[\s-]?(?:analysis|review|audit))\b",
    re.IGNORECASE,
)


def _looks_like_deep_agent_request(text: str) -> bool:
    """Pre-think hint for deep_agent escalation. Used alongside
    `_looks_like_self_analysis_request` (which is more specific —
    self-analysis is one TYPE of deep_agent task, but reviews of
    third-party code or research questions also belong in
    deep_agent without being self-analysis)."""
    if not text:
        return False
    return bool(_DEEP_AGENT_HINT_RE.search(text))


def _pick_pipeline_mode(
    intent: str,
    task: str,
    thinking: "ThinkingResult | None",
) -> str:
    """Decide which pipeline tier should handle this turn. Cheap to
    call — no LLM. The verifier-side cost saving comes from the
    task_mode path skipping `_verify` (which is normally a 4-8 KB
    prompt + a separate LLM call); typical Q&A turns are
    `task_mode` and pay nothing extra for verification.
    """
    if intent == "chat" or intent == "preference":
        return PIPELINE_FAST_CHAT
    # intent == "task" from here on. Decide between task_mode and
    # deep_agent.
    if _looks_like_self_analysis_request(task) or _looks_like_deep_agent_request(task):
        return PIPELINE_DEEP_AGENT
    if thinking is not None:
        if thinking.question_type == "self_analysis":
            return PIPELINE_DEEP_AGENT
        if thinking.subtasks:
            return PIPELINE_DEEP_AGENT
        if isinstance(thinking.confidence, int) and thinking.confidence < 60:
            return PIPELINE_DEEP_AGENT
    return PIPELINE_TASK_MODE


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
from .pipeline.critic import SelfCriticMixin  # noqa: E402
from .pipeline.intent import IntentClassifierMixin  # noqa: E402
from .pipeline.preferences import PreferenceHandlerMixin  # noqa: E402
from .pipeline.thinking import ThinkingMixin  # noqa: E402


class Agent(
    IntentClassifierMixin,
    PreferenceHandlerMixin,
    ThinkingMixin,
    SelfCriticMixin,
):
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
                speaker_id=getattr(self, "_speaker_id", None),
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
    # `_classify_intent` is provided by IntentClassifierMixin
    # (backend/pipeline/intent.py). The mixin reads `self._t0` /
    # `self._llm_calls` etc. from this Agent instance — extracted
    # here for code-organisation, no behaviour change.

    # Быстрый chat-ответ: один LLM-вызов с identity preamble.
    def _chat_reply(self, task: str, core: str) -> str:
        self.progress("chat", "chatting...")
        system = _with_identity(
            CHAT_SYSTEM_BASE, speaker_id=getattr(self, "_speaker_id", None),
        )
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

    # `_save_preference` is provided by PreferenceHandlerMixin
    # (backend/pipeline/preferences.py). Extracted from this file
    # for code organisation — behaviour unchanged.

    # Step 2 — universal thinking
    # `_think` is provided by ThinkingMixin
    # (backend/pipeline/thinking.py). Extracted from this file
    # for code organisation — behaviour unchanged.

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
            # min_raw_score: rejects weak matches where the KB has
            # nothing actually relevant but min-max normalization would
            # scale the top noise hit to 1.0 and load it as a "best"
            # match. Bumped from 0.4 → 0.55 after the production audit
            # saw `mercury boiling point` pull in `Scary Movie` and
            # `German vocabulary` (both passed the 0.4 floor on weak
            # vector overlap). 0.55 sits comfortably above the bge-m3
            # noise band; legitimate cross-domain matches typically
            # land >0.6.
            hit = HYBRID.find_best(topic, min_raw_score=0.55)
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
        system = _with_identity(
            SOLVER_SYSTEM_BASE, speaker_id=getattr(self, "_speaker_id", None),
        )
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

        # Round F: progressive reveal. Wrap registry.execute so we
        # emit a `tool_starting` event BEFORE the tool runs (just
        # name + args, no result yet). The existing `_on_tool_call`
        # still fires AFTER completion as the Tool output event. The
        # frontend renders them as two separate OpenClaw-style pills:
        # the "Tool call" card appears immediately when the LLM
        # decides to invoke the tool, then the matching "Tool output"
        # card appears when the result returns. Big UX win on slow
        # tools (read_file of huge files, run_python with long
        # execution) where the gap is many seconds.
        def _execute_with_progress(name: str, args: dict):
            preview = ", ".join(str(k) for k in (args or {}).keys())
            self.progress(
                "tool_starting",
                f"{name}({preview})",
                tool_call=ToolCallDetail(
                    name=name,
                    args=args or {},
                    result="",
                    result_truncated=False,
                    result_full_len=0,
                    is_error=False,
                ),
            )
            return registry.execute(name, args)

        import time as _t
        _t0 = _t.monotonic()
        usage_before = TOKENS.request_usage()
        answer = router().call_with_tools(
            TaskType.COMPLEX_SOLVING,
            system,
            user,
            tools=tools,
            execute_tool=_execute_with_progress,
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
    # `_build_critique` + `_verify` are provided by SelfCriticMixin
    # (backend/pipeline/critic.py). Extracted from this file for
    # code organisation — behaviour unchanged.

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

        Skipped entirely on profile-recall turns ('what is my color?',
        'do you remember X'). Those questions are recall-only — they
        don't introduce new facts about the user, but pre-fix the
        extractor was scanning the agent's answer ('Your color is
        teal') and creating a duplicate of an already-stored fact on
        every recall. The memory log doubled in size on every
        conversation about previously-stored topics.
        """
        if _looks_like_profile_recall(user_msg):
            return
        try:
            facts = MEMORY.extract_and_store(
                user_msg, answer, intent=intent,
                confidence=confidence, contradictions=contradictions,
                speaker_id=getattr(self, "_speaker_id", None),
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
        speaker_id: str = "webui:default",
    ) -> AgentAnswer:
        import time as _time
        from .sessions import normalize_speaker
        # Audit #14: agent.run state used to be set unconditionally
        # on `self`, so a re-entrant call (tool handler that itself
        # invokes another agent.run) would clobber the outer call's
        # state silently. Today no tool handler does that, but the
        # design is brittle. Snapshot the prior state on entry and
        # restore in `finally` so re-entrancy is safe even though
        # it's still not a documented pattern.
        _prev_state = {
            attr: getattr(self, attr, None)
            for attr in (
                "_trace", "_llm_calls", "_request_id",
                "_self_analysis_unverified", "_t0", "_attachments",
                "_channel", "_speaker_id", "_role", "_role_token",
                "_sticky_info", "_sticky_token", "_mode",
            )
        }
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
        # Speaker — '<channel>:<user_id>'. Primary routing key for
        # sessions + conversation memory + per-speaker user profile.
        # Stays on `self` so every add_turn / context_block in this
        # run picks up the same speaker without each call passing it.
        self._speaker_id = normalize_speaker(speaker_id)
        # Phase 11: per-speaker role gating. Read once at run start
        # so tool checks below don't re-read the roles file per call.
        # Set the ContextVar so deeply-nested tool handlers
        # (run_python, schedule_message, …) can see who's asking
        # without every signature having to thread `speaker_id`
        # through.
        from . import roles as _roles
        self._role: str = _roles.role_of(self._speaker_id)
        self._role_token = _roles.set_current_speaker(self._speaker_id)
        # Pipeline mode picked for this turn (fast_chat / task_mode /
        # deep_agent). Set by the branches below so AgentAnswer can
        # report which tier ran.
        self._mode: str = ""

        # Context compaction: when the speaker's recent history
        # exceeds the chars budget, summarise the middle band into
        # one structured `[CONTEXT COMPACTION]` turn so the tail
        # context (recent list + active sub-thread) survives. The
        # compactor is cheap when nothing needs compacting (just a
        # sum check). Runs BEFORE sticky detection so the detector
        # sees the post-compaction history.
        try:
            from . import context_compressor as _cc
            _cc.maybe_compact(speaker_id=self._speaker_id)
        except Exception as e:
            log.debug("context_compressor failed (non-fatal): %s", e)

        # Sticky-request detection: inspect this speaker's recent
        # conversation. If the SAME system attribute (voice / language
        # / model / …) was raised in M of the last K turns and the
        # agent's prior answers were short acks without tool calls,
        # we plant a `# STICKY REQUEST DETECTED` block in the system
        # prompt + force task-mode routing so this turn actually
        # applies the change. The pattern matches the production case
        # ("Измени голос" × 4) that motivated all of Phase 2.
        from . import sticky_requests as _sticky
        sticky_info = _sticky.detect_sticky_request(
            current_user_message=task,
            speaker_id=self._speaker_id,
        )
        self._sticky_info = sticky_info
        self._sticky_token = _current_sticky_block.set(
            _sticky.render_sticky_block(sticky_info)
        )
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
                    speaker_id=getattr(self, "_speaker_id", None),
                )
                # Memory extraction gate: short acks have nothing to
                # extract — skipping saves another LLM call. The
                # lookups in EVALUATOR / GOALS still get the turn
                # via add_turn → recall_block.
                self._cleanup()
                self._tick_goals()
                self._persist_dev_capture(task, micro, 100)
                self._mode = PIPELINE_FAST_CHAT
                return AgentAnswer(
                    answer=micro,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    mode=PIPELINE_FAST_CHAT,
                    token_usage=self._get_token_usage(),
                    thinking_trace=self._trace,
                    llm_calls=self._llm_calls,
                )

            intent = self._classify_intent(task)
            # Sticky override: when the detector caught the user
            # repeating a system-setting request, force task path
            # regardless of what the LLM classifier said. Same
            # destination as `_looks_like_system_directive` /
            # `_looks_like_system_setting_preference` — the sticky
            # path is the BEHAVIOURAL backstop for the cases those
            # syntactic guards miss.
            if self._sticky_info.get("sticky"):
                if intent != "task":
                    self.progress(
                        "sticky_override",
                        f"sticky request on {self._sticky_info.get('attribute')!r}; "
                        f"forcing task path",
                    )
                intent = "task"

            # Branch 1: chitchat / small-talk. One warm reply, no pipeline.
            if intent == "chat":
                answer = self._chat_reply(task, core)
                CONVERSATION.add_turn(
                    task, answer, intent="chat", is_chat=True,
                    channel=self._channel,
                    speaker_id=getattr(self, "_speaker_id", None),
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
                self._mode = PIPELINE_FAST_CHAT
                return AgentAnswer(
                    answer=answer,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    mode=PIPELINE_FAST_CHAT,
                    token_usage=self._get_token_usage(),
                    thinking_trace=self._trace,
                    llm_calls=self._llm_calls,
                )

            # Branch 2: preference — user configures the agent or shares
            # personal info. Save to user.md and acknowledge briefly.
            #
            # Pre-ack guard: if the preference touches a SYSTEM
            # ATTRIBUTE the agent owns (voice, model, language, …),
            # escalate to the task path instead of just saving a
            # fact. Pre-fix, "respond using a male voice" got saved
            # as a preference in user_profile.md and nothing
            # changed; the user repeated the request 4 times in
            # production. The directive regex catches imperative
            # form; this catches the preference form ("I want X /
            # always X / use X") that mentions a system attribute
            # the agent can mutate. Same destination — task path
            # with tools — different entry signal.
            if intent == "preference" and _looks_like_system_setting_preference(task):
                self.progress(
                    "preference_to_task",
                    "preference touches a system setting; "
                    "running as a task so tools fire",
                )
                intent = "task"

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
                    speaker_id=getattr(self, "_speaker_id", None),
                )
                self._extract_memories(task, reply, "preference")
                self._cleanup()
                self._tick_goals()
                self._persist_dev_capture(task, reply, 100)
                self._mode = PIPELINE_FAST_CHAT
                return AgentAnswer(
                    answer=reply,
                    verification=VerificationResult(confidence=100),
                    learned_topics=[],
                    used_topics=[],
                    project=project,
                    is_chat=True,
                    mode=PIPELINE_FAST_CHAT,
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

            # Pipeline tier decision. task_mode (default) skips
            # verifier + retry + heavy learning — that's the cost
            # saving for normal Q&A. deep_agent gets the full cycle
            # and is reserved for self-analysis, code review, and
            # complex investigations where the extra rounds pay off.
            pipeline = _pick_pipeline_mode(intent, task, thinking)
            self._mode = pipeline
            self.progress("mode", pipeline)
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

            if pipeline == PIPELINE_TASK_MODE:
                # task_mode short-circuit: no verifier LLM call, no
                # self-critic retry, no inline learning. The placeholder
                # VR uses thinking.confidence (the cheapest honest
                # number we have) so downstream code that reads
                # vr.confidence (conversation row, claims builder,
                # AgentAnswer) keeps working. Anything that genuinely
                # needs the verifier is routed to deep_agent upstream
                # by _pick_pipeline_mode.
                vr = VerificationResult(
                    confidence=int(thinking.confidence) if thinking else 75,
                    notes_used=[n.frontmatter.topic for n in notes],
                )
            else:
                # === deep_agent path: full verify + retry + learning ===
                # Skip verification for creative / meta / pure self_analysis where
                # there's no factual ground truth — saves 1 LLM call. BUT when
                # self_analysis came with tool_context (the agent actually
                # read_file'd its own code) the tool output IS evidence and
                # must be checked against — otherwise we ship hallucinated
                # claims about file contents.
                qtype = thinking.question_type if thinking else ""
                no_evidence_types = ("creative", "meta", "self_analysis")
                skip_verify = (
                    qtype in no_evidence_types
                    and not (tool_context or "").strip()
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

            # Self-critic loop + inline learning + meta-learner —
            # ALL deep_agent only. task_mode skipped the verifier
            # above and has nothing useful to retry / learn from.
            if pipeline != PIPELINE_DEEP_AGENT:
                # task_mode flow stops after answer + placeholder vr.
                # Light memory extraction still runs below for both
                # paths (it picks up user-stated facts independently
                # of verifier confidence).
                pass
            else:
                critic_threshold = CONFIG.verification.get("critic_threshold", 50)
                max_retries = CONFIG.verification.get("critic_max_retries", 2)
                # Hard token budget for retries — keeps a low-confidence
                # answer from burning $0.50 across retry × 2.
                retry_token_budget = CONFIG.verification.get(
                    "critic_retry_token_budget", 60000
                )
                retry = 0
                no_notes = not notes and not tool_context
                should_retry = (
                    vr.confidence < critic_threshold
                    and not no_notes  # don't retry if there's no evidence at all
                )
                # Self-analysis retry guard: the audit caught a single
                # "how are you built?" turn racking up 11 LLM calls /
                # $0.26 because the verifier returned 67% confidence
                # (under the 70 threshold), triggering a retry that
                # re-ran think + solve + verify. For self-analysis
                # turns, the first answer already had access to up to
                # 60K of source code via read_file/view_file — a
                # retry won't discover new code that wasn't
                # accessible before, so it just burns cost without
                # improving the answer. Cap retries to 0 here.
                if is_self_analysis:
                    should_retry = False
                # Accumulate tool_context across retry attempts so the
                # verifier on attempt N still sees evidence collected
                # on attempts 1..N-1.
                accumulated_tool_context = tool_context or ""
                _MAX_ACCUMULATED_CTX = 80000
                while should_retry and retry < max_retries:
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

                # Inline learning + meta-learner failure analysis only
                # in deep_agent. task_mode skips both — those flows
                # spend an LLM call each on top of a turn that's
                # already happy with its answer.
                self._learn_from_experience(task, answer, notes, vr, project)

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

            # Compact summary fields stamped on the conversation row
            # so the WebUI restores token bar + tool / LLM counts on
            # page refresh without a lazy /api/turns/<id> roundtrip.
            _tu = self._get_token_usage()
            _n_tools = sum(
                1 for s in self._trace
                if s.tool_call and (s.event == "tool" or s.event == "tool_error")
            )
            CONVERSATION.add_turn(
                task, answer,
                intent=thinking.question_type if thinking else "task",
                confidence=vr.confidence,
                topics_used=used,
                channel=getattr(self, "_channel", "webui"),
                speaker_id=getattr(self, "_speaker_id", None),
                turn_id=turn_id,
                token_usage=_tu.model_dump() if _tu else None,
                n_tool_calls=_n_tools,
                n_llm_calls=len(self._llm_calls or []),
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
                mode=getattr(self, "_mode", "") or pipeline,
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
                mode=getattr(self, "_mode", "") or "",
                token_usage=self._get_token_usage(),
                thinking_trace=self._trace,
                llm_calls=self._llm_calls,
            )
        finally:
            # Phase 11: always restore the prior speaker ContextVar
            # even on early return / exception, so a long-lived
            # process never accidentally serves the next request
            # with leaked role context.
            try:
                _roles.reset_current_speaker(self._role_token)
            except Exception:
                pass
            # Same reset for the sticky-block ContextVar — a leaked
            # block would carry into the next request and tell that
            # request's LLM about a "sticky" pattern that wasn't
            # its conversation.
            try:
                _current_sticky_block.reset(self._sticky_token)
            except Exception:
                pass
            # Audit #14: restore instance state captured at the top
            # of run(). Makes re-entrancy safe — an inner run()'s
            # state doesn't survive past its own `finally`.
            for _attr, _val in _prev_state.items():
                if _val is None:
                    try:
                        delattr(self, _attr)
                    except AttributeError:
                        pass
                else:
                    setattr(self, _attr, _val)
