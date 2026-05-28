"""Main agent loop.

The agent first classifies the incoming message into three categories:

  * chat       — short small-talk (greeting, farewell, gratitude,
                 "how are you", "who are you"). One warm reply, no topic
                 analysis, no notes, no verification.
  * preference — the user communicates how to interact or what to remember
                 about them ("answer in Russian", "my name is Armen",
                 "be brief", "don't add disclaimers"). We extract a
                 structured fact and save it to user.md, reply with a short
                 confirmation.
  * task       — everything else: a real task, question, code, calculation.
                 The full 7-step cycle runs: topic analysis, knowledge-base
                 search, verification, and experience collection.

The idea is that "deep thinking" only kicks in when it is actually needed.
Casual exchanges are handled warmly and quickly; behavioural instructions
are remembered and applied; tasks are solved strictly from knowledge.
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
# `from .analogy_engine import ANALOGIES` was retired 2026-05-27
# (audit T3.1). Pattern extraction fired only in the legacy
# non-unified Agent.run path; unified_agent never wired it in.
# The patterns.json file went stale 2026-05-16. Removed module +
# WebUI endpoint with this commit.
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

# ---------- system prompts ----------
#
# Prompts live in backend/prompts.py (easier to edit without scrolling
# through the 2000-line agent.py). Re-import keeps test/channel callers
# that do `from backend.agent import SOLVER_SYSTEM_BASE` working. The
# identity preamble and capabilities block are added at runtime via
# `_with_identity()` and `_capabilities_block()` below.
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


def _with_identity(
    base_system: str,
    *,
    speaker_id: str | None = None,
) -> str:
    """Stitches identity preamble + agent state snapshot +
    per-speaker permissions together with a specific step's system prompt.

    `speaker_id` selects which user_profile and role/permission
    block to inject. Default None falls back to the WebUI speaker
    (which is always owner) for back-compat.

    The STATE SNAPSHOT block (`backend.self_state`) is what gives
    the agent line-of-sight to its own current settings — active
    TTS voice, active model, config file paths, tools registered.
    Without it, "change your voice to male" became "Understood" + a
    stored preference fact + no actual change; with it, the LLM
    sees the current voice + config path + the `set_setting`
    tool it can use to mutate the file.

    Note: per-turn `sticky_block` plumbing was removed 2026-05-23
    (audit Important #8) along with `backend.sticky_requests`.
    The TSP rule "Apply, don't acknowledge" plus the new
    `re_prompt_resilience` section cover the original use case.
    """
    from . import roles as _roles
    from . import self_state as _ss
    perms = _roles.permissions_block(speaker_id)
    try:
        state = _ss.current_self_state(speaker_id=speaker_id)
        snapshot = _ss.render_snapshot(state)
    except Exception:
        snapshot = ""  # never block a turn over the bookkeeping block
    return (
        f"{IDENTITY.preamble(speaker_id=speaker_id)}\n\n"
        + (f"---\n\n{snapshot}\n\n" if snapshot else "")
        + f"---\n\n{base_system}\n\n"
        + f"---\n\n{perms}"
    )


def _capabilities_block(compact: bool = False) -> str:
    """Dynamic "MY CAPABILITIES" block for the system prompt.

    Lists the concrete things the agent has and can do right now:
    registered tools, loaded skills, connected MCP servers, and a map
    of its own source code on disk — so that when asked about itself
    the agent knows where to look (read_file) instead of hallucinating.

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
        lines.append("\n## Tools")
        desc_cap = 60 if compact else 100
        for name, tool in sorted(tools.items()):
            origin_label = f"  [{tool.origin}]" if tool.origin != "builtin" else ""
            lines.append(f"- `{name}` — {tool.description[:desc_cap]}{origin_label}")

    # --- Skills ---
    if SKILLS.skills:
        lines.append("\n## Skills")
        for sk in SKILLS.skills:
            triggers = ", ".join(sk.triggers[:5]) if sk.triggers else "—"
            desc_cap = 60 if compact else 80
            lines.append(f"- **{sk.name}**: {sk.description[:desc_cap]}  (triggers: {triggers})")

    # --- MCP ---
    if MCP.servers and not compact:
        lines.append("\n## Connected MCP servers")
        for srv_name, srv in MCP.servers.items():
            tool_count = len([t for t in tools if t.startswith(f"mcp_{srv_name}__")])
            lines.append(f"- `{srv_name}` ({tool_count} tools)")

    if compact:
        # Compact stops here. The big source map is only useful when
        # the model is going to actually read its own code; chat /
        # think don't need it and pay for those ~3k chars on every turn.
        return "\n".join(lines)

    # --- Source map (for self-referential questions) ---
    root = Path(__file__).resolve().parent.parent
    lines.append("\n## My source code (source map)")
    lines.append(f"Root directory: `{root}`")
    lines.append("Key files:")
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
        # analogy_engine retired 2026-05-27 — see commit log.
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
    lines.append("\nTo learn how I am built, call `read_file` on any of these files.")
    lines.append("Do NOT make claims about your own code without reading the file first.")

    return "\n".join(lines)


# Quick filter for the most obvious cases — without a single LLM call.
# Deliberately narrow: only catches unambiguous small-talk; anything
# ambiguous goes to the LLM classifier.
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

# Word-form arithmetic (e.g. "сколько будет два плюс два", "calculate 2+2",
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
      - word form (e.g. "сколько будет 25 умножить на 3", "calculate 12 times 4")
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
# "what is my X" / "do you remember Y" / etc. The agent already has user
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
    # Russian: "что мой/моя/моё/мои X" (what is my X), "помнишь(/-ите) мой X" (do you remember my X),
    # "что ты обо мне знаешь" (what do you know about me), "что я тебе говорил" (what did I tell you)
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
# (change voice to male) routed to preference → saved to user_profile.md
# → no actual change. `_handle_preference` and the LLM classifier both
# saw it as "preference" because the wording ("change X") superficially
# matches how users phrase preferences. This regex catches the imperative-
# directive shape and forces it onto the task path where tools fire.
#
# Pattern intent:
#   - Imperative verb at the start of the message
#   - Optionally addressed to the agent ("ты" / "you")
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
    # Increase/decrease/raise/lower/bump/slow down/speed up etc.
    # Caught in the 2nd Telegram audit — "Increase voice speed by
    # 30%" missed every prior pattern and landed on the preference
    # path, which acked without applying.
    r"increase|decrease|raise|lower|bump|reduce|speed\s*up|slow\s*down|"
    # "Make X faster/slower" — common phrasing.
    r"give\s+(?:me\s+)?(?:a\s+|the\s+)?[a-z]+\s+(?:voice|model|setting))\b"
    # Russian directives — most-common imperative forms the user
    # actually types in chat (the production audit caught
    # "измени голос на мужской" (change voice to male) repeated 4 times).
    r"|(?:пожалуйста\s+)?(?:измен[иьяет]+|"
    r"смени|поменя[йли]+|"
    r"переключ[иьи]+|перенастро[йли]+|"
    r"сдела[йли]+|"
    r"поста[вьи]+|постав[ьи]+|"
    r"включ[иьи]+|выключ[иьи]+|"
    r"настро[йли]+|"
    # Increase/decrease/speed-up/slow-down — also caught in 2nd audit:
    # "увеличь скорость" (increase speed), "ускорь голос" (speed up voice), "замедли" (slow down).
    r"увелич[ьитл]+|уменьш[итеь]+|"
    r"ускор[ьитл]+|замедл[ьитл]+|"
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
    r"voice|tts|голос|озвучк[ауи]+|speech|речь|реч[иь]|"
    r"language|язык|"
    r"model|модель|"
    r"provider|провайдер|"
    r"channel|канал|"
    r"tone|тон|стиль|"
    r"speed|скорост|rate|темп|"
    r"volume|громкост|"
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
      "Измени голос на мужской" (change voice to male) → change voice
      "switch model to gpt-4o"                         → change model
      "поставь язык русский" (set language to Russian) → change language
      "Set the TTS voice to a male voice"              → change voice

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


# Detector for questions about the agent itself — used in _chat_reply
# to inject the capabilities block even in the lightweight path.
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
#                   - task contains explicit "review/audit/исследуй" (investigate)
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


# ---------- progress callback type ----------
ProgressCB = Callable[..., None]  # (event, message, tool_call: ToolCallDetail|None=None)


def _noop(_evt: str, _msg: str, _tc: "ToolCallDetail | None" = None) -> None:
    pass


# Lazily connect MCP servers once per process. If nothing is configured in
# config.yaml this is a no-op. Done at module level (not in Agent.__init__)
# so tests with a mocked agent do not attempt to spin up subprocesses.
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


# ---------- agent ----------
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
        # Mirror this progress event onto the LogBus so the WebUI
        # Logs tab sees it alongside Python logging + tool calls +
        # job events. Best-effort — never break the agent on a
        # logging concern.
        try:
            from .log_bus import publish_agent_event as _pub_agent
            _pub_agent(
                event=event,
                message=message,
                request_id=getattr(self, "_last_turn_id", "") or "",
            )
        except Exception:
            pass

    # Step 1
    def _load_core(self) -> str:
        self.progress("core", "loading core memory")
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

    # Step 1.5 — intent classification (chat | preference | task)
    # `_classify_intent` is provided by IntentClassifierMixin
    # (backend/pipeline/intent.py). The mixin reads `self._t0` /
    # `self._llm_calls` etc. from this Agent instance — extracted
    # here for code-organisation, no behaviour change.

    # Quick chat reply: one LLM call with identity preamble.
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
            return "Hi! The API is temporarily unavailable, but I am here. Try again in a moment."
        if any(w in t for w in ("пока", "bye", "до свидания", "goodbye")):
            return "Goodbye! Talk soon."
        if any(w in t for w in ("спасибо", "thanks", "thank")):
            return "You're welcome!"
        if any(w in t for w in ("как дела", "как ты", "how are you")):
            return "The API is currently overloaded, but otherwise I'm fine. Please try again in a minute."
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
                    return f"{short}\n\n_(API unavailable — answer from identity.md)_"
            except Exception:
                pass
            return "I am a self-learning AI agent. The API is currently unavailable, but ask again later — I will tell you more."
        return "The API is currently unavailable. Please try again in a minute — I will be ready to help."

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

    # Step 3
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
        # Deduplicate the input topic list
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
            self.progress("learning", f"learning: {topic}")
            try:
                note = learn_topic(topic, depth="quick", project=project)
                note_slug = _slug(note.frontmatter.topic)
                if note_slug not in loaded_slugs:
                    loaded.append(note)
                    loaded_slugs.add(note_slug)
                learned.append(topic)
                self.progress("learned", f"note created: {topic}")
            except Exception as e:
                self.progress("error", f"failed to learn {topic}: {e}")
        return loaded, learned

    # helper: collect note text
    def _notes_block(self, notes: list[Note], max_total_chars: int = 12000) -> str:
        """Assembles the notes block for context with a soft size cap.

        If the combined size of the notes exceeds the cap, the longest
        notes are trimmed (from the end) to fit. This prevents context
        blow-up when the analyser requested 6 large topics at once.
        """
        if not notes:
            return "(no notes loaded)"

        # Deduplicate by slug just in case.
        seen: set[str] = set()
        unique: list[Note] = []
        for n in notes:
            s = _slug(n.frontmatter.topic)
            if s in seen:
                continue
            seen.add(s)
            unique.append(n)

        # Simple strategy: equal quota per note.
        quota = max(400, max_total_chars // max(1, len(unique)))
        parts: list[str] = []
        for n in unique:
            body = n.body
            if len(body) > quota:
                body = body[:quota].rstrip() + "\n… [truncated for context]"
            parts.append(
                f"### {n.frontmatter.topic} "
                f"(source: {n.frontmatter.source or 'internal'})\n{body}"
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

    # Step 6
    def _learn_from_experience(
        self,
        task: str,
        answer: str,
        notes: list[Note],
        vr: VerificationResult,
        project: str | None,
    ) -> None:
        self.progress("experience", "saving experience")
        # access_count is already tracked in KM.get_note
        # Auto-collect into finetune queue: confidence >= 85%, notes present, verified
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
                f"Q&A added to finetune queue [{added.metadata.category}]",
            )

        # analogy_engine pattern extraction retired 2026-05-27.

    # Step 7 — cleanup (nothing to unload, notes are already on disk)
    def _cleanup(self) -> None:
        self.progress("cleanup", "done")

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

    # ---------- public entry point ----------
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
        session_key: str | None = None,
        job_id: str | None = None,
        supervisor_mode: bool = False,
        supervisor_job_id: str | None = None,
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
                "_channel", "_speaker_id", "_session_key", "_job_id",
                "_role", "_role_token", "_mode",
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
        # Speaker — '<channel>:<user_id>'. Identity key for roles,
        # profile, knowledge. Same person across every chat thread.
        self._speaker_id = normalize_speaker(speaker_id)
        # Session key — thread isolation key. Same person in two
        # different Telegram chats (or different bots) gets distinct
        # session_keys, so their conversation buffers don't bleed.
        # Defaults to speaker_id when the caller hasn't been updated
        # to pass a chat-scoped key (e.g. the WebUI single-user path).
        self._session_key = (session_key or "").strip() or self._speaker_id
        # Job id — set by run_tracked BEFORE agent.run so unified_agent
        # can deep-link the SESSIONS turn record to the durable Job.
        # None for CLI / fast-path callers that don't track jobs.
        self._job_id = job_id or None
        # Phase 11: per-speaker role gating. Read once at run start
        # so tool checks below don't re-read the roles file per call.
        # Set the ContextVar so deeply-nested tool handlers
        # (run_python, schedule_message, …) can see who's asking
        # without every signature having to thread `speaker_id`
        # through.
        from . import roles as _roles
        self._role: str = _roles.role_of(self._speaker_id)
        self._role_token = _roles.set_current_speaker(self._speaker_id)
        # Phase C: unified path is the ONLY path. The legacy
        # classifier / preference branch / pipeline tier code that
        # used to live below this point is now dead — Agent class
        # still has the mixin methods (`_chat_reply`, `_solve`,
        # `_think`, `_save_preference`, `_classify_intent`) for
        # source-code stability + reverter-friendliness, but
        # `run()` never calls them. A later cleanup sprint will
        # delete the mixins + their backing pipeline/*.py files.
        # See backend/unified_agent.py for the architecture.
        self._mode = "unified"
        from . import unified_agent as _ua
        try:
            return _ua.run_unified(
                agent=self,
                task=task,
                project=project,
                attachments=attachments,
                channel=channel,
                speaker_id=self._speaker_id,
                session_key=self._session_key,
                job_id=self._job_id,
                supervisor_mode=supervisor_mode,
                supervisor_job_id=supervisor_job_id,
            )
        finally:
            try:
                _roles.reset_current_speaker(self._role_token)
            except Exception:
                pass
            # Re-entrancy state restore — same shape the legacy
            # finally{} used.
            for _attr, _val in _prev_state.items():
                if _val is None:
                    try:
                        delattr(self, _attr)
                    except AttributeError:
                        pass
                else:
                    setattr(self, _attr, _val)

