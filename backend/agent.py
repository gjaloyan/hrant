"""Main agent loop.

The live turn path is `unified_agent.run_unified` (single tool-loop).
`Agent.run` delegates directly to it; the legacy pre-unified pipeline
(classify → think → solve → verify) has been removed.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Callable, Optional

from .config import CONFIG
from .conversation import CONVERSATION
from .core_memory import CORE
from .knowledge_manager import _slug
from .llm import TOKENS
from .models import (
    AgentAnswer,
    Note,
    LLMCallDetail,
    ThinkingStep,
    ToolCallDetail,
    TokenUsage,
)
from .dev_capture import redact_prompt, new_request_id
from .goals import GOALS
from .memory_extractor import MEMORY
from .project_mode import PROJECTS
from .mcp_client import MCP, MCPServerConfig
from .skills import SKILLS
from .tool_registry import get_registry
from .verifier import verify

# Re-export SOLVER_SYSTEM_BASE so callers that do
# `from backend.agent import SOLVER_SYSTEM_BASE` keep working.
from .prompts import SOLVER_SYSTEM_BASE  # noqa: F401


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


# Tools that count as "reading the source" for self-analysis guard.
SOURCE_READ_TOOLS = frozenset({"read_file", "view_file"})


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


class Agent(
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
        """Common per-turn context block (state snapshot).

        Soul / identity / user profile / name+language overrides are
        already in the system prompt via `IDENTITY.preamble()`. THIS
        block carries the per-turn state that tells the model "what we
        are doing right now":

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
        # Stash attachments so unified_agent helpers can pick them up
        # without threading the kwarg through every helper.
        self._attachments = attachments or None
        # Channel tag for conversation memory + turn record.
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
        # Unified path — the only path. See backend/unified_agent.py.
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

