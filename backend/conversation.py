"""Conversation memory: persists recent exchanges across messages.

The problem: each agent.run() call is stateless. When the user says
"continue" after a rate-limit error, the agent has no idea what to
continue. This module gives the agent a sliding window of recent
conversation turns, persisted to disk so it survives restarts.

Design:
  - Stores the last N turns (user message + agent answer summary)
  - Persisted as a single JSON file in the knowledge directory
  - Auto-trims to max_turns (default 20) and max_chars (default 12000)
  - Each turn stores: user message, agent answer (truncated), timestamp,
    intent classification, and whether it was a chat/task
  - The conversation block is injected into the unified agent turn's
    system prompt so the agent knows what was discussed
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG


class ConversationMemory:
    """Sliding window of recent conversation turns."""

    def __init__(
        self,
        path: Optional[Path] = None,
        max_turns: int = 20,
        max_answer_chars: int = 500,
    ):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "conversation.json")
        self.max_turns = max_turns
        self.max_answer_chars = max_answer_chars
        self._turns: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._turns = data if isinstance(data, list) else []
            except Exception:
                self._turns = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._turns, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # conversation memory is best-effort

    def add_turn(
        self,
        user_message: str,
        agent_answer: str,
        intent: str = "task",
        is_chat: bool = False,
        confidence: int = 0,
        topics_used: list[str] | None = None,
        *,
        channel: str = "webui",
        speaker_id: str = "webui:default",
        session_key: str | None = None,
        turn_id: str = "",
        token_usage: Optional[dict] = None,
        n_tool_calls: int = 0,
        n_llm_calls: int = 0,
    ) -> None:
        """Record one conversation turn (user question + agent response).

        `speaker_id` ('<channel>:<user_id>') uniquely identifies WHO
        the agent talked to. Each speaker has their own conversation
        thread — turns from `telegram:123` (a real person) never
        appear in the context window of a `telegram:456` chat
        (a different person), even on the same Telegram bot.

        `channel` is retained as a coarse label (webui / telegram /
        ...) for UI grouping and back-compat; the primary scoping
        key is `speaker_id`.

        `turn_id` references the on-disk TurnWorkspace artifact at
        `workspace/turns/<turn_id>.json` (P1). Empty when the caller
        ran outside the main `Agent.run` path (chat fast-path,
        preference branch — these don't write a turn record). The
        WebUI lazy-loads the full thinking_trace + claims + evidence
        from that file when the user expands the message.
        """
        # Truncate answer for storage efficiency
        answer_preview = agent_answer
        if len(answer_preview) > self.max_answer_chars:
            answer_preview = answer_preview[: self.max_answer_chars] + "..."

        from .sessions import normalize_speaker
        sp = normalize_speaker(speaker_id)
        # session_key isolates conversation threads (e.g. Wife DM vs
        # Wife in a group are distinct threads sharing the same
        # speaker_id). When the caller hasn't been updated yet, we
        # default to the speaker_id — one thread per speaker, the
        # pre-refactor behaviour.
        skey = (session_key or "").strip() or sp
        turn = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_message.strip(),
            "answer": answer_preview.strip(),
            "intent": intent,
            "is_chat": is_chat,
            "confidence": confidence,
            "channel": channel or "webui",
            "speaker_id": sp,
            "session_key": skey,
        }
        if topics_used:
            turn["topics"] = topics_used
        if turn_id:
            turn["turn_id"] = turn_id
        # Round F-pre: stamp summary fields so WebUI badges (token
        # bar + tool / LLM call counts) survive page refresh without
        # the lazy /api/turns/<id> fetch. Heavy data (full
        # thinking_trace) is still fetched on expand.
        if token_usage:
            turn["token_usage"] = token_usage
        if n_tool_calls:
            turn["n_tool_calls"] = int(n_tool_calls)
        if n_llm_calls:
            turn["n_llm_calls"] = int(n_llm_calls)

        self._turns.append(turn)
        # Trim to max
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]
        self._save()

    def recent(
        self,
        n: int = 10,
        *,
        channel: Optional[str] = None,
        speaker_id: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> list[dict]:
        """Get the last N turns.

        Filter precedence:
          1. `session_key` (the PRIMARY thread filter — e.g. Wife's
             DM thread vs Wife's group-chat thread are separate even
             though both have speaker_id `telegram:<wife>`).
          2. `speaker_id` (identity filter, used when the caller
             cares about "everything this person said anywhere").
          3. `channel` (coarsest, legacy).

        Legacy turns saved without `session_key` fall back to their
        speaker_id when session_key filtering is in effect, so old
        per-speaker buffers keep working as before until they age
        out of the rolling window.
        """
        from .sessions import normalize_speaker
        turns = self._turns
        if session_key is not None:
            wanted = (session_key or "").strip()
            turns = [
                t for t in turns
                if (t.get("session_key") or t.get("speaker_id") or "") == wanted
            ]
        elif speaker_id is not None:
            wanted = normalize_speaker(speaker_id)
            turns = [
                t for t in turns
                if normalize_speaker(
                    t.get("speaker_id") or f"{t.get('channel') or 'webui'}:legacy"
                ) == wanted
            ]
        elif channel is not None:
            turns = [t for t in turns if (t.get("channel") or "webui") == channel]
        return turns[-n:]

    def recent_full(
        self,
        n: int = 50,
        *,
        channel: Optional[str] = None,
        speaker_id: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> list[dict]:
        """Like `recent` but bigger default — for the WebUI history
        list. Returns full turn dicts so the frontend can render
        without re-querying. Heavy data lazy-loaded via
        GET /api/turns/<id>."""
        return self.recent(
            n, channel=channel, speaker_id=speaker_id, session_key=session_key,
        )

    def context_block(
        self,
        n: int = 6,
        *,
        channel: Optional[str] = None,
        speaker_id: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> str:
        """Build a conversation context block for injection into prompts.

        Returns a formatted string showing recent exchanges. If
        there's no history, returns an empty string.

        `session_key` is the primary filter — when set, only turns
        from that specific conversation thread are included. This
        is how the agent avoids bleeding "wife's DM" into "wife in
        the group chat" or "wife on bot-B" — same identity, different
        thread. `speaker_id` is the identity-level fallback (still
        used by callers that don't yet have a session_key).
        Knowledge (KG + notes) stays SHARED across threads — only
        the recent buffer is scoped.

        `channel` is the legacy coarse filter, kept for back-compat.
        """
        turns = self.recent(
            n, channel=channel, speaker_id=speaker_id, session_key=session_key,
        )
        if not turns:
            return ""

        lines = [
            "# RECENT CONVERSATION",
            "(Previous exchanges in this session. Use this context when the "
            "user says 'continue', 'go on', refers to 'it', 'that', etc.)",
            "",
        ]
        for t in turns:
            ts = t.get("ts", "")
            intent_tag = f" [{t.get('intent', '')}]" if t.get("intent") else ""
            lines.append(f"**User** ({ts}{intent_tag}):")
            lines.append(t.get("user", ""))
            lines.append("")
            lines.append("**Agent:**")
            lines.append(t.get("answer", "(no response recorded)"))
            # Confidence label removed from context_block 2026-05-23:
            # the next turn's LLM was reading `_(confidence: 16%)_` on
            # the prior assistant message as evidence the agent should
            # keep hedging, which reinforced the verifier-refusal loop
            # caught in the audit. Confidence is still surfaced in the
            # WebUI (read from CONVERSATION rows / SESSIONS), just not
            # injected into the next LLM call's context.
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns = []
        self._save()

    def count(self) -> int:
        return len(self._turns)


CONVERSATION = ConversationMemory()
