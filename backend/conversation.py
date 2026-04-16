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
  - The conversation block is injected into _think and _solve prompts
    so the agent knows what was discussed
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
    ) -> None:
        """Record one conversation turn (user question + agent response)."""
        # Truncate answer for storage efficiency
        answer_preview = agent_answer
        if len(answer_preview) > self.max_answer_chars:
            answer_preview = answer_preview[: self.max_answer_chars] + "..."

        turn = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_message.strip(),
            "answer": answer_preview.strip(),
            "intent": intent,
            "is_chat": is_chat,
            "confidence": confidence,
        }
        if topics_used:
            turn["topics"] = topics_used

        self._turns.append(turn)
        # Trim to max
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]
        self._save()

    def recent(self, n: int = 10) -> list[dict]:
        """Get the last N turns."""
        return self._turns[-n:]

    def context_block(self, n: int = 6) -> str:
        """Build a conversation context block for injection into prompts.

        Returns a formatted string showing recent exchanges. If there's
        no history, returns an empty string (so it doesn't clutter prompts
        for the first message of a session).
        """
        turns = self.recent(n)
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
            if t.get("confidence"):
                lines.append(f"_(confidence: {t['confidence']}%)_")
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
