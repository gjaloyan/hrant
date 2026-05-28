"""Interactive REPL for the agent.

Entered via `hrant chat` (the unified CLI dispatcher routes here
through `backend/cli.py:cmd_chat`).

    hrant chat                   # interactive chat
    hrant chat 'your question'     # one-shot query

Thin loop: every line goes straight to the unified agent, which sees
all tools and decides what to do (remember a fact, learn a topic,
answer a question, ...). There is no command parser — read-out
utilities live in the `hrant` CLI subcommands (`hrant status`,
`hrant graph`, `hrant jobs`, ...).
"""
from __future__ import annotations
import sys

from backend.agent import Agent
from backend.models import AgentAnswer
from backend.project_mode import PROJECTS


def _progress(event: str, msg: str) -> None:
    markers = {
        "core": "💾",
        "think": "🧠",
        "chat": "💬",
        "found": "✓",
        "learning": "📖",
        "learned": "📚",
        "solve": "✍️ ",
        "skill": "⚡",
        "tool": "🔧",
        "tool_error": "🔧⚠️",
        "verify": "🔍",
        "memory_save": "📝",
        "error": "⚠️ ",
    }
    print(f"  {markers.get(event, '·')} {msg}", flush=True)


def _print_answer(res: AgentAnswer) -> None:
    # Lightweight output for small-talk: no box and no verification footer.
    if res.is_chat:
        print(f"\n{res.answer}\n")
        return

    print("\n" + "=" * 60)
    print(res.answer)
    print("=" * 60)
    vr = res.verification
    badge = "🟢" if vr.confidence >= 90 else ("🟡" if vr.confidence >= 70 else "🔴")
    print(f"{badge} confidence: {vr.confidence}%")
    if res.used_topics:
        print(f"📂 topics used: {', '.join(res.used_topics)}")
    if res.learned_topics:
        print(f"📚 newly learned: {', '.join(res.learned_topics)}")
    if vr.unverified_claims:
        print(f"⚠️  unverified: {len(vr.unverified_claims)} claims")
        for c in vr.unverified_claims[:5]:
            print(f"    · {c}")
    if vr.contradictions:
        print(f"❌ contradictions: {len(vr.contradictions)}")
        for c in vr.contradictions[:5]:
            print(f"    · {c}")
    print()


def main() -> None:
    agent = Agent(progress=_progress)

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        _print_answer(agent.run(task, project=PROJECTS.current))
        return

    print("🧠 Self-Learning Agent — CLI")
    print("   exit / quit — exit")
    if PROJECTS.current:
        print(f"   active project: {PROJECTS.current}")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            return
        _print_answer(agent.run(text, project=PROJECTS.current))


if __name__ == "__main__":
    main()
