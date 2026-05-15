"""Memory extractor: picks up facts from conversations and stores them in the graph.

After every conversation turn the extractor decides:
  1. Does the message contain any memorable facts? (prices, dates, preferences,
     technical specs, personal info, events, opinions...)
  2. If yes — extract structured triples and store them as "memory facts" in the
     knowledge graph with source="conversation".

The graph becomes the agent's long-term memory: not just a note index,
but a living store of everything the agent learned from talking to the user.

On the retrieval side, _think() queries the graph for facts relevant to
the current question BEFORE looking at notes — so the agent can recall
"tomatoes cost 2 USD/kg in Armenia" when the topic is Armenian markets.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .knowledge_graph import GRAPH
from .llm import LLMError, TaskType, router

# ---- Prompts ----

EXTRACT_FACTS_SYSTEM = """You are a memory extraction module for an AI assistant.
Given a conversation turn (user message + agent answer), extract FACTS worth remembering.

What counts as a memorable fact:
- Prices, numbers, measurements ("tomatoes cost 2 USD/kg in Armenia")
- Personal facts about the user ("user lives in Yerevan", "user is a developer")
- Events and dates ("project deadline is March 15")
- Technical specifications ("server has 32GB RAM")
- Preferences and opinions ("user prefers Python over Java")
- Locations and geography ("office is on 5th floor")
- Relationships between things ("Redis is used as cache for the main app")
- Rules and constraints ("API rate limit is 100 req/min")
- Any concrete, specific information that could be useful later

What is NOT a fact:
- Greetings, thanks, small talk
- Vague statements without specifics
- The agent's own reasoning process
- Questions without answers

For each fact, extract it as entity-relation-entity triples that can be stored
in a knowledge graph. Also provide tags/keywords for retrieval.

Return strictly JSON:
{
  "has_facts": true/false,
  "facts": [
    {
      "summary": "short human-readable summary of the fact",
      "triples": [
        ["entity1", "relation", "entity2"]
      ],
      "replaces": [
        ["old_subject", "old_relation", "old_target"]
      ],
      "tags": ["tag1", "tag2"],
      "category": "price" | "personal" | "technical" | "event" | "location" | "preference" | "relationship" | "rule" | "general",
      "confidence": 0.0-1.0
    }
  ]
}

Rules:
- Entity names should be lowercase, concise, noun phrases
- Relations should be short verbs or prepositions: "costs", "lives_in", "has", "is", "prefers", "located_at", "deadline", "uses", etc.
- Extract ALL facts, even small ones — memory should be comprehensive
- If nothing is worth remembering, return {"has_facts": false, "facts": []}
- Max 8 facts per turn

CORRECTIONS — IMPORTANT:
- When the user CORRECTS a previous fact ("not Tigran, my brother is Arman",
  "actually I live in Berlin now, not Yerevan", "ignore that, I meant 100 not 10"),
  fill the `replaces` array with the OLD triple(s) the new fact supersedes.
- Watch for correction signals in the user message:
  "not X", "actually", "I meant", "ignore that", "wait", "no, X is Y", "correction"
- The OLD triple in `replaces` must use the SAME canonical lowercase entity names
  as the corresponding `triples` entry — only the target changes.
- If the new fact is purely additive (no correction), leave `replaces` empty or omit it.
- Multi-valued relations (`brother_of`, `friend_of`, `child_of`) are NOT auto-corrected
  by the graph — the `replaces` hint is the ONLY way to invalidate the old fact, so
  emit it whenever the conversation makes the correction explicit."""


def _looks_like_agent_self_rule(
    raw: dict, triples: list[tuple[str, str, str]],
) -> bool:
    """True for extracted facts that describe how the AGENT should
    behave (especially agent name/addressing rules) rather than facts
    about the user. These belong in identity.md, not user_profile.md.

    The prompt later reads user_profile under a `# USER PROFILE`
    header, and rules like 'Respond to the name X' get
    re-interpreted as 'the user is named X', causing the agent to
    address the user by its own name.

    Heuristics:
      - summary mentions agent-side patterns ("respond to the name",
        "your name is", "answer to <name>", "откликаешься на имя")
      - any triple's subject is the agent itself
        ("you", "agent", "assistant", "hrant" in lowercase) and the
        relation is identity-shaped (`is_named`, `name`, `responds_to`,
        `answers_to`)
    """
    summary = (raw.get("summary") or "").lower()
    patterns = (
        "respond to the name",
        "answer to the name",
        "your name is",
        "you answer to",
        "you are called",
        "reply as ",
        "откликаешься на имя",
    )
    if any(p in summary for p in patterns):
        return True
    agent_aliases = {
        "you", "agent", "assistant", "the agent", "the assistant",
        "hrant", "ассистент", "агент", "бот", "bot",
    }
    name_relations = {
        "is_named", "named", "name", "responds_to", "responds",
        "answers_to", "answer_to", "called",
    }
    for subj, rel, _ in triples:
        if subj.strip().lower() in agent_aliases and rel.strip().lower() in name_relations:
            return True
    return False


class MemoryFact:
    """A single fact extracted from conversation. Tagged with the
    `speaker_id` that produced it so per-speaker filtering is
    possible at retrieval time."""

    def __init__(
        self,
        summary: str,
        triples: list[tuple[str, str, str]],
        tags: list[str],
        category: str = "general",
        confidence: float = 0.8,
        ts: str | None = None,
        source_turn: str = "",
        speaker_id: str | None = None,
    ):
        self.summary = summary
        self.triples = triples
        self.tags = tags
        self.category = category
        self.confidence = confidence
        self.ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.source_turn = source_turn[:200]
        # Empty string when extracted before Phase 10 (no speaker
        # routing yet) or by a non-speaker-aware caller (legacy CLI).
        self.speaker_id = speaker_id or ""

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "triples": self.triples,
            "tags": self.tags,
            "category": self.category,
            "confidence": self.confidence,
            "ts": self.ts,
            "source_turn": self.source_turn,
            "speaker_id": self.speaker_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MemoryFact:
        return cls(
            summary=d.get("summary", ""),
            triples=[tuple(t) for t in d.get("triples", [])],
            tags=d.get("tags", []),
            category=d.get("category", "general"),
            confidence=d.get("confidence", 0.8),
            ts=d.get("ts"),
            source_turn=d.get("source_turn", ""),
            speaker_id=d.get("speaker_id", ""),
        )


class MemoryExtractor:
    """Extracts and stores conversation facts in the knowledge graph."""

    GRAPH_SOURCE = "_memory"  # Special source marker for conversation-derived facts

    def __init__(self, log_path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.log_path = log_path or (kb_dir / "memory_facts.jsonl")
        self._fact_count = 0
        # Count existing facts
        if self.log_path.exists():
            try:
                self._fact_count = sum(
                    1 for line in self.log_path.read_text(encoding="utf-8").strip().split("\n")
                    if line.strip()
                )
            except Exception:
                pass

    def extract_and_store(
        self,
        user_message: str,
        agent_answer: str,
        intent: str = "task",
        *,
        confidence: int = 100,
        contradictions: int = 0,
        speaker_id: str | None = None,
    ) -> list[MemoryFact]:
        """Extract facts from a conversation turn and store in graph.

        Called after every agent response. Skips pure chat/greetings
        (handled by intent filter). Returns list of extracted facts.

        `confidence` and `contradictions` are the verifier's output
        for THIS turn. When the answer is low-confidence (<60) or has
        any contradictions, we DROP the agent_answer side from the
        extraction prompt and only mine the user_message — otherwise
        the graph would absorb claims the verifier already flagged
        as wrong, polluting future retrieval.
        """
        # Skip trivial chat — no facts to extract
        if intent == "chat" and len(user_message.strip()) < 30:
            return []

        # Confidence gate: if the answer didn't hold up, don't mine
        # facts from it. User-stated facts still safe to extract.
        trust_answer = confidence >= 60 and contradictions == 0
        answer_for_extraction = agent_answer if trust_answer else ""

        try:
            user_prompt = (
                f"USER MESSAGE:\n{user_message[:1000]}\n\n"
                f"AGENT ANSWER:\n{answer_for_extraction[:1500]}"
            )
            data = router().call_json(
                TaskType.CLASSIFICATION,
                EXTRACT_FACTS_SYSTEM,
                user_prompt,
                max_tokens=800,
                temperature=0.1,
            )

            if not data.get("has_facts"):
                return []

            facts: list[MemoryFact] = []
            for raw in data.get("facts", []):
                raw_triples = raw.get("triples", [])
                triples = []
                for t in raw_triples:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        triples.append((str(t[0]), str(t[1]), str(t[2])))

                if not triples:
                    continue

                # Drop facts that describe AGENT behavior (e.g. how the
                # agent should address itself) — those belong in
                # identity.md, not user_profile.md. Without this filter
                # the prompt later reads "the user is named Hrant" and
                # the agent starts calling the user by its own name. The
                # bug was a real production incident with a fact
                # `Respond to the name Hrant.` landing under `## Правила
                # взаимодействия` of the Telegram speaker profile.
                if _looks_like_agent_self_rule(raw, triples):
                    continue

                fact = MemoryFact(
                    summary=raw.get("summary", ""),
                    triples=triples,
                    tags=raw.get("tags", []),
                    category=raw.get("category", "general"),
                    confidence=float(raw.get("confidence", 0.8)),
                    source_turn=user_message[:200],
                    speaker_id=speaker_id,
                )
                facts.append(fact)

                # Corrections — close the superseded triples BEFORE adding
                # the new ones. Without this, a "not Tigran, my brother is
                # Arman" turn leaves both edges open and recall returns
                # both names. Multi-valued relations like `has_brother`
                # aren't in KG.SINGLE_VALUED_RELATIONS, so the graph
                # itself can't auto-invalidate — the extractor's hint
                # is the only signal.
                for old in raw.get("replaces", []):
                    if (
                        isinstance(old, (list, tuple))
                        and len(old) >= 3
                        and all(str(x).strip() for x in old[:3])
                    ):
                        try:
                            GRAPH.invalidate(str(old[0]), str(old[1]), str(old[2]))
                        except Exception:
                            # Invalidation is best-effort — we'd rather
                            # have a stale-but-additive graph than a
                            # crashed extraction.
                            pass

                # Store in knowledge graph
                graph_triples = list(triples)
                # Also add tag edges for better retrieval
                for tag in fact.tags:
                    for subj, _, _ in triples[:1]:  # Link first entity to tags
                        graph_triples.append((subj.lower(), "tagged", tag.lower()))

                GRAPH.add_relations(
                    graph_triples,
                    source_note=self.GRAPH_SOURCE,
                    weight=fact.confidence,
                )

                # Append to log
                self._append_log(fact)

            return facts

        except LLMError:
            return []

    def _append_log(self, fact: MemoryFact) -> None:
        """Append a fact to the persistent log."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
            self._fact_count += 1
        except Exception:
            pass

    def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Recall facts relevant to a query by searching the graph.

        Returns fact summaries from memory edges that match the query.
        This is the agent's "remember" function — it finds conversation
        facts stored in the graph.
        """
        # Search graph for memory-sourced edges
        query_terms = GRAPH._extract_query_entities(query)
        if not query_terms:
            return []

        relevant_facts: list[dict] = []
        seen_summaries: set[str] = set()

        for term in query_terms:
            term_n = GRAPH._normalize(term)
            # Direct entity match
            if term_n in GRAPH._edges:
                for edge in GRAPH._edges[term_n]:
                    if edge.get("note") == self.GRAPH_SOURCE:
                        # Found a memory edge — look up the full fact
                        fact_info = {
                            "entity": term_n,
                            "relation": edge["relation"],
                            "target": edge["target"],
                            "weight": edge.get("weight", 1.0),
                        }
                        key = f"{term_n}|{edge['relation']}|{edge['target']}"
                        if key not in seen_summaries:
                            seen_summaries.add(key)
                            relevant_facts.append(fact_info)

            # Also check if term appears as a target in memory edges.
            # Old impl scanned the entire graph (`for entity, edges in
            # GRAPH._edges.items()` × inner loop) which was O(N×M) per
            # query term — a real cost as the graph grows. Now we hit
            # the reverse target index in O(1) per term.
            for entity, edge in GRAPH.find_facts_by_target(term_n):
                if edge.get("note") != self.GRAPH_SOURCE:
                    continue
                key = f"{entity}|{edge['relation']}|{edge['target']}"
                if key not in seen_summaries:
                    seen_summaries.add(key)
                    relevant_facts.append({
                        "entity": entity,
                        "relation": edge["relation"],
                        "target": edge["target"],
                        "weight": edge.get("weight", 1.0),
                    })

        # Sort by weight and return top results
        relevant_facts.sort(key=lambda f: -f["weight"])
        return relevant_facts[:limit]

    def recall_block(self, query: str, max_facts: int = 8) -> str:
        """Build a memory context block for injection into prompts.

        Returns a formatted string of recalled facts, or "" if nothing found.
        """
        facts = self.recall(query, limit=max_facts)
        if not facts:
            return ""

        lines = [
            "# REMEMBERED FACTS (from previous conversations)",
            "These facts were learned from earlier conversations with the user.",
            "",
        ]
        for f in facts:
            rel = f["relation"].replace("_", " ")
            lines.append(f"- {f['entity']} {rel} {f['target']}")

        return "\n".join(lines)

    def recent_facts(self, limit: int = 30) -> list[dict]:
        """Return recent facts from the log."""
        if not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
            facts = []
            for line in lines[-limit:]:
                if line.strip():
                    facts.append(json.loads(line))
            facts.reverse()
            return facts
        except Exception:
            return []

    def stats(self) -> dict:
        """Memory statistics."""
        # Count memory edges in graph
        memory_edges = 0
        memory_entities: set[str] = set()
        for entity, edges in GRAPH._edges.items():
            for edge in edges:
                if edge.get("note") == self.GRAPH_SOURCE:
                    memory_edges += 1
                    memory_entities.add(entity)
                    memory_entities.add(edge["target"])

        return {
            "total_facts_logged": self._fact_count,
            "memory_edges_in_graph": memory_edges,
            "memory_entities": len(memory_entities),
            "graph_total_entities": GRAPH.entity_count(),
            "graph_total_edges": GRAPH.edge_count(),
        }


MEMORY = MemoryExtractor()
