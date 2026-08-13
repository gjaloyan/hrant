"""MCP server exposing the agent's memory + KB to any MCP-compatible client.

Wraps existing in-process modules (KM, GRAPH, MEMORY, GOALS, CORE) so the
agent's data is reachable from Claude Code, Codex, Gemini, Cursor, etc.
without going through HTTP.

Run:
    .venv/Scripts/python.exe -m backend.mcp_server

Client install (Claude Code example):
    claude mcp add agi -- python -m backend.mcp_server

Tools exposed (read-mostly, with a few writes):
    knowledge_search          semantic + text + graph hybrid search
    knowledge_get             fetch a specific note by topic
    knowledge_learn           kick off learn_topic for a new topic
    core_memory_get           read core_memory.md
    core_memory_add           append a fact
    graph_query_entity        triples for an entity, optional as_of
    graph_timeline            full history of an entity
    goals_list                list active goals
    memory_recall             recall facts from memory_facts.jsonl
"""
from __future__ import annotations

import asyncio
import json
import logging

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions

log = logging.getLogger(__name__)


server: Server = Server("agi-memory")


# ---------- tool list ----------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="knowledge_search",
            description=(
                "Hybrid search across the agent's knowledge base. Combines "
                "text matching, graph traversal, and (when available) "
                "vector similarity. Returns top-K notes ranked by relevance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="knowledge_get",
            description="Fetch the full body of a knowledge note by exact topic name.",
            inputSchema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        ),
        types.Tool(
            name="knowledge_learn",
            description=(
                "Trigger active learning on a new topic. The agent researches "
                "and writes a curated note. Returns the note path. Slow — uses "
                "the configured LLM provider."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "depth": {"type": "string", "enum": ["quick", "deep"], "default": "quick"},
                    "category": {"type": "string", "default": "profession"},
                },
                "required": ["topic"],
            },
        ),
        types.Tool(
            name="core_memory_get",
            description="Read the agent's core memory (compact always-loaded facts).",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="core_memory_add",
            description="Append a fact to the core memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "source": {"type": "string", "default": "mcp"},
                },
                "required": ["fact"],
            },
        ),
        types.Tool(
            name="graph_query_entity",
            description=(
                "Return knowledge-graph edges for an entity. If `as_of` is "
                "set (ISO date), returns only edges valid at that instant. "
                "Otherwise returns currently-open edges."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "as_of": {"type": "string", "description": "Optional ISO date"},
                    "direction": {"type": "string", "enum": ["out", "in", "both"], "default": "both"},
                },
                "required": ["entity"],
            },
        ),
        types.Tool(
            name="graph_timeline",
            description="All edges (open + closed) for an entity, sorted by valid_from.",
            inputSchema={
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"],
            },
        ),
        types.Tool(
            name="goals_list",
            description="List the agent's active goals with stats.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="memory_recall",
            description="Recall conversation-extracted facts (memory_facts.jsonl).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "speaker_id": {
                        "type": "string",
                        "default": "webui:default",
                        "description": "Exact private-memory owner scope.",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


# ---------- tool dispatch ----------

def _ok(payload) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _err(message: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps({"error": message}, ensure_ascii=False))]


@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    args = arguments or {}
    try:
        if name == "knowledge_search":
            from .hybrid_searcher import HYBRID
            limit = int(args.get("limit", 5))
            hits = HYBRID.search(args["query"], k=limit)
            return _ok({
                "query": args["query"],
                "results": [
                    {"topic": h.topic, "score": getattr(h, "score", None), "snippet": getattr(h, "preview", "")}
                    for h in hits
                ],
            })

        if name == "knowledge_get":
            from .knowledge_manager import KM
            note = KM.get_note(args["topic"])
            if not note:
                return _err(f"note not found: {args['topic']}")
            return _ok({
                "topic": args["topic"],
                "category": note.frontmatter.category,
                "body": note.body,
                "keywords": note.frontmatter.keywords,
                "updated": note.frontmatter.updated,
            })

        if name == "knowledge_learn":
            from .note_creator import learn_topic
            note = learn_topic(
                args["topic"],
                depth=args.get("depth", "quick"),
                category=args.get("category", "profession"),
                project=None,
            )
            return _ok({"topic": note.frontmatter.topic, "path": str(note.path)})

        if name == "core_memory_get":
            from .core_memory import CORE
            return _ok({
                "content": CORE.read(),
                "tokens": CORE.tokens(),
                "max": CORE.max_tokens,
            })

        if name == "core_memory_add":
            from .core_memory import CORE
            msg = CORE.add_fact(args["fact"], args.get("source", "mcp"))
            return _ok({"message": msg})

        if name == "graph_query_entity":
            from .knowledge_graph import GRAPH
            edges = GRAPH.query_entity(
                args["entity"],
                as_of=args.get("as_of"),
                direction=args.get("direction", "both"),
            )
            return _ok({"entity": args["entity"], "edges": edges})

        if name == "graph_timeline":
            from .knowledge_graph import GRAPH
            return _ok({"entity": args["entity"], "timeline": GRAPH.timeline(args["entity"])})

        if name == "goals_list":
            from .goals import GOALS
            return _ok({
                "goals": [g.to_dict() for g in GOALS.all_goals()],
                "stats": GOALS.stats(),
            })

        if name == "memory_recall":
            from .memory_extractor import MEMORY
            facts = MEMORY.recall(
                args["query"],
                limit=int(args.get("limit", 10)),
                speaker_id=str(args.get("speaker_id") or "webui:default"),
            )
            return _ok({"query": args["query"], "facts": facts})

        return _err(f"unknown tool: {name}")
    except Exception as e:
        log.exception("MCP tool %s failed", name)
        return _err(f"{type(e).__name__}: {e}")


async def amain() -> None:
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="agi-memory",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
