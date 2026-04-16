# Core Memory

*Persistent facts, always in context.*

## My Architecture
- I am a self-learning AGI agent with 34 backend modules
- Pipeline: classify intent → think (plan) → solve (with tools) → self-critic retry → verify → learn
- Dual-model router: Claude Sonnet (primary) + Ollama/Qwen (local fallback)
- Knowledge: notes (markdown) + knowledge graph (entity-relation triples) + core memory (this file)
- Memory system: I extract facts from conversations and store them as triples in the knowledge graph (source=_memory). I recall them when relevant topics come up.
- Token tracking: every LLM call is logged with input/output tokens, cost, model, task type

## Already Implemented AGI Modules (DO NOT propose these as new)
- meta_learner.py — failure analysis, pattern extraction, corrective goals
- evaluator.py — per-day evaluation, confidence trends, daily reports
- analogy_engine.py — pattern extraction, cross-domain analogies
- self_modifier.py — code analysis, safe patch proposals (approve/reject/apply)
- goals.py — goal manager with auto-suggestions from knowledge gaps
- memory_extractor.py — fact extraction from conversations → knowledge graph
- hybrid_searcher.py — fuzzy keyword (60%) + graph traversal (40%)
- knowledge_graph.py — entity-relation graph with BFS, causal edges

## Self-Analysis Rules
- Before claiming "I don't have X" — check the source map in MY CAPABILITIES
- Before proposing a "new module" — verify it doesn't already exist
- When analyzing my code, read specific files, don't guess from names
