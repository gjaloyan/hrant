"""Knowledge graph — Phase 16C.

Glues together everything the agent knows about a single user:
  - Facts from memory_facts.jsonl
  - Topics (tags) referenced by facts and skills
  - Skills (with their triggers as topic edges)
  - Projects/goals

Storage: `~/.hrant/data/knowledge/graph.json`
   Single JSON file with `nodes` + `edges` arrays. Small (~10–100KB
   at the personal-agent scale), fast to load, easy to inspect.

Phase 16C scope (this commit):
  - Data model + persistence
  - Builder: derive initial graph from existing sources
  - Query API: neighborhood, search, top topics
  - Integration: consolidation pipeline updates the graph each run
  - REST + CLI + WebUI explorer with a simple SVG node-link diagram
"""
