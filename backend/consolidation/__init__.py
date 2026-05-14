"""Daily memory consolidation — Phase 16.

What this package does:
  - At ~3am local time (adaptive: fires when the agent's been idle
    for 15 min AND it's been >=24h since the last run), summarises
    the past 24h of activity, extracts durable facts, updates the
    user profile, and writes a digest file.
  - Digests live at `~/.hrant/data/knowledge/memory_digests/<YYYY-MM-DD>.json`
  - Phase 16A (this commit): scheduler + pipeline + digest + surfaces.
    NO pruning, NO knowledge graph, NO rollback — those are 16B/16C.

Why a new module rather than extending the autonomic lever:
  The existing `FIRE_MEMORY_CONSOLIDATION` lever in `autonomic/levers/`
  is session-scoped, opportunistic, fires whenever the autonomic
  scheduler picks it. We want a daily, deterministic, full-day-scope
  consolidation with persistent digest files for the WebUI to render.
  Both can coexist — the lever produces facts during the day, daily
  consolidation produces the narrative + cross-links + open threads
  digest at night. Phase 16B may unify them.
"""
