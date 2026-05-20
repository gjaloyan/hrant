# `knowledge/self/` — agent's self-knowledge

This directory is the agent's own internal documentation:

- `architecture.md` — high-level overview of how the engine works
- `cheatsheet.md`   — filesystem map + common CLI commands
- `modules/`        — per-module notes written by `FIRE_SELF_STUDY` over time

The starter `architecture.md` and `cheatsheet.md` ship in
`knowledge_templates/self/` and get copied here on `hrant init`.
`FIRE_SELF_STUDY` (an autonomic lever) reads `backend/**/*.py` and
writes one note per module under `modules/` — by default this kicks
in after the agent has been running for a while.

The agent reads these when answering questions about itself (`how
are you implemented?`, `what does X do?`, `where is Y in the code?`).
