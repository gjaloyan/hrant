# Skills — what they are and how to add them

A **skill** is a markdown plugin Hrant loads at startup. Each skill
declares triggers — keywords the agent watches for in your messages
— and a body of instructions the LLM follows when those triggers
match.

## Anatomy

```
<skills_dir>/<name>/
  SKILL.md      ← required: frontmatter + instructions
  handler.py    ← optional: Python module that registers tools
```

`SKILL.md`:

```markdown
---
name: pdf_summary
description: Read a local PDF and produce a structured summary.
triggers: [pdf, summarize, документ, конспект]
when_to_use: |
  User asks to summarize, extract, or analyze a local PDF file.
---

# PDF Summary

Free-form Markdown body. When the agent matches a trigger, this
whole block is injected into the system prompt for that turn.
```

If `handler.py` exists, it must expose `register(registry)` and is
free to add tools to the agent's `ToolRegistry`. Those tools become
available in the normal tool-use loop. Without a handler, a skill
is just a system-prompt block.

## Two tiers

Skills come from two places:

| tier      | location                            | survives `hrant update` |
|-----------|-------------------------------------|-------------------------|
| `builtin` | `backend/skills/` in the engine repo| no (refreshed)         |
| `user`    | `~/.hrant/data/skills/`             | yes                    |

A `user` skill with the same name as a `builtin` **overrides** it
— that's how you tweak a shipped skill without forking the engine.

## Soft kill-switch

`~/.hrant/data/skills_disabled.json` is the list of skill names the
WebUI has flipped off. Disabled skills still appear in the skills
list (so you can toggle them back on) but their triggers don't
match and their `handler.py` doesn't register tools.

## WebUI

Settings → **Skills**:

- Left column: every skill on disk with its `source` badge, trigger
  count, and on/off state.
- Right pane: SKILL.md editor (full-text). Save creates a user
  override if you're editing a built-in. Toggle enables/disables.
  Delete removes a user skill (built-ins protected — they'd just
  come back on next `hrant update`).
- **+ New skill**: writes a starter SKILL.md template into the user
  dir under the name you pick.
- **Install from URL / path**: pulls a skill from `git`, `zip`, or
  a local directory. Owner-only — installing runs `handler.py`
  inside the agent process at next load, so only do it from sources
  you trust. The UI shows that warning before confirming.

## API

```
GET    /api/skills                    — list everything
GET    /api/skills/{name}             — one skill with raw SKILL.md
PUT    /api/skills/{name}             — upsert user skill body
DELETE /api/skills/{name}             — delete user skill
POST   /api/skills/{name}/enabled     body {enabled: bool}
POST   /api/skills/reload             — re-scan from disk
POST   /api/skills/install            body {source_type, source, name?, subdir?}
```

Owner-gate enforced on every mutation via the Phase 11 ContextVar.

## Install sources

- `git`: `source` is a clone URL. Optional `subdir` if the skill
  lives in a sub-path of the repo. Uses `git clone --depth=1`.
- `zip`: `source` is an HTTP(S) URL to a `.zip` file. Auto-flattens
  GitHub-style "single top-level dir" archives. Path-traversal
  attacks are refused (any entry starting with `/` or containing
  `..` aborts the install).
- `local`: `source` is an absolute path on the server's filesystem.
  Copies the directory into `~/.hrant/data/skills/<name>/`.

## Heartbeat (related)

Phase 12 also added a heartbeat signal for the autonomic scheduler,
available at `/api/health` → `components.autonomic`:

- `ok` — last tick within `2× tick_interval_seconds`
- `degraded` — within `10× tick_interval_seconds`
- `down` — older, or no ticks ever
- `not_configured` — kill-switch disabled

`tick_interval_seconds` is read from `~/.hrant/data/autonomic_settings.json`
(editable via Settings → Autonomic) so a slower hand-tuned loop
doesn't trigger false `degraded` alerts.
