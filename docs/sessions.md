# Sessions & speaker_id

Hrant treats every distinct user as an independent **speaker**. Each
speaker gets their own session, conversation context, and user
profile — nothing leaks between them.

## The model

```
speaker_id = "<channel>:<user_id>"

  webui:default     ← the local user via the WebUI (one per install)
  telegram:111      ← Gor on Telegram (his TG user id)
  telegram:222      ← his wife on Telegram (her TG user id)
  whatsapp:444      ← future channels follow the same shape
```

Three things are partitioned per speaker:

1. **Sessions** — `knowledge/sessions.json` keeps a
   `current_by_speaker: {speaker_id: session_id}` map. The agent's
   "current session" is always speaker-scoped.
2. **Conversation memory** (recent turns used as in-prompt context)
   — turns carry `speaker_id`, and the context block for speaker X
   only contains turns from X.
3. **User profile** (`knowledge/identity/`):
   - `webui:default` → `user.md` (legacy path, back-compat)
   - everyone else → `profiles/<sanitized>.md` (e.g.
     `profiles/telegram_111.md`)

Three things stay **shared**:

- `knowledge/<category>/*.md` (notes about the world)
- `knowledge/core_memory.md` (pinned context the agent always sees)
- `knowledge/identity/soul.md` + `identity.md` (the agent's own personality)

The split is intentional: knowledge about the world is one body of
information; what the agent remembers about *you specifically* lives
on your own conversation thread.

## In practice

**Telegram:** When a message arrives, the bot derives
`speaker_id = f"telegram:{update.message.from_user.id}"` and
hands that to `agent.run(...)`. The first time a user texts the
bot, a fresh profile + session are auto-created. From then on, their
context is theirs alone.

**WebUI:** Sends `speaker_id = "webui:default"` (the local user)
on every `/api/chat` POST. A future multi-WebUI-user feature could
override this with `webui:alice`, `webui:bob`, etc.

## Endpoints

```
GET  /api/sessions?speaker_id=…              — list, filtered by speaker
GET  /api/sessions/speakers                  — every speaker with stats
GET  /api/sessions/current?speaker_id=…      — current session for a speaker
POST /api/sessions/new?speaker_id=…          — new session for a speaker

GET  /api/conversation?speaker_id=…          — recent turns for a speaker
GET  /api/conversation?channel=telegram      — coarser filter, legacy

GET  /api/identity?speaker_id=…              — soul/identity + profile for that speaker
GET  /api/identity/profiles                  — list every per-speaker profile file
PUT  /api/identity body:{file:"user", content, speaker_id} — edit a speaker's profile
```

## WebUI

- **Sessions** panel: dropdown at the top to filter by speaker. Every
  session row shows the owning `speaker_id` in violet.
- **Settings → User Profile**: speaker selector at the top. Switching
  loads that speaker's `profiles/<sanitized>.md` for editing. The
  default (`webui:default`) still uses the legacy `user.md` file for
  back-compat with existing data.

## Migration

Phase 10 ships with the partition model in place. Existing data
that pre-dates speaker_id was wiped during the upgrade (conversation,
sessions, memory_facts) per the user's choice; identity stays where
it was. New turns carry `speaker_id` from the moment the upgrade
lands.
