# Roles, relationships, scheduled messages

Phase 11 makes Hrant safe to share. The model: same agent, same
shared world-knowledge, but every speaker (you on the WebUI, each
Telegram user, future channels) has a **role** that gates dangerous
actions, and the owner can schedule messages **across** speakers.

## Roles

Three tiers:

| role    | chat | self-mod | code exec | config | cross-speaker scheduling |
|---------|------|----------|-----------|--------|--------------------------|
| owner   | ✓    | ✓        | ✓         | ✓      | to anyone                |
| trusted | ✓    | ✗        | ✗         | ✗      | **only to the owner**    |
| guest   | ✓    | ✗        | ✗         | ✗      | ✗                        |

The WebUI default speaker (`webui:default`) is **always** owner —
that's the local-machine user, who can't be locked out of their
own box even by a bad edit of `roles.json`.

Storage: `~/.hrant/data/knowledge/identity/roles.json`

```json
{
  "owner_speaker_ids": ["webui:default", "telegram:111111111"],
  "speakers": {
    "telegram:222222222": {"role": "trusted", "label": "Wife"},
    "telegram:333333333": {"role": "guest",   "label": "Friend"}
  }
}
```

## How the gate works

Three lines of defence, in order:

1. **System prompt**: every request to the agent ships with a
   `# SPEAKER PERMISSIONS` block at the END of the prompt
   (highest attention weight). For non-owners it explicitly says
   "refuse self-modification, code execution, config changes". The
   model is told to politely explain it can only do those at the
   owner's request.
2. **Tool gate**: dangerous tools (`run_python`, `schedule_message`)
   check the speaker's role via a `ContextVar` set at the top of
   `Agent.run()`. Non-owner callers get an immediate refusal
   stamped into the tool output. No subprocess, no side-effect.
3. **API gate**: `self_modifier.apply()` reads the same ContextVar
   and refuses if the current speaker isn't owner. Last line of
   defence in case the model is talked into trying anyway.

The WebUI's "Settings → Roles & Contacts" tab is the user-facing
control: list every speaker the system has seen, set their role,
add labels. Promoting a Telegram user_id to **owner** lets that
person drive self-modification from a phone the same way the
local user can from the browser.

## Cross-speaker messaging

The owner asks: "remind my wife to call me at 10:00 tomorrow".
The agent uses the **`schedule_message`** tool:

```
schedule_message(target="wife", text="Gor asks you to call him", due_at="2026-05-14T07:00:00Z")
```

What happens under the hood:

1. **Resolve the target** — `relationships.json` maps the alias to
   a speaker_id. `"wife"` → `"telegram:222"`. Already-qualified
   speaker_ids pass through unchanged so the LLM can use either form.
2. **Permission check** — owner can target anyone. Trusted can
   target **only the owner**. Guests can't call the tool at all.
3. **Persist** — append a row to
   `~/.hrant/data/knowledge/scheduled_messages.jsonl` with
   `status="pending"`.
4. **Wait** — the `FIRE_SCHEDULED_MESSAGES` autonomic lever runs
   every minute, scans for `due_at <= now`, delivers each one via
   the appropriate channel.
5. **Deliver** — for Telegram, look up the recipient's `chat_id`
   in `~/.hrant/data/telegram_chat_ids.json` (auto-populated when
   they message the bot) and call `bot.send_message(chat_id, text)`.

### "But I've never messaged the bot" — the chat_id problem

We can only deliver to a Telegram user we already have a `chat_id`
for. The bot stores `chat_id` automatically on every incoming
message — so once your wife has texted the bot **once**, she's
addressable. If she's never pinged it, the dispatcher marks the
message `failed` with a clear "no chat_id" reason; the owner can
ask her to ping the bot once and re-schedule.

## Endpoints

```
GET    /api/roles                          — list speakers + roles + seen
PUT    /api/roles/{speaker_id}             body {role, label?}
GET    /api/relationships                  — alias → speaker_id map
PUT    /api/relationships                  body {relationships: {alias: speaker_id, ...}}
GET    /api/contacts/telegram              — auto-captured chat_id store
GET    /api/scheduled-messages?status=…    — ledger, optionally filtered
DELETE /api/scheduled-messages/{id}        — cancel a pending row
```

The agent's `schedule_message` tool is the LLM-callable surface;
the WebUI uses the REST endpoints above directly.

## WebUI

Settings → **Roles & Contacts** has four sections:

1. **Speakers** — every speaker_id we've seen, with a role badge,
   editable label, and three role buttons (owner / trusted / guest).
2. **Relationships** — alias → speaker_id editor. Add/remove rows
   so the agent can resolve "wife" / "mom" / "kid" / ... .
3. **Telegram contacts** — read-only view of `chat_id` capture —
   useful for verifying you can deliver to that user.
4. **Scheduled messages** — full ledger (newest first) with status
   badges, cancel button for pending rows, error inline for failed
   rows.

## Examples

**Owner self-mod from Telegram:**

You (owner via TG): "Modify backend/X to do Y."
Agent: `[diff]. Reply 'apply <id>' or 'reject'.`
You: "apply abc"
→ Patch saved to `data_dir/self_mods/0003-…patch`, applied to engine.

**Trusted user trying to self-mod:**

Wife (trusted): "Modify your code to save my recipes."
Agent: "I can only do code modifications at my owner's request.
Ask Gor."

**Guest scheduling a cross-channel message:**

Friend (guest): "Tell Gor I'll be late."
Agent: "I can't schedule messages from your account. Only the owner
and trusted users can do that."

**Owner schedules a reminder:**

You (owner): "Remind my wife to call me at 10am tomorrow."
Agent: "Scheduled for 2026-05-14T07:00:00Z. Visible in
Settings → Roles & Contacts → Scheduled messages."
[At 10am] Bot DMs wife: "Gor asks you to call him."
