---
name: debug_telegram_callbacks
description: Diagnose Telegram inline-button failures (taps that don't work, spinner that never stops, "buttons don't work"). Reads the journal first, checks ApplicationBuilder config, traces the callback path. Born from the May 19, 2026 incident where concurrent_updates(True) was missing and queries expired in the update queue.
triggers: [кнопки не работают, кнопка не работает, button does not work, buttons don't work, buttons dont work, callback не работает, спиннер не уходит, approve не срабатывает, fix button]
tags: [telegram, button, callback, callback_query, answerCallbackQuery, PTB, concurrent_updates, install_approve, prop_apply, kbd, inline-keyboard, spinner]
when_to_use: |
  Load this when the user reports any of:
    - "buttons don't work" / "кнопки не работают"
    - tap on Approve / Reject / Show stops on spinner
    - install proposals don't apply after tap
    - pair / prop / sched / skill / install inline-keyboard taps
      look dead.

  DON'T load this for unrelated Telegram issues (sendMessage failures,
  voice transcription, file uploads) — those have different root
  causes. This skill is specifically for inline-keyboard callbacks.
required_tools: []
---

# Debug Telegram inline-button failures

A button "not working" almost always means one of three things:
1. The bot's `answerCallbackQuery` is returning HTTP 400 (the spinner
   on the user's side never stops because Telegram never got an
   "I handled it" response).
2. The bot's `editMessageText` is returning 400 (the original
   message keeps its old buttons because the post-action edit
   never lands).
3. The callback handler is never reached at all (dispatcher
   routing miss, wrong prefix, callback_data malformed).

Walk the diagnosis in this order — don't skip steps.

## Phase 1 — read the journal FIRST (mandatory)

Before reading code, before checking handler logic, run:

```
terminal_exec("journalctl --user -u hrant --since '30 min ago' --no-pager | grep -iE 'callback|answerCallback|editMessageText|400 Bad Request' | tail -60")
```

Look for the failing endpoint:

- `POST .../answerCallbackQuery "HTTP/1.1 400 Bad Request"` → query
  expired or already-answered. Jump to Phase 2.
- `POST .../editMessageText "HTTP/1.1 400 Bad Request"` → "message
  is not modified" or "message to edit not found". Jump to Phase 4.
- No 400s at all + user still says buttons dead → handler isn't
  running. Jump to Phase 5.

If the journal grep returned nothing, widen the window
(`--since '2h ago'`) or drop the grep entirely
(`journalctl --user -u hrant -n 300 --no-pager | tail -200`).

## Phase 2 — `answerCallbackQuery → 400`: check concurrent_updates

This was the May 19 root cause. Telegram callback queries have a
~15s server-side expiration. If PTB processes updates **sequentially**
(the default), a callback tap that arrives while a chat handler is
busy queues up, expires, and Telegram refuses every
`answerCallbackQuery` afterwards.

```
terminal_exec("grep -n 'ApplicationBuilder' /home/hrant/hrant/backend/channels.py")
```

Look for the builder chain. If you see `.concurrent_updates(True)`
between `.token(...)` and `.build()` — this isn't your bug, move
to Phase 3. If it's MISSING, that's the fix:

```python
app = (
    ApplicationBuilder()
    .token(self.token)
    .concurrent_updates(True)   # <-- add this
    .build()
)
```

Apply via `run_python` (preferred for one-line edits) or
`terminal_exec` with `sed -i`. Then restart the service:

```
terminal_exec("systemctl --user restart hrant")
```

(or wait for `hrant update` if you're applying via the normal
deploy path).

## Phase 3 — `answerCallbackQuery → 400` AND concurrent_updates is on

The other reasons Telegram refuses are:

1. **Already answered.** The handler called `query.answer()` twice
   on the same query. PTB only accepts one answer per query.
   - Grep `await query.answer(` in `backend/channels.py` and trace
     both call sites in `handle_callback_query`. The `pre_answered`
     flag should gate the second call.
2. **Malformed callback_query_id.** Rare — would be a bug in the
   wrapper. Check the python-telegram-bot version is current
   (`~/.local/share/pipx/venvs/agi-agent/bin/python -c "import
   telegram; print(telegram.__version__)"`); ≥ 22.0 is the
   supported floor.

## Phase 4 — `editMessageText → 400`: identical text or missing target

Telegram refuses an edit if:
- The new content is byte-identical to the current message (returns
  "message is not modified"). Fix: only edit when content changed,
  or strip leading/trailing whitespace before the equality check.
- The original message is gone (deleted by the user, or older than
  48h). Fix: catch the BadRequest and fall back to sending a new
  message.

```
terminal_exec("journalctl --user -u hrant --since '30 min ago' --no-pager | grep -i 'editMessageText\\|not modified\\|message to edit'")
```

If the journal shows the exact reason ("message is not modified" /
"message to edit not found"), patch the call site in
`handle_callback_query` accordingly.

## Phase 5 — handler never runs

If the journal shows ZERO callback log lines after a tap:

1. The `CallbackQueryHandler` isn't registered. Grep
   `add_handler(CallbackQueryHandler` in `backend/channels.py` —
   confirm one is wired in around the application setup.
2. The callback_data prefix has no handler. The dispatcher
   (`backend/tg_interactive.py`) routes by prefix
   (`install:`, `prop:`, `sched:`, `skill:`, `pair:`). If a button
   was created with a prefix that no handler subscribes to, the
   dispatch drops it. Grep `register_callback_handler(` in the
   backend to see what prefixes are registered.
3. The bot polling died. `journalctl --user -u hrant -n 50 | grep
   -i 'polling\|getUpdates'` — if polling stopped, the service
   needs a restart.

## Phase 6 — apply the fix

For small one-line / one-flag patches (the May 19 case), DO NOT
wrap them in `propose_self_modification` — that adds friction
without adding safety. Use:

```
run_python("""
from pathlib import Path
p = Path('/home/hrant/hrant/backend/channels.py')
src = p.read_text()
new = src.replace(
    '.token(self.token).build()',
    '.token(self.token).concurrent_updates(True).build()',
)
assert new != src, 'pattern not found — check current code'
p.write_text(new)
print('patched OK; diff line count:', new.count(chr(10)) - src.count(chr(10)))
""")
```

Then restart:

```
terminal_exec("systemctl --user restart hrant")
```

Then verify by sending the user a test approve request and asking
them to tap. Or directly check the next callback tap in the journal
shows `answerCallbackQuery → 200` (not 400).

## Phase 7 — final verification

After the fix is deployed, the journal of a fresh button tap should
show:

```
sendChatAction  HTTP/1.1 200 OK
answerCallbackQuery  HTTP/1.1 200 OK       <-- was 400 before
editMessageText  HTTP/1.1 200 OK            <-- was 400 before
```

If `answerCallbackQuery` returned 200 — buttons are confirmed
working. Tell the user "fixed and verified — last tap returned
HTTP 200 on answerCallbackQuery".

If you applied the fix but the journal still shows 400s on the
next tap, the diagnosis was wrong — go back to Phase 1 and look
for a different pattern in the journal.

## What NOT to do

- DON'T spend 30+ minutes reading `handle_callback_query` /
  `dispatch_callback` source before checking the journal. The
  journal tells you which Telegram API call is failing within
  one tool call.
- DON'T call `propose_self_modification` for a one-flag fix.
  That's PSM ceremony for an architectural change; this is a
  config flag.
- DON'T refuse with "I can't write to the filesystem" —
  `run_python` and `terminal_exec` are both write-capable.
  See the TSP "Pick the right tool" → Self-mod rule.
- DON'T claim the fix is applied without verifying the next tap
  shows HTTP 200 in the journal. The May 19 incident left a
  full hour of "I checked the handler, looks OK" — none of that
  fixed the bug, because the bug wasn't in the handler.
