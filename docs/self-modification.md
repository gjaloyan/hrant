# Self-Modification

Hrant is a commercial product. Each user gets their own copy of the
engine and can modify it on their own machine. **Self-modifications
are strictly local** — they never touch the official engine
distribution and there is no way to publish them back from the
agent. The same machinery acts as the safety net: if a self-mod
breaks something, **one click resets the engine to the exact official
version**.

Treat self-modification as a power-tool with a built-in undo, not as
a way to fork. The expected pattern is:

1. User asks for a tweak.
2. Agent applies it locally with full audit trail.
3. If things look broken later → **Revert all → official**.
4. Done — engine matches what the user originally installed.

## The flow

1. **User asks for a change.** Example: «save memory in a local SQLite
   database, I don't want to use the RAG system».

2. **Agent proposes a patch.** It analyses the relevant module(s),
   produces a diff, and surfaces it to the user with a risk
   assessment. Typical risks:
   - Breaks a contract some other module relied on
   - Requires an external dependency (SQLite is bundled with Python;
     redis isn't)
   - Diverges from upstream — future `hrant update` may conflict if
     the upstream version touches the same lines

3. **User reviews + approves.** Without explicit approval the
   modification doesn't happen.

4. **Patch recorded + applied.** The agent:
   - Writes a unified diff to `~/.hrant/data/self_mods/NNNN-<slug>.patch`
   - Appends a `PatchEntry` to `~/.hrant/data/self_mods/applied.json`
   - `git apply`s the diff so the running engine reflects the change

5. **Settings → Self-Modifications.** The WebUI shows every patch
   with status, file, created timestamp, and per-patch revert.

## What persists where

```
~/.hrant/data/self_mods/
  0001-add-sqlite-memory.patch       ← unified diff
  0002-skip-rag-on-read.patch
  applied.json                       ← manifest (order + status)
```

Storage is in `data_dir` (not the engine repo) so the patches survive
`hrant update`, `hrant rollback`, and any `git reset` that touches
the engine tree.

## Three statuses

- **`applied`** — currently active. The engine reflects the patch.
- **`needs_review`** — the last `hrant update` brought new engine code
  that the patch can't cleanly apply onto, even with 3-way merge.
  The engine is at the official version for the affected lines; the
  user fixes the patch manually or reverts it.
- **`reverted`** — user clicked Revert. Kept in manifest for audit
  trail until the next manifest write prunes it.

## `hrant update` integration

When you run `hrant update`:

1. Working tree dirty check.
2. `git pull --ff-only origin master`.
3. `pip install -e .`, `npm run build`.
4. **`self_mods.reapply_all()`** — walks `applied.json` in order:
   - `git apply --check` (dry-run)
   - If clean → apply.
   - If conflict → `git apply --3way` (attempts a merge).
   - If both fail → mark `needs_review`, leave engine alone.
5. UpdateResult carries `self_mods_reapplied` + `self_mods_needs_review`
   lists. The CLI prints "self-mods needing review: N" so you know
   to check the Settings tab.

The conservative choice: **engine stability over preserving the
user's local mod**. If a patch conflicts, the engine stays at the
official version — never half-applied broken code. The user
intervenes via the WebUI.

## Reverting

### One patch

In Settings → Self-Modifications, click **Revert** on a row. Under
the hood:

```
git apply -R <patch>
```

If the patch you're reverting isn't the most recent, the UI warns
that later patches built on top may now conflict (because they
diff against state your reverted patch produced). For complex
stacks, `Revert all → official` is more predictable.

### All

The **Revert all → official** button:

```
git reset --hard origin/master
rm -rf ~/.hrant/data/self_mods/*.patch
rm ~/.hrant/data/self_mods/applied.json
```

After this the engine is byte-identical to the GitHub remote at
HEAD. User data (knowledge, workspace, settings, conversation,
identity, channels) is untouched.

## API

```
GET    /api/self-mods                  list every patch in manifest order
POST   /api/self-mods/{id}/revert      reverse-apply one patch
POST   /api/self-mods/revert-all       hard reset to origin/master
```

Plus the legacy proposal flow (`/api/self-modifier/*` — see
[backend/api/intel.py](../backend/api/intel.py)) for the
analyse → propose → approve → apply lifecycle. That flow now
automatically captures the resulting diff as a self-mod entry.

## Risks (be honest about them)

- **Drift from upstream.** Every self-mod increases the chance of
  a conflict at the next `hrant update`. For divergent feature
  requests (e.g. replacing RAG with SQLite), assume future updates
  in the affected modules will need manual reconciliation.

- **Stacked patches.** Patch N+1 diffs against the state patch N
  produced. Reverting patch N out of order may render patch N+1
  unapplyable. The UI warns; in practice, `Revert all → official`
  is the safe escape hatch.

- **Code that breaks the running process.** The legacy
  `self_modifier.apply()` path runs `py_compile` and the
  proposal's `test_commands` before committing — invalid Python
  is rolled back automatically. Patches coming from other flows
  (manual diff drop-in) don't go through that gate.

- **No syncing across machines.** Self-mods live in one specific
  `data_dir`. If the user has two installs, they're independent.
  Self-modifications are personal and stay personal — there's no
  built-in path to publish them.

## When to revert instead

If a self-modification is causing problems, the safe escape is
**Revert all → official**:

- Settings → Self-Modifications → "Revert all → official"
- Behind the scenes: `git reset --hard origin/master` + wipe of
  every `.patch` file in `~/.hrant/data/self_mods/`.
- User data (knowledge, workspace, settings) is untouched.

This is the primary reason self-modifications exist in this form
— so the user always has a known-good fallback to a clean engine.
