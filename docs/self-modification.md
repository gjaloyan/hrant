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
- **`needs_review`** — a previous attempt to apply a patch failed
  (e.g. one piece of a restored archive conflicted on top of another).
  Engine is at the official version for the affected lines; the user
  reverts or re-applies a different version from the archive.
- **`reverted`** — user clicked Revert. Kept in manifest for audit
  trail until the next manifest write prunes it.

## `hrant update` integration — archive, don't re-apply

This is the deliberate design choice:

> `hrant update` always brings the engine to **exactly** the
> official `origin/master` head. The user's active self-mods are
> NOT re-applied automatically. Instead they're **archived** so
> the user can re-apply them by hand from the WebUI afterwards.

The reason: auto-reapply leads to half-broken states when a patch
partially conflicts with new engine code. Surfacing the choice to
the user keeps every install in a predictable, known-good state.

When you run `hrant update`:

1. **Pre-flight consent** — if active self-mods exist, the CLI
   prints how many will be archived and prompts `Continue? [y/N]`.
   Non-TTY runs (cron, systemd `ExecStartPre`) must pass `--yes`
   explicitly so an unattended update can't silently archive.
2. Working tree dirty check.
3. `git pull --ff-only origin master`.
4. `pip install -e .`, `npm run build`.
5. **`self_mods.archive_all_active()`** — moves every active patch
   into a fresh timestamped directory:
   ```
   ~/.hrant/data/self_mods/history/2026-05-13T15-22-08Z/
     0001-add-sqlite-memory.patch
     0002-skip-rag.patch
     manifest.json          (snapshot of the archived bundle)
   ```
6. The top-level `applied.json` becomes `{"entries": []}` — engine
   is clean.
7. The UpdateResult shows `self_mods_archived: N` and
   `self_mods_archive_id: <ts>` so the UI can deep-link to that
   bundle in the History panel.

## Re-applying from history

Settings → Self-Modifications → **History** shows every archived
bundle (newest first), each with:

- **Re-apply** on a single patch — applies it as a *new* active
  entry. The archive is unchanged; the same patch can be re-applied
  any number of times.
- **Re-apply all (N)** on a bundle — attempts every archived patch
  in order. Stops at the first conflict and reports which patch
  failed; later patches stay archived so the user can fix or skip
  the broken one.

If the engine has drifted enough that an archived patch no longer
applies, the user sees `git apply failed: …` and the patch file is
left in the archive untouched. There's no force-apply path — a
patch either applies cleanly or not at all.

## Reverting

### One active patch

Settings → Self-Modifications, click **Revert** on a row. Under
the hood: `git apply -R <patch>` + remove from manifest. If the
reverted patch isn't the most recent, the UI warns that later
patches built on top may now conflict.

### All — reset to official

The **Revert all → official** button:

```
git reset --hard origin/master
rm -rf ~/.hrant/data/self_mods/*.patch
rm ~/.hrant/data/self_mods/applied.json
```

After this the engine is byte-identical to the GitHub remote at
HEAD. User data (knowledge, workspace, settings, conversation,
identity, channels) is untouched. The **History** is also kept —
even after revert-all, the archive bundles remain available for
re-apply.

## API

```
GET    /api/self-mods                                 — list active patches
POST   /api/self-mods/{id}/revert                     — revert one
POST   /api/self-mods/revert-all                      — engine -> official

GET    /api/self-mods/history                         — list archives
POST   /api/self-mods/history/{archive_id}/restore    — re-apply whole bundle
POST   /api/self-mods/history/{archive_id}/{patch_filename}/restore
                                                      — re-apply one patch
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
