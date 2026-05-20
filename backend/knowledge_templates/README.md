# Starter content for a fresh Hrant install

`hrant init` copies everything in this directory into the user's
`knowledge_dir` (default: `~/.hrant/data/knowledge/`) the first time
it runs against an empty data directory.

These files are **starter templates**, not the agent's live memory:

  - `identity/identity.md` — who Hrant is (brand-level; doesn't
    reference any specific user)
  - `identity/soul.md`     — tone & character defaults
  - `identity/user_profile.md` — empty stub; the wizard or the WebUI
    User Profile tab fills it in
  - `core_memory.md`       — empty pinned-context file
  - `goals.json`           — `{"goals": []}`
  - `autonomic/.gitkeep`   — preserves the autonomic log directory

`hrant update` ships newer templates with each engine release.
Already-initialised data dirs do NOT get re-copied — the wizard
would overwrite user customisations. To pull in a new template
field after an upgrade, the user copies it manually (or we add
a `--diff-templates` flag later).

What's deliberately NOT here:

  - `providers.json`, `channels.json`, `active_model.json`,
    `oauth_tokens.json` — auth-bearing files. The wizard creates
    them empty or via the Providers tab.
  - `config.yaml` — copied from `config.example.yaml` in the repo
    root by the wizard.
  - `.env` — written by the wizard from user input.
