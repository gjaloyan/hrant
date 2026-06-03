# Harbor Adapters

Source-of-truth for any Harbor agent adapter that wraps Hrant.

## Why this lives here

Harbor's plugin loader reads adapter classes from
`harbor/agents/installed/` inside its venv. If we only edited the
file in-place there, our changes would be lost on the next `pip
install --upgrade harbor` (or any `hrant update` that rebuilds the
harbor venv). Keeping the source in this repo lets us version it
alongside the matching gateway changes (`/api/exec-protocol`,
ContextVar override) and ensures a `harbor` reinstall can be
followed by a deterministic redeploy step.

## Files

- `hrant_agent.py` — `HrantAgent(BaseInstalledAgent)` adapter that
  routes Harbor terminal-bench trials through Hrant's gateway.
- `__init__.py` — package marker (also points readers at this README).

## Manual deploy (until `hrant update` automates this)

Run on the host where Harbor's venv lives:

```bash
cp harbor_adapter/hrant_agent.py \
  ~/.hrant/data/workspace/terminal_bench_2_1/.venv/lib/python3.12/site-packages/harbor/agents/installed/hrant_agent.py
```

Verify Harbor sees it:

```bash
~/.hrant/data/workspace/terminal_bench_2_1/.venv/bin/harbor run --help \
  | grep -i hrant || echo "adapter NOT registered"
```

(The agent should appear in the `--agent` enum.)

## Running a trial

```bash
~/.hrant/data/workspace/terminal_bench_2_1/.venv/bin/harbor run \
  --dataset terminal-bench \
  --n-tasks 2 \
  --agent hrant \
  --agent-timeout-multiplier 10 \
  --n-concurrent 1
```

`--agent-timeout-multiplier 10` because Hrant turns can legitimately
take 5–15 minutes (cold tool loop with skill load + KG search). The
default per-trial timeout in Harbor is much tighter.

`--n-concurrent 1` because Hrant's module-level state (skills,
in-progress jobs) hasn't been audited for parallel `Agent.run`
safety yet.

## Architecture (one-liner)

Adapter starts a loopback aiohttp server → POSTs `{task,
callback_url}` to `http://localhost:3333/api/exec-protocol` → gateway
sets a ContextVar so Hrant's `terminal_exec` POSTs each command back
to the adapter → adapter awaits `environment.exec` (Harbor's docker
exec) and returns stdout/stderr/return_code. Full design:
`docs/superpowers/specs/2026-06-03-hrant-harbor-adapter-design.md`.