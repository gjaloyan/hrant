"""Multi-role subagent pool.

Lets the main agent delegate a focused subtask to a specialised
"role" instance with a tighter system prompt and a restricted tool
allowlist. The child runs an LLM tool loop independent of the
parent's pipeline (no full Agent.run cycle, no verifier, no memory
writes) — keeps cost predictable and isolation strong.

Pattern modelled after the hermes-agent `delegate_task` tool (see
agent_examples/hermes-agent-main/tools/delegate_tool.py for the
reference implementation). Differences from hermes:
  - No grandchildren — depth capped at 1 by `MAX_DEPTH`.
  - No ThreadPoolExecutor — subagents run sequentially in the
    parent's thread. Parallel batching is a future addition; this
    keeps the first version simple to reason about.
  - Role registry is hard-coded (researcher / coder / reviewer)
    rather than dynamic from a config — the set is small and the
    prompts are sensitive to LLM quirks, so we treat them as code.
"""
from .dispatch import (
    SubagentResult,
    available_roles,
    run_subagent,
)
from .roles import ROLE_REGISTRY, RoleConfig

__all__ = [
    "ROLE_REGISTRY",
    "RoleConfig",
    "SubagentResult",
    "available_roles",
    "run_subagent",
]
