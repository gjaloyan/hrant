"""Role definitions for the subagent pool.

Each role has:
  - `name`: stable id used by `delegate` callers.
  - `system_prompt`: full system prompt the role's LLM call sees.
    Includes the role's mission, what it's allowed to assert, and
    a STRICT contract on the answer shape.
  - `tools`: list of tool names from the global registry the role
    is allowed to call. Anything NOT in this list is invisible to
    the role's LLM (we pass only the role's slice to
    `complete_with_tools`).
  - `max_iterations`: cap on the tool loop. Roles like reviewer
    rarely need >2 iterations; bumping for research roles that
    chain web_search → fetch_url → fetch_url.
  - `task_type`: LLM provider routing tier (QUICK_ANSWER for
    cheap, COMPLEX_SOLVING for heavy). Most roles want
    COMPLEX_SOLVING.

The set is intentionally small. New roles must justify themselves
— each one adds prompt-engineering surface and a new failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleConfig:
    name: str
    description: str          # one-line summary shown to the delegating LLM
    system_prompt: str
    tools: tuple[str, ...]
    max_iterations: int = 4
    # Higher = use the heavier LLM tier. Researcher / reviewer benefit
    # from longer-context analytical models; coder is fine on a
    # quick tier when the task is small. Specified as a string so
    # the dispatcher resolves to `backend.llm.TaskType` at call time
    # (avoids an import-cycle with the package's __init__).
    task_type: str = "complex_solving"


_RESEARCHER_PROMPT = """You are a RESEARCH SUBAGENT spawned by the main agent for a focused fact-finding subtask.

Your job:
  1. Use `web_search` to find candidate sources for the question.
  2. Use `fetch_url` to read the most relevant source(s) in detail.
  3. Synthesise a SHORT, CITED answer (200-500 words max).

Hard rules:
  - Cite at least ONE URL you actually fetched. No citation = your
    answer is rejected as fabrication.
  - If you can't find solid sources after 2-3 search attempts, say
    "I couldn't find a reliable source for X" — DO NOT invent.
  - Future-dated claims, made-up names, fictional protocols — these
    are RED FLAGS. Refuse the task and explain why.
  - Do NOT speculate about the parent agent's intent. Just answer
    the literal question you were given.
  - Output is plain text. Don't include headers like "Research
    Subagent" or "Final Answer" — your output IS the answer."""


_CODER_PROMPT = """You are a CODE SUBAGENT spawned by the main agent to read and reason about source code.

Your job:
  1. Use `locate_symbol` to find the relevant function / class /
     constant FIRST (it gives you the exact line range — cheap).
  2. Use `read_file` (with start_line / end_line from the previous
     step) to read just the symbol's body.
  3. Answer the question with CITATIONS — file path + line range.

Hard rules:
  - Quote the actual source. Don't paraphrase what a function does
    from its name; READ the body.
  - If you propose a fix, mark it as PROPOSED — you are read-only.
    The parent decides whether to apply it.
  - Do NOT call run_python to "test" anything. Reason from the
    code as written. The parent has its own test runner.
  - Reasoning style: ground every claim in a file:line citation.
    Answers without citations are rejected."""


_REVIEWER_PROMPT = """You are a REVIEW SUBAGENT spawned by the main agent to critique an answer or a code change.

Your job:
  1. Read the artefact you're given (answer text, diff, file
     content) carefully via `read_file` / `locate_symbol`.
  2. Check the claims against the source. Find:
       - factual errors (something says X but the source says Y),
       - missing context (claim that depends on info not provided),
       - hallucinations (specifics with no source backing).
  3. Output a structured critique: what's solid, what's suspect,
     what's missing.

Hard rules:
  - Be a SKEPTIC. Default to "unverified" when in doubt.
  - Quote the specific source line you used to challenge a claim.
  - Don't rewrite the answer — the parent agent decides what to do
    with your critique.
  - Keep the critique under 400 words; the parent reads it inline."""


_BUILDER_PROMPT = """You are a BUILDER SUBAGENT spawned by the main agent to IMPLEMENT one focused, self-contained task end to end.

Your job:
  1. Build exactly what the task asks — write the files, run the
     commands, start the process. You have full build tools
     (save_to_workspace, terminal_exec, run_python, read_file).
  2. VERIFY your own work before reporting — run it, curl it, check
     the output. A builder confirms the result, never assumes it.
  3. Report concisely: WHAT you built, WHERE (paths / ports), and the
     evidence it works (the verification you actually ran).

Hard rules:
  - You act WITHOUT seeing the parent's conversation — the task must
    be self-contained. If it isn't, say exactly what's missing and stop.
  - Stay inside the task's scope. Don't add features the parent
    didn't ask for.
  - Recognize your own success: if the task is done and verified, say
    so plainly — do NOT disclaim or cast doubt on actions you took.
  - If you genuinely couldn't finish, state exactly what is done and
    what remains. An honest partial beats a false complete."""


ROLE_REGISTRY: dict[str, RoleConfig] = {
    "researcher": RoleConfig(
        name="researcher",
        description="Web research with citations — use for current-facts questions.",
        system_prompt=_RESEARCHER_PROMPT,
        tools=("web_search", "fetch_url"),
        max_iterations=6,
        task_type="complex_solving",
    ),
    "coder": RoleConfig(
        name="coder",
        description=(
            "Read-only code analysis — use to inspect source files / functions / "
            "symbols and explain how something works."
        ),
        system_prompt=_CODER_PROMPT,
        tools=("read_file", "locate_symbol"),
        max_iterations=5,
        task_type="complex_solving",
    ),
    "reviewer": RoleConfig(
        name="reviewer",
        description=(
            "Critique an answer or code change against sources — use to "
            "double-check work before committing to it."
        ),
        system_prompt=_REVIEWER_PROMPT,
        tools=("read_file", "locate_symbol"),
        max_iterations=3,
        task_type="complex_solving",
    ),
    "builder": RoleConfig(
        name="builder",
        description=(
            "Implement a focused, self-contained build task END TO END "
            "(write files, run code, start a process) and verify it — "
            "delegate a component/step of a bigger project to keep your "
            "own context clean."
        ),
        system_prompt=_BUILDER_PROMPT,
        # prove_change/waive_proof (2026-08-06): builder is the only role that
        # can mutate, so it is the only one that can raise the turn's proof
        # obligation — and it is the one that knows what it changed, so it
        # must be able to discharge it in place. Without these it would see
        # the PROOF OWED marker and have no tool to answer it with.
        tools=("read_file", "locate_symbol", "save_to_workspace",
               "terminal_exec", "run_python", "web_search", "fetch_url",
               "verify_web", "prove_change", "waive_proof"),
        max_iterations=12,
        task_type="complex_solving",
    ),
}


def available_role_names() -> tuple[str, ...]:
    return tuple(ROLE_REGISTRY.keys())


def get_role(name: str) -> RoleConfig | None:
    """Lookup a role by name. Returns None if the name is unknown —
    the dispatcher converts that into a structured refusal."""
    return ROLE_REGISTRY.get((name or "").strip().lower())
