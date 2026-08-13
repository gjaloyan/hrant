"""Handler for the `calc` skill — registers a safe arithmetic evaluator.

The evaluator itself lives in `backend.tools.calc` (used to live inline
here; consolidated so there's one source of truth and one set of tests).
The skill module exists to wire it up via the SkillsManager — handler
files like this one are how skills declare their tools, and that
mechanism stays unchanged.
"""
from __future__ import annotations

from ...tools.calc import CalcError, calc as _calc_eval


def calc(expression: str) -> str:
    """Tool handler — returns a printable result string.

    Backwards-compatible with the previous in-skill implementation:
    on success returns just the number ("5", "17.0"), on failure
    returns a `[calc error: ...]` line. Tests pin this contract.
    """
    try:
        result = _calc_eval(expression)
    except CalcError as e:
        return f"[calc error: {e}]"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def register(registry) -> None:
    from ...tool_registry import ToolEffect
    if "calc" in registry.tools:
        return
    registry.register_func(
        name="calc",
        description=(
            "Safely evaluate an arithmetic expression. Supports +, -, *, "
            "/, //, %, **, parentheses, numeric literals, and a small math "
            "whitelist (sqrt, abs, pow, round, floor, ceil, log, log10, "
            "log2, exp, sin, cos, tan, min, max) plus constants `pi`, `e`. "
            "NO imports, NO attribute access, NO names beyond the whitelist. "
            "Use this for arithmetic; use run_python for anything more complex."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '4500 * 0.17' or '2 ** 32' or 'sqrt(16)'.",
                },
            },
            "required": ["expression"],
        },
        handler=calc,
        origin="skill:calc",
        effect=ToolEffect.READ,
        audit_visible=True,
    )
