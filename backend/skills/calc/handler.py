"""Handler for the `calc` skill — registers a safe arithmetic-only evaluator.

This skill exists mainly to demonstrate the handler.py mechanism: a skill
can ship its own narrowly-scoped tool with a tighter contract than the
generic `run_python`. Here `calc` only allows arithmetic operators and
numeric literals — no names, no calls, no attribute access. That's safer
and gives the LLM a clear signal: "use this for math, run_python for code".
"""
from __future__ import annotations
import ast
import operator as op

# Map AST node types to operator functions. Only arithmetic.
_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def calc(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except (SyntaxError, ValueError) as e:
        return f"[calc error: {e}]"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def register(registry) -> None:
    if "calc" in registry.tools:
        return
    registry.register_func(
        name="calc",
        description=(
            "Safely evaluate an arithmetic expression. Supports +, -, *, /, //, %, **, "
            "parentheses and numeric literals. NO names, calls, or attribute access. "
            "Use this for arithmetic; use run_python for anything more complex."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '4500 * 0.17' or '2 ** 32'.",
                },
            },
            "required": ["expression"],
        },
        handler=calc,
        origin="skill:calc",
    )
