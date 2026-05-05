"""Safe arithmetic evaluator — used by the agent's `calc` tool.

We don't go through `subprocess + run_python` for arithmetic because:

  * subprocess startup on Windows is ~150-300 ms wall time per call —
    expensive for an answer that should be instant.
  * `run_python` is a full Python interpreter; for "what is 2+2" we
    don't want to expose `os`, `subprocess`, `socket`, etc. (the agent's
    own self-review correctly flagged this naming as misleading).

So `calc` parses the expression with `ast` and walks the tree,
allowing only:

  * Numeric literals (int, float).
  * Binary ops: + - * / // % **
  * Unary ops: + - (negation).
  * Parentheses (free via AST).
  * A short whitelist of math functions: sqrt, abs, pow, round,
    floor, ceil, log, log10, exp, sin, cos, tan.
  * Constants: `pi`, `e`.

Anything else (function calls, attribute access, names, comprehensions,
imports, …) raises a CalcError with a short reason. Time and memory
bounds are unnecessary because the AST shape forbids loops and
recursion — the only operations available run in O(1) on bounded
operand sizes.
"""
from __future__ import annotations
import ast
import math
import operator as op


class CalcError(ValueError):
    """Raised when the input isn't a safe arithmetic expression."""


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
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "min": min,
    "max": max,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

# Cap operand magnitude so a Pow-of-doom (10**10**10) can't run away
# the interpreter even though the AST whitelist already forbids loops.
_MAX_ABS = 1e308


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalcError(f"constant of unsupported type {type(node.value).__name__}")
    # `ast.Num` was removed in Python 3.12; only reach for it on older
    # interpreters via getattr so the code stays import-safe everywhere.
    _Num = getattr(ast, "Num", None)
    if _Num is not None and isinstance(node, _Num):
        return node.n  # type: ignore[attr-defined]
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise CalcError(f"binary op {type(node.op).__name__} not allowed")
        a = _eval(node.left)
        b = _eval(node.right)
        if isinstance(node.op, ast.Pow):
            # Cheap guard against 10**10**10 — pow with huge exponent
            # is rejected before we hand it to op.pow.
            if isinstance(b, (int, float)) and abs(b) > 1000:
                raise CalcError("exponent too large")
        result = _BIN_OPS[type(node.op)](a, b)
        if isinstance(result, (int, float)) and abs(result) > _MAX_ABS:
            raise CalcError("result overflow")
        return result
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise CalcError(f"unary op {type(node.op).__name__} not allowed")
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalcError("only direct function calls allowed")
        name = node.func.id
        fn = _FUNCTIONS.get(name)
        if fn is None:
            raise CalcError(f"function '{name}' not allowed")
        if node.keywords:
            raise CalcError("keyword arguments not allowed")
        args = [_eval(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Name):
        val = _CONSTANTS.get(node.id)
        if val is None:
            raise CalcError(f"name '{node.id}' not allowed")
        return val
    raise CalcError(f"node type {type(node).__name__} not allowed")


def calc(expression: str) -> float | int:
    """Evaluate a single arithmetic expression. Returns the result.

    Raises CalcError on any unsupported syntax / disallowed operations.
    Examples:
        calc("2 + 2")                   -> 4
        calc("(5 + 3) / 2")             -> 4.0
        calc("sqrt(16)")                -> 4.0
        calc("2 ** 10")                 -> 1024
        calc("100 * 0.17")              -> 17.0
        calc("import os")               -> CalcError
        calc("__import__('os').system") -> CalcError
    """
    if not expression or not expression.strip():
        raise CalcError("empty expression")
    if len(expression) > 1000:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise CalcError(f"syntax error: {e.msg}") from e
    return _eval(tree)
