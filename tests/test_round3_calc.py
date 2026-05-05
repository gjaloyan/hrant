"""Round 3 / #1: `calc` is a real builtin tool — safe AST evaluator,
no subprocess, no module imports, no filesystem. Used by the agent
when arithmetic is detected so we don't spawn run_python for "2+2".
"""
from __future__ import annotations
import math

import pytest

from backend.tools.calc import CalcError, calc


def test_basic_arithmetic():
    assert calc("2+2") == 4
    assert calc("15 * 3") == 45
    assert calc("(5+3)/2") == 4.0
    assert calc("10 - 7") == 3
    assert calc("17 % 5") == 2
    assert calc("17 // 5") == 3


def test_unary_minus_and_plus():
    assert calc("-5") == -5
    assert calc("+5") == 5
    assert calc("-(2+3)") == -5


def test_power():
    assert calc("2 ** 10") == 1024
    assert calc("2**0.5") == pytest.approx(1.4142135, abs=1e-6)


def test_decimals_and_percent_style():
    assert calc("100 * 0.17") == pytest.approx(17.0)
    assert calc("250 * 0.10") == pytest.approx(25.0)


def test_whitelisted_functions():
    assert calc("sqrt(16)") == 4.0
    assert calc("abs(-7)") == 7
    assert calc("round(3.7)") == 4
    assert calc("min(3, 1, 2)") == 1
    assert calc("max(3, 1, 2)") == 3
    assert calc("log10(1000)") == pytest.approx(3.0)
    assert calc("exp(0)") == 1.0


def test_constants():
    assert calc("pi") == math.pi
    assert calc("2 * pi") == pytest.approx(2 * math.pi)
    assert calc("e") == math.e


def test_rejects_imports_and_attribute_access():
    for src in [
        "__import__('os')",
        "open('/etc/passwd')",
        "os.system('rm -rf /')",
    ]:
        with pytest.raises(CalcError):
            calc(src)


def test_rejects_unknown_function():
    with pytest.raises(CalcError):
        calc("system('echo hi')")


def test_rejects_keyword_args():
    with pytest.raises(CalcError):
        calc("pow(2, exp=3)")


def test_rejects_huge_exponent():
    """Pow-of-doom guard — `10**10**10` would lock up the interpreter."""
    with pytest.raises(CalcError):
        calc("10 ** 10000")


def test_rejects_empty_or_oversized():
    with pytest.raises(CalcError):
        calc("")
    with pytest.raises(CalcError):
        calc("   ")
    with pytest.raises(CalcError):
        calc("1+1" + " + 1" * 5000)  # over 1000-char cap


def test_syntax_error_message_is_short():
    """SyntaxError is raised as CalcError with a short reason — we
    don't dump a full traceback through the tool result."""
    with pytest.raises(CalcError) as exc:
        calc("2 +")
    # The message should be human-readable, not a stack trace.
    assert "syntax error" in str(exc.value).lower()


def test_calc_reachable_via_skills():
    """`calc` lives in a skill (backend/skills/calc/), not in
    builtin_tools.py — but the agent loads skills before the tool
    loop, so it ends up in the registry just the same. Verify the
    full path: load skills → registry has `calc` → handler delegates
    to the new shared evaluator."""
    from backend.tool_registry import get_registry, reset_registry
    from backend.builtin_tools import register_builtin_tools
    from backend.skills import SKILLS

    reset_registry()
    # reset_registry clears `tools`, but SKILLS singleton may have
    # _loaded=True from an earlier test — force a reload.
    SKILLS._loaded = False
    register_builtin_tools()
    SKILLS.ensure_loaded()
    reg = get_registry()
    assert "calc" in reg.names()

    out, is_err = reg.execute("calc", {"expression": "2 + 2"})
    assert out == "4"
    assert is_err is False
    # Errors come back as `[calc error: ...]` per the existing skill
    # contract — the new evaluator's CalcError is wrapped to keep that.
    out_err, err_flag = reg.execute("calc", {"expression": "__import__('os')"})
    assert "calc error" in out_err.lower()
