"""Round 1 hardening:

- Self-analysis with NO read_file in the trace must trigger one
  forced retry with a strict critique. Without this, the model can
  answer review questions purely from its own narrative and bypass
  verification (the `skip_verify` short-circuit fires when
  tool_context is empty).
- The `_tool_cap` for read_file goes wider on self_analysis turns
  so the verifier sees enough of the actual source to catch
  hallucinations.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import Agent
from backend.models import ThinkingResult, VerificationResult


def test_arithmetic_word_form_detects_russian_phrasing():
    from backend.agent import _looks_like_arithmetic
    assert _looks_like_arithmetic("сколько будет 25 умножить на 3")
    assert _looks_like_arithmetic("посчитай 10 процентов от 250")
    assert _looks_like_arithmetic("два в степени 10")  # has digit
    assert _looks_like_arithmetic("корень из 16")


def test_arithmetic_word_form_detects_english_phrasing():
    from backend.agent import _looks_like_arithmetic
    assert _looks_like_arithmetic("calculate 12 times 4")
    assert _looks_like_arithmetic("what is 100 divided by 5")
    assert _looks_like_arithmetic("sum of 7 and 13")
    assert _looks_like_arithmetic("square root of 256")
    assert _looks_like_arithmetic("2 to the power of 10")


def test_arithmetic_word_form_requires_digit():
    """Bare conversational use of operator words must NOT misfire."""
    from backend.agent import _looks_like_arithmetic
    assert not _looks_like_arithmetic("это просто минус для меня")
    assert not _looks_like_arithmetic("plus or minus, depending")
    assert not _looks_like_arithmetic("давай посчитаем варианты")  # no digit
    assert not _looks_like_arithmetic("calculate the trade-offs")  # no digit


def test_arithmetic_symbolic_form_still_works():
    from backend.agent import _looks_like_arithmetic
    assert _looks_like_arithmetic("2+2")
    assert _looks_like_arithmetic("15 * 3")
    assert _looks_like_arithmetic("(5+3)/2")
