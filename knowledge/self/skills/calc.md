---
skill: calc
category: self
kind: skill
updated: 2026-05-07T15:10:54.518235+00:00
file_count: 2
---

# skill: calc

## Description
---
name: calc
description: Evaluate arithmetic and short numeric expressions safely (no eval).
triggers: ["посчитай", "calc", "calculate", "сколько будет", "=", "+", "*", "/", "^"]
when_to_use: |
  User asks for an arithmetic calculation, percentage, conversion, or any
  short numeric expression that doesn't need full Python.
---

# How to use the calc skill

## When
- Простая арифметика, проценты, единицы измерения, длинные числа.
- Когда пользователю нужен быстрый и надёжный счёт без галлюцинаций.

## How
1. Зови `calc(expression="...")` — он принимает Python-подобное арифметическое
   выражение (только +, -, *, /, **, %, скобки и числа).
2. Если выражение сложнее (нужны функции, переменные, циклы) — используй
   обычный `run_python` вместо `calc`.
3. Возвращай результат коротко: одно число, а не лекцию.

## Examples
- "сколько будет 17% от 4500" → calc("4500 * 0.17") → 765
- "2^32" → calc("2 ** 32") → 4294967296

## Files
- `SKILL.md`
- `handler.py`
