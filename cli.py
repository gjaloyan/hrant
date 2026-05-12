"""Backwards-compatibility shim — the REPL itself lives in
`backend/repl.py` now.

Why keep this file: `python cli.py` and `python cli.py "вопрос"` were
the documented invocations for over a year. Tearing them out would
break service scripts, README examples, and finger-memory. So the
shim stays — it's three lines that just re-export.

For new code, prefer:
  hrant chat                — the unified CLI dispatcher
  hrant chat "вопрос"       — one-shot question
  python -m backend.repl    — direct REPL invocation
"""
from __future__ import annotations
import sys

from backend.repl import main

if __name__ == "__main__":
    main()
    sys.exit(0)
