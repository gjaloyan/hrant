"""read_captcha — recognise distorted characters in a challenge image.

Runs the recogniser in a SUBPROCESS rather than in-process, for two
reasons that both come from measurement on the deployment box:

  * The agent's venv has no torch, and installing it there would add
    ~800 MB to a process that needs it for a few seconds per turn.
  * The weights are 1.3 GB resident. Loading them into the long-lived
    agent would make every turn pay for a capability most turns never
    use. A subprocess that exits releases it all.

The price is the model load, measured at ~7 s, on top of ~6 s of
inference. That is the right trade for a tool called a handful of times
per turn, and it is why this is a separate tool rather than something
folded into the vision path.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL = "hakim77/trocr-captcha-v4-massive-2.4M"
_WORKER = Path(__file__).with_name("captcha_worker.py")
_TIMEOUT = 240

# Resolved once per process: probing interpreters costs a subprocess each.
_interpreter: str | None = None


def _candidate_interpreters() -> list[str]:
    """Interpreters that might carry torch, best guess first.

    An explicit override wins, so a box with the dependencies in an
    unusual place is configuration rather than a code change.
    """
    out: list[str] = []
    env = os.environ.get("HRANT_VISION_PYTHON", "").strip()
    if env:
        out.append(env)
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            out.append(found)
    out.append(sys.executable)
    seen, uniq = set(), []
    for p in out:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _find_interpreter() -> tuple[str, str]:
    """Return (interpreter, error). Probes for torch + transformers."""
    global _interpreter
    if _interpreter:
        return _interpreter, ""
    tried = []
    for exe in _candidate_interpreters():
        try:
            r = subprocess.run(
                [exe, "-c", "import torch, transformers, PIL, numpy"],
                capture_output=True, timeout=120,
            )
        except Exception as e:
            tried.append(f"{exe}: {type(e).__name__}")
            continue
        if r.returncode == 0:
            _interpreter = exe
            return exe, ""
        tried.append(f"{exe}: missing deps")
    return "", (
        "no interpreter with torch+transformers found (tried "
        + "; ".join(tried)
        + "). Install them, or set HRANT_VISION_PYTHON to one that has them."
    )


def read_captcha(
    path: str,
    *,
    expected_length: int = 0,
    min_length: int = 0,
    max_length: int = 0,
    max_candidates: int = 6,
    model: str = "",
) -> dict:
    """Read the characters in a CAPTCHA image.

    Returns a dict with `ok`, `best`, `candidates`, `readings` and
    `agreement`. Never raises: failures come back as `ok=False` with an
    `error` string, so the tool loop does not trip on them.

    Args:
        path: image file on disk (png/jpg/gif/webp/bmp).
        expected_length: exact character count, for generators that emit
            a fixed number. Filters out readings of the wrong length —
            the most common failure mode.
        min_length, max_length: bounds instead of an exact count, for
            generators whose length varies between challenges.
        max_candidates: how many ranked alternatives to return.
        model: HuggingFace repo id; defaults to DEFAULT_MODEL.

    All three length arguments default to 0, meaning no constraint. That
    is the honest state when the generator has not been observed, and it
    is better than a guess: filtering on a wrong length would discard the
    correct reading outright.
    """
    p = Path(str(path or "").strip()).expanduser()
    if not p.is_file():
        return {"ok": False, "error": f"no such image: {p}"}
    if not _WORKER.is_file():
        return {"ok": False, "error": f"worker missing: {_WORKER}"}

    exe, err = _find_interpreter()
    if err:
        return {"ok": False, "error": err}

    payload = json.dumps({
        "path": str(p),
        "expected_length": int(expected_length or 0),
        "min_length": int(min_length or 0),
        "max_length": int(max_length or 0),
        "max_candidates": int(max_candidates or 6),
        "model": (model or "").strip() or DEFAULT_MODEL,
    })
    try:
        r = subprocess.run(
            [exe, str(_WORKER)],
            input=payload, capture_output=True, text=True, timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False,
                "error": f"recogniser timed out after {_TIMEOUT}s"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    out = (r.stdout or "").strip()
    if not out:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        return {"ok": False,
                "error": "recogniser produced no output: " + " | ".join(tail)}
    try:
        return json.loads(out)
    except Exception:
        # The worker only ever prints JSON; anything else means it died
        # in a way it could not catch. Show what it actually said.
        return {"ok": False, "error": f"unparseable worker output: {out[:300]}"}
