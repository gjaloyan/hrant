"""Standalone speech recogniser for a model the agent's venv cannot load.

Run as a subprocess, never imported — the same arrangement as
`captcha_worker.py`, and for the same reason. The agent runs in a venv
with `faster_whisper` but without `transformers` or `torch`, and the
Armenian model ships plain transformers weights with no CTranslate2
build. Installing torch into that venv to serve one language would add
~800 MB to a process that needs it for a few seconds per voice note.

Talks JSON over stdout. Free of `backend` imports so whichever
interpreter does have the dependencies can execute it.
"""
from __future__ import annotations

import json
import sys


def main():
    args = json.loads(sys.stdin.read() or "{}")
    path = args.get("path") or ""
    model = args.get("model") or ""
    language = args.get("language") or None
    if not path or not model:
        json.dump({"ok": False, "error": "path and model are required"},
                  sys.stdout, ensure_ascii=False)
        return

    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model, device=-1)
    kw = {"task": "transcribe"}
    if language:
        kw["language"] = language
    out = asr(path, generate_kwargs=kw)
    json.dump({"ok": True, "text": ((out or {}).get("text") or "").strip()},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:      # the caller shows this verbatim
        json.dump({"ok": False, "error": f"{type(e).__name__}: {e}"},
                  sys.stdout, ensure_ascii=False)
