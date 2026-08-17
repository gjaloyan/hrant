"""Standalone CAPTCHA recogniser. Run as a subprocess, never imported.

Deliberately free of `backend` imports: the agent runs in a venv without
torch, so this file is executed by whichever interpreter DOES have it
(see captcha_reader._find_interpreter). Talks JSON over stdout.

Two things make a distorted-glyph reader usable in practice, and both
live here rather than in the caller:

  * Background removal. Most CAPTCHAs draw coloured ink over a fixed
    decorative plate. Keeping only saturated, non-white pixels leaves
    clean strokes and costs nothing. It is not always a win — heavy
    anti-aliasing can eat thin strokes — so the image is read BOTH ways
    and both readings are returned.
  * Ambiguity that pixels cannot settle. In brush/handwriting fonts
    O/0/Q and I/1/L are the same shape plus noise; magnifying does not
    help. Emitting one confident answer would be a lie, so the reader
    ranks substitutions and lets the caller retry — CAPTCHA reloads are
    free, a wrong single answer is not.
"""
from __future__ import annotations

import json
import sys

# Glyph shapes that genuinely collide once a font distorts them. Written
# as groups and expanded below so the relation cannot go one-way: an
# asymmetric table silently strands one reading direction without
# alternatives, which is how a retry loop runs out of guesses early.
# Only shape collisions belong here — this is not a spellcheck table.
_CONFUSION_GROUPS = (
    "0OQD",   # closed ovals; the tail on Q vanishes into neighbouring ink
    "1IL", "17", "7T",
    "5S", "2Z", "8B", "6G",
    "UV", "VW", "MN",
)


def _build_confusions(groups):
    table: dict[str, str] = {}
    for group in groups:
        for ch in group:
            others = [c for c in group if c != ch]
            existing = table.get(ch, "")
            table[ch] = existing + "".join(c for c in others if c not in existing)
    return table


CONFUSIONS = _build_confusions(_CONFUSION_GROUPS)


def isolate_ink(img):
    """Drop the decorative plate, keep the ink.

    Saturation, not darkness, is the discriminator: plates are usually
    grey/beige textures while the glyphs are printed in a colour. Falls
    back to a darkness threshold when the image is genuinely greyscale,
    so a monochrome CAPTCHA still gets a usable second reading.
    """
    import numpy as np
    from PIL import Image

    a = np.asarray(img.convert("RGB")).astype(int)
    mx, mn = a.max(2), a.min(2)
    ink = ((mx - mn) > 20) & (mx < 235)
    if ink.sum() < 40:
        ink = a.mean(2) < 128
    return Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).convert("RGB")


def rank_candidates(votes, expected_length=0, limit=6):
    """Order plausible strings: agreed readings first, then substitutions.

    `expected_length`, when the caller knows it, is a hard filter and the
    single most valuable input this function takes — a reader that drops
    or invents one character is the common failure, and length is the one
    property a caller can usually establish from a handful of samples.
    """
    seen, out = set(), []

    def push(s):
        if not s or s in seen:
            return
        if expected_length and len(s) != expected_length:
            return
        seen.add(s)
        out.append(s)

    for v in votes:
        push(v)
    base = next((v for v in votes if not expected_length
                 or len(v) == expected_length), votes[0] if votes else "")
    for i, ch in enumerate(base):
        for alt in CONFUSIONS.get(ch, ""):
            push(base[:i] + alt + base[i + 1:])
    return out[:limit]


def main():
    args = json.loads(sys.stdin.read() or "{}")
    path = args.get("path") or ""
    repo = args.get("model") or "hakim77/trocr-captcha-v4-massive-2.4M"
    expected = int(args.get("expected_length") or 0)
    limit = int(args.get("max_candidates") or 6)

    from PIL import Image
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    proc = TrOCRProcessor.from_pretrained(repo)
    model = VisionEncoderDecoderModel.from_pretrained(repo).eval()

    img = Image.open(path)
    votes = []
    for variant in (isolate_ink(img), img.convert("RGB")):
        px = proc(images=variant, return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(px, max_new_tokens=12)
        votes.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip().upper())

    cands = rank_candidates(votes, expected, limit)
    json.dump({
        "ok": True,
        "readings": votes,
        # Both passes returning the same string is the reader's own
        # confidence signal: measured on real samples, agreement tracked
        # correctness far better than any per-token score.
        "agreement": len(set(votes)) == 1,
        "best": cands[0] if cands else "",
        "candidates": cands,
        "length_filtered": bool(expected),
    }, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # surface the real cause; the caller shows it verbatim
        json.dump({"ok": False, "error": f"{type(e).__name__}: {e}"},
                  sys.stdout, ensure_ascii=False)
