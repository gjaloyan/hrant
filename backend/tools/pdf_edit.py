"""Replace text inside an existing PDF, then prove the replacement happened.

Why a dedicated tool instead of hand-rolled PyMuPDF: on 2026-07-21 the agent was
asked to change an invoice recipient. It drew the new text ON TOP of the old
one without removing it, so the page carried "Gor Jaloyan", a second
"Gor Jaloyan" and "Jaloyan PE" overlapping at the same coordinates — and it
reported success because it only checked that the NEW string was present, never
that the OLD one was gone. This module does the redaction properly, matches the
original font size, and verifies BOTH directions before claiming anything.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "/home/hrant/.local/share/pipx/venvs/agi-agent/bin/python -m pip install pymupdf"
)
_DEFAULT_FONTSIZE = 9.0
_LINE_GAP = 1.25          # multiplier of the font size between stacked lines


@dataclass
class ReplacementReport:
    find: str
    replace: str
    occurrences: int = 0
    verified: bool = False
    detail: str = ""


@dataclass
class EditReport:
    ok: bool = False
    out_path: str = ""
    replacements: list = field(default_factory=list)
    verified: bool = False
    verification_detail: str = ""
    error: str = ""
    install_hint: str = ""


def _load_fitz():
    """Import PyMuPDF or return None — callers fail honestly, never pretend."""
    try:
        import fitz  # PyMuPDF
        return fitz
    except Exception:
        return None


def _fontsize_at(page, rect, default: float = _DEFAULT_FONTSIZE) -> float:
    """Font size of the span sitting at `rect`, so the replacement is not
    visibly larger or smaller than the text it stands in for."""
    try:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sb = span.get("bbox")
                    if not sb:
                        continue
                    # overlap test against the hit rectangle
                    if (sb[0] < rect.x1 and sb[2] > rect.x0
                            and sb[1] < rect.y1 and sb[3] > rect.y0):
                        size = float(span.get("size") or 0)
                        if size > 0:
                            return size
    except Exception as e:  # pragma: no cover — defensive
        log.debug("fontsize probe failed: %s", e)
    return default


def default_out_path(path: str) -> str:
    """Sibling file in the workspace outbox (the directory the Telegram bridge
    is allowed to attach from), so the result is deliverable by construction."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    try:
        from ..paths import workspace_dir
        outbox = workspace_dir() / "outbox"
    except Exception:  # pragma: no cover — tests without a data dir
        return os.path.join(os.path.dirname(path) or ".", f"{stem}_edited.pdf")
    outbox.mkdir(parents=True, exist_ok=True)
    return str(outbox / f"{stem}_edited.pdf")


def media_line(path: str) -> str:
    """The exact line the model must put in its answer to deliver the file."""
    return f"MEDIA:{path}"


def replace_text(path: str, replacements: list, out_path: str = "") -> EditReport:
    """Replace each `find` with its `replace` in `path`, write `out_path`.

    `replacements`: [{"find": str, "replace": str}]. A `replace` containing
    newlines is laid out as stacked lines starting at the original position.
    The original text is REDACTED (removed from the content stream), not
    covered — otherwise both strings stay extractable and overlap on screen.
    """
    rep = EditReport(out_path=out_path or "")
    fitz = _load_fitz()
    if fitz is None:
        rep.error = "PyMuPDF (fitz) is not installed on this box"
        rep.install_hint = INSTALL_HINT
        return rep
    if not os.path.exists(path):
        rep.error = f"input pdf not found: {path}"
        return rep
    pairs = [
        (str(r.get("find") or ""), str(r.get("replace") or ""))
        for r in (replacements or [])
        if isinstance(r, dict) and str(r.get("find") or "").strip()
    ]
    if not pairs:
        rep.error = "no usable replacements (each needs a non-empty 'find')"
        return rep

    rep.out_path = out_path or default_out_path(path)
    try:
        doc = fitz.open(path)
    except Exception as e:
        rep.error = f"cannot open pdf: {e}"
        return rep

    reports = {find: ReplacementReport(find=find, replace=replace)
               for find, replace in pairs}
    try:
        for page in doc:
            # Collect hits BEFORE redacting: apply_redactions() rewrites the
            # page, invalidating positions found afterwards.
            planned = []
            for find, replace in pairs:
                for rect in page.search_for(find):
                    planned.append((rect, _fontsize_at(page, rect), find, replace))
                    reports[find].occurrences += 1
            if not planned:
                continue
            for rect, _size, _find, _replace in planned:
                page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions()
            for rect, size, _find, replace in planned:
                lines = [ln for ln in replace.split("\n")] or [""]
                for i, line in enumerate(lines):
                    if not line:
                        continue
                    page.insert_text(
                        (rect.x0, rect.y1 + i * size * _LINE_GAP),
                        line, fontsize=size,
                    )
        doc.save(rep.out_path)
    except Exception as e:
        rep.error = f"edit failed: {e}"
        return rep
    finally:
        try:
            doc.close()
        except Exception:
            pass

    rep.replacements = list(reports.values())
    _verify(rep, pairs, fitz)
    rep.ok = rep.verified
    return rep


def extract_text(path: str, fitz=None) -> str:
    """Full text of a PDF (empty string on failure) — used for verification."""
    fitz = fitz or _load_fitz()
    if fitz is None:
        return ""
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        log.debug("extract_text failed for %s: %s", path, e)
        return ""


def _verify(rep: EditReport, pairs: list, fitz=None) -> None:
    """Both directions: every replacement must be present AND every original
    must be gone. Checking only the first is how the 2026-07-21 overlap
    shipped as a success."""
    text = extract_text(rep.out_path, fitz)
    if not text:
        rep.verified = False
        rep.verification_detail = "could not re-extract text from the output pdf"
        return
    problems = []
    for r in rep.replacements:
        find, replace = r.find, r.replace
        # A replacement may legitimately CONTAIN the original ("Gor Jaloyan" ->
        # "Gor Jaloyan PE"), so "old gone" cannot be a plain substring test.
        # The original is gone iff every surviving occurrence is accounted for
        # by a replacement that embeds it.
        expected = replace.count(find) * r.occurrences
        actual = text.count(find)
        old_gone = actual <= expected
        new_present = all(part.strip() in text
                          for part in replace.split("\n") if part.strip())
        r.verified = old_gone and new_present and r.occurrences > 0
        if r.occurrences == 0:
            r.detail = f"'{find}' was not found in the document"
        elif not old_gone:
            r.detail = (
                f"'{find}' still appears {actual} time(s), expected at most "
                f"{expected} — the original was drawn over, not replaced"
            )
        elif not new_present:
            r.detail = f"'{replace}' is not present in the output"
        else:
            r.detail = f"replaced {r.occurrences} occurrence(s)"
        if not r.verified:
            problems.append(r.detail)
    rep.verified = not problems
    rep.verification_detail = ("; ".join(problems) if problems
                               else "all replacements verified: originals gone, "
                                    "new text present")
