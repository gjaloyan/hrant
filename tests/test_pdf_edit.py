"""pdf_edit must REPLACE text, not draw over it — and prove both directions.

Prod incident 2026-07-21 / verified 2026-08-05: the hand-rolled edit left the
page carrying the original "Gor Jaloyan" AND a second copy AND "Jaloyan PE"
overlapping at the same coordinates (x=405 and x=424, same y), because the
original was never redacted. It was reported as success since only the presence
of the NEW text was checked.
"""
from __future__ import annotations

import json

import pytest

from backend.tools import pdf_edit as pe


fitz = pe._load_fitz()
requires_fitz = pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")


def _make_pdf(path, lines):
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 20
    doc.save(str(path))
    doc.close()
    return str(path)


# ── honest failure when the library is missing ────────────────────────
def test_missing_library_fails_honestly(monkeypatch):
    monkeypatch.setattr(pe, "_load_fitz", lambda: None)
    rep = pe.replace_text("/nope.pdf", [{"find": "a", "replace": "b"}])
    assert rep.ok is False
    assert "not installed" in rep.error
    assert "pip install pymupdf" in rep.install_hint


def test_missing_input_is_reported(monkeypatch):
    if fitz is None:
        monkeypatch.setattr(pe, "_load_fitz", lambda: object())
    rep = pe.replace_text("/definitely/missing.pdf", [{"find": "a", "replace": "b"}])
    assert rep.ok is False and "not found" in rep.error


def test_empty_replacements_rejected(monkeypatch):
    if fitz is None:
        monkeypatch.setattr(pe, "_load_fitz", lambda: object())
    rep = pe.replace_text(__file__, [{"find": "  ", "replace": "x"}])
    assert rep.ok is False and "no usable replacements" in rep.error


def test_media_line_shape():
    assert pe.media_line("/tmp/x.pdf") == "MEDIA:/tmp/x.pdf"


# ── the real thing ────────────────────────────────────────────────────
@requires_fitz
def test_original_text_is_gone_not_overlapped(tmp_path):
    src = _make_pdf(tmp_path / "in.pdf", ["Bill to:", "Gor Jaloyan", "Total: 100"])
    out = str(tmp_path / "out.pdf")
    rep = pe.replace_text(src, [{"find": "Gor Jaloyan",
                                 "replace": "Gor Jaloyan PE\nTax Number:25391688"}],
                          out_path=out)
    assert rep.ok is True, rep.verification_detail
    text = pe.extract_text(out)
    assert "Gor Jaloyan PE" in text
    assert "Tax Number:25391688" in text
    # the exact failure of 2026-07-21: the bare original must NOT survive
    assert text.count("Gor Jaloyan") == 1        # only inside "Gor Jaloyan PE"
    assert rep.replacements[0].occurrences == 1
    assert rep.replacements[0].verified is True


@requires_fitz
def test_overlap_is_caught_when_the_original_survives(tmp_path, monkeypatch):
    """The Jul-21 failure mode itself: original left in place, new text added
    beside it. Verification must REFUSE that, not call it done."""
    src = _make_pdf(tmp_path / "in.pdf", ["Gor Jaloyan"])
    out = str(tmp_path / "out.pdf")
    # simulate the broken hand-rolled edit: draw over without redacting
    doc = fitz.open(src)
    # the full new string drawn on top while the original stays -> the page
    # now carries the original AND the replacement (which embeds it)
    doc[0].insert_text((72, 130), "Gor Jaloyan PE", fontsize=11)
    doc.save(out)
    doc.close()

    rep = pe.EditReport(out_path=out)
    rep.replacements = [pe.ReplacementReport(
        find="Gor Jaloyan", replace="Gor Jaloyan PE", occurrences=1)]
    pe._verify(rep, [("Gor Jaloyan", "Gor Jaloyan PE")], fitz)
    assert rep.verified is False
    assert "drawn over, not replaced" in rep.verification_detail


@requires_fitz
def test_reports_when_the_target_is_absent(tmp_path):
    src = _make_pdf(tmp_path / "in.pdf", ["Nothing to see"])
    rep = pe.replace_text(src, [{"find": "Missing Name", "replace": "X"}],
                          out_path=str(tmp_path / "out.pdf"))
    assert rep.ok is False
    assert rep.replacements[0].occurrences == 0
    assert "was not found" in rep.replacements[0].detail


@requires_fitz
def test_multiple_occurrences_all_replaced(tmp_path):
    src = _make_pdf(tmp_path / "in.pdf", ["ACME Ltd", "invoice", "ACME Ltd"])
    out = str(tmp_path / "out.pdf")
    rep = pe.replace_text(src, [{"find": "ACME Ltd", "replace": "ACME PLC"}],
                          out_path=out)
    assert rep.ok is True, rep.verification_detail
    text = pe.extract_text(out)
    assert "ACME Ltd" not in text and text.count("ACME PLC") == 2


@requires_fitz
def test_default_out_path_lands_in_outbox(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    import importlib
    from backend import config
    importlib.reload(config)
    out = pe.default_out_path("/somewhere/invoice.pdf")
    assert out.endswith("invoice_edited.pdf")
    assert "outbox" in out          # deliverable by construction


# ── tool layer ────────────────────────────────────────────────────────
@requires_fitz
def test_handler_returns_media_hint_on_success(tmp_path, monkeypatch):
    import backend.builtin_tools as bt
    monkeypatch.setattr(bt, "_check_owner", lambda *a, **k: (False, "webui:default"))
    src = _make_pdf(tmp_path / "in.pdf", ["Gor Jaloyan"])
    out = str(tmp_path / "out.pdf")
    payload = json.loads(bt._pdf_edit_handler(
        src, [{"find": "Gor Jaloyan", "replace": "Gor Jaloyan PE"}], out))
    assert payload["ok"] is True
    assert payload["media_hint"] == f"MEDIA:{out}"
    assert "failed task" in payload["note"]


def test_handler_is_owner_gated(monkeypatch):
    import backend.builtin_tools as bt
    monkeypatch.setattr(bt, "_check_owner", lambda *a, **k: ("refused", None))
    payload = json.loads(bt._pdf_edit_handler("/x.pdf", [{"find": "a", "replace": "b"}]))
    assert payload["ok"] is False and "owner" in payload["error"]
