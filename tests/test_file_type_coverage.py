"""Tests for the file-type-coverage fixes (audit from the
"are hrant can send and process all types of file?" question).

  F1 — read_file whitelist now spans most code & text suffixes
       instead of «неподдерживаемый формат».
  F2 — read_file for image extensions saves into AttachmentStore
       and returns a redirect message pointing at `analyze_image`.
  F3 — msg.audio files auto-transcribe just like msg.voice.
  F4 — _UNIFIED_RULES carries the file-type → tool mini-table.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ─── F1: text-like whitelist ─────────────────────────────────────────


def test_read_file_csv_returns_text(tmp_path):
    from backend.tools.file_reader import read_file
    p = tmp_path / "data.csv"
    p.write_text("name,score\nlogo,42\nwatermark,7\n", encoding="utf-8")
    out = read_file(p)
    assert "name,score" in out
    assert "logo,42" in out


def test_read_file_html_returns_text(tmp_path):
    from backend.tools.file_reader import read_file
    p = tmp_path / "page.html"
    p.write_text("<html><body>Hello UNIQUE_HTML_MARKER</body></html>", encoding="utf-8")
    out = read_file(p)
    assert "UNIQUE_HTML_MARKER" in out


def test_read_file_shell_script(tmp_path):
    from backend.tools.file_reader import read_file
    p = tmp_path / "deploy.sh"
    p.write_text("#!/bin/bash\necho UNIQUE_SH_MARKER", encoding="utf-8")
    out = read_file(p)
    assert "UNIQUE_SH_MARKER" in out


def test_read_file_c_source(tmp_path):
    from backend.tools.file_reader import read_file
    p = tmp_path / "x.c"
    p.write_text("int main(){ return 42; /* UNIQUE_C_MARKER */ }", encoding="utf-8")
    out = read_file(p)
    assert "UNIQUE_C_MARKER" in out


def test_read_file_rust_go_java(tmp_path):
    from backend.tools.file_reader import read_file
    for ext, marker in [(".rs", "MARK_RS"), (".go", "MARK_GO"), (".java", "MARK_JAVA")]:
        p = tmp_path / f"f{ext}"
        p.write_text(f"// {marker}\n", encoding="utf-8")
        out = read_file(p)
        assert marker in out, f"{ext} should read as text"


def test_read_file_toml_and_ini(tmp_path):
    from backend.tools.file_reader import read_file
    for ext in (".toml", ".ini", ".cfg"):
        p = tmp_path / f"conf{ext}"
        p.write_text(f"[section]\nkey = value_for_{ext.strip('.')}\n", encoding="utf-8")
        out = read_file(p)
        assert "[section]" in out


def test_read_file_dockerfile_no_extension(tmp_path):
    from backend.tools.file_reader import read_file
    p = tmp_path / "Dockerfile"
    p.write_text("FROM python:3.12\nRUN echo UNIQUE_DOCKER_MARKER", encoding="utf-8")
    out = read_file(p)
    # Extension-less files matched by stem should be read as text.
    assert "UNIQUE_DOCKER_MARKER" in out


def test_read_file_unknown_format_message_is_actionable(tmp_path):
    """The fall-through error message should hint at `run_python`
    rather than just saying 'unsupported' — that's the audit fix
    for the silent 'I have no tool' failure mode."""
    from backend.tools.file_reader import read_file
    p = tmp_path / "weird.xyz"
    p.write_bytes(b"\x89\x42\x53unknown_format")
    out = read_file(p)
    assert "run_python" in out


# ─── F2: image → analyze_image redirect ──────────────────────────────


def test_read_file_image_saves_and_redirects(tmp_path, monkeypatch):
    """An image passed to read_file should land in the AttachmentStore
    and return a message naming the resulting sha256 + pointing at
    `analyze_image`."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    img = tmp_path / "logo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"FAKE_PNG_BODY")

    from backend.tools.file_reader import read_file
    out = read_file(img)
    assert "analyze_image" in out
    assert "sha256=" in out
    # Confirm bytes actually landed in the store.
    from backend.attachments import ATTACHMENTS
    # The output text contains the truncated sha; find it back via list_all.
    found = [a for a in ATTACHMENTS.list_all() if a.kind == "image" and a.filename == "logo.png"]
    assert found
    assert found[0].mime_type == "image/png"


def test_read_file_image_jpg_uses_jpeg_mime(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    img = tmp_path / "thumb.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0JPEG_BODY")
    from backend.tools.file_reader import read_file
    out = read_file(img)
    assert "analyze_image" in out
    from backend.attachments import ATTACHMENTS
    rec = next(a for a in ATTACHMENTS.list_all() if a.filename == "thumb.jpg")
    assert rec.mime_type == "image/jpeg"


# ─── F3: audio auto-transcribe code shape ────────────────────────────


def test_audio_handler_calls_transcriber():
    """The `msg.audio` branch in _gather_attachments must invoke
    TRANSCRIBER + ATTACHMENTS.set_transcript. We pin the source
    shape because mocking the full Telegram event loop in tests is
    impractical and a regression here means voice transcripts come
    back but audio-file transcripts silently don't."""
    import inspect
    import backend.channels as ch_mod
    src = inspect.getsource(ch_mod)
    # Find the msg.audio branch
    audio_branch_idx = src.find('if getattr(msg, "audio", None):')
    assert audio_branch_idx > 0
    # 800 bytes window is more than the branch body length.
    window = src[audio_branch_idx:audio_branch_idx + 1200]
    assert "TRANSCRIBER.transcribe" in window, (
        "msg.audio branch must call TRANSCRIBER.transcribe — "
        "audit gap from voice/audio asymmetry"
    )
    assert "set_transcript" in window


# ─── F4: RULES carry the file-type → tool mini-table ─────────────────


def test_rules_have_file_type_table():
    from backend.unified_agent import _UNIFIED_RULES
    rules = _UNIFIED_RULES
    # Section header
    assert "File types" in rules
    # Mentions of the three main multi-modal tools by name
    assert "analyze_image" in rules
    assert "preprocess_video" in rules
    assert "read_file" in rules
    # PDF / DOCX explicitly named
    assert "PDF" in rules and "DOCX" in rules
    # Fallback hint for unknown
    assert "run_python" in rules
