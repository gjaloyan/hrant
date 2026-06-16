"""Tests for `strip_markdown_for_speech` — the TTS input cleaner.

A voice reply used to feed the raw answer (full of Telegram emphasis
`*_..._*`, `### headers`, `` `code` ``, bullet markers) straight into
the synthesizer, which then vocalized the markup symbols. The cleaner
removes the formatting MARKERS while keeping the words and the sentence
punctuation that gives the voice its prosody.
"""
from __future__ import annotations

from backend.tts import strip_markdown_for_speech


def test_strips_telegram_emphasis_keeps_words():
    assert strip_markdown_for_speech("technically PEPE looks *_bearish_*") == (
        "technically PEPE looks bearish"
    )


def test_strips_bold_italic_code_and_links():
    src = "see **bold**, *italic*, `code` and [the docs](https://x.io/y)"
    assert strip_markdown_for_speech(src) == (
        "see bold, italic, code and the docs"
    )


def test_strips_headings_and_bullets_at_line_start():
    src = "### Main verdict\n- first point\n- second point"
    out = strip_markdown_for_speech(src)
    assert "#" not in out
    assert "Main verdict" in out
    assert "first point" in out and "second point" in out
    assert "- " not in out


def test_keeps_sentence_punctuation():
    # Commas / periods / question marks shape prosody — they stay.
    assert strip_markdown_for_speech("Done. Fixed it, right?") == (
        "Done. Fixed it, right?"
    )


def test_preserves_underscores_inside_words():
    # file_name is a word, not emphasis — leave it intact.
    assert strip_markdown_for_speech("edited backend/builtin_tools.py") == (
        "edited backend/builtin_tools.py"
    )


def test_drops_fenced_code_blocks():
    src = "Here is the fix:\n```python\nx = 1\n```\nDone."
    out = strip_markdown_for_speech(src)
    assert "x = 1" not in out
    assert "Done." in out
    assert "`" not in out


def test_empty_and_none_safe():
    assert strip_markdown_for_speech("") == ""
    assert strip_markdown_for_speech(None) == ""
