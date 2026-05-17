"""Tests for the inline-keyboard / callback dispatcher foundation.

Pinned behaviour:
  - InlineButton refuses oversized callback_data (TG 64-byte limit).
  - InlineButtonSet.to_markup builds the expected python-telegram-bot
    keyboard layout (rows of buttons).
  - register_callback_handler + dispatch_callback round-trip.
  - dispatch_callback returns a "no handler" CallbackResult for
    unknown prefixes (without raising).
  - dispatch_callback survives a handler that raises.
  - register_state / consume_state / peek_state semantics.
  - escape_html escapes the unsafe characters.

These tests do NOT touch the Telegram network — `to_markup` is the
only path that imports telegram.* and we exercise it through a
late-import in a separate test that only runs when the package is
installed.
"""
from __future__ import annotations

import pytest

from backend import tg_interactive as tg


# ─── InlineButton / InlineButtonSet ─────────────────────────────────


def test_button_requires_callback_or_url():
    with pytest.raises(ValueError):
        tg.InlineButton("label")


def test_button_rejects_oversized_callback_data():
    too_long = "x" * (tg.CALLBACK_DATA_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="callback_data"):
        tg.InlineButton("label", callback_data=too_long)


def test_button_to_dict_callback():
    b = tg.InlineButton("OK", callback_data="pair:approve:always:ABCD1234")
    assert b.to_dict() == {
        "text": "OK",
        "callback_data": "pair:approve:always:ABCD1234",
    }


def test_button_to_dict_url():
    b = tg.InlineButton("Docs", url="https://example.com")
    assert b.to_dict() == {"text": "Docs", "url": "https://example.com"}


def test_buttonset_to_markup_shape():
    """Late-import test — requires python-telegram-bot installed.
    Asserts the markup carries the 2x2 layout we built."""
    pytest.importorskip("telegram")
    bs = (
        tg.InlineButtonSet()
        .row(
            tg.InlineButton("Once", callback_data="pair:approve:once:AB"),
            tg.InlineButton("Session", callback_data="pair:approve:session:AB"),
        )
        .row(
            tg.InlineButton("Always", callback_data="pair:approve:always:AB"),
            tg.InlineButton("Deny", callback_data="pair:deny:AB"),
        )
    )
    markup = bs.to_markup()
    rows = markup.inline_keyboard
    assert len(rows) == 2
    assert [b.text for b in rows[0]] == ["Once", "Session"]
    assert [b.text for b in rows[1]] == ["Always", "Deny"]
    assert rows[0][0].callback_data == "pair:approve:once:AB"


# ─── Callback dispatcher ────────────────────────────────────────────


@pytest.fixture
def clean_dispatcher():
    """Snapshot the dispatcher state, restore after the test."""
    saved = dict(tg._HANDLERS)
    saved_state = dict(tg._STATE)
    yield
    tg._HANDLERS.clear()
    tg._HANDLERS.update(saved)
    tg._STATE.clear()
    tg._STATE.update(saved_state)


def test_register_and_dispatch_round_trip(clean_dispatcher):
    captured = {}

    def handler(parts, ctx):
        captured["parts"] = parts
        captured["ctx"] = ctx
        return tg.CallbackResult(ok=True, edited_text="done", toast="ok")

    tg.register_callback_handler("test", handler)
    res = tg.dispatch_callback("test:do:thing:42", {"u": 1})
    assert res.ok
    assert res.edited_text == "done"
    assert captured["parts"] == ["do", "thing", "42"]
    assert captured["ctx"]["u"] == 1


def test_dispatch_unknown_prefix(clean_dispatcher):
    res = tg.dispatch_callback("nope:something", {})
    assert res.ok is False
    assert "no handler" in (res.toast or "")
    # Don't clear keyboard for unknown — the user might have an
    # old message and we don't want to silently drop their buttons.
    assert res.clear_keyboard is False


def test_dispatch_handler_raises_yields_failed_result(clean_dispatcher):
    def bad(parts, ctx):
        raise RuntimeError("boom")

    tg.register_callback_handler("crash", bad)
    res = tg.dispatch_callback("crash:test", {})
    assert res.ok is False
    assert "boom" in (res.toast or "")


def test_dispatch_invalid_callback_data(clean_dispatcher):
    """No colon at all -> can't be split into prefix + tail."""
    res = tg.dispatch_callback("nocolon", {})
    assert res.ok is False
    assert "invalid" in (res.toast or "").lower()


def test_register_handler_rejects_colon_in_prefix(clean_dispatcher):
    with pytest.raises(ValueError):
        tg.register_callback_handler("bad:prefix", lambda parts, ctx: tg.CallbackResult(ok=True))


# ─── State table ────────────────────────────────────────────────────


def test_state_register_consume_round_trip(clean_dispatcher):
    sid = tg.register_state({"diff": "huge diff payload"})
    assert len(sid) == 8
    assert tg.peek_state(sid) == {"diff": "huge diff payload"}
    consumed = tg.consume_state(sid)
    assert consumed == {"diff": "huge diff payload"}
    # Single-shot — second consume returns None.
    assert tg.consume_state(sid) is None


def test_state_consume_unknown_returns_none(clean_dispatcher):
    assert tg.consume_state("ZZZZZZZZ") is None


# ─── Text helpers ───────────────────────────────────────────────────


def test_escape_html_blocks_tag_injection():
    assert tg.escape_html("<script>alert(1)</script>") == \
        "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_escape_html_handles_none():
    assert tg.escape_html("") == ""
    assert tg.escape_html(None) == ""  # type: ignore[arg-type]


def test_fmt_user_handle_with_username():
    out = tg.fmt_user_handle(123, username="lusine", full_name="Lusine S")
    assert "Lusine S" in out
    assert "@lusine" in out


def test_fmt_user_handle_no_username_falls_back_to_id():
    out = tg.fmt_user_handle(456, username="", full_name="")
    assert "456" in out


def test_fmt_user_handle_escapes_unsafe_chars():
    out = tg.fmt_user_handle(789, username="<evil>", full_name="<b>injection</b>")
    assert "<script>" not in out
    assert "&lt;" in out
