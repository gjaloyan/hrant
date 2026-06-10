"""Telegram inline-keyboard helpers — Hermes-style approval prompts.

Two pieces:

  - `InlineButtonSet`: declarative builder for an inline keyboard.
    Each button carries a `callback_data` string ≤ 64 bytes (the
    Telegram limit). The set serialises into the python-telegram-bot
    `InlineKeyboardMarkup` shape on demand, so the rest of the code
    can construct keyboards without importing `telegram.*` directly.

  - A module-level callback dispatcher. Each callback_data starts
    with a short prefix (e.g. `pair:`); handlers registered with
    `register_callback_handler("pair", fn)` get invoked with the
    parsed parts. Used by `channels.py::_handle_callback_query`.

The Telegram callback_data limit is 64 BYTES, not characters — keep
prefixes short. The audit checked our pairing codes are 8 chars from
a 32-char alphabet, so `pair:approve:once:ABCDEFGH` is 25 bytes.
Plenty of headroom.

For payloads too large for callback_data (long diffs, multi-file
proposals), use `register_state(payload) → short_id` + dispatch on
the short_id. State is in-memory only (lost across restarts); for
durable approvals stash to the relevant on-disk store
(pairing.json, jobs/, etc.) and reference that.
"""
from __future__ import annotations

import html as _html
import logging
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# Telegram BotAPI limit on callback_data.
CALLBACK_DATA_MAX_BYTES = 64


# ─── Button + ButtonSet builders ────────────────────────────────────


@dataclass(frozen=True)
class InlineButton:
    """A single inline keyboard button.

    Two shapes — text-driven with `callback_data` (sends a
    CallbackQuery event when pressed) or `url` (opens browser).
    Telegram allows additional kinds (webapp, login_url, switch_inline),
    omitted here until we actually need them."""
    label: str
    callback_data: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self):
        if not self.callback_data and not self.url:
            raise ValueError("InlineButton needs callback_data or url")
        if self.callback_data and len(self.callback_data.encode("utf-8")) > CALLBACK_DATA_MAX_BYTES:
            raise ValueError(
                f"callback_data {self.callback_data!r} exceeds {CALLBACK_DATA_MAX_BYTES} bytes"
            )

    def to_dict(self) -> dict:
        d: dict = {"text": self.label}
        if self.callback_data:
            d["callback_data"] = self.callback_data
        if self.url:
            d["url"] = self.url
        return d


@dataclass
class InlineButtonSet:
    """Builder for a multi-row inline keyboard."""
    rows: list[list[InlineButton]] = field(default_factory=list)

    def row(self, *buttons: InlineButton) -> "InlineButtonSet":
        """Append a row of buttons. Returns self for chaining."""
        self.rows.append(list(buttons))
        return self

    def to_markup(self):
        """Convert to a python-telegram-bot `InlineKeyboardMarkup`.

        Late-imported so the module remains importable in tests that
        don't have telegram-bot installed (and so the import cycle
        with channels.py stays one-way)."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [
                InlineKeyboardButton(
                    text=b.label,
                    callback_data=b.callback_data,
                    url=b.url,
                )
                for b in row
            ]
            for row in self.rows
        ]
        return InlineKeyboardMarkup(keyboard)


# ─── Callback dispatcher ────────────────────────────────────────────


_CallbackHandler = Callable[[list[str], dict], "CallbackResult"]


@dataclass
class CallbackResult:
    """Returned by a handler. The bot uses these fields to update the
    original message (or push a transient toast)."""
    ok: bool
    # Replacement text for the original message. None → don't edit.
    edited_text: Optional[str] = None
    # Transient pop-up shown to the clicker (max ~200 chars).
    toast: Optional[str] = None
    # When True the bot tells the dispatcher to drop the inline
    # keyboard from the original message (a no-op when edited_text
    # is provided, since editMessageText replaces the keyboard too).
    clear_keyboard: bool = True
    # Extra message to send AFTER the edit/toast — used when the
    # handler wants to surface additional content (e.g. a "Show diff"
    # button that returns the full diff in a follow-up <pre> block
    # while keeping the original approval prompt intact). HTML-mode.
    followup_text: Optional[str] = None


_HANDLERS: dict[str, _CallbackHandler] = {}
_HANDLERS_LOCK = threading.Lock()


def register_callback_handler(prefix: str, handler: _CallbackHandler) -> None:
    """Register a handler for callback_data values starting with
    `<prefix>:`. The handler receives `(parts, ctx)` where `parts`
    is the colon-split tail of the data and `ctx` is a free-form
    context dict (the dispatcher injects fields like `user_id`,
    `chat_id`, `message_id`). Idempotent — registering twice replaces.
    """
    if not prefix or ":" in prefix:
        raise ValueError("prefix must be non-empty and contain no ':'")
    with _HANDLERS_LOCK:
        _HANDLERS[prefix] = handler


def unregister_callback_handler(prefix: str) -> None:
    with _HANDLERS_LOCK:
        _HANDLERS.pop(prefix, None)


def dispatch_callback(callback_data: str, ctx: dict) -> CallbackResult:
    """Find the right handler for `callback_data` and invoke it.

    Returns a CallbackResult the channel layer uses to edit the
    original message and/or show a transient toast. Unknown prefix
    or handler-raised exceptions surface as a failed CallbackResult
    (with `toast` set to the error string) rather than propagating
    upstream — a single bad button shouldn't kill the bot.
    """
    if not callback_data or ":" not in callback_data:
        return CallbackResult(ok=False, toast="invalid callback data", clear_keyboard=False)
    prefix, _, _ = callback_data.partition(":")
    with _HANDLERS_LOCK:
        handler = _HANDLERS.get(prefix)
    if handler is None:
        return CallbackResult(ok=False, toast=f"no handler for {prefix!r}", clear_keyboard=False)
    parts = callback_data.split(":")[1:]
    try:
        return handler(parts, ctx)
    except Exception as e:
        log.exception("callback %r failed", callback_data)
        return CallbackResult(ok=False, toast=f"error: {type(e).__name__}: {e}", clear_keyboard=False)


# ─── In-memory state table (for callbacks too big for callback_data) ─


_STATE: dict[str, Any] = {}
_STATE_LOCK = threading.Lock()


def register_state(payload: Any) -> str:
    """Stash a Python value, return a short id usable in callback_data.

    Used when the action payload (e.g. a self-mod diff, a scheduled
    message preview) doesn't fit into the 64-byte callback_data limit.
    Pair with `consume_state(id)` from inside a handler.

    Returns an 8-char base32 id (32^8 ≈ 10^12 — plenty for the small
    window an approval stays open). The id is intentionally NOT
    sequential so a stale screenshot of a previous approval can't be
    used to fish for in-flight ones.
    """
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    sid = "".join(secrets.choice(alphabet) for _ in range(8))
    with _STATE_LOCK:
        _STATE[sid] = payload
    return sid


def consume_state(sid: str) -> Optional[Any]:
    """Pop and return the stashed payload, or None if not found.

    Approvals are single-shot — once a button is pressed the state
    is gone. If the user re-presses the same button on a now-stale
    message, consume_state returns None and the handler should
    show a toast explaining the link has expired."""
    with _STATE_LOCK:
        return _STATE.pop(sid, None)


def peek_state(sid: str) -> Optional[Any]:
    with _STATE_LOCK:
        return _STATE.get(sid)


# ─── Text helpers ───────────────────────────────────────────────────


def escape_html(text: str) -> str:
    """Escape text for parse_mode=HTML. Re-export of stdlib
    html.escape so callers don't have to import it separately."""
    return _html.escape(text or "", quote=False)


def fmt_user_handle(user_id: int | str, username: str = "", full_name: str = "") -> str:
    """Render a Telegram user identity for HTML messages. Prefers
    a clickable username; falls back to id."""
    name = full_name or username or str(user_id)
    safe_name = escape_html(name)
    handle = f"@{escape_html(username)}" if username else f"id {escape_html(str(user_id))}"
    return f"<b>{safe_name}</b> ({handle})"
