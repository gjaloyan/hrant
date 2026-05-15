"""Tiny ANSI color helper for the CLI surface.

Inspired by openclaw's Lobster palette — a warm-orange accent that
reads as "Hrant" without dragging in chalk/rich as a dependency.
Functions degrade to plain strings when:

  - stdout is not a TTY (so `hrant config list > file` stays clean)
  - $NO_COLOR is set (https://no-color.org)
  - the platform reports it can't render 24-bit color

Designed for one-liner ergonomics:

    from .cli_colors import c
    print(c.heading("Hrant configuration"))
    print(f"  {c.muted('anthropic.api_key')}  {c.success('sk-***xxxx')}")
"""
from __future__ import annotations

import os
import re
import sys


# Lobster-ish palette — warm orange (Hrant's color), with semantic
# status colors that match what users see in modern dev tools.
_PALETTE = {
    "accent":        (0xFF, 0x8C, 0x00),  # warm orange
    "accent_bright": (0xFF, 0xA8, 0x33),  # highlight orange
    "accent_dim":    (0xC8, 0x6A, 0x00),  # dim orange (links)
    "success":       (0x4F, 0xCF, 0x6A),  # green
    "warn":          (0xFF, 0xC1, 0x07),  # amber
    "error":         (0xE2, 0x3D, 0x2D),  # red
    "info":          (0x5F, 0xB3, 0xFF),  # blue
    "muted":         (0x80, 0x80, 0x80),  # gray
}


def _supports_unicode() -> bool:
    """True when stdout can render the Unicode glyphs we use for the
    config menu (checkmarks, arrows, box-drawing). False on legacy
    Windows code pages — fall back to ASCII so the terminal doesn't
    crash with UnicodeEncodeError."""
    enc = getattr(sys.stdout, "encoding", None) or ""
    enc = enc.lower()
    if "utf" in enc:
        return True
    # cp65001 is Windows UTF-8 codepage label; cp1252 etc. → ASCII.
    return "65001" in enc


# Glyphs used across the CLI. Single source of truth so an ASCII
# fallback flips them all at once.
_UNICODE = _supports_unicode()

g = type("Glyphs", (), {
    "check":      "✓"     if _UNICODE else "[ok]",
    "cross":      "✗"     if _UNICODE else "[x]",
    "arrow":      "→"     if _UNICODE else "->",
    "bullet":     "·"     if _UNICODE else "*",
    "ellipsis":   "…"     if _UNICODE else "...",
    "rule":       "─"     if _UNICODE else "-",
    "warn":       "⚠"     if _UNICODE else "!",
})()


def _supports_color() -> bool:
    """True only when emitting ANSI is safe and useful."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    # Most modern terminals on Win/macOS/Linux do truecolor. We don't
    # try to detect — if NO_COLOR isn't set and we're a TTY, ship it.
    return True


_ENABLED = _supports_color()


class _AnsiAware(str):
    """A string carrying ANSI escape sequences that knows its
    *visible* width for f-string padding. Without this, `f"{c.muted(s):<10}"`
    counted the ~17 invisible chars of an SGR escape against the
    target width and column tables looked staggered. The class
    overrides `__format__` to honour `<`/`>`/`^` alignment + width
    based on what the terminal will actually render."""

    def __format__(self, spec: str) -> str:  # noqa: D401
        if not spec:
            return str(self)
        m = _PAD_SPEC_RE.match(spec)
        if not m:
            return str(self).__format__(spec)
        align = m.group("align") or "<"
        width = int(m.group("width"))
        visible = visible_len(self)
        deficit = width - visible
        if deficit <= 0:
            return str(self)
        if align == ">":
            return (" " * deficit) + str(self)
        if align == "^":
            left = deficit // 2
            right = deficit - left
            return (" " * left) + str(self) + (" " * right)
        return str(self) + (" " * deficit)


# Spec pattern: optional alignment (<|>|^), required width.
# Doesn't handle fill chars / precision — those are rare for
# colored output and the caller can fall back to plain str.
import re as _re_for_pad
_PAD_SPEC_RE = _re_for_pad.compile(r"^(?P<align>[<>^])?(?P<width>\d+)$")


def _wrap(code: str, text: str) -> _AnsiAware:
    if not _ENABLED:
        return _AnsiAware(text)
    return _AnsiAware(f"\033[{code}m{text}\033[0m")


def _fg(rgb: tuple[int, int, int], text: str) -> _AnsiAware:
    if not _ENABLED:
        return _AnsiAware(text)
    r, g, b = rgb
    return _AnsiAware(f"\033[38;2;{r};{g};{b}m{text}\033[0m")


class _Palette:
    """Namespace-style accessor — `c.accent("hi")` reads cleanly at the
    call site, no need to import each color individually."""

    def accent(self, text: str) -> str:
        return _fg(_PALETTE["accent"], text)

    def accent_bright(self, text: str) -> str:
        return _fg(_PALETTE["accent_bright"], text)

    def accent_dim(self, text: str) -> str:
        return _fg(_PALETTE["accent_dim"], text)

    def success(self, text: str) -> str:
        return _fg(_PALETTE["success"], text)

    def warn(self, text: str) -> str:
        return _fg(_PALETTE["warn"], text)

    def error(self, text: str) -> str:
        return _fg(_PALETTE["error"], text)

    def info(self, text: str) -> str:
        return _fg(_PALETTE["info"], text)

    def muted(self, text: str) -> str:
        return _fg(_PALETTE["muted"], text)

    def bold(self, text: str) -> str:
        return _wrap("1", text)

    def dim(self, text: str) -> str:
        return _wrap("2", text)

    def heading(self, text: str) -> str:
        """Bold + accent. Use for section titles."""
        return self.bold(self.accent(text))

    @property
    def enabled(self) -> bool:
        return _ENABLED

    def hr(self, width: int = 60) -> str:
        """Horizontal rule used between sections."""
        return self.muted(g.rule * width)


c = _Palette()


# ─── ANSI-aware string padding ─────────────────────────────────────


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def visible_len(text: str) -> int:
    """Length of `text` after stripping ANSI escape sequences.
    Use when padding colored strings to fixed widths — `f"{s:<10}"`
    counts ANSI codes (~10 extra chars per color) which throws off
    column alignment. `visible_len(c.muted('abc')) == 3` even
    though the wrapped string is ~17 chars."""
    if not text:
        return 0
    return len(_ANSI_RE.sub("", text))


def pad_visible(text: str, width: int, *, align: str = "left") -> str:
    """Right- or left-pad `text` to `width` VISIBLE columns,
    ignoring ANSI escape codes. Audit #22 fix — without this the
    `hrant jobs list` / `graph stats` tables misaligned whenever
    colors were enabled."""
    deficit = width - visible_len(text)
    if deficit <= 0:
        return text
    if align == "right":
        return (" " * deficit) + text
    return text + (" " * deficit)
