"""Arrow-key navigable menu for the CLI — what `openclaw` uses via
inquirer, rebuilt here without adding a dependency.

Behaviour:
  - ↑ / ↓  move selection
  - Enter  pick the highlighted row
  - q / Esc / Ctrl-C  cancel (returns -1)
  - On non-TTY, falls back to a numbered prompt so cron / pipes /
    captured-stdin tests still work.

Cross-platform single-key read:
  - Linux/macOS: termios + sys.stdin (raw mode), escape sequences
    (\\x1b[A = up, \\x1b[B = down)
  - Windows: msvcrt.getch, special-key prefix \\xe0 followed by H/P

Rendering:
  Each frame redraws all option rows in place. Cursor is moved up by
  the number of lines printed last time, then each line is cleared
  with \\033[K and re-printed. Cursor is hidden during navigation so
  the redraw isn't visually noisy.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from .cli_colors import c, g


# Sentinel returned when the user cancels (q / Esc / Ctrl-C).
CANCELLED = -1


# ─── Single-key read ──────────────────────────────────────────────────


def _read_key_unix() -> str:
    """Linux/macOS: raw terminal, read one key. Returns one of:
    'up', 'down', 'enter', 'esc', 'q', 'ctrl_c', or the raw char."""
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Escape sequence — peek two more chars without blocking
            # if no follow-up arrives (lone Esc).
            import select
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                return "esc"
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            if seq == "[C":
                return "right"
            if seq == "[D":
                return "left"
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl_c"
        if ch in ("q", "Q"):
            return "q"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_windows() -> str:
    import msvcrt
    ch = msvcrt.getch()
    # Special-key prefix on Windows: \xe0 (arrows / function keys),
    # \x00 (extended keys).
    if ch in (b"\xe0", b"\x00"):
        ch2 = msvcrt.getch()
        return {
            b"H": "up",
            b"P": "down",
            b"K": "left",
            b"M": "right",
        }.get(ch2, "")
    if ch in (b"\r", b"\n"):
        return "enter"
    if ch == b"\x1b":
        return "esc"
    if ch == b"\x03":
        return "ctrl_c"
    try:
        s = ch.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    if s in ("q", "Q"):
        return "q"
    return s


def _read_key() -> str:
    if os.name == "nt":
        return _read_key_windows()
    return _read_key_unix()


# ─── Cursor + line control ────────────────────────────────────────────


def _hide_cursor() -> None:
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def _show_cursor() -> None:
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def _clear_lines(n: int) -> None:
    """Move cursor up `n` lines and clear each one."""
    for _ in range(n):
        sys.stdout.write("\033[1A\033[K")
    sys.stdout.flush()


# ─── Numbered fallback (non-TTY) ──────────────────────────────────────


def _numbered_fallback(
    prompt: str,
    options: list[tuple[str, str]],
    default_idx: int,
) -> int:
    """Non-TTY path: print the menu once + a `[default]` prompt, read
    a single line. Returns default on empty input / EOF. Cancel is
    not available without a real terminal."""
    print(f"  {prompt}")
    print()
    for i, (label, desc) in enumerate(options, start=1):
        marker = c.accent_bright(g.arrow) if (i - 1) == default_idx else " "
        line = f"    {marker} {i}) {c.bold(label)}"
        if desc:
            line += f"  {c.muted(desc)}"
        print(line)
    print()
    try:
        raw = input(f"  {c.muted('Choice')} [{c.accent(str(default_idx + 1))}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default_idx
    if not raw:
        return default_idx
    try:
        idx = int(raw) - 1
    except ValueError:
        return default_idx
    if 0 <= idx < len(options):
        return idx
    return default_idx


# ─── Public API ───────────────────────────────────────────────────────


def select(
    prompt: str,
    options: list[tuple[str, str]],
    default_idx: int = 0,
    *,
    allow_cancel: bool = True,
) -> int:
    """Render an arrow-key navigated menu. Returns the chosen index
    or CANCELLED (-1) when the user presses q / Esc / Ctrl-C.

    `options` is a list of (label, description) tuples. Description
    may be empty.

    Falls back to a numbered prompt when stdin/stdout aren't TTYs
    (cron, piped capture, tests) — uses the input() readline path
    so existing test fixtures that stub stdin still work."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _numbered_fallback(prompt, options, default_idx)

    sel = max(0, min(default_idx, len(options) - 1))
    print(f"  {prompt}")
    print()
    lines_drawn = 0

    def draw(first: bool = False) -> None:
        nonlocal lines_drawn
        if not first:
            _clear_lines(lines_drawn)
        lines_drawn = 0
        for i, (label, desc) in enumerate(options):
            if i == sel:
                arrow = c.accent_bright(g.arrow)
                lbl = c.accent_bright(c.bold(label))
            else:
                arrow = " "
                lbl = c.bold(label)
            line = f"    {arrow} {lbl}"
            if desc:
                line += f"  {c.muted(desc)}"
            sys.stdout.write(line + "\n")
            lines_drawn += 1
        hint_parts = [
            f"{c.muted('↑/↓')} {c.muted('navigate')}",
            f"{c.muted('Enter')} {c.muted('select')}",
        ]
        if allow_cancel:
            hint_parts.append(f"{c.muted('q')} {c.muted('cancel')}")
        sys.stdout.write("  " + "   ".join(hint_parts) + "\n")
        lines_drawn += 1
        sys.stdout.flush()

    _hide_cursor()
    try:
        draw(first=True)
        while True:
            key = _read_key()
            if key == "up":
                sel = (sel - 1) % len(options)
                draw()
            elif key == "down":
                sel = (sel + 1) % len(options)
                draw()
            elif key == "enter":
                # Repaint without the hint line so the chosen row stays
                # visible in scrollback as a record of what was picked.
                _clear_lines(lines_drawn)
                lines_drawn = 0
                for i, (label, desc) in enumerate(options):
                    if i == sel:
                        marker = c.success(g.check)
                        lbl = c.success(c.bold(label))
                    else:
                        marker = " "
                        lbl = c.muted(label)
                    line = f"    {marker} {lbl}"
                    if desc and i == sel:
                        line += f"  {c.muted(desc)}"
                    sys.stdout.write(line + "\n")
                    lines_drawn += 1
                sys.stdout.flush()
                return sel
            elif key in ("q", "esc", "ctrl_c"):
                if not allow_cancel and key != "ctrl_c":
                    continue
                # Repaint as muted (nothing chosen).
                _clear_lines(lines_drawn)
                for i, (label, _) in enumerate(options):
                    sys.stdout.write(f"    {c.muted(label)}\n")
                sys.stdout.write(f"  {c.muted('cancelled')}\n")
                sys.stdout.flush()
                if key == "ctrl_c":
                    # Re-raise so the wizard loop knows to fully exit.
                    raise KeyboardInterrupt
                return CANCELLED
    finally:
        _show_cursor()
