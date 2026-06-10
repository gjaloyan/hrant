"""Catastrophic denylist must split on newline / carriage-return / NBSP.

Audit 2026-06-10 (C1, N4) caught two bypass classes:

  1. `terminal_exec("echo ok\\nrm -rf /")` — `/bin/bash -c` treats
     newlines as command separators, so `rm -rf /` actually executes.
     `_command_segments` only split on `; | & && ||` and missed `\\n` /
     `\\r`. After the fix the per-segment scanners see `["rm", "-rf", "/"]`
     and the rm-rf danger check fires.

  2. `terminal_exec("rm\\u00a0-rf\\u00a0/")` — NBSP separators are not
     ASCII whitespace, so `shlex.split` fuses them into a single token
     `"rm -rf /"` and `inner[0] != "rm"`. Bash itself tokenizes on
     Unicode-whitespace, so the command would run. `_normalize_whitespace`
     coerces every `Zs` codepoint to ASCII space before shlex.
"""
from __future__ import annotations


def test_newline_separates_segments_for_rm_rf_root():
    """`echo ok\\nrm -rf /` must be refused on the rm-rf segment."""
    from backend.tools.terminal_exec import _check_dangerous_command
    reason = _check_dangerous_command("echo ok\nrm -rf /")
    assert reason is not None
    assert "rm -rf" in reason.lower() or "rm" in reason.lower()


def test_carriage_return_also_separates():
    """`\\r` is the other common shell separator (legacy Windows-style
    line endings) — bash treats it the same as `\\n`."""
    from backend.tools.terminal_exec import _check_dangerous_command
    reason = _check_dangerous_command("echo ok\rrm -rf ~")
    assert reason is not None
    assert "rm" in reason.lower()


def test_safe_newline_command_still_allowed():
    """A multi-line script that doesn't contain catastrophic patterns
    is still allowed — splitting on newline doesn't false-positive
    on normal heredocs / multi-line commands."""
    from backend.tools.terminal_exec import _check_dangerous_command
    reason = _check_dangerous_command("echo line1\necho line2\nls -la")
    assert reason is None


def test_nbsp_separator_still_caught():
    """`rm\\u00a0-rf\\u00a0/` with NBSP separators must be refused.
    Without the Unicode-whitespace normalize step, shlex would fuse
    the tokens and `inner[0] != 'rm'` would let it through."""
    from backend.tools.terminal_exec import _check_dangerous_command
    reason = _check_dangerous_command("rm -rf /")
    assert reason is not None
    assert "rm" in reason.lower()


def test_en_space_separator_still_caught():
    """En-space (U+2002) is another Zs-category codepoint — same
    bypass as NBSP. Pin the normalizer covers the whole Zs class."""
    from backend.tools.terminal_exec import _check_dangerous_command
    reason = _check_dangerous_command("rm -rf /")
    assert reason is not None
    assert "rm" in reason.lower()


def test_normalize_whitespace_helper_preserves_ascii():
    """The normalizer is a no-op on ASCII text — pin the contract
    so a future caller doesn't accidentally use it on something it
    shouldn't touch."""
    from backend.tools.terminal_exec import _normalize_whitespace
    assert _normalize_whitespace("rm -rf /") == "rm -rf /"
    assert _normalize_whitespace("") == ""
    assert _normalize_whitespace("a\tb\nc") == "a\tb\nc"  # tab/newline are Cc, not Zs


def test_dd_block_device_after_newline_caught():
    """The per-segment scanners (dd, mkfs, shred, kill, chmod) all
    benefit from the newline split. Pin one representative."""
    from backend.tools.terminal_exec import _check_dangerous_command
    reason = _check_dangerous_command("echo ok\ndd if=/dev/zero of=/dev/sda")
    assert reason is not None
    assert "dd" in reason.lower() or "block device" in reason.lower()
