"""The output cap must bound the buffer, not just the reply.

From the GPT-6 Astra audit, 2026-09-05, security finding 6. Every exec
tool called `subprocess.run(capture_output=True)` and clipped
afterwards. The auditor printed 1 MiB and caught the parent holding all
1,048,576 characters before reducing them to 204,820 for the answer.
`run_python`'s own comment said the 200 KB limit prevented an OOM; it
described the answer size. `terminal_exec` caps at 16 KB, so the same
input was a 64x overshoot.

A runaway print loop or a chatty CLI is a likelier cause than an
attacker, which is why this is a robustness fix.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from backend.tools.bounded_capture import run_capped


def _py(code: str):
    return [sys.executable, "-c", code]


def test_a_megabyte_of_output_does_not_reach_the_buffer():
    cap = 4096
    r = run_capped(_py("print('x' * 1_000_000)"), max_bytes=cap, timeout=60)
    assert len(r.stdout) <= cap, f"kept {len(r.stdout)} bytes for a {cap} cap"
    assert r.stdout_truncated is True
    assert r.stdout_dropped > 900_000
    assert r.returncode == 0


def test_both_streams_are_bounded_at_once():
    """The audit asked for this case by name: loud on stdout AND stderr."""
    code = (
        "import sys;"
        "sys.stdout.write('o' * 500_000);"
        "sys.stderr.write('e' * 500_000)"
    )
    r = run_capped(_py(code), max_bytes=2048, stderr_max_bytes=1024, timeout=60)
    assert len(r.stdout) <= 2048
    assert len(r.stderr) <= 1024
    assert r.stdout_truncated and r.stderr_truncated


def test_the_child_is_not_deadlocked_by_a_full_pipe():
    """Past the cap the reader keeps reading and discards. If it simply
    stopped, the child would block on a full pipe and only the timeout
    would end it — the process must exit on its own, well inside it."""
    r = run_capped(_py("print('y' * 2_000_000)"), max_bytes=256, timeout=30)
    assert r.returncode == 0, "the child finished rather than being killed"
    assert len(r.stdout) <= 256


def test_short_output_is_untouched_and_not_flagged():
    r = run_capped(_py("print('hello')"), max_bytes=4096, timeout=30)
    assert r.stdout.strip() == b"hello"
    assert r.truncated is False
    assert r.stdout_dropped == 0


def test_a_nonzero_exit_is_reported():
    r = run_capped(_py("import sys; sys.exit(3)"), max_bytes=1024, timeout=30)
    assert r.returncode == 3


def test_timeout_kills_the_child_and_raises_like_run():
    with pytest.raises(subprocess.TimeoutExpired):
        run_capped(_py("import time; time.sleep(30)"),
                   max_bytes=1024, timeout=1)


def test_terminal_exec_output_is_bounded_end_to_end():
    """Through the real tool, not the helper."""
    from backend.tools import terminal_exec as te
    res = te.run_terminal(
        "python -c \"print('z' * 400000)\"", timeout_seconds=60)
    if res.exit_code != 0:          # no `python` on this box
        pytest.skip("no python on PATH for the child process")
    assert len(res.stdout) <= te.MAX_OUTPUT_BYTES
    assert res.truncated is True
