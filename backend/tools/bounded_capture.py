"""Run a subprocess without letting its output into the agent's memory.

2026-09-05 audit, security finding 6. Every exec tool used
`subprocess.run(capture_output=True)` and clipped afterwards, so the
cap described the ANSWER, not the buffer. Measured by the auditor: a
child printing 1 MiB was held whole — 1,048,576 characters — and then
reduced to 204,820 for the reply. `run_python`'s comment about the
200 KB limit preventing an OOM described something the code did not do,
and `terminal_exec`'s 16 KB cap meant a 64x overshoot on the same input.

That is a robustness problem before it is a security one: a chatty CLI,
a runaway print loop, or several jobs at once can cost the agent far
more memory than the visible output suggests.

Here the pipes are drained on their own threads into bounded buffers.
Past the cap the reader keeps reading and throws the bytes away — it
must not simply stop, or the child blocks forever on a full pipe and
the timeout becomes the only exit.
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

_CHUNK = 65536


@dataclass
class CappedOutput:
    stdout: bytes
    stderr: bytes
    returncode: int
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_dropped: int = 0
    stderr_dropped: int = 0

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated


def _drain(pipe, cap: int, out: list) -> None:
    """Read `pipe` to EOF, keeping at most `cap` bytes."""
    kept = bytearray()
    dropped = 0
    try:
        while True:
            chunk = pipe.read(_CHUNK)
            if not chunk:
                break
            room = cap - len(kept)
            if room > 0:
                kept += chunk[:room]
                dropped += len(chunk) - min(room, len(chunk))
            else:
                dropped += len(chunk)
    except (OSError, ValueError):
        # Pipe closed under us (kill on timeout). Keep what we have.
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass
    out.append((bytes(kept), dropped))


def run_capped(
    args,
    *,
    max_bytes: int,
    timeout: float | None = None,
    stderr_max_bytes: int | None = None,
    **popen_kwargs,
) -> CappedOutput:
    """`subprocess.run(capture_output=True)` with a hard memory ceiling.

    Raises `subprocess.TimeoutExpired` on timeout, having killed the
    child first — same contract as `subprocess.run`, so callers keep
    their existing handling. `text`/`capture_output`/`stdout`/`stderr`
    are ignored if passed: output is always captured, always as bytes.
    """
    for key in ("capture_output", "text", "stdout", "stderr", "encoding",
                "errors", "universal_newlines"):
        popen_kwargs.pop(key, None)
    err_cap = max_bytes if stderr_max_bytes is None else stderr_max_bytes

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    out_sink: list = []
    err_sink: list = []
    t_out = threading.Thread(
        target=_drain, args=(proc.stdout, max_bytes, out_sink), daemon=True)
    t_err = threading.Thread(
        target=_drain, args=(proc.stderr, err_cap, err_sink), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        t_out.join(5)
        t_err.join(5)
        raise
    # The child is gone; the readers see EOF promptly.
    t_out.join(10)
    t_err.join(10)

    stdout, out_dropped = out_sink[0] if out_sink else (b"", 0)
    stderr, err_dropped = err_sink[0] if err_sink else (b"", 0)
    return CappedOutput(
        stdout=stdout,
        stderr=stderr,
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout_truncated=out_dropped > 0,
        stderr_truncated=err_dropped > 0,
        stdout_dropped=out_dropped,
        stderr_dropped=err_dropped,
    )
