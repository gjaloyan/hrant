"""A sandbox that isn't one must say so before it runs anything.

From the GPT-6 Astra audit, 2026-09-05, security finding 5. With no
isolator on PATH the degraded tier ran `sh -c` immediately: real
filesystem, real network, even when the caller passed `network=False`.
The result's `network` field echoed the REQUEST, so a caller reading it
back saw its own wish reflected as a guarantee, and the warning about
containment arrived in `notes` after the command had already run.

That matters more here than in most tools because this one's own
description offers it for unknown archives and freshly downloaded
binaries — exactly the cases where the decision has to be made before
execution, not after.

Refusing costs the agent nothing: `terminal_exec` is a full shell with
no gate, so anything it genuinely wants to run uncontained it can still
run deliberately. What it can no longer do is believe it was protected.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.tools import sandbox as sb


def test_no_isolator_means_the_command_is_not_run():
    """The audit's acceptance test: subprocess is never reached."""
    with patch.object(sb, "detect_tier", return_value=sb.TIER_DEGRADED), \
         patch("backend.tools.bounded_capture.run_capped") as fake_run:
        res = sb.sandbox_exec("echo hi")
    fake_run.assert_not_called()
    assert res.ok is False
    assert res.isolation == sb.TIER_UNAVAILABLE
    assert "NOT RUN" in res.stderr
    assert res.fs_isolated is False and res.network_contained is False


def test_the_refusal_names_the_ways_forward():
    """A refusal the agent cannot act on is just a dead end."""
    with patch.object(sb, "detect_tier", return_value=sb.TIER_DEGRADED), \
         patch("backend.tools.bounded_capture.run_capped"):
        res = sb.sandbox_exec("echo hi")
    for hint in ("bubblewrap", "terminal_exec", "allow_degraded"):
        assert hint in res.stderr, hint


def test_an_explicit_opt_in_still_runs():
    """The owner's rule is that the agent stays free. It may accept an
    uncontained run — knowingly, and on the record."""
    with patch.object(sb, "detect_tier", return_value=sb.TIER_DEGRADED):
        res = sb.sandbox_exec("echo hi", allow_degraded=True)
    assert res.isolation == sb.TIER_DEGRADED
    assert res.ok is True
    assert res.network_contained is False
    assert any("DEGRADED tier accepted" in n for n in res.notes)


def test_network_false_is_not_reported_as_contained_when_it_is_not():
    """`network` echoes the request; `network_contained` reports what
    the tier enforced. They used to be the same field."""
    with patch.object(sb, "detect_tier", return_value=sb.TIER_DEGRADED):
        res = sb.sandbox_exec("echo hi", network=False, allow_degraded=True)
    assert res.network is False, "the request is preserved"
    assert res.network_contained is False, "and it was not honoured"
    assert any("not enforced" in n for n in res.notes)
    d = res.to_dict()
    assert d["requested_network"] is False and d["network_contained"] is False


@pytest.mark.parametrize("tier,fs,net", [
    (sb.TIER_BWRAP, True, True),
    (sb.TIER_FIREJAIL, True, True),
    # unshare gets a fresh netns and leaves the real filesystem visible —
    # `_unshare_argv` says so in its own docstring.
    (sb.TIER_UNSHARE, False, True),
    (sb.TIER_DEGRADED, False, False),
])
def test_each_tier_reports_what_it_actually_enforces(tier, fs, net):
    from backend.tools.bounded_capture import CappedOutput
    done = CappedOutput(stdout=b"", stderr=b"", returncode=0)
    with patch.object(sb, "detect_tier", return_value=tier), \
         patch("backend.tools.bounded_capture.run_capped", return_value=done):
        res = sb.sandbox_exec("echo hi", network=False, allow_degraded=True)
    assert res.fs_isolated is fs
    assert res.network_contained is net
