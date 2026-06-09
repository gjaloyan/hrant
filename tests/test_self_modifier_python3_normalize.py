"""self_modifier normalizes `python ...` -> `python3 ...` in test commands.

Background: the 2026-06-09 self-modification trial (proposal e7dcd0e3aa)
rolled back its patch because the agent's `test_commands` specified
`python -m py_compile backend/api/status.py` but the host had no
`python` symlink — only `python3`. The applier ran the command, hit
`[Errno 2] No such file or directory: 'python'`, and rolled back what
was otherwise a perfectly valid patch.

Fix:
  1. `_normalize_test_command(cmd)` rewrites a `python` first-token to
     `python3`. Tokens past position 0 are left alone (so `pytest
     some/python.py` is unchanged).
  2. `Proposal.__init__` runs every entry of `test_commands` through
     this normalizer — covers both LLM-generated proposals and legacy
     ones loaded from disk.
  3. `_ALLOWED_TEST_PREFIXES` accepts both `python` and `python3`
     prefixes so unmigrated proposals still pass the validator (the
     applier ends up running the normalized command anyway).
"""
from __future__ import annotations


def test_normalize_python_to_python3_simple():
    """`python -m py_compile <file>` -> `python3 -m py_compile <file>`."""
    from backend.self_modifier import _normalize_test_command
    out = _normalize_test_command("python -m py_compile backend/api/status.py")
    assert out == "python3 -m py_compile backend/api/status.py"


def test_normalize_python_module_with_args():
    """`python -m pytest tests/test_X.py -q` -> python3 form."""
    from backend.self_modifier import _normalize_test_command
    out = _normalize_test_command("python -m pytest tests/test_X.py -q")
    assert out == "python3 -m pytest tests/test_X.py -q"


def test_normalize_leaves_python3_unchanged():
    """An already-python3 command passes through untouched."""
    from backend.self_modifier import _normalize_test_command
    cmd = "python3 -m py_compile backend/api/status.py"
    assert _normalize_test_command(cmd) == cmd


def test_normalize_leaves_pytest_unchanged():
    """`pytest tests/...` is not rewritten — only first-token `python`
    triggers the rewrite."""
    from backend.self_modifier import _normalize_test_command
    cmd = "pytest tests/test_X.py -q"
    assert _normalize_test_command(cmd) == cmd


def test_normalize_does_not_touch_python_in_arg_position():
    """`pytest some/python.py` keeps the `python` substring inside an
    argument — only the FIRST token is rewritten."""
    from backend.self_modifier import _normalize_test_command
    cmd = "pytest some/python.py"
    assert _normalize_test_command(cmd) == cmd


def test_normalize_handles_empty_and_malformed():
    """Empty / whitespace / unparseable input -> returned unchanged."""
    from backend.self_modifier import _normalize_test_command
    assert _normalize_test_command("") == ""
    # Unclosed quote -> shlex raises -> we return as-is for the
    # validator to reject downstream.
    assert _normalize_test_command('python "unclosed') == 'python "unclosed'


def test_proposal_init_normalizes_test_commands():
    """Every entry of `test_commands` passed to Proposal.__init__ is
    normalized in place. Covers both LLM-generated proposals and
    legacy proposals loaded from disk via Proposal.from_dict."""
    from backend.self_modifier import Proposal
    p = Proposal(
        description="add /api/uptime",
        test_commands=[
            "python -m py_compile backend/api/status.py",
            "python -m pytest tests/test_uptime.py -q",
            "pytest -q",  # already correct shape
            "python3 -c 'import backend.main'",  # already python3
        ],
    )
    assert p.test_commands == [
        "python3 -m py_compile backend/api/status.py",
        "python3 -m pytest tests/test_uptime.py -q",
        "pytest -q",
        "python3 -c 'import backend.main'",
    ]


def test_proposal_init_with_empty_test_commands():
    """No test_commands -> empty list. None -> empty list."""
    from backend.self_modifier import Proposal
    assert Proposal(description="x").test_commands == []
    assert Proposal(description="x", test_commands=[]).test_commands == []
    assert Proposal(description="x", test_commands=None).test_commands == []


def test_allowed_prefixes_accept_python3():
    """The validator must accept `python3` prefixes so normalized
    commands pass through without being rejected."""
    from backend.self_modifier import _validate_test_command
    ok, _reason, _argv = _validate_test_command(
        "python3 -m py_compile backend/api/status.py"
    )
    assert ok is True
    ok, _reason, _argv = _validate_test_command(
        "python3 script.py"
    )
    assert ok is True
    ok, _reason, _argv = _validate_test_command(
        "python3 -c 'print(1)'"
    )
    assert ok is True


def test_allowed_prefixes_still_accept_legacy_python():
    """Back-compat: a legacy proposal loaded from disk that wasn't
    normalized must still pass validation (the validator runs BEFORE
    the apply path normalizes via Proposal.__init__, but defense-in-
    depth: also accept the un-normalized form)."""
    from backend.self_modifier import _validate_test_command
    ok, _reason, _argv = _validate_test_command(
        "python -m py_compile backend/api/status.py"
    )
    assert ok is True


def test_analyze_system_prompt_says_python3():
    """Sanity: the LLM-facing ANALYZE_SYSTEM prompt instructs the
    model to use `python3`, not `python`. This is the second prong of
    the fix — the LLM ought to emit python3 from the start; the
    normalizer is the safety net."""
    from backend.self_modifier import ANALYZE_SYSTEM
    assert "python3" in ANALYZE_SYSTEM
    # The example test_commands line must use python3, not python.
    assert '"python3 -m pytest' in ANALYZE_SYSTEM
