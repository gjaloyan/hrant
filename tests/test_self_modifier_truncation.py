"""self_modifier: large modules must be truncated to a useful slice
of head + tail, not just the head. Module-level singletons, registry
hooks, and `if __name__ == "__main__"` blocks live at the bottom of
files and matter for self-analysis suggestions."""
from __future__ import annotations
from unittest.mock import patch

import pytest

from backend.self_modifier import SelfModifier


def _module_text(*, head_marker: str, body_filler_chars: int, tail_marker: str) -> str:
    """Build a synthetic module with a recognizable head, a long
    middle, and a recognizable tail. Used to verify that truncation
    keeps both ends."""
    middle = "x" * body_filler_chars
    return f"# HEAD\n{head_marker}\n\n{middle}\n\n# TAIL\n{tail_marker}\n"


def test_short_files_are_not_truncated(tmp_path, monkeypatch):
    code = "class Foo:\n    pass\n"
    f = tmp_path / "small_module.py"
    f.write_text(code, encoding="utf-8")

    sm = SelfModifier()
    captured: dict[str, str] = {}

    class _Router:
        def call_json(self, task_type, system, user, **kw):
            captured["user"] = user
            return {"proposals": []}

    sm._backend_dir = f.parent
    with patch("backend.self_modifier.router", return_value=_Router()):
        sm.analyze_module("small_module")

    assert "class Foo" in captured["user"]
    assert "(truncated)" not in captured["user"]
    assert "omitted" not in captured["user"]


def test_large_files_keep_both_head_and_tail(tmp_path, monkeypatch):
    """When a file exceeds the cap, both the head AND the tail must
    survive truncation, with a marker explaining what's missing."""
    code = _module_text(
        head_marker="HEAD_SENTINEL_VERY_FIRST_FUNCTION",
        body_filler_chars=40000,  # well over the 30000 cap
        tail_marker="TAIL_SENTINEL_MODULE_LEVEL_SINGLETON = X()",
    )
    f = tmp_path / "big_module.py"
    f.write_text(code, encoding="utf-8")

    sm = SelfModifier()
    captured: dict[str, str] = {}

    class _Router:
        def call_json(self, task_type, system, user, **kw):
            captured["user"] = user
            return {"proposals": []}

    sm._backend_dir = f.parent
    with patch("backend.self_modifier.router", return_value=_Router()):
        sm.analyze_module("big_module")

    user = captured["user"]
    assert "HEAD_SENTINEL_VERY_FIRST_FUNCTION" in user, "head must be kept"
    assert "TAIL_SENTINEL_MODULE_LEVEL_SINGLETON" in user, "tail must be kept"
    assert "omitted from middle of file" in user, "marker must explain truncation"


def test_truncation_produces_total_under_max(tmp_path, monkeypatch):
    """The post-truncation prompt body for the source code MUST be
    under the documented cap (30000) — otherwise the bump didn't
    actually bound anything."""
    code = "x" * 100_000
    f = tmp_path / "huge_module.py"
    f.write_text(code, encoding="utf-8")

    sm = SelfModifier()
    captured: dict[str, str] = {}

    class _Router:
        def call_json(self, task_type, system, user, **kw):
            captured["user"] = user
            return {"proposals": []}

    sm._backend_dir = f.parent
    with patch("backend.self_modifier.router", return_value=_Router()):
        sm.analyze_module("huge_module")

    # Extract the SOURCE CODE block from the user prompt and verify length.
    user = captured["user"]
    body = user.split("```python\n", 1)[1].split("\n```", 1)[0]
    assert len(body) <= 30200, f"truncated body {len(body)} chars exceeds cap"
