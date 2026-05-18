"""Tests for the analyze_image tool + iteration-cap recovery path.

Two-thread fix from the logo-task post-mortem:
  - analyze_image: a routable multimodal-LLM call. Replaces hand-
    rolled OpenCV pixel classifiers in skills that need to read
    overlay coordinates / colours / text from a frame.
  - _rewrite_xml_tool_call_dump: when the LLM hits max_iterations
    and dumps the next intended call as `<tool_call name="…">` XML
    text, we rewrite the answer to a status report so the user
    sees actionable text instead of broken-looking markup.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ─── analyze_image (function + tool handler) ─────────────────────────


@pytest.fixture
def isolated_attachments(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    return tmp_path


def test_analyze_image_missing_sha_returns_error_text(isolated_attachments, monkeypatch):
    from backend.tools.analyze_image import analyze_image
    out = analyze_image("deadbeef" * 8, "where is the logo?")
    assert "not in attachment store" in out


def test_analyze_image_video_sha_rejected(isolated_attachments, monkeypatch):
    """A video sha must NOT be passed to analyze_image — the LLM
    needs a frame, not the encoded clip. Rejection should be loud."""
    from backend.attachments import ATTACHMENTS
    from backend.tools.analyze_image import analyze_image
    vid = ATTACHMENTS.save(b"fake-mp4-bytes", "video/mp4", filename="x.mp4")
    out = analyze_image(vid.sha256, "what is in here?")
    assert "preprocess_video" in out
    assert "frame" in out.lower()


def test_analyze_image_calls_router_with_attachments(isolated_attachments, monkeypatch):
    from backend.attachments import ATTACHMENTS
    from backend.tools import analyze_image as mod
    img = ATTACHMENTS.save(b"\xff\xd8\xff\xe0fakejpg", "image/jpeg", filename="frame.jpg")

    captured = {}

    class FakeRouter:
        def call(self, task_type, system, user, *, max_tokens=None,
                 temperature=None, attachments=None):
            captured["task_type"] = task_type
            captured["system"] = system
            captured["user"] = user
            captured["attachments"] = list(attachments or [])
            return "x=100 y=200 w=80 h=80 — red logo bottom-left"

    monkeypatch.setattr(mod, "__name__", mod.__name__)  # no-op, keeps lint quiet
    # Patch the lazy import inside the function: the function does
    # `from ..llm import router`. We patch `backend.llm.router`.
    from backend import llm as _llm
    monkeypatch.setattr(_llm, "router", lambda: FakeRouter())

    out = mod.analyze_image(img.sha256, "Where is the logo? Return x,y,w,h.")
    assert "x=100" in out
    assert captured["attachments"] == [img.sha256]
    assert "logo" in captured["user"].lower()


def test_analyze_image_handler_returns_ok_json(isolated_attachments, monkeypatch):
    from backend.attachments import ATTACHMENTS
    img = ATTACHMENTS.save(b"\xff\xd8\xff\xe0jpg", "image/jpeg", filename="f.jpg")

    from backend import llm as _llm
    fake_router = MagicMock()
    fake_router.call.return_value = "x=10 y=20 w=30 h=40"
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import builtin_tools
    out = builtin_tools._analyze_image_handler(img.sha256, "where?")
    data = json.loads(out)
    assert data["ok"] is True
    assert "x=10" in data["answer"]
    assert data["sha256"] == img.sha256


def test_analyze_image_handler_surfaces_error_as_ok_false(isolated_attachments):
    from backend import builtin_tools
    out = builtin_tools._analyze_image_handler("not-a-real-sha", "where?")
    data = json.loads(out)
    assert data["ok"] is False
    assert "not in attachment store" in data["answer"]


def test_analyze_image_tool_is_registered():
    from backend import builtin_tools
    from backend.tool_registry import get_registry
    builtin_tools.register_builtin_tools()
    assert "analyze_image" in get_registry().tools


# ─── _rewrite_xml_tool_call_dump ─────────────────────────────────────


def _fake_step(event: str, tool_name: str):
    s = MagicMock()
    s.event = event
    s.tool_call = MagicMock()
    s.tool_call.name = tool_name
    return s


def test_rewrite_passes_through_plain_answer():
    from backend.unified_agent import _rewrite_xml_tool_call_dump
    agent = MagicMock()
    agent._trace = [_fake_step("tool", "terminal_exec")]
    msg = "All done. The logo is gone."
    assert _rewrite_xml_tool_call_dump(msg, agent) == msg


def test_rewrite_detects_xml_dump_and_rewrites():
    from backend.unified_agent import _rewrite_xml_tool_call_dump
    agent = MagicMock()
    agent._trace = [
        _fake_step("tool", "terminal_exec"),
        _fake_step("tool", "run_python"),
        _fake_step("tool", "run_python"),
    ]
    msg = (
        '<tool_call name="terminal_exec">\n'
        '  <arg name="cmd">ffmpeg ...</arg>\n'
        '</tool_call>'
    )
    out = _rewrite_xml_tool_call_dump(msg, agent)
    assert "iteration ceiling" in out
    assert "terminal_exec" in out  # intended tool name surfaced
    assert "run_python" in out      # what already ran
    assert "<tool_call" not in out  # raw XML stripped from the user-facing text


def test_rewrite_handles_leading_whitespace():
    """The detector must work even when the dump has leading
    whitespace / newlines from the model."""
    from backend.unified_agent import _rewrite_xml_tool_call_dump
    agent = MagicMock()
    agent._trace = []
    msg = '\n\n  <tool_call name="set_setting"><arg name="key">tts.rate</arg></tool_call>'
    out = _rewrite_xml_tool_call_dump(msg, agent)
    assert "iteration ceiling" in out
    assert "set_setting" in out


def test_rewrite_does_not_match_prose_mentioning_tool_call():
    """A plain mention of 'tool_call' in prose should NOT trigger
    the rewrite — only an actual leading XML element does."""
    from backend.unified_agent import _rewrite_xml_tool_call_dump
    agent = MagicMock()
    agent._trace = []
    msg = "I would normally make a tool_call here, but I can't."
    assert _rewrite_xml_tool_call_dump(msg, agent) == msg


def test_rewrite_empty_answer_passthrough():
    from backend.unified_agent import _rewrite_xml_tool_call_dump
    agent = MagicMock()
    agent._trace = []
    assert _rewrite_xml_tool_call_dump("", agent) == ""
    assert _rewrite_xml_tool_call_dump(None, agent) == ""  # type: ignore[arg-type]


# ─── video_overlay_removal skill — discovery ─────────────────────────


def test_video_overlay_removal_skill_loaded():
    """The skill we put back in backend/skills/ must auto-load and
    its triggers must match plain-English logo requests."""
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    sk = skills.SKILLS.get("video_overlay_removal")
    assert sk is not None
    matched = skills.SKILLS.match("remove the logo from this video")
    assert any(s.name == "video_overlay_removal" for s in matched)
    # And Russian.
    matched_ru = skills.SKILLS.match("убери лого из видео")
    assert any(s.name == "video_overlay_removal" for s in matched_ru)


def test_video_overlay_removal_skill_mentions_analyze_image():
    """The body must point the LLM at `analyze_image` for coord
    detection — that's the whole reason we re-added the skill."""
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.ensure_loaded()
    sk = skills.SKILLS.get("video_overlay_removal")
    assert sk is not None
    assert "analyze_image" in sk.body


# ─── max_iterations bumped to 20 ─────────────────────────────────────


def test_run_unified_max_iterations_is_20(monkeypatch):
    """Pin the iteration cap so a future audit doesn't silently
    cut it back down. The skill workflow needs the headroom."""
    import inspect
    from backend import unified_agent
    src = inspect.getsource(unified_agent)
    assert "max_iterations=20" in src
    assert "max_iterations=10" not in src
