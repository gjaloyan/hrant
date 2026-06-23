"""frame_problem captures a component map + scope and returns ask_user-ready
scope options, persisting a durable frame artifact."""
from __future__ import annotations

import json
import importlib
import pytest


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, knowledge_manager, builtin_tools
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    importlib.reload(builtin_tools)
    monkeypatch.setattr(builtin_tools, "_check_owner",
                        lambda *a, **k: (False, "webui:default"))
    return builtin_tools


def test_frame_problem_persists_and_returns_scope_options(tools, tmp_path):
    out = json.loads(tools._frame_problem_handler(
        title="Online shop",
        domain="ecommerce",
        components=[
            {"name": "catalog", "role": "list products", "mvp": True,
             "source": "baymard", "confidence": "high"},
            {"name": "payments", "role": "take money", "mvp": False,
             "source": "stripe docs", "confidence": "med"},
        ],
        proposed_scope="MVP: catalog + cart + checkout; defer payments/auth.",
        open_questions=["Real payments or stubbed?"],
    ))
    assert out["ok"] is True
    assert out["frame_id"].startswith("frame_")
    # MVP vs fuller scope options, ready for ask_user
    labels = [o["label"] for o in out["scope_options"]]
    assert any("MVP" in l for l in labels)
    # component fields normalized
    comp = out["frame"]["components"][0]
    assert comp["mvp"] is True and comp["confidence"] == "high"
    # durable artifact on disk
    frames = list((tmp_path / "workspace" / "frames").glob("*.json"))
    assert len(frames) == 1
    saved = json.loads(frames[0].read_text(encoding="utf-8"))
    assert saved["title"] == "Online shop"
    assert saved["open_questions"] == ["Real payments or stubbed?"]


def test_frame_problem_owner_gated(tools):
    import backend.builtin_tools as bt
    bt._check_owner = lambda *a, **k: ("refused", None)
    out = json.loads(bt._frame_problem_handler(title="x", components=[{"name": "y"}]))
    assert out["ok"] is False


def test_frame_problem_skips_nameless_components(tools):
    out = json.loads(tools._frame_problem_handler(
        title="t", components=[{"role": "no name"}, {"name": "real"}]))
    assert [c["name"] for c in out["frame"]["components"]] == ["real"]
