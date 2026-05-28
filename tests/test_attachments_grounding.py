"""Regression: when the user attaches an image/audio/file, the textual
prompt sent to classifier / thinker / solver / chat must say so.

Without this, the model only sees the user's question text — and a
short question like "what do you think about that?" plus prior
conversation history naturally biases the answer toward whatever was
last discussed, ignoring the actual attachment.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import Agent


def _attach_one_image_meta(monkeypatch):
    """Stub ATTACHMENTS so _attachment_marker classifies the SHA as image
    without writing real bytes. Returns the SHA list to pass via run()."""
    fake_sha = "a" * 64

    class _FakeMeta:
        kind = "image"

    class _FakeStore:
        def get_meta(self, sha):
            return _FakeMeta() if sha == fake_sha else None

    monkeypatch.setattr("backend.agent.ATTACHMENTS", _FakeStore(), raising=False)
    # The marker also imports ATTACHMENTS lazily inside the helper:
    import backend.attachments as _a_mod
    monkeypatch.setattr(_a_mod, "ATTACHMENTS", _FakeStore())
    return [fake_sha]


def test_attachment_marker_is_empty_without_attachments():
    agent = Agent()
    agent._attachments = None
    assert agent._attachment_marker() == ""
    agent._attachments = []
    assert agent._attachment_marker() == ""


def test_attachment_marker_describes_image(monkeypatch):
    shas = _attach_one_image_meta(monkeypatch)
    agent = Agent()
    agent._attachments = shas
    marker = agent._attachment_marker()
    assert "ATTACHMENT NOTICE" in marker
    assert "1 image" in marker
    assert "look at" in marker.lower()


