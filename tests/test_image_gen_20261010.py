"""generate_image — the Codex-subscription image model, ported from Hermes.

The 2026-09-28 turn asked for room renders and found no image tool: 32 LLM
calls looking for ComfyUI, a headless browser on chatgpt.com, then a question
to the owner about which service to use. These pin the tool's contract
without touching the network.
"""
from __future__ import annotations

import base64
import json
import types

import pytest

from backend.tools import image_gen as ig

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def api(monkeypatch, tmp_path):
    import backend.providers as prov
    calls = []
    auth = types.SimpleNamespace(refreshed=0)

    def _get():
        return (f"tok-{auth.refreshed}", "acc")

    def _force():
        auth.refreshed += 1
        return _get()

    monkeypatch.setattr(prov, "CODEX_AUTH", types.SimpleNamespace(
        get_access_token=_get, force_refresh=_force))
    monkeypatch.setattr(ig, "_base_url", lambda: "https://codex.test")

    class _WS:
        def save_outbox(self, name, data):
            p = tmp_path / name
            p.write_bytes(data)
            return p

    import backend.workspace as ws
    monkeypatch.setattr(ws, "get_workspace", lambda: _WS())
    responses = []

    def _post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "body": json, "headers": headers})
        return responses.pop(0)

    import httpx
    monkeypatch.setattr(httpx, "post", _post)
    return calls, responses, auth


def _ok():
    return _Resp(200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}],
                       "size": "1024x1024"}, {"x-codex-imagegen-request-id": "r1"})


def test_a_prompt_alone_goes_to_generations(api):
    calls, responses, _ = api
    responses.append(_ok())
    out = ig.generate_image("a loft with a concrete ceiling", aspect="landscape")
    assert out["ok"] is True
    assert calls[0]["url"] == "https://codex.test/images/generations"
    assert calls[0]["body"]["model"] == "gpt-image-2"
    assert calls[0]["body"]["size"] == "1536x1024"
    assert out["media_line"].startswith("MEDIA:")
    assert out["mode"] == "generate"


def test_photos_go_to_edits_as_data_urls(api, tmp_path):
    calls, responses, _ = api
    photo = tmp_path / "room.png"
    photo.write_bytes(PNG)
    responses.append(_ok())
    out = ig.generate_image("keep the concrete ceiling, warm loft",
                            input_images=[str(photo)])
    assert out["ok"] and out["mode"] == "edit"
    assert calls[0]["url"].endswith("/images/edits")
    assert calls[0]["body"]["images"][0]["image_url"].startswith("data:image/png;base64,")


def test_the_agent_identifies_itself_not_as_the_codex_client(api):
    calls, responses, _ = api
    responses.append(_ok())
    ig.generate_image("x")
    h = calls[0]["headers"]
    assert h["originator"] == "hrant-agent"
    assert h["User-Agent"].startswith("HrantAgent/")
    assert h["ChatGPT-Account-ID"] == "acc"


def test_a_401_refreshes_the_login_once(api):
    calls, responses, auth = api
    responses.extend([_Resp(401, {"error": {"message": "token_expired"}}), _ok()])
    out = ig.generate_image("x")
    assert out["ok"] is True
    assert auth.refreshed == 1
    assert [c["headers"]["Authorization"] for c in calls] == ["Bearer tok-0", "Bearer tok-1"]


def test_an_api_error_is_reported_not_raised(api):
    _calls, responses, _ = api
    responses.append(_Resp(400, {"error": {"message": "content policy"}}))
    out = ig.generate_image("x")
    assert out["ok"] is False and "content policy" in out["error"]


def test_bad_inputs_are_refused_before_spending_a_call(api, tmp_path):
    calls, _responses, _ = api
    junk = tmp_path / "notes.txt"
    junk.write_text("hello")
    assert ig.generate_image("")["ok"] is False
    assert ig.generate_image("x", input_images=[str(junk)])["ok"] is False
    assert ig.generate_image("x", input_images=["/no/such/file.png"])["ok"] is False
    assert calls == []


def test_guests_cannot_spend_the_owners_subscription(monkeypatch):
    from backend import builtin_tools as bt
    import backend.roles as roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:stranger")
    monkeypatch.setattr(roles, "is_at_least", lambda sp, r: False)
    out = json.loads(bt._generate_image_handler(prompt="x"))
    assert out["ok"] is False and "permission" in out["error"]


def test_the_tool_is_always_visible():
    from backend.tool_bundles import BASE_TOOLS
    from backend.tool_registry import get_registry
    assert "generate_image" in BASE_TOOLS
    assert "generate_image" in get_registry().names()
